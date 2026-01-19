import os
import pathlib

import click
import numpy as np
import onnxruntime as ort
import torchaudio
import yaml
from tqdm import tqdm

import modules.AP_detector
import modules.g2p
import numba

from modules.utils.export_tool import Exporter
from modules.utils.post_processing import post_processing


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_inference(session, waveform, num_frames, ph_seq_id):
    output_names = [output.name for output in session.get_outputs()]

    input_data = {
        'waveform': waveform,
        'num_frames': np.array(num_frames, dtype=np.int64),
        'ph_seq_id': ph_seq_id
    }

    # 运行推理
    try:
        results = session.run(output_names, input_data)
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        raise

    # 将结果转换为字典形式
    output_dict = {name: result for name, result in zip(output_names, results)}

    return output_dict


def create_session(onnx_model_path):
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'
                 ]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
    except Exception as e:
        print(f"An error occurred while creating ONNX Runtime session: {e}")
        raise

    return session


@numba.jit
def forward_pass(T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
                 prob3_pad_len):
    for t in range(1, T):
        # [t-1,s] -> [t,s]
        prob1 = dp[t - 1, :] + prob_log[t, :] + not_edge_prob_log[t]

        prob2 = np.empty(S, dtype=np.float32)
        prob2[0] = -np.inf
        for i in range(1, S):
            prob2[i] = (
                    dp[t - 1, i - 1]
                    + prob_log[t, i - 1]
                    + edge_prob_log[t]
                    + curr_ph_max_prob_log[i - 1] * (T / S)
            )

        # [t-1,s-2] -> [t,s]
        prob3 = np.empty(S, dtype=np.float32)
        for i in range(prob3_pad_len):
            prob3[i] = -np.inf
        for i in range(prob3_pad_len, S):
            if i - prob3_pad_len + 1 < S - 1 and ph_seq_id[i - prob3_pad_len + 1] != 0:
                prob3[i] = -np.inf
            else:
                prob3[i] = (
                        dp[t - 1, i - prob3_pad_len]
                        + prob_log[t, i - prob3_pad_len]
                        + edge_prob_log[t]
                        + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                )

        stacked_probs = np.empty((3, S), dtype=np.float32)
        for i in range(S):
            stacked_probs[0, i] = prob1[i]
            stacked_probs[1, i] = prob2[i]
            stacked_probs[2, i] = prob3[i]

        for i in range(S):
            max_idx = 0
            max_val = stacked_probs[0, i]
            for j in range(1, 3):
                if stacked_probs[j, i] > max_val:
                    max_val = stacked_probs[j, i]
                    max_idx = j
            dp[t, i] = max_val
            backtrack_s[t, i] = max_idx

        for i in range(S):
            if backtrack_s[t, i] == 0:
                curr_ph_max_prob_log[i] = max(curr_ph_max_prob_log[i], prob_log[t, i])
            elif backtrack_s[t, i] > 0:
                curr_ph_max_prob_log[i] = prob_log[t, i]

        for i in range(S):
            if ph_seq_id[i] == 0:
                curr_ph_max_prob_log[i] = 0

    return dp, backtrack_s, curr_ph_max_prob_log


def decode(ph_seq_id, ph_prob_log, edge_prob):
    # ph_seq_id: (S)
    # ph_prob_log: (T, vocab_size)
    # edge_prob: (T,2)
    T = ph_prob_log.shape[0]
    S = len(ph_seq_id)
    # not_SP_num = (ph_seq_id > 0).sum()
    prob_log = ph_prob_log[:, ph_seq_id]

    edge_prob_log = np.log(edge_prob + 1e-6).astype("float32")
    not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float32")

    # init
    curr_ph_max_prob_log = np.full(S, -np.inf)
    dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
    backtrack_s = np.full_like(dp, -1, dtype="int32")

    dp[0, 0] = prob_log[0, 0]
    curr_ph_max_prob_log[0] = prob_log[0, 0]
    if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
        dp[0, 1] = prob_log[0, 1]
        curr_ph_max_prob_log[1] = prob_log[0, 1]

    # forward
    prob3_pad_len = 2 if S >= 2 else 1
    dp, backtrack_s, curr_ph_max_prob_log = forward_pass(
        T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
        prob3_pad_len
    )

    # backward
    ph_idx_seq = []
    ph_time_int = []
    frame_confidence = []

    # 如果mode==forced，只能从最后一个音素或者SP结束
    if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
        s = S - 2
    else:
        s = S - 1

    for t in np.arange(T - 1, -1, -1):
        assert backtrack_s[t, s] >= 0 or t == 0
        frame_confidence.append(dp[t, s])
        if backtrack_s[t, s] != 0:
            ph_idx_seq.append(s)
            ph_time_int.append(t)
            s -= backtrack_s[t, s]
    ph_idx_seq.reverse()
    ph_time_int.reverse()
    frame_confidence.reverse()
    frame_confidence = np.exp(
        np.diff(
            np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
        )
    )

    return (
        np.array(ph_idx_seq),
        np.array(ph_time_int),
        np.array(frame_confidence),
    )


@click.command()
@click.option(
    "--onnx",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the onnx",
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class"
)
@click.option(
    "--ap_detector",
    "-a",
    default="LoudnessSpectralcentroidAPDetector",  # "NoneAPDetector",
    type=str,
    help="name of the AP detector class",
)
@click.option(
    "--in_format",
    "-if",
    default="lab",
    required=False,
    type=str,
    help="File extension of input transcriptions. Default: lab",
)
@click.option(
    "--out_formats",
    "-of",
    default="textgrid,htk,trans",
    required=False,
    type=str,
    help="Types of output file, separated by comma. Supported types:"
         "textgrid(praat),"
         " htk(lab,nnsvs,sinsy),"
         " transcriptions.csv(diffsinger,trans,transcription,transcriptions)",
)
@click.option(
    "--save_confidence",
    "-sc",
    is_flag=True,
    default=False,
    show_default=True,
    help="save confidence.csv",
)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
@click.option(
    "--split_all_diphthong",
    is_flag=True,
    default=False,
    show_default=True,
    help="Split all compound vowels (diphthongs) into their component vowels during inference.",
)
@click.option(
    "--split_diphthong_rate",
    type=float,
    default=None,
    help="Probability (0.0-1.0) of splitting each individual diphthong instance.",
)
@click.option(
    "--split_dict",
    type=str,
    default="dictionary/vowel_split_example.txt",
    show_default=True,
    help="Path to the diphthong split rules dictionary.",
)
@click.option(
    "--modify_type",
    type=click.Choice(["tg", "csv"]),
    default=None,
    help="Modify existing annotation files instead of running inference. "
         "'tg' for TextGrid, 'csv' for transcriptions.csv.",
)
@click.option(
    "--combine_all_diphthong",
    is_flag=True,
    default=False,
    show_default=True,
    help="When modifying annotations, combine all split vowel sequences back into compound vowels.",
)
@click.option(
    "--batch_subfolders",
    is_flag=True,
    default=False,
    show_default=True,
    help="Process each subdirectory of the input folder as a separate batch.",
)
def infer(onnx,
          folder,
          g2p,
          ap_detector,
          in_format,
          out_formats,
          save_confidence,
          split_all_diphthong,
          split_diphthong_rate,
          split_dict,
          modify_type,
          combine_all_diphthong,
          batch_subfolders,
          **kwargs, ):
    folder_path = pathlib.Path(folder)
    
    # Handle annotation modification mode (modifying existing files)
    if modify_type is not None:
        from modules.utils.diphthong_split import load_split_rules
        from modules.utils.annotation_modifier import process_folder_annotations
        
        split_rules = load_split_rules(split_dict)
        if not split_rules:
            raise ValueError(f"No split rules loaded from {split_dict}")
        
        # Determine the modification mode
        if combine_all_diphthong:
            mod_mode = "combine_all"
            print(f"Combining all split vowels back into compound vowels...")
        elif split_all_diphthong:
            mod_mode = "split_all"
            print(f"Splitting all compound vowels...")
        elif split_diphthong_rate is not None:
            if not 0.0 <= split_diphthong_rate <= 1.0:
                raise ValueError("--split_diphthong_rate must be between 0.0 and 1.0")
            mod_mode = "split_rate"
            print(f"Splitting {split_diphthong_rate*100:.1f}% of compound vowels...")
        else:
            raise ValueError("When using --modify_type, you must also specify --combine_all_diphthong, "
                           "--split_all_diphthong, or --split_diphthong_rate")
        
        # Process folders
        if batch_subfolders:
            subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
            print(f"Batch processing {len(subfolders)} subdirectories...")
            for subfolder in subfolders:
                print(f"\nProcessing: {subfolder}")
                process_folder_annotations(
                    subfolder, modify_type, split_rules, mod_mode,
                    split_diphthong_rate or 1.0, recursive=True
                )
        else:
            process_folder_annotations(
                folder_path, modify_type, split_rules, mod_mode,
                split_diphthong_rate or 1.0, recursive=True
            )
        
        print("\nAnnotation modification complete.")
        return
    
    # Standard ONNX inference mode
    config_file = pathlib.Path(onnx).with_name('config.yaml')
    assert os.path.exists(onnx), f"Onnx file does not exist: {onnx}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    config = load_config_from_yaml(config_file)
    melspec_config = config['melspec_config']
    
    # Load split rules if diphthong splitting is enabled
    split_rules = None
    if split_all_diphthong or split_diphthong_rate is not None:
        from modules.utils.diphthong_split import load_split_rules
        split_rules = load_split_rules(split_dict)
        if split_rules:
            if split_all_diphthong:
                print(f"Diphthong splitting: all qualifying diphthongs will be split ({len(split_rules)} rules)")
            else:
                print(f"Diphthong splitting: {split_diphthong_rate*100:.1f}% of qualifying diphthongs will be split")
        else:
            print(f"Warning: No split rules loaded from {split_dict}")
    
    def run_onnx_inference_on_folder(target_folder):
        session = create_session(onnx)
        
        if not g2p.endswith("G2P"):
            g2p_name = g2p + "G2P"
        else:
            g2p_name = g2p
        g2p_class = getattr(modules.g2p, g2p_name)
        grapheme_to_phoneme = g2p_class(**kwargs)
        out_format_list = [i.strip().lower() for i in out_formats.split(",")]

        if not ap_detector.endswith("APDetector"):
            ap_name = ap_detector + "APDetector"
        else:
            ap_name = ap_detector
        AP_detector_class = getattr(modules.AP_detector, ap_name)
        get_AP = AP_detector_class(**kwargs)

        grapheme_to_phoneme.set_in_format(in_format)
        dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(target_folder).rglob("*.wav"))
        predictions = []

        for i in tqdm(range(len(dataset)), desc="Processing", unit="sample"):
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = dataset[i]

            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform[0][None, :][0]
            if sr != melspec_config['sample_rate']:
                waveform = torchaudio.transforms.Resample(sr, melspec_config['sample_rate'])(waveform)

            wav_length = waveform.shape[0] / melspec_config["sample_rate"]
            ph_seq_id = np.array([config['vocab'][ph] for ph in ph_seq], dtype=np.int64)
            num_frames = int(
                (wav_length * melspec_config["scale_factor"] * melspec_config["sample_rate"] + 0.5) / melspec_config[
                    "hop_length"]
            )
            results = run_inference(session, [waveform.numpy()], num_frames, [ph_seq_id])

            edge_diff = results['edge_diff']
            edge_prob = results['edge_prob']
            ph_prob_log = results['ph_prob_log']
            T = results['T']

            ph_idx_seq, ph_time_int_pred, frame_confidence = decode(ph_seq_id, ph_prob_log, edge_prob, )
            total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

            # postprocess
            frame_length = melspec_config["hop_length"] / (
                    melspec_config["sample_rate"] * melspec_config["scale_factor"]
            )
            ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
            ph_time_pred = frame_length * (
                np.concatenate(
                    [
                        ph_time_int_pred.astype("float32") + ph_time_fractional,
                        [T],
                    ]
                )
            )
            ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

            ph_seq_pred = []
            ph_intervals_pred = []
            word_seq_pred = []
            word_intervals_pred = []

            word_idx_last = -1
            for j, ph_idx in enumerate(ph_idx_seq):
                if ph_seq[ph_idx] == "SP":
                    continue
                ph_seq_pred.append(ph_seq[ph_idx])
                ph_intervals_pred.append(ph_intervals[j, :])

                word_idx = ph_idx_to_word_idx[ph_idx]
                if word_idx == word_idx_last:
                    word_intervals_pred[-1][1] = ph_intervals[j, 1]
                else:
                    word_seq_pred.append(word_seq[word_idx])
                    word_intervals_pred.append([ph_intervals[j, 0], ph_intervals[j, 1]])
                    word_idx_last = word_idx
            
            ph_seq_pred = list(ph_seq_pred)
            ph_intervals_pred = [list(interval) for interval in ph_intervals_pred]
            
            # Apply diphthong splitting if enabled
            if split_rules and (split_all_diphthong or split_diphthong_rate is not None):
                from modules.utils.diphthong_split import apply_split_to_annotations
                
                if split_all_diphthong:
                    ph_seq_pred, ph_intervals_pred = apply_split_to_annotations(
                        ph_seq_pred, [(s, e) for s, e in ph_intervals_pred], 
                        split_rules, "all"
                    )
                elif split_diphthong_rate is not None:
                    ph_seq_pred, ph_intervals_pred = apply_split_to_annotations(
                        ph_seq_pred, [(s, e) for s, e in ph_intervals_pred],
                        split_rules, "rate", split_diphthong_rate
                    )
            
            ph_seq_pred = np.array(ph_seq_pred)
            ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
            word_seq_pred = np.array(word_seq_pred)
            word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

            predictions.append((wav_path,
                                wav_length,
                                total_confidence,
                                ph_seq_pred,
                                ph_intervals_pred,
                                word_seq_pred,
                                word_intervals_pred))

        predictions = get_AP.process(predictions)
        predictions, log = post_processing(predictions)
        exporter = Exporter(predictions, log)

        if save_confidence:
            out_format_list.append('confidence')

        exporter.export(out_format_list)
    
    if batch_subfolders:
        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
        print(f"Batch processing {len(subfolders)} subdirectories...")
        for subfolder in subfolders:
            print(f"\nProcessing: {subfolder}")
            run_onnx_inference_on_folder(subfolder)
    else:
        run_onnx_inference_on_folder(folder_path)
    
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == '__main__':
    infer()
