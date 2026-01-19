import pathlib

import click
import lightning as pl
import torch

import modules.AP_detector
import modules.g2p
from modules.utils.export_tool import Exporter
from modules.utils.post_processing import post_processing
from train import LitForcedAlignmentTask


@click.command()
@click.option(
    "--ckpt",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the checkpoint",
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--mode", "-m", default="force", type=click.Choice(["force", "match"])
)  # TODO: add asr mode
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
    help="Probability (0.0-1.0) of splitting each individual diphthong instance. "
         "Each diphthong is independently selected for splitting with this probability.",
)
@click.option(
    "--split_dict",
    type=str,
    default="dictionary/vowel_split_example.txt",
    show_default=True,
    help="Path to the diphthong split rules dictionary (used with --split_all_diphthong or --split_diphthong_rate).",
)
def main(
        ckpt,
        folder,
        mode,
        g2p,
        ap_detector,
        in_format,
        out_formats,
        save_confidence,
        split_all_diphthong,
        split_diphthong_rate,
        split_dict,
        **kwargs,
):
    # Handle diphthong splitting mode
    diphthong_split_mode = None
    diphthong_mapping = None
    
    if split_all_diphthong or split_diphthong_rate is not None:
        from modules.utils.diphthong_split import load_split_rules, build_diphthong_mapping
        
        split_rules = load_split_rules(split_dict)
        if not split_rules:
            print(f"Warning: No split rules loaded from {split_dict}")
        else:
            if split_all_diphthong:
                diphthong_split_mode = "all"
                print(f"Diphthong splitting: all qualifying diphthongs will be split ({len(split_rules)} rules)")
            elif split_diphthong_rate is not None:
                if not 0.0 <= split_diphthong_rate <= 1.0:
                    raise ValueError("--split_diphthong_rate must be between 0.0 and 1.0")
                diphthong_split_mode = "rate"
                print(f"Diphthong splitting: {split_diphthong_rate*100:.1f}% of qualifying diphthongs will be split")
    
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    out_formats = [i.strip().lower() for i in out_formats.split(",")]

    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    
    # If diphthong splitting is enabled, set up the mapping using model's vocab
    if diphthong_split_mode and split_rules:
        diphthong_mapping = build_diphthong_mapping(model.vocab, split_rules)
        model.set_diphthong_split_mode(diphthong_split_mode, diphthong_mapping, split_diphthong_rate)
    
    model.set_inference_mode(mode)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions = get_AP.process(predictions)
    predictions, log = post_processing(predictions)
    exporter = Exporter(predictions, log)

    if save_confidence:
        out_formats.append('confidence')

    exporter.export(out_formats)

    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
