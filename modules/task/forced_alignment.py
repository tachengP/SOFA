from typing import Any
import random

import lightning as pl
import numpy as np
import numba
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import rearrange, repeat

import modules.scheduler as scheduler_module
from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling
from modules.loss.BinaryEMDLoss import BinaryEMDLoss
from modules.loss.GHMLoss import CTCGHMLoss, GHMLoss, MultiLabelGHMLoss
from modules.utils.get_melspec import MelSpecExtractor
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_valid
from modules.utils.diphthong_split import apply_split_to_annotations


@numba.jit
def forward_pass(
    T,
    S,
    prob_log,
    not_edge_prob_log,
    edge_prob_log,
    curr_ph_max_prob_log,
    dp,
    backtrack_s,
    ph_seq_id,
    prob3_pad_len,
):
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


class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
        self,
        vocab_text,
        model_config,
        melspec_config,
        optimizer_config,
        loss_config,
        data_augmentation_enabled,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = yaml.safe_load(vocab_text)

        self.backbone = UNetBackbone(
            melspec_config["n_mels"],
            model_config["hidden_dims"],
            model_config["hidden_dims"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["down_sampling_factor"],  # 3
            down_sampling_times=model_config["down_sampling_times"],  # 7
            channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
        )
        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 2
        )
        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.pseudo_label_ratio = loss_config["function"]["pseudo_label_ratio"]
        self.pseudo_label_auto_theshold = 0.5

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_EMD_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "consistency_loss",
            "pseudo_label_loss",
            "compound_vowel_loss",
            "total_loss",
        ]
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(
                        max_steps=optimizer_config["total_steps"]
                    )
                )
            else:
                self.losses_schedulers.append(scheduler_module.NoneScheduler())
        self.data_augmentation_enabled = data_augmentation_enabled

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(
            self.vocab["<vocab_size>"],
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )
        self.pseudo_label_GHM_loss_fn = MultiLabelGHMLoss(
            self.vocab["<vocab_size>"],
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )
        self.ph_edge_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.EMD_loss_fn = BinaryEMDLoss()
        self.ph_edge_diff_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=1 - 1e-3)
        self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"
        
        # Diphthong split settings (for inference)
        self.diphthong_split_mode = None
        self.diphthong_mapping = None
        self.diphthong_split_rate = None
        self.diphthong_split_rules = None  # phoneme name -> component names (for pre-inference transformation)
        
        # Split rules for validation visualization
        self.split_rules = None

    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["<vocab_size>"] == pretrained_model.vocab["<vocab_size>"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["<vocab_size>"] + 2
            )

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        # resume loss schedulers
        for scheduler in self.losses_schedulers:
            scheduler.resume(self.global_step)
        self.losses_weights = self.losses_weights.to(self.device)

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(
            self.device
        )

    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
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
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = prob_log[0, 0]
            curr_ph_max_prob_log[0] = prob_log[0, 0]
            if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
                dp[0, 1] = prob_log[0, 1]
                curr_ph_max_prob_log[1] = prob_log[0, 1]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            for i, ph_id in enumerate(ph_seq_id):
                dp[0, i] = prob_log[0, i]
                curr_ph_max_prob_log[i] = prob_log[0, i]

        # forward
        prob3_pad_len = 2 if S >= 2 else 1
        dp, backtrack_s, curr_ph_max_prob_log = forward_pass(
            T,
            S,
            prob_log,
            not_edge_prob_log,
            edge_prob_log,
            curr_ph_max_prob_log,
            dp,
            backtrack_s,
            ph_seq_id,
            prob3_pad_len,
        )

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []
        # 如果mode==forced，只能从最后一个音素或者SP结束
        if self.inference_mode == "force":
            if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
                s = S - 2
            else:
                s = S - 1
        # 如果mode==match，可以从任意音素结束
        elif self.inference_mode == "match":
            s = np.argmax(dp[-1, :])
        else:
            raise ValueError("inference_mode must be 'force' or 'match'")

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

    def _infer_once(
        self,
        melspec,
        wav_length,
        ph_seq,
        word_seq=None,
        ph_idx_to_word_idx=None,
        return_ctc=False,
        return_plot=False,
    ):
        ph_seq_id = np.array([self.vocab[ph] for ph in ph_seq])
        ph_mask = np.zeros(self.vocab["<vocab_size>"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1
        ph_mask = torch.from_numpy(ph_mask)
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        # forward
        with torch.no_grad():
            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))
        if wav_length is not None:
            num_frames = int(
                (
                    (
                        wav_length
                        * self.melspec_config["scale_factor"]
                        * self.melspec_config["sample_rate"]
                        + 0.5
                    )
                )
                / self.melspec_config["hop_length"]
            )
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ph_edge_logits = ph_edge_logits[:, :num_frames]
            ctc_logits = ctc_logits[:, :num_frames, :]

        ph_mask = (
            ph_mask.to(torch.float32)
            .to(ph_frame_logits.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .logical_not()
            * 1e9
        )
        ph_frame_pred = (
            torch.nn.functional.softmax(
                ph_frame_logits.float() - ph_mask.float(), dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_prob_log = (
            torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_edge_pred = (
            (torch.nn.functional.sigmoid(ph_edge_logits.float()) - 0.1) / 0.8
        ).clamp(0.0, 1.0)
        ph_edge_pred = ph_edge_pred.squeeze(0).cpu().numpy().astype("float32")
        ctc_logits = (
            ctc_logits.float().squeeze(0).cpu().numpy().astype("float32")
        )  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)
        edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)
        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
            edge_prob,
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        # postprocess
        frame_length = self.melspec_config["hop_length"] / (
            self.melspec_config["sample_rate"] * self.melspec_config["scale_factor"]
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
        for i, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP":
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[i, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[i, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[i, 0], ph_intervals[i, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = np.argmax(ctc_logits, axis=-1)
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        fig = None
        ph_intervals_pred_int = (
            (ph_intervals_pred / frame_length).round().astype("int32")
        )
        if return_plot:
            ph_idx_frame = np.zeros(T).astype("int32")
            last_ph_idx = 0
            for ph_idx, ph_time in zip(ph_idx_seq, ph_time_int_pred):
                ph_idx_frame[ph_time] += ph_idx - last_ph_idx
                last_ph_idx = ph_idx
            ph_idx_frame = np.cumsum(ph_idx_frame)
            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_intervals": ph_intervals_pred_int,
                "frame_confidence": frame_confidence,
                "ph_frame_prob": ph_frame_pred[:, ph_seq_id],
                "ph_frame_id_gt": ph_idx_frame,
                "edge_prob": edge_prob,
            }
            fig = plot_for_valid(**args)

        return (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            total_confidence,
            ctc,
            fig,
        )

    def set_inference_mode(self, mode):
        self.inference_mode = mode
    
    def set_diphthong_split_mode(self, mode, mapping, rate=None, split_rules=None):
        """
        Set the diphthong split mode for inference.
        
        Args:
            mode: Either "all" (split all diphthongs) or "rate" (probabilistic splitting)
            mapping: Dictionary mapping compound vowel IDs to component vowel IDs
            rate: Split probability when mode is "rate" (0.0-1.0)
            split_rules: Dictionary mapping compound phoneme names to component phoneme names
        """
        self.diphthong_split_mode = mode
        self.diphthong_mapping = mapping
        self.diphthong_split_rate = rate
        self.diphthong_split_rules = split_rules

    def set_split_rules(self, split_rules):
        """
        Set split rules for validation visualization.
        
        Args:
            split_rules: Dictionary mapping compound vowels to their component vowels
        """
        self.split_rules = split_rules

    def on_predict_start(self):
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

    def _transform_ph_seq_for_splitting(self, ph_seq, word_seq, ph_idx_to_word_idx):
        """
        Transform phoneme sequence BEFORE inference by replacing compound vowels 
        with their component sequences. This allows the model to predict actual 
        boundaries for the components based on acoustic features.
        
        Args:
            ph_seq: List of phoneme names
            word_seq: List of word names
            ph_idx_to_word_idx: Mapping from phoneme index to word index
            
        Returns:
            Tuple of (new_ph_seq, new_word_seq, new_ph_idx_to_word_idx)
        """
        if self.diphthong_split_mode is None or self.diphthong_split_rules is None:
            return ph_seq, word_seq, ph_idx_to_word_idx
        
        # Find which phonemes can be split (check if components exist in vocab)
        splittable = {}
        for ph in set(ph_seq):
            if ph in self.diphthong_split_rules:
                components = self.diphthong_split_rules[ph]
                # Check if all components exist in vocabulary
                all_exist = all(comp in self.vocab for comp in components)
                if all_exist:
                    splittable[ph] = components
        
        if not splittable:
            return ph_seq, word_seq, ph_idx_to_word_idx
        
        # Determine which phoneme instances to actually split
        split_indices = set()
        for idx, ph in enumerate(ph_seq):
            if ph in splittable:
                if self.diphthong_split_mode == "all":
                    split_indices.add(idx)
                elif self.diphthong_split_mode == "rate":
                    if random.random() < self.diphthong_split_rate:
                        split_indices.add(idx)
        
        if not split_indices:
            return ph_seq, word_seq, ph_idx_to_word_idx
        
        # Build new sequences
        new_ph_seq = []
        new_ph_idx_to_word_idx = []
        
        for idx, ph in enumerate(ph_seq):
            if idx in split_indices:
                components = splittable[ph]
                word_idx = ph_idx_to_word_idx[idx] if ph_idx_to_word_idx is not None else idx
                for comp in components:
                    new_ph_seq.append(comp)
                    new_ph_idx_to_word_idx.append(word_idx)
            else:
                new_ph_seq.append(ph)
                if ph_idx_to_word_idx is not None:
                    new_ph_idx_to_word_idx.append(ph_idx_to_word_idx[idx])
                else:
                    new_ph_idx_to_word_idx.append(idx)
        
        return new_ph_seq, word_seq, np.array(new_ph_idx_to_word_idx)

    def predict_step(self, batch, batch_idx):
        try:
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
            waveform = load_wav(
                wav_path, self.device, self.melspec_config["sample_rate"]
            )
            wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
            melspec = self.get_melspec(waveform).detach().unsqueeze(0)
            melspec = (melspec - melspec.mean()) / melspec.std()
            melspec = repeat(
                melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
            )
            
            # Transform phoneme sequence BEFORE inference if diphthong splitting is enabled
            # This allows the model to predict actual boundaries for component phonemes
            if self.diphthong_split_mode is not None:
                ph_seq, word_seq, ph_idx_to_word_idx = self._transform_ph_seq_for_splitting(
                    ph_seq, word_seq, ph_idx_to_word_idx
                )

            (
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
                confidence,
                _,
                _,
            ) = self._infer_once(
                melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, False, False
            )

            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        except Exception as e:
            e.args += (f"{str(wav_path)}",)
            raise e

    def _get_full_label_loss(
        self,
        ph_frame_logits,
        ph_edge_logits,
        ph_frame_gt,
        ph_edge_gt,
        input_feature_lengths,
        ph_mask,
        valid,
    ):
        T = ph_frame_logits.shape[1]

        # ph_frame_prob_gt = nn.functional.one_hot(
        #     ph_frame_gt.long(), num_classes=self.vocab["<vocab_size>"]
        # ).float()

        # calculate mask matrix
        # (B, T)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=ph_frame_logits.shape[0])
        mask = (mask < input_feature_lengths.unsqueeze(1)).to(ph_frame_logits.dtype)

        # ph_frame_loss
        # print((mask.unsqueeze(-1) * ph_mask.unsqueeze(1)).shape, ph_frame_pred.shape)
        ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
            ph_frame_logits,
            ph_frame_gt,
            (mask.unsqueeze(-1) * ph_mask.unsqueeze(1)),
            valid,
        )

        # ph_edge loss
        # BCE_GHM loss
        ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(
            ph_edge_logits.unsqueeze(-1), ph_edge_gt.unsqueeze(-1), mask, valid
        )

        # EMD loss
        ph_edge_pred = torch.nn.functional.sigmoid(ph_edge_logits.float())
        ph_edge_EMD_loss = self.EMD_loss_fn(ph_edge_pred * mask, ph_edge_gt * mask)

        # diff loss
        ph_edge_diff_loss = self.ph_edge_diff_GHM_loss_fn(
            (torch.diff(ph_edge_logits, 1, dim=-1) + 1).unsqueeze(-1) / 2,
            (torch.diff(ph_edge_gt, 1, dim=-1) + 1).unsqueeze(-1) / 2,
            mask[:, 1:],
            valid,
        )
        return ph_frame_GHM_loss, ph_edge_GHM_loss, ph_edge_EMD_loss, ph_edge_diff_loss

    def _get_weak_label_loss(
        self,
        ctc_logits,
        ph_mask,
        ph_seq_gt,
        ph_seq_lengths_gt,
        input_feature_lengths,
        valid,
    ):
        ctc_logits = ctc_logits - ph_mask.unsqueeze(1).logical_not().float() * 1e9
        log_probs_pred = nn.functional.log_softmax(ctc_logits, dim=-1)
        # ctc loss
        log_probs_pred = rearrange(log_probs_pred, "B T C -> T B C")
        ctc_GHM_loss = self.CTC_GHM_loss_fn(
            log_probs_pred,
            ph_seq_gt,
            input_feature_lengths,
            ph_seq_lengths_gt,
            valid,
        )

        return ctc_GHM_loss

    def _get_consistency_loss(
        self, ph_frame_logits, ph_edge_logits, input_feature_lengths
    ):
        output_tensors = torch.cat(
            [ph_frame_logits, ph_edge_logits.unsqueeze(-1)], dim=-1
        )
        output_tensors = torch.nn.functional.sigmoid(output_tensors.float())
        B = output_tensors.shape[0]
        T = output_tensors.shape[1]

        # calculate mask matrix
        # (B//2, T, 1)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=B // 2)
        mask = (
            (mask < input_feature_lengths[: B // 2].unsqueeze(1))
            .to(torch.bool)
            .unsqueeze(-1)
        )

        # consistency loss
        consistency_loss = self.MSE_loss_fn(
            output_tensors[: B // 2, :, :] * mask,
            output_tensors[B // 2 :, :, :] * mask,
        )

        return consistency_loss

    def _get_pseudo_label_loss(self, ph_frame_logits, input_feature_lengths, valid):
        B = ph_frame_logits.shape[0]
        T = ph_frame_logits.shape[1]

        ph_edge_prob = torch.nn.functional.sigmoid(ph_frame_logits.float())

        pred1 = ph_edge_prob[: B // 2, :]
        pred2 = ph_edge_prob[B // 2 :, :]
        pseudo_label1 = (pred1 >= 0.5).float()
        pseudo_label2 = (pred2 >= 0.5).float()
        gradient_magnitude1 = torch.abs(pred1 - pseudo_label1)
        gradient_magnitude2 = torch.abs(pred2 - pseudo_label2)
        gradient_magnitude = (gradient_magnitude1 + gradient_magnitude2) / 2

        # calculate mask matrix
        # (B//2, T, 1)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=B // 2)
        mask = (
            (mask < input_feature_lengths[: B // 2].unsqueeze(1))
            .to(torch.bool)
            .unsqueeze(-1)
        )
        pseudo_label_mask = (  # (B//2, T)
            mask
            & (pseudo_label1 == pseudo_label2)
            & (gradient_magnitude < self.pseudo_label_auto_theshold)
        )

        if pseudo_label_mask.sum() / mask.sum() < self.pseudo_label_ratio:
            self.pseudo_label_auto_theshold += 0.005
        else:
            self.pseudo_label_auto_theshold -= 0.005

        if pseudo_label_mask.any():
            pseudo_label_loss = self.pseudo_label_GHM_loss_fn(
                ph_frame_logits,
                torch.cat([pseudo_label1, pseudo_label2], dim=0),
                torch.cat([pseudo_label_mask, pseudo_label_mask], dim=0),
                valid,
            )
        else:
            pseudo_label_loss = torch.tensor(0).to(self.device)

        return pseudo_label_loss

    def _get_compound_vowel_loss(
        self,
        ph_frame_logits,  # (B, T, vocab_size)
        ph_frame_gt,  # (B, T) - component phoneme IDs
        ph_frame_compound_gt,  # (B, T) - compound vowel IDs (0 where no compound)
        input_feature_lengths,  # (B)
    ):
        """
        Compute compound vowel association loss.
        
        This loss encourages the model to predict high probability for the compound vowel
        when it already predicts high probability for its component phonemes. 
        Specifically, for frames where component phonemes are labeled AND a compound vowel 
        association exists, we encourage the compound vowel logit to be similar to the 
        average of its component logits.
        
        Good trend: This loss should decrease during training, indicating that the model
        learns to associate component phoneme features with compound vowel features.
        A well-trained model should have this loss converge to a low stable value.
        """
        B, T, vocab_size = ph_frame_logits.shape
        
        # Create mask for frames where compound vowel association exists
        compound_mask = (ph_frame_compound_gt > 0).float()  # (B, T)
        
        # Also mask by input feature lengths
        length_mask = torch.arange(T, device=self.device).unsqueeze(0) < input_feature_lengths.unsqueeze(1)
        compound_mask = compound_mask * length_mask.float()
        
        if compound_mask.sum() == 0:
            return torch.tensor(0.0).to(self.device)
        
        # Get probability distributions
        ph_frame_prob = torch.nn.functional.softmax(ph_frame_logits.float(), dim=-1)  # (B, T, vocab_size)
        
        # For each frame with compound vowel association:
        # - The target is to have high probability for the compound vowel
        # - We use the ground truth compound vowel ID from ph_frame_compound_gt
        
        # Gather the compound vowel probabilities
        compound_ids = ph_frame_compound_gt.long()  # (B, T)
        compound_probs = torch.gather(ph_frame_prob, dim=2, index=compound_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
        
        # We want to maximize the probability of compound vowels in frames where components are
        # This is equivalent to minimizing -log(prob), i.e., cross entropy
        # But we use soft target: encourage compound prob to be at least as high as component prob
        component_ids = ph_frame_gt.long()  # (B, T)
        component_probs = torch.gather(ph_frame_prob, dim=2, index=component_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
        
        # Loss: encourage compound_prob >= component_prob
        # Using a margin loss: max(0, component_prob - compound_prob + margin)
        margin = 0.0
        loss_per_frame = torch.nn.functional.relu(component_probs - compound_probs + margin)
        
        # Apply mask and compute mean
        loss = (loss_per_frame * compound_mask).sum() / (compound_mask.sum() + 1e-6)
        
        return loss

    def _get_loss(
        self,
        ph_frame_logits,  # (B, T, vocab_size)
        ph_edge_logits,  # (B, T)
        ctc_logits,  # (B, T, vocab_size)
        ph_frame_gt,  # (B, T)
        ph_edge_gt,  # (B, T)
        ph_seq_gt,  # (B S)
        ph_seq_lengths_gt,  # (B)
        ph_mask,  # (B vocab_size)
        input_feature_lengths,  # (B)
        label_type,  # (B)
        ph_frame_compound_gt=None,  # (B, T) - compound vowel associations
        valid=False,
    ):
        full_label_idx = label_type >= 2
        weak_label_idx = label_type >= 1
        not_full_label_idx = label_type < 2
        ZERO = torch.tensor(0).to(self.device)

        if (full_label_idx).any():
            (
                ph_frame_GHM_loss,
                ph_edge_GHM_loss,
                ph_edge_EMD_loss,
                ph_edge_diff_loss,
            ) = self._get_full_label_loss(
                ph_frame_logits[full_label_idx, :, :],
                ph_edge_logits[full_label_idx, :],
                ph_frame_gt[full_label_idx, :],
                ph_edge_gt[full_label_idx, :],
                input_feature_lengths[full_label_idx],
                ph_mask[full_label_idx, :],
                valid,
            )
        else:
            ph_frame_GHM_loss = ph_edge_GHM_loss = ZERO
            ph_edge_EMD_loss = ph_edge_diff_loss = ZERO

        # TODO:这种pack方式无法处理只有batch中的一部分需要计算Loss的情况，改掉
        if (weak_label_idx).any():
            ctc_GHM_loss = ZERO
            ctc_GHM_loss = self._get_weak_label_loss(
                ctc_logits[weak_label_idx, :, :],
                ph_mask[weak_label_idx, :],
                ph_seq_gt[weak_label_idx, :],
                ph_seq_lengths_gt[weak_label_idx],
                input_feature_lengths[weak_label_idx],
                valid,
            )
        else:
            ctc_GHM_loss = ZERO

        if not valid and self.data_augmentation_enabled:
            consistency_loss = self._get_consistency_loss(
                ph_frame_logits, ph_edge_logits, input_feature_lengths
            )
            pseudo_label_loss = ZERO
            # pseudo_label_loss = self._get_pseudo_label_loss(
            #     ph_frame_logits[not_full_label_idx, :, :],
            #     input_feature_lengths[not_full_label_idx],
            #     valid,
            # )
        else:
            consistency_loss = ZERO
            pseudo_label_loss = ZERO
        
        # Compound vowel association loss
        if ph_frame_compound_gt is not None and (full_label_idx).any():
            compound_vowel_loss = self._get_compound_vowel_loss(
                ph_frame_logits[full_label_idx, :, :],
                ph_frame_gt[full_label_idx, :],
                ph_frame_compound_gt[full_label_idx, :],
                input_feature_lengths[full_label_idx],
            )
        else:
            compound_vowel_loss = ZERO

        losses = [
            ph_frame_GHM_loss,
            ph_edge_GHM_loss,
            ph_edge_EMD_loss,
            ph_edge_diff_loss,
            ctc_GHM_loss,
            consistency_loss,
            pseudo_label_loss,
            compound_vowel_loss,
        ]

        return losses

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        h = self.backbone(*args, **kwargs)
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 2:]
        ph_edge_logits = logits[:, :, 0]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)
        return ph_frame_logits, ph_edge_logits, ctc_logits

    def training_step(self, batch, batch_idx):
        try:
            (
                input_feature,  # (B, n_mels, T)
                input_feature_lengths,  # (B)
                ph_seq,  # (B S)
                ph_seq_lengths,  # (B)
                ph_edge,  # (B, T)
                ph_frame,  # (B, T)
                ph_mask,  # (B vocab_size)
                label_type,  # (B)
                ph_frame_compound,  # (B, T) - compound vowel associations
            ) = batch

            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(input_feature.transpose(1, 2))

            losses = self._get_loss(
                ph_frame_logits,
                ph_edge_logits,
                ctc_logits,
                ph_frame,
                ph_edge,
                ph_seq,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                ph_frame_compound_gt=ph_frame_compound,
                valid=False,
            )

            schedule_weight = self._losses_schedulers_call()
            self._losses_schedulers_step()
            total_loss = (
                torch.stack(losses) * self.losses_weights * schedule_weight
            ).sum()
            losses.append(total_loss)

            log_dict = {
                f"train_loss/{k}": v
                for k, v in zip(self.losses_names, losses)
                if v != 0
            }
            log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_dict.update(
                {
                    f"scheduler/{k}": v
                    for k, v in zip(self.losses_names, schedule_weight)
                    if v != 1
                }
            )
            self.log_dict(log_dict)
            return total_loss
        except Exception as e:
            print(f"Error: {e}. skip this batch.")
            return torch.tensor(torch.nan).to(self.device)

    def validation_step(self, batch, batch_idx):
        (
            input_feature,  # (B, n_mels, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            ph_edge,  # (B, T)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            label_type,  # (B)
            ph_frame_compound,  # (B, T) - compound vowel associations
        ) = batch

        ph_seq_g2p = ["SP"]
        for ph in ph_seq.squeeze(0).cpu().numpy():
            if ph == 0:
                continue
            ph_seq_g2p.append(self.vocab[ph])
            ph_seq_g2p.append("SP")
        
        # Run inference for original (compound vowel) version
        ph_seq_pred, ph_intervals_pred, _, _, _, ctc, fig = self._infer_once(
            input_feature,
            None,
            ph_seq_g2p,
            None,
            None,
            True,
            True,
        )
        self.logger.experiment.add_text(
            f"valid/ctc_predict_{batch_idx}", " ".join(ctc), self.global_step
        )
        self.logger.experiment.add_figure(
            f"valid/plot_{batch_idx}", fig, self.global_step
        )
        
        # If split_rules are available, also show the split version
        if self.split_rules is not None and len(ph_seq_pred) > 0:
            # Check if any phoneme can be split
            has_splittable = any(ph in self.split_rules for ph in ph_seq_pred)
            
            if has_splittable:
                # Convert intervals to list of tuples for apply_split_to_annotations
                intervals_as_tuples = [(float(s), float(e)) for s, e in ph_intervals_pred]
                
                # Apply split rules to get split version
                split_ph_seq, split_intervals = apply_split_to_annotations(
                    list(ph_seq_pred),
                    intervals_as_tuples,
                    self.split_rules,
                    split_mode="all"
                )
                
                # Log the split version
                split_text = " ".join(f"{ph}[{s:.3f}-{e:.3f}]" for ph, (s, e) in zip(split_ph_seq, split_intervals))
                self.logger.experiment.add_text(
                    f"valid/split_align_{batch_idx}", split_text, self.global_step
                )

        (
            ph_frame_logits,  # (B, T, vocab_size)
            ph_edge_logits,  # (B, T)
            ctc_logits,  # (B, T, vocab_size)
        ) = self.forward(input_feature.transpose(1, 2))

        losses = self._get_loss(
            ph_frame_logits,
            ph_edge_logits,
            ctc_logits,
            ph_frame,
            ph_edge,
            ph_seq,
            ph_seq_lengths,
            ph_mask,
            input_feature_lengths,
            label_type,
            ph_frame_compound_gt=ph_frame_compound,
            valid=True,
        )

        weights = self._losses_schedulers_call() * self.losses_weights
        total_loss = (torch.stack(losses) * weights).sum()
        losses.append(total_loss)
        losses = torch.stack(losses)

        self.validation_step_outputs["losses"].append(losses)

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs["losses"], dim=0)
        losses = (losses / ((losses > 0).sum(dim=0, keepdim=True) + 1e-6)).sum(dim=0)
        self.log_dict(
            {f"valid/{k}": v for k, v in zip(self.losses_names, losses) if v != 0}
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone.parameters(),
                    "lr": self.optimizer_config["lr"]["backbone"],
                },
                {
                    "params": self.head.parameters(),
                    "lr": self.optimizer_config["lr"]["head"],
                },
            ],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.OneCycleLR(
                optimizer,
                max_lr=[
                    self.optimizer_config["lr"]["backbone"],
                    self.optimizer_config["lr"]["head"],
                ],
                total_steps=self.optimizer_config["total_steps"],
            ),
            "interval": "step",
        }

        for k, v in self.optimizer_config["freeze"].items():
            if v:
                getattr(self, k).requires_grad_(False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
