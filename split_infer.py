"""
Split Inference Script

This script performs compound vowel splitting based on forced alignment.
It uses the alignment model to find optimal split points within compound vowels.

Usage:
    python split_infer.py --ckpt model.ckpt --folder segments --split_dictionary dictionary/vowel_split_example.txt
"""

import pathlib

import click
import lightning as pl
import torch

import modules.AP_detector
import modules.g2p
from modules.utils.export_tool import Exporter
from modules.utils.post_processing import post_processing
from train import LitForcedAlignmentTask


class SplitExporter(Exporter):
    """Extended exporter that also outputs split information."""
    
    def __init__(self, predictions, log, split_info=None):
        super().__init__(predictions, log)
        self.split_info = split_info or {}


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
    "--folder", 
    "-f", 
    default="segments", 
    type=str, 
    help="path to the input folder"
)
@click.option(
    "--mode", 
    "-m", 
    default="force", 
    type=click.Choice(["force", "match"])
)
@click.option(
    "--g2p", 
    "-g", 
    default="Split", 
    type=str, 
    help="name of the g2p class (default: Split for split inference)"
)
@click.option(
    "--ap_detector",
    "-a",
    default="LoudnessSpectralcentroidAPDetector",
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
    help="Types of output file, separated by comma.",
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
    help="path to the standard dictionary",
)
@click.option(
    "--split_dictionary",
    "-sd",
    default=None,
    required=True,
    type=str,
    help="path to the split rules dictionary (required for split inference)",
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
    dictionary,
    split_dictionary,
):
    """
    Perform split inference on audio files.
    
    This script uses a forced alignment model trained with split phonemes
    to split compound vowels in the input audio into their component vowels.
    """
    # Create kwargs dict for G2P initialization
    kwargs = {
        "dictionary": dictionary,
        "split_dictionary": split_dictionary,
    }
    
    # Ensure split_dictionary is provided
    if split_dictionary is None:
        raise ValueError("--split_dictionary is required for split inference")
    
    # Initialize the SplitG2P
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    
    out_formats = [i.strip().lower() for i in out_formats.split(",")]

    # Initialize AP detector
    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    # Set input format and get dataset
    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    # Load model and run inference
    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    model.set_inference_mode(mode)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    # Process predictions
    predictions = get_AP.process(predictions)
    predictions, log = post_processing(predictions)
    
    # Export results
    exporter = SplitExporter(
        predictions, 
        log, 
        split_info={"split_rules": grapheme_to_phoneme.get_split_rules()}
    )

    if save_confidence:
        out_formats.append('confidence')

    exporter.export(out_formats)

    print("Output files are saved to the same folder as the input wav files.")
    print(f"Split rules applied: {len(grapheme_to_phoneme.get_split_rules())} rules")


if __name__ == "__main__":
    main()
