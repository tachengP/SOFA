"""
Prepare Split Training Data

This utility script helps prepare training data for compound vowel split models.
It converts existing transcriptions.csv files by applying split rules to phoneme sequences.

Usage:
    python prepare_split_data.py --input data/full_label --output data/split_label \
        --split_dictionary dictionary/vowel_split_example.txt

The script will:
1. Read transcriptions.csv files from the input directory
2. Apply split rules to the phoneme sequences
3. Adjust phoneme durations proportionally
4. Write the converted files to the output directory
"""

import argparse
import os
import pathlib
import shutil
from typing import Dict, List, Tuple

import pandas as pd


def load_split_rules(split_dict_path: str) -> Dict[str, List[str]]:
    """Load split rules from a dictionary file."""
    split_rules = {}
    
    with open(split_dict_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        
        compound = parts[0].strip()
        components = parts[1].strip().split(" ")
        
        if len(components) >= 2:
            split_rules[compound] = components
    
    print(f"Loaded {len(split_rules)} split rules")
    return split_rules


def split_phoneme_sequence(
    ph_seq: str,
    ph_dur: str,
    split_rules: Dict[str, List[str]]
) -> Tuple[str, str]:
    """
    Apply split rules to a phoneme sequence and adjust durations.
    
    Args:
        ph_seq: Space-separated phoneme sequence
        ph_dur: Space-separated duration sequence
        split_rules: Dictionary of split rules
        
    Returns:
        Tuple of (new_ph_seq, new_ph_dur)
    """
    if not isinstance(ph_seq, str) or not isinstance(ph_dur, str):
        return ph_seq, ph_dur
    
    phonemes = ph_seq.split(" ")
    durations = [float(d) for d in ph_dur.split(" ")]
    
    if len(phonemes) != len(durations):
        print(f"Warning: phoneme count ({len(phonemes)}) != duration count ({len(durations)})")
        return ph_seq, ph_dur
    
    new_phonemes = []
    new_durations = []
    
    for ph, dur in zip(phonemes, durations):
        if ph in split_rules:
            components = split_rules[ph]
            # Distribute duration equally among components
            component_dur = dur / len(components)
            for comp in components:
                new_phonemes.append(comp)
                new_durations.append(component_dur)
        else:
            new_phonemes.append(ph)
            new_durations.append(dur)
    
    new_ph_seq = " ".join(new_phonemes)
    new_ph_dur = " ".join([f"{d:.5f}" for d in new_durations])
    
    return new_ph_seq, new_ph_dur


def process_transcriptions(
    input_csv: pathlib.Path,
    output_csv: pathlib.Path,
    split_rules: Dict[str, List[str]]
) -> int:
    """
    Process a transcriptions.csv file and apply split rules.
    
    Returns the number of rows processed.
    """
    df = pd.read_csv(input_csv, dtype=str)
    
    if "ph_seq" not in df.columns:
        print(f"Warning: {input_csv} does not have 'ph_seq' column, skipping")
        return 0
    
    has_ph_dur = "ph_dur" in df.columns
    
    for idx, row in df.iterrows():
        ph_seq = row["ph_seq"]
        ph_dur = row.get("ph_dur", "")
        
        if has_ph_dur and ph_dur:
            new_ph_seq, new_ph_dur = split_phoneme_sequence(ph_seq, ph_dur, split_rules)
            df.at[idx, "ph_seq"] = new_ph_seq
            df.at[idx, "ph_dur"] = new_ph_dur
        else:
            # Just split phonemes without adjusting durations
            phonemes = ph_seq.split(" ") if isinstance(ph_seq, str) else []
            new_phonemes = []
            for ph in phonemes:
                if ph in split_rules:
                    new_phonemes.extend(split_rules[ph])
                else:
                    new_phonemes.append(ph)
            df.at[idx, "ph_seq"] = " ".join(new_phonemes)
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for compound vowel split models"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing transcriptions.csv files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--split_dictionary", "-sd",
        required=True,
        help="Path to the split rules dictionary"
    )
    parser.add_argument(
        "--copy_wavs",
        action="store_true",
        help="Also copy wav files to the output directory"
    )
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Load split rules
    split_rules = load_split_rules(args.split_dictionary)
    
    if not split_rules:
        print("Error: No split rules loaded")
        return
    
    # Find and process all transcriptions.csv files
    trans_files = list(input_dir.rglob("transcriptions.csv"))
    print(f"Found {len(trans_files)} transcriptions.csv files")
    
    total_rows = 0
    for trans_file in trans_files:
        # Calculate relative path
        rel_path = trans_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        print(f"Processing: {rel_path}")
        rows = process_transcriptions(trans_file, output_file, split_rules)
        total_rows += rows
        
        # Copy wavs if requested
        if args.copy_wavs:
            wavs_dir = trans_file.parent / "wavs"
            if wavs_dir.exists():
                output_wavs_dir = output_file.parent / "wavs"
                output_wavs_dir.mkdir(parents=True, exist_ok=True)
                for wav_file in wavs_dir.glob("*.wav"):
                    shutil.copy2(wav_file, output_wavs_dir / wav_file.name)
    
    print(f"\nDone! Processed {total_rows} total rows across {len(trans_files)} files")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
