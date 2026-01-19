"""
Annotation Modifier Utilities

This module provides utilities for reading and modifying existing annotation files
(TextGrid and CSV formats) to apply diphthong splitting or combining operations.
"""

import pathlib
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import textgrid

from modules.utils.diphthong_split import (
    load_split_rules,
    apply_split_to_annotations,
    apply_combine_to_annotations
)


def find_textgrid_for_wav(wav_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find the TextGrid file corresponding to a wav file.
    
    Searches in two locations:
    1. Same directory as wav file with same name
    2. TextGrid subdirectory with same name
    
    Args:
        wav_path: Path to the wav file
        
    Returns:
        Path to TextGrid file or None if not found
    """
    # Try same directory
    tg_path = wav_path.with_suffix(".TextGrid")
    if tg_path.exists():
        return tg_path
    
    # Try TextGrid subdirectory
    tg_path = wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name
    if tg_path.exists():
        return tg_path
    
    return None


def read_textgrid_annotations(tg_path: pathlib.Path) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Read phoneme annotations from a TextGrid file.
    
    Args:
        tg_path: Path to TextGrid file
        
    Returns:
        Tuple of (ph_seq, ph_intervals)
    """
    tg = textgrid.TextGrid.fromFile(str(tg_path))
    
    # Find the phones tier
    ph_tier = None
    for tier in tg:
        if tier.name.lower() in ["phones", "phone", "phonemes", "phoneme"]:
            ph_tier = tier
            break
    
    if ph_tier is None:
        # Try to find any interval tier
        for tier in tg:
            if isinstance(tier, textgrid.IntervalTier):
                ph_tier = tier
                break
    
    if ph_tier is None:
        raise ValueError(f"No suitable tier found in {tg_path}")
    
    ph_seq = []
    ph_intervals = []
    
    for interval in ph_tier:
        mark = interval.mark.strip()
        if mark:  # Skip empty intervals
            ph_seq.append(mark)
            ph_intervals.append((interval.minTime, interval.maxTime))
    
    return ph_seq, ph_intervals


def write_textgrid_annotations(
    tg_path: pathlib.Path,
    ph_seq: List[str],
    ph_intervals: List[Tuple[float, float]],
    word_seq: Optional[List[str]] = None,
    word_intervals: Optional[List[Tuple[float, float]]] = None
):
    """
    Write phoneme annotations to a TextGrid file.
    
    Args:
        tg_path: Path to output TextGrid file
        ph_seq: List of phoneme names
        ph_intervals: List of (start, end) intervals
        word_seq: Optional list of word names
        word_intervals: Optional list of word intervals
    """
    if not ph_intervals:
        return
    
    max_time = max(interval[1] for interval in ph_intervals)
    
    tg = textgrid.TextGrid(maxTime=max_time)
    
    if word_seq and word_intervals:
        word_tier = textgrid.IntervalTier(name="words", maxTime=max_time)
        for word, (start, end) in zip(word_seq, word_intervals):
            word_tier.add(start, end, word)
        tg.append(word_tier)
    
    ph_tier = textgrid.IntervalTier(name="phones", maxTime=max_time)
    for ph, (start, end) in zip(ph_seq, ph_intervals):
        ph_tier.add(start, end, ph)
    tg.append(ph_tier)
    
    tg_path.parent.mkdir(parents=True, exist_ok=True)
    tg.write(str(tg_path))


def find_csv_for_folder(folder_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Find the transcriptions.csv file for a folder.
    
    Args:
        folder_path: Path to folder containing wavs
        
    Returns:
        Path to transcriptions.csv or None if not found
    """
    # Check if folder has wavs subfolder
    if (folder_path / "wavs").exists():
        csv_path = folder_path / "transcriptions.csv"
        if csv_path.exists():
            return csv_path
    
    # Check current folder
    csv_path = folder_path / "transcriptions.csv"
    if csv_path.exists():
        return csv_path
    
    # Check transcriptions subfolder
    csv_path = folder_path / "transcriptions" / "transcriptions.csv"
    if csv_path.exists():
        return csv_path
    
    return None


def read_csv_annotations(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Read annotations from a transcriptions.csv file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with annotations
    """
    return pd.read_csv(csv_path, dtype=str)


def parse_csv_row_phonemes(row: pd.Series) -> Tuple[List[str], List[float]]:
    """
    Parse phoneme sequence and durations from a CSV row.
    
    Args:
        row: DataFrame row
        
    Returns:
        Tuple of (ph_seq, ph_dur)
    """
    ph_seq = row["ph_seq"].split(" ") if pd.notna(row.get("ph_seq")) else []
    ph_dur = [float(d) for d in row["ph_dur"].split(" ")] if pd.notna(row.get("ph_dur")) else []
    return ph_seq, ph_dur


def durations_to_intervals(durations: List[float]) -> List[Tuple[float, float]]:
    """
    Convert duration list to interval list.
    
    Args:
        durations: List of durations
        
    Returns:
        List of (start, end) intervals
    """
    intervals = []
    current_time = 0.0
    for dur in durations:
        intervals.append((current_time, current_time + dur))
        current_time += dur
    return intervals


def intervals_to_durations(intervals: List[Tuple[float, float]]) -> List[float]:
    """
    Convert interval list to duration list.
    
    Args:
        intervals: List of (start, end) intervals
        
    Returns:
        List of durations
    """
    return [end - start for start, end in intervals]


def modify_csv_annotations(
    csv_path: pathlib.Path,
    split_rules: Dict[str, List[str]],
    mode: str,  # "split_all", "combine_all", or "split_rate"
    split_rate: float = 1.0,
    output_path: Optional[pathlib.Path] = None
):
    """
    Modify annotations in a CSV file.
    
    Args:
        csv_path: Path to input CSV file
        split_rules: Dictionary of split rules
        mode: "split_all", "combine_all", or "split_rate"
        split_rate: Probability for rate-based splitting
        output_path: Optional output path (overwrites input if None)
    """
    df = read_csv_annotations(csv_path)
    
    for idx, row in df.iterrows():
        ph_seq, ph_dur = parse_csv_row_phonemes(row)
        
        if not ph_seq or not ph_dur or len(ph_seq) != len(ph_dur):
            continue
        
        ph_intervals = durations_to_intervals(ph_dur)
        
        if mode == "split_all":
            new_ph_seq, new_ph_intervals = apply_split_to_annotations(
                ph_seq, ph_intervals, split_rules, "all"
            )
        elif mode == "split_rate":
            new_ph_seq, new_ph_intervals = apply_split_to_annotations(
                ph_seq, ph_intervals, split_rules, "rate", split_rate
            )
        elif mode == "combine_all":
            new_ph_seq, new_ph_intervals = apply_combine_to_annotations(
                ph_seq, ph_intervals, split_rules
            )
        else:
            continue
        
        new_ph_dur = intervals_to_durations(new_ph_intervals)
        
        df.at[idx, "ph_seq"] = " ".join(new_ph_seq)
        df.at[idx, "ph_dur"] = " ".join([f"{d:.5f}" for d in new_ph_dur])
    
    output = output_path or csv_path
    df.to_csv(output, index=False)


def modify_textgrid_annotations(
    tg_path: pathlib.Path,
    split_rules: Dict[str, List[str]],
    mode: str,  # "split_all", "combine_all", or "split_rate"
    split_rate: float = 1.0,
    output_path: Optional[pathlib.Path] = None
):
    """
    Modify annotations in a TextGrid file.
    
    Args:
        tg_path: Path to input TextGrid file
        split_rules: Dictionary of split rules
        mode: "split_all", "combine_all", or "split_rate"
        split_rate: Probability for rate-based splitting
        output_path: Optional output path (overwrites input if None)
    """
    ph_seq, ph_intervals = read_textgrid_annotations(tg_path)
    
    if mode == "split_all":
        new_ph_seq, new_ph_intervals = apply_split_to_annotations(
            ph_seq, ph_intervals, split_rules, "all"
        )
    elif mode == "split_rate":
        new_ph_seq, new_ph_intervals = apply_split_to_annotations(
            ph_seq, ph_intervals, split_rules, "rate", split_rate
        )
    elif mode == "combine_all":
        new_ph_seq, new_ph_intervals = apply_combine_to_annotations(
            ph_seq, ph_intervals, split_rules
        )
    else:
        return
    
    output = output_path or tg_path
    write_textgrid_annotations(output, new_ph_seq, new_ph_intervals)


def process_folder_annotations(
    folder_path: pathlib.Path,
    modify_type: str,  # "tg" or "csv"
    split_rules: Dict[str, List[str]],
    mode: str,  # "split_all", "combine_all", or "split_rate"
    split_rate: float = 1.0,
    recursive: bool = True
):
    """
    Process all annotations in a folder.
    
    Args:
        folder_path: Path to folder
        modify_type: "tg" for TextGrid, "csv" for CSV
        split_rules: Dictionary of split rules
        mode: "split_all", "combine_all", or "split_rate"
        split_rate: Probability for rate-based splitting
        recursive: Whether to process subdirectories
    """
    folder_path = pathlib.Path(folder_path)
    
    if modify_type == "tg":
        # Find all TextGrid files
        pattern = "**/*.TextGrid" if recursive else "*.TextGrid"
        tg_files = list(folder_path.glob(pattern))
        
        print(f"Found {len(tg_files)} TextGrid files to process")
        
        for tg_path in tg_files:
            try:
                modify_textgrid_annotations(tg_path, split_rules, mode, split_rate)
                print(f"  Processed: {tg_path}")
            except Exception as e:
                print(f"  Error processing {tg_path}: {e}")
    
    elif modify_type == "csv":
        # Find all transcriptions.csv files
        pattern = "**/transcriptions.csv" if recursive else "transcriptions.csv"
        csv_files = list(folder_path.glob(pattern))
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        for csv_path in csv_files:
            try:
                modify_csv_annotations(csv_path, split_rules, mode, split_rate)
                print(f"  Processed: {csv_path}")
            except Exception as e:
                print(f"  Error processing {csv_path}: {e}")
