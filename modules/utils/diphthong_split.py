"""
Diphthong Split Utilities

This module provides utilities for handling diphthong (compound vowel) splitting.
It loads split rules from a dictionary and provides functions to check and apply splits.

Split rules dictionary format:
    compound_vowel<TAB>component1 component2 [component3 ...]

Example:
    ai	a i
    ao	a o
    iao	i a o
"""

import random
from typing import Dict, List, Tuple
import warnings


def load_split_rules(split_dict_path: str) -> Dict[str, List[str]]:
    """
    Load diphthong split rules from a dictionary file.
    
    Args:
        split_dict_path: Path to the split rules dictionary file
        
    Returns:
        Dictionary mapping compound vowels to their component vowels
    """
    split_rules = {}
    
    try:
        with open(split_dict_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
    except FileNotFoundError:
        warnings.warn(f"Split dictionary file not found: {split_dict_path}")
        return split_rules
    
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
    
    return split_rules


def build_diphthong_mapping(
    vocab: Dict,
    split_rules: Dict[str, List[str]]
) -> Dict[int, List[int]]:
    """
    Build a mapping from compound vowel IDs to their component vowel IDs.
    
    This is used during training to link compound vowels with their split forms.
    
    Args:
        vocab: The vocabulary dictionary (phoneme <-> ID mapping)
        split_rules: Dictionary of split rules
        
    Returns:
        Dictionary mapping compound vowel ID to list of component vowel IDs
    """
    diphthong_mapping = {}
    
    for compound, components in split_rules.items():
        if compound not in vocab:
            continue
        
        compound_id = vocab[compound]
        component_ids = []
        
        all_components_exist = True
        for comp in components:
            if comp not in vocab:
                all_components_exist = False
                break
            component_ids.append(vocab[comp])
        
        if all_components_exist:
            diphthong_mapping[compound_id] = component_ids
    
    return diphthong_mapping


def get_splittable_phonemes(
    ph_seq: List[int],
    diphthong_mapping: Dict[int, List[int]]
) -> List[int]:
    """
    Get indices of phonemes in a sequence that can be split.
    
    Args:
        ph_seq: List of phoneme IDs
        diphthong_mapping: Mapping from compound to component IDs
        
    Returns:
        List of indices where splittable diphthongs occur
    """
    splittable_indices = []
    for idx, ph_id in enumerate(ph_seq):
        if ph_id in diphthong_mapping:
            splittable_indices.append(idx)
    return splittable_indices


def apply_split_to_sequence(
    ph_seq: List[int],
    ph_dur: List[float],
    split_indices: List[int],
    diphthong_mapping: Dict[int, List[int]]
) -> Tuple[List[int], List[float]]:
    """
    Apply splits to a phoneme sequence at specified indices.
    
    Args:
        ph_seq: List of phoneme IDs
        ph_dur: List of phoneme durations
        split_indices: Indices where splits should be applied
        diphthong_mapping: Mapping from compound to component IDs
        
    Returns:
        Tuple of (new_ph_seq, new_ph_dur) with splits applied
    """
    new_ph_seq = []
    new_ph_dur = []
    
    for idx, (ph_id, dur) in enumerate(zip(ph_seq, ph_dur)):
        if idx in split_indices and ph_id in diphthong_mapping:
            components = diphthong_mapping[ph_id]
            component_dur = dur / len(components)
            for comp_id in components:
                new_ph_seq.append(comp_id)
                new_ph_dur.append(component_dur)
        else:
            new_ph_seq.append(ph_id)
            new_ph_dur.append(dur)
    
    return new_ph_seq, new_ph_dur


def select_splits_by_rate(
    splittable_indices: List[int],
    rate: float
) -> List[int]:
    """
    Select which diphthongs to split based on a probability rate.
    
    Each splittable diphthong instance is independently selected with
    the given probability, ensuring uniform distribution across all
    phoneme types.
    
    Args:
        splittable_indices: List of indices where splits can occur
        rate: Probability of splitting each individual diphthong (0.0-1.0)
        
    Returns:
        List of indices that were selected for splitting
    """
    selected = []
    for idx in splittable_indices:
        if random.random() < rate:
            selected.append(idx)
    
    return selected


def build_reverse_mapping(split_rules: Dict[str, List[str]]) -> Dict[Tuple[str, ...], str]:
    """
    Build a reverse mapping from component vowel sequences to compound vowels.
    
    This is used for combining split vowels back into compound vowels.
    
    Args:
        split_rules: Dictionary mapping compound vowels to component lists
        
    Returns:
        Dictionary mapping component sequence (as tuple) to compound vowel
    """
    reverse_mapping = {}
    for compound, components in split_rules.items():
        # Use tuple of components as key
        key = tuple(components)
        reverse_mapping[key] = compound
    return reverse_mapping


def find_combinable_sequences(
    ph_seq: List[str],
    reverse_mapping: Dict[Tuple[str, ...], str]
) -> List[Tuple[int, int, str]]:
    """
    Find sequences of phonemes that can be combined into compound vowels.
    
    Args:
        ph_seq: List of phoneme names
        reverse_mapping: Mapping from component sequences to compound vowels
        
    Returns:
        List of (start_idx, end_idx, compound_vowel) tuples
    """
    combinable = []
    
    # Check for sequences of length 2, 3, 4 (typical diphthong/triphthong lengths)
    for seq_len in [4, 3, 2]:
        i = 0
        while i <= len(ph_seq) - seq_len:
            seq = tuple(ph_seq[i:i+seq_len])
            if seq in reverse_mapping:
                combinable.append((i, i + seq_len, reverse_mapping[seq]))
                i += seq_len  # Skip the matched sequence
            else:
                i += 1
    
    # Remove overlapping matches (keep longer matches)
    combinable.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = -1
    for start, end, compound in combinable:
        if start >= last_end:
            non_overlapping.append((start, end, compound))
            last_end = end
    
    return non_overlapping


def apply_combine_to_sequence(
    ph_seq: List[str],
    ph_intervals: List[Tuple[float, float]],
    combinable: List[Tuple[int, int, str]]
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Apply combines to a phoneme sequence.
    
    Args:
        ph_seq: List of phoneme names
        ph_intervals: List of (start, end) intervals
        combinable: List of (start_idx, end_idx, compound_vowel) tuples
        
    Returns:
        Tuple of (new_ph_seq, new_ph_intervals)
    """
    if not combinable:
        return ph_seq, ph_intervals
    
    new_ph_seq = []
    new_ph_intervals = []
    
    # Create a set of indices to skip
    skip_until = -1
    combine_dict = {c[0]: c for c in combinable}
    
    for i, (ph, interval) in enumerate(zip(ph_seq, ph_intervals)):
        if i < skip_until:
            continue
        
        if i in combine_dict:
            start_idx, end_idx, compound = combine_dict[i]
            # Combine intervals: use start of first, end of last
            combined_start = ph_intervals[start_idx][0]
            combined_end = ph_intervals[end_idx - 1][1]
            new_ph_seq.append(compound)
            new_ph_intervals.append((combined_start, combined_end))
            skip_until = end_idx
        else:
            new_ph_seq.append(ph)
            new_ph_intervals.append(interval)
    
    return new_ph_seq, new_ph_intervals


def apply_split_to_annotations(
    ph_seq: List[str],
    ph_intervals: List[Tuple[float, float]],
    split_rules: Dict[str, List[str]],
    split_mode: str = "all",
    split_rate: float = 1.0
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Apply splits to phoneme annotations.
    
    Args:
        ph_seq: List of phoneme names
        ph_intervals: List of (start, end) intervals
        split_rules: Dictionary of split rules
        split_mode: "all" to split all, "rate" for probabilistic
        split_rate: Probability of splitting each diphthong (only used when split_mode="rate")
        
    Returns:
        Tuple of (new_ph_seq, new_ph_intervals)
    """
    new_ph_seq = []
    new_ph_intervals = []
    
    for ph, interval in zip(ph_seq, ph_intervals):
        if ph in split_rules:
            should_split = (split_mode == "all") or (split_mode == "rate" and random.random() < split_rate)
            if should_split:
                components = split_rules[ph]
                start, end = interval
                duration = end - start
                component_duration = duration / len(components)
                
                for i, comp in enumerate(components):
                    comp_start = start + i * component_duration
                    comp_end = start + (i + 1) * component_duration
                    new_ph_seq.append(comp)
                    new_ph_intervals.append((comp_start, comp_end))
            else:
                new_ph_seq.append(ph)
                new_ph_intervals.append(interval)
        else:
            new_ph_seq.append(ph)
            new_ph_intervals.append(interval)
    
    return new_ph_seq, new_ph_intervals


def apply_combine_to_annotations(
    ph_seq: List[str],
    ph_intervals: List[Tuple[float, float]],
    split_rules: Dict[str, List[str]]
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Combine split vowels back into compound vowels in annotations.
    
    Args:
        ph_seq: List of phoneme names
        ph_intervals: List of (start, end) intervals
        split_rules: Dictionary of split rules
        
    Returns:
        Tuple of (new_ph_seq, new_ph_intervals)
    """
    reverse_mapping = build_reverse_mapping(split_rules)
    combinable = find_combinable_sequences(ph_seq, reverse_mapping)
    return apply_combine_to_sequence(ph_seq, ph_intervals, combinable)


def build_reverse_mapping_with_ids(
    split_rules: Dict[str, List[str]],
    vocab: Dict
) -> Dict[Tuple[int, ...], int]:
    """
    Build a reverse mapping from component vowel ID sequences to compound vowel IDs.
    
    This is used during binarization to detect split component sequences and 
    associate them with their compound vowel.
    
    Args:
        split_rules: Dictionary mapping compound vowels to component lists
        vocab: The vocabulary dictionary (phoneme <-> ID mapping)
        
    Returns:
        Dictionary mapping component ID sequence (as tuple) to compound vowel ID
    """
    reverse_mapping = {}
    for compound, components in split_rules.items():
        # Check if compound and all components exist in vocab
        if compound not in vocab:
            continue
        
        all_exist = True
        component_ids = []
        for comp in components:
            if comp not in vocab:
                all_exist = False
                break
            component_ids.append(vocab[comp])
        
        if all_exist:
            reverse_mapping[tuple(component_ids)] = vocab[compound]
    
    return reverse_mapping


def find_component_sequences_in_ids(
    ph_seq_ids: List[int],
    reverse_id_mapping: Dict[Tuple[int, ...], int]
) -> List[Tuple[int, int, int]]:
    """
    Find sequences of phoneme IDs that form component sequences of compound vowels.
    
    Args:
        ph_seq_ids: List of phoneme IDs
        reverse_id_mapping: Mapping from component ID sequences to compound vowel IDs
        
    Returns:
        List of (start_idx, end_idx, compound_vowel_id) tuples
    """
    matches = []
    
    # Check for sequences of length 2, 3, 4 (typical diphthong/triphthong lengths)
    for seq_len in [4, 3, 2]:
        i = 0
        while i <= len(ph_seq_ids) - seq_len:
            seq = tuple(ph_seq_ids[i:i+seq_len])
            if seq in reverse_id_mapping:
                matches.append((i, i + seq_len, reverse_id_mapping[seq]))
                i += seq_len  # Skip the matched sequence
            else:
                i += 1
    
    # Remove overlapping matches (keep longer matches)
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = -1
    for start, end, compound_id in matches:
        if start >= last_end:
            non_overlapping.append((start, end, compound_id))
            last_end = end
    
    return non_overlapping
