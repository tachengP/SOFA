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
