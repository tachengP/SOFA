"""
Phoneme Merge Utilities

This module provides utilities for merging phonemes that share the same acoustic features.
When phonemes from different languages have identical acoustic characteristics (e.g., yue/a, ko/a, zh/a#),
they can be merged into a single canonical phoneme to reduce vocabulary size and improve model training.

Configuration format:
    merged_phoneme_groups:
      - [SP, cl]                           # Merge cl into SP
      - [yue/a, ja/a, ko/a, zh/a#]         # Merge all into yue/a (first one)
      - [yue/iy, en/iy, ja/i, ko/i]        # Merge all into yue/iy

The first phoneme in each group is the canonical form that all others map to.
"""

from typing import Dict, List, Set, Tuple


def build_phoneme_merge_mapping(
    merged_phoneme_groups: List[List[str]],
    existing_phonemes: Set[str] = None
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Build a mapping from phonemes to their canonical (merged) forms.
    
    Args:
        merged_phoneme_groups: List of phoneme groups, where phonemes in the same group
                               will be mapped to the first phoneme (canonical form)
        existing_phonemes: Optional set of phonemes that exist in the data.
                          If provided, only groups where at least one phoneme exists
                          will be processed.
    
    Returns:
        Tuple of:
            - merge_mapping: Dict mapping each phoneme to its canonical form
            - canonical_phonemes: Set of canonical phonemes (first in each group)
    """
    merge_mapping = {}
    canonical_phonemes = set()
    
    for group in merged_phoneme_groups:
        if len(group) < 2:
            continue
        
        # If existing_phonemes is provided, check if any phoneme in the group exists
        if existing_phonemes is not None:
            group_phonemes_in_data = [p for p in group if p in existing_phonemes]
            if len(group_phonemes_in_data) == 0:
                continue
        
        canonical = group[0]
        canonical_phonemes.add(canonical)
        
        for phoneme in group:
            merge_mapping[phoneme] = canonical
    
    return merge_mapping, canonical_phonemes


def apply_merge_to_phoneme(phoneme: str, merge_mapping: Dict[str, str]) -> str:
    """
    Apply merge mapping to a single phoneme.
    
    Args:
        phoneme: The phoneme to potentially merge
        merge_mapping: Dict mapping phonemes to their canonical forms
        
    Returns:
        The canonical form of the phoneme, or the original if not in mapping
    """
    return merge_mapping.get(phoneme, phoneme)


def apply_merge_to_phoneme_list(
    phonemes: List[str],
    merge_mapping: Dict[str, str]
) -> List[str]:
    """
    Apply merge mapping to a list of phonemes.
    
    Args:
        phonemes: List of phonemes
        merge_mapping: Dict mapping phonemes to their canonical forms
        
    Returns:
        List of phonemes with merging applied
    """
    return [apply_merge_to_phoneme(p, merge_mapping) for p in phonemes]


def get_merged_vocab(
    original_phonemes: Set[str],
    merge_mapping: Dict[str, str],
    ignored_phonemes: List[str]
) -> Dict:
    """
    Generate a vocabulary with merged phonemes.
    
    Args:
        original_phonemes: Set of original phonemes from the data
        merge_mapping: Dict mapping phonemes to their canonical forms
        ignored_phonemes: List of phonemes to ignore
        
    Returns:
        Vocabulary dictionary with phoneme <-> ID mappings
    """
    # Apply merging to get the set of canonical phonemes
    merged_phonemes = set()
    for p in original_phonemes:
        canonical = merge_mapping.get(p, p)
        merged_phonemes.add(canonical)
    
    # Remove ignored phonemes
    for p in ignored_phonemes:
        if p in merged_phonemes:
            merged_phonemes.remove(p)
    
    # Sort and add SP at the beginning
    phonemes = sorted(merged_phonemes)
    if "SP" in phonemes:
        phonemes.remove("SP")
    phonemes = ["SP", *phonemes]
    
    # Create vocab
    vocab = dict(zip(phonemes, range(len(phonemes))))
    vocab.update(dict(zip(range(len(phonemes)), phonemes)))
    vocab.update({i: 0 for i in ignored_phonemes})
    vocab.update({"<vocab_size>": len(phonemes)})
    
    return vocab


def build_phoneme_id_merge_mapping(
    merge_mapping: Dict[str, str],
    vocab: Dict
) -> Dict[int, int]:
    """
    Build a mapping from phoneme IDs to their canonical (merged) IDs.
    
    This is useful for converting between vocabularies where merging wasn't applied
    during vocab creation.
    
    Args:
        merge_mapping: Dict mapping phoneme names to their canonical forms
        vocab: The vocabulary dictionary
        
    Returns:
        Dict mapping phoneme IDs to their canonical IDs
    """
    id_mapping = {}
    
    for phoneme, canonical in merge_mapping.items():
        if phoneme in vocab and canonical in vocab:
            phoneme_id = vocab[phoneme]
            canonical_id = vocab[canonical]
            if phoneme_id != canonical_id:
                id_mapping[phoneme_id] = canonical_id
    
    return id_mapping


def summarize_merge_groups(
    merged_phoneme_groups: List[List[str]],
    existing_phonemes: Set[str] = None
) -> str:
    """
    Generate a summary of the merge groups for logging.
    
    Args:
        merged_phoneme_groups: List of phoneme groups
        existing_phonemes: Optional set of phonemes that exist in the data
        
    Returns:
        Summary string
    """
    lines = []
    total_merged = 0
    
    for group in merged_phoneme_groups:
        if len(group) < 2:
            continue
        
        if existing_phonemes is not None:
            group_in_data = [p for p in group if p in existing_phonemes]
            if len(group_in_data) == 0:
                continue
            lines.append(f"  {group[0]} <- {group[1:]} (found: {len(group_in_data)}/{len(group)})")
            total_merged += len(group_in_data) - 1
        else:
            lines.append(f"  {group[0]} <- {group[1:]}")
            total_merged += len(group) - 1
    
    if lines:
        header = f"Phoneme merge groups (total {total_merged} phonemes merged):"
        return header + "\n" + "\n".join(lines)
    else:
        return "No phoneme merge groups applied."
