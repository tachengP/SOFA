"""
Compound Vowel Split G2P Module

This module provides a G2P class that handles compound vowel splitting.
It loads split rules from a dictionary and applies them during inference
to split compound vowels into their component simple vowels.
"""

import warnings
from typing import Dict, List, Tuple

from modules.g2p.base_g2p import BaseG2P


class SplitG2P(BaseG2P):
    """
    G2P class that splits compound vowels into component vowels.
    
    This class works in two modes:
    1. Standard mode: Similar to DictionaryG2P, converts words to phonemes
    2. Split mode: Takes phoneme sequences and splits compound vowels
    
    The split rules are loaded from a dictionary file with format:
    compound_vowel<TAB>component1 component2 [component3 ...]
    
    Example:
        ai	a i
        ao	a o
        iao	i a o
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the SplitG2P with a split dictionary.
        
        Args:
            split_dictionary: Path to the split rules dictionary file
            dictionary: (Optional) Path to the standard dictionary for word to phoneme conversion
        """
        super().__init__(**kwargs)
        
        # Load split rules
        split_dict_path = kwargs.get("split_dictionary")
        if split_dict_path is None:
            raise ValueError("split_dictionary parameter is required for SplitG2P")
        
        self.split_rules: Dict[str, List[str]] = {}
        self._load_split_rules(split_dict_path)
        
        # Optionally load standard dictionary for word-to-phoneme conversion
        self.dictionary: Dict[str, List[str]] = {}
        dict_path = kwargs.get("dictionary")
        if dict_path:
            self._load_dictionary(dict_path)
    
    def _load_split_rules(self, split_dict_path: str) -> None:
        """Load split rules from the dictionary file."""
        with open(split_dict_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                warnings.warn(f"Invalid split rule format: {line}")
                continue
            
            compound = parts[0].strip()
            components = parts[1].strip().split(" ")
            
            if len(components) < 2:
                warnings.warn(f"Split rule must have at least 2 components: {line}")
                continue
            
            self.split_rules[compound] = components
        
        print(f"Loaded {len(self.split_rules)} split rules")
    
    def _load_dictionary(self, dict_path: str) -> None:
        """Load standard dictionary for word-to-phoneme conversion."""
        with open(dict_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            
            word = parts[0].strip()
            phonemes = parts[1].strip().split(" ")
            self.dictionary[word] = phonemes
    
    def get_split_rules(self) -> Dict[str, List[str]]:
        """Return the loaded split rules."""
        return self.split_rules
    
    def can_split(self, phoneme: str) -> bool:
        """Check if a phoneme can be split according to the rules."""
        return phoneme in self.split_rules
    
    def get_split_components(self, phoneme: str) -> List[str]:
        """Get the component phonemes for a compound vowel."""
        return self.split_rules.get(phoneme, [phoneme])
    
    def split_phoneme_sequence(
        self, 
        ph_seq: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Split compound vowels in a phoneme sequence.
        
        Args:
            ph_seq: List of phonemes
            
        Returns:
            Tuple of:
            - List of phonemes with compound vowels split
            - List of original phoneme indices for each output phoneme
        """
        result = []
        original_indices = []
        
        for idx, ph in enumerate(ph_seq):
            if self.can_split(ph):
                components = self.get_split_components(ph)
                for comp in components:
                    result.append(comp)
                    original_indices.append(idx)
            else:
                result.append(ph)
                original_indices.append(idx)
        
        return result, original_indices
    
    def _g2p(self, input_text: str) -> Tuple[List[str], List[str], List[int]]:
        """
        Convert input text to phoneme sequence with compound vowel splitting.
        
        If a standard dictionary is loaded, converts words to phonemes first.
        Then applies split rules to any compound vowels.
        
        Args:
            input_text: Space-separated words or phonemes
            
        Returns:
            Tuple of (ph_seq, word_seq, ph_idx_to_word_idx)
        """
        word_seq_raw = input_text.strip().split(" ")
        word_seq = []
        word_seq_idx = 0
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        
        for word in word_seq_raw:
            if not word or word == "SP":
                continue
            
            # Get phonemes for the word
            if self.dictionary and word in self.dictionary:
                phonemes = self.dictionary[word]
            else:
                # If no dictionary or word not found, treat word as phoneme
                phonemes = [word]
            
            word_seq.append(word)
            
            for i, ph in enumerate(phonemes):
                if (i == 0 or i == len(phonemes) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                
                # Apply split rules if phoneme is a compound vowel
                if self.can_split(ph):
                    components = self.get_split_components(ph)
                    for comp in components:
                        ph_seq.append(comp)
                        ph_idx_to_word_idx.append(word_seq_idx)
                else:
                    ph_seq.append(ph)
                    ph_idx_to_word_idx.append(word_seq_idx)
            
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)
            
            word_seq_idx += 1
        
        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    # Test the SplitG2P class
    g2p = SplitG2P(
        split_dictionary="dictionary/vowel_split_example.txt",
        dictionary="dictionary/opencpop-extension.txt"
    )
    
    # Test with a sample text
    text = "ai shi xiao niao"
    ph_seq, word_seq, ph_idx_to_word_idx = g2p(text)
    print(f"Input: {text}")
    print(f"Phonemes: {ph_seq}")
    print(f"Words: {word_seq}")
    print(f"Mapping: {ph_idx_to_word_idx}")
    
    # Test split rules
    print(f"\nSplit rules: {g2p.get_split_rules()}")
    print(f"Can split 'ai': {g2p.can_split('ai')}")
    print(f"Components of 'ai': {g2p.get_split_components('ai')}")
