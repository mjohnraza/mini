import json
import os
from typing import List, Dict, Optional

class CharTokenizer:
    """
    Deterministic character-level tokenizer for causal language modeling.
    Extracts all unique characters from the corpus, assigns unique deterministic integer IDs,
    and provides bi-directional encoding/decoding.
    """
    def __init__(self, chars: Optional[List[str]] = None):
        self.chars = sorted(list(set(chars))) if chars is not None else []
        self.vocab_size = len(self.chars)
        self.char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        """Build tokenizer vocabulary deterministically from the corpus text."""
        chars = sorted(list(set(text)))
        tokenizer = cls(chars)
        print(f"[Tokenizer] Trained vocabulary size: {tokenizer.vocab_size} unique characters.")
        return tokenizer

    def encode(self, text: str) -> List[int]:
        """Convert a string of text into a list of token integers."""
        encoded = []
        for ch in text:
            if ch in self.char_to_id:
                encoded.append(self.char_to_id[ch])
            else:
                # Fallback to space or closest known char if unknown
                encoded.append(self.char_to_id.get(' ', 0))
        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list or tensor of token integers back into string text."""
        chars = [self.id_to_char.get(int(idx), '') for idx in token_ids]
        return ''.join(chars)

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "chars": self.chars,
            "char_to_id": self.char_to_id
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(data["chars"])
        return tokenizer
