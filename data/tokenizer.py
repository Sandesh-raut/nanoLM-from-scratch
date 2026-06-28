"""
Character-level tokenizer.

Maps each unique character in the corpus to an integer id.
Vocab is derived entirely from the training text — no pre-built word list.

Real-world analogue: BPE tokenizers (tiktoken, SentencePiece) do the same thing
at the sub-word level. Flagged as a Phase 9 upgrade path.
"""
import json
from pathlib import Path


class CharTokenizer:
    """
    Build vocab from text, then encode/decode freely.

    vocab  : sorted list of unique characters  e.g. [' ', 'a', 'b', ...]
    stoi   : char  -> int   (string-to-index)
    itos   : int   -> char  (index-to-string)
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(chars)}

    # ── Encode / decode ──────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token ids."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token ids back to a string."""
        return ''.join(self.itos.get(i, '?') for i in ids)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save vocab to JSON so the same tokenizer can be reloaded later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'vocab': self.vocab}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'CharTokenizer':
        """Reload a saved tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.vocab = data['vocab']
        tok.vocab_size = len(tok.vocab)
        tok.stoi = {c: i for i, c in enumerate(tok.vocab)}
        tok.itos = {i: c for i, c in enumerate(tok.vocab)}
        return tok

    def __repr__(self) -> str:
        preview = ''.join(self.vocab[:10])
        return f"CharTokenizer(vocab_size={self.vocab_size}, chars='{preview}...')"
