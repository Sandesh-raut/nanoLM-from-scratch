"""
Batch loader: slices the encoded corpus into (input, target) windows.

For a sequence of tokens  [t0, t1, t2, ..., tN]:
  - A window of length block_size starting at offset i gives:
      x = [t_i,   t_{i+1}, ..., t_{i+B-1}]   ← input context
      y = [t_{i+1}, ..., t_{i+B}]              ← targets (shifted by 1)
  - At every position k in the window, the model sees x[k] and must
    predict y[k] = x[k+1].

This is "next-token prediction" — the training objective of every LLM.
"""
import numpy as np


class BatchLoader:
    """
    Returns random batches of (x, y) windows from the encoded corpus.

    Parameters
    ----------
    data       : 1-D int array of token ids
    block_size : context length B
    batch_size : number of windows per batch
    seed       : for reproducible sampling
    """

    def __init__(
        self,
        data: np.ndarray,
        block_size: int,
        batch_size: int,
        seed: int = 42,
    ):
        assert len(data) > block_size, (
            f"Corpus too short ({len(data)} tokens) for block_size={block_size}. "
            "Use a larger corpus or smaller block_size."
        )
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.n_positions = len(data) - block_size  # valid start positions

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch.
        Returns x, y each of shape (batch_size, block_size).
        """
        starts = self.rng.integers(0, self.n_positions, size=self.batch_size)
        x = np.stack([self.data[i : i + self.block_size] for i in starts])
        y = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in starts])
        return x, y

    def split(self, val_fraction: float = 0.1) -> tuple['BatchLoader', 'BatchLoader']:
        """
        Split corpus into train / val loaders.  (Used from Phase 5 onward.)
        Split is done on the raw data array, not on batches.
        """
        n = int(len(self.data) * (1 - val_fraction))
        train_loader = BatchLoader(self.data[:n], self.block_size, self.batch_size, seed=self.rng.integers(1e9))
        val_loader   = BatchLoader(self.data[n:], self.block_size, self.batch_size, seed=self.rng.integers(1e9))
        return train_loader, val_loader
