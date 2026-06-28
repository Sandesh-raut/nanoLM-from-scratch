"""
Phase 2 — Baseline NumPy model with hand-written backprop.

Architecture: Token → Embedding → Linear → Logits  (no attention yet)

This is a bigram model: to predict the next token at position t, the model
uses ONLY the embedding of the token AT t. It cannot see tokens before t.
Phase 3 adds self-attention so every position can attend to its full history.

Parameters
----------
  E : (vocab_size, embed_dim)   embedding table
  W : (embed_dim,  vocab_size)  output projection
  b : (vocab_size,)             output bias

Total params = 2 × vocab_size × embed_dim + vocab_size

Math reference
--------------
Forward:
  h       = E[x]             # embedding lookup, shape (B, T, D)
  logits  = h @ W + b        # linear projection, shape (B, T, V)
  probs   = softmax(logits)  # probabilities, shape (B, T, V)
  loss    = -mean( log p[target] )   # cross-entropy

Backward (softmax + cross-entropy combined):
  dlogits = (probs - one_hot(targets)) / N   where N = B*T
  dW      = h_flat.T  @ dlogits             # (D, V)
  db      = dlogits.sum(0)                  # (V,)
  dh      = dlogits   @ W.T                 # (N, D)
  dE      = scatter_add(dh, indices=x)      # (vocab_size, D)
"""
import numpy as np


class BigramModel:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.01  # small init keeps softmax near-uniform at step 0

        self.E = rng.standard_normal((vocab_size, embed_dim)).astype(np.float64) * scale
        self.W = rng.standard_normal((embed_dim, vocab_size)).astype(np.float64) * scale
        self.b = np.zeros(vocab_size, dtype=np.float64)

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x : (B, T)  integer token ids
        Returns logits (B, T, V) and hidden states h (B, T, D).

        Each position t is independent — no token sees its neighbours.
        Phase 3 will replace the mean-pool here with self-attention.
        """
        h      = self.E[x]           # (B, T, D)  — embedding lookup
        logits = h @ self.W + self.b # (B, T, V)  — linear projection
        return logits, h

    # ── Loss + gradients ─────────────────────────────────────────────────────

    def loss_and_grads(
        self,
        x:       np.ndarray,   # (B, T) int  — input tokens
        targets: np.ndarray,   # (B, T) int  — target tokens (x shifted right by 1)
    ) -> tuple[float, dict[str, np.ndarray]]:
        """
        One forward pass, then full backprop in closed form.
        Returns (scalar loss, gradient dict keyed by param name).
        """
        B, T = x.shape
        N    = B * T   # total predictions in this batch

        # ── Forward ──
        logits, h = self.forward(x)                # logits (B,T,V), h (B,T,D)
        logits_flat = logits.reshape(N, self.vocab_size)   # (N, V)
        h_flat      = h.reshape(N, self.embed_dim)         # (N, D)
        tgts_flat   = targets.reshape(N)                   # (N,)

        # Numerically stable softmax: subtract row max before exp
        shifted = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, V)

        # Cross-entropy loss
        correct_log_p = np.log(probs[np.arange(N), tgts_flat] + 1e-8)
        loss = float(-correct_log_p.mean())

        # ── Backward ──
        # Gradient of (softmax + cross-entropy) combined:
        #   dL/dlogits = (probs - one_hot(target)) / N
        dlogits = probs.copy()
        dlogits[np.arange(N), tgts_flat] -= 1.0
        dlogits /= N                                      # (N, V)

        dW = h_flat.T @ dlogits                           # (D, V)
        db = dlogits.sum(axis=0)                          # (V,)

        dh_flat = dlogits @ self.W.T                      # (N, D)
        dh      = dh_flat.reshape(B, T, self.embed_dim)

        # Scatter-add: accumulate gradients into embedding rows
        # dE[x[b,t]] += dh[b,t]  for all b,t
        dE = np.zeros_like(self.E)
        np.add.at(dE, x, dh)

        return loss, {'E': dE, 'W': dW, 'b': db}

    # ── SGD update ───────────────────────────────────────────────────────────

    def sgd_step(self, grads: dict[str, np.ndarray], lr: float):
        """Plain gradient descent: θ ← θ − lr · ∇θ."""
        self.E -= lr * grads['E']
        self.W -= lr * grads['W']
        self.b -= lr * grads['b']

    # ── Inspection helpers ───────────────────────────────────────────────────

    description = "Phase 2 · NumPy Bigram · hand-written backprop"

    def param_table(self) -> list[tuple[str, str, int]]:
        V, D = self.vocab_size, self.embed_dim
        return [
            ('embed.E', f'({V} × {D})', self.E.size),
            ('proj.W',  f'({D} × {V})', self.W.size),
            ('proj.b',  f'({V},)',       self.b.size),
        ]

    def param_count(self) -> dict[str, int]:
        """Return parameter count broken down by layer."""
        counts = {name: n for name, _, n in self.param_table()}
        counts['total'] = sum(counts.values())
        return counts

    def tracked_weight(self, row: int = 0, col: int = 0) -> float:
        """Return E[row, col] — a single scalar to watch move during training."""
        return float(self.E[row, col])

    # ── Flat iterators for AdamW ──────────────────────────────────────────────

    def _flat_params(self):
        yield 'E', self.E
        yield 'W', self.W
        yield 'b', self.b

    def _flat_grads(self, grads: dict):
        yield 'E', grads['E']
        yield 'W', grads['W']
        yield 'b', grads['b']

    def compute_loss(self, x: np.ndarray, targets: np.ndarray) -> float:
        """Loss in inference mode (no dropout in bigram — identical to training)."""
        B, T = x.shape
        N    = B * T
        logits, _ = self.forward(x)
        logits_flat = logits.reshape(N, self.vocab_size)
        tgts_flat   = targets.reshape(N)
        shifted     = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l       = np.exp(shifted)
        probs       = exp_l / exp_l.sum(axis=1, keepdims=True)
        return float(-np.log(probs[np.arange(N), tgts_flat] + 1e-8).mean())
