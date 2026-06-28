"""
SwiGLU — Swish-Gated Linear Unit FFN (Noam Shazeer, 2020).

Replaces the two-layer ReLU feed-forward network with a gated variant:

  Standard FFN (Phase 3):   out = ReLU(x · W1) · W2
  SwiGLU FFN  (Phase 9):    out = (SiLU(x · Wg) ⊙ (x · Wv)) · W2

Where:
  SiLU(z) = z · σ(z)     (Sigmoid Linear Unit, smooth approximation of ReLU)
  ⊙        = elementwise product (the "gate")
  Wg       = gate weight   (D → inner)
  Wv       = value weight  (D → inner)
  W2       = down weight   (inner → D)

Why the gate?
-------------
The gate selects which features to pass through based on input content.
ReLU always zeroes the same positions; SiLU + gate makes the filtering
input-dependent. This consistently improves loss at the same param budget.

Inner dimension
---------------
Standard FFN uses inner = 4D, giving param count: 2 · D · 4D = 8D².
SwiGLU uses three matrices (Wg, Wv, W2), so to match param count:
  3 · D · inner ≈ 8D²  →  inner ≈ 8D/3 ≈ 2.67D.
We round up to the nearest multiple of 64 for hardware alignment
(LLaMA does the same).

Used in: LLaMA, LLaMA 2/3, Mistral, PaLM 2, Gemma.

Backward pass derivation
-------------------------
Forward:
  gate   = x · Wg
  val    = x · Wv
  sig    = σ(gate)         = 1 / (1 + exp(−gate))
  silu   = gate · sig      (SiLU)
  swiglu = silu ⊙ val
  out    = swiglu · W2

Backward (given dout):
  d_swiglu = dout · W2ᵀ                                          [inner]
  dW2      = swiglu.T · dout

  d_silu   = d_swiglu ⊙ val
  d_val    = d_swiglu ⊙ silu

  d_gate = d_silu · ∂SiLU/∂gate
         = d_silu · (sig + gate · sig · (1 − sig))
         = d_silu · sig · (1 + gate · (1 − sig))

  dWg = xᵀ · d_gate
  dWv = xᵀ · d_val
  dx  = d_gate · Wgᵀ + d_val · Wvᵀ
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class SwiGLUFFN:
    """
    SwiGLU feed-forward block.

    Same external interface as FFN:
      forward(x)     → out
      backward(dout) → (dx, grads)
      grads keys: {'Wg', 'Wv', 'W2'}   (no biases — common in LLaMA-style models)
    """

    def __init__(self, dim: int, seed: int = 0):
        rng   = np.random.default_rng(seed)
        scale = 0.02

        # inner ≈ 8D/3, rounded to nearest 64 for alignment
        raw_inner = int(dim * 8 / 3)
        inner     = max(64, ((raw_inner + 63) // 64) * 64)

        self.dim   = dim
        self.inner = inner

        self.Wg = rng.standard_normal((dim, inner)).astype(np.float64) * scale  # gate
        self.Wv = rng.standard_normal((dim, inner)).astype(np.float64) * scale  # value
        self.W2 = rng.standard_normal((inner, dim)).astype(np.float64) * scale  # down
        self._c = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, D) → out: (B, T, D)"""
        gate   = x @ self.Wg           # (B, T, inner)
        val    = x @ self.Wv           # (B, T, inner)
        sig    = _sigmoid(gate)
        silu   = gate * sig            # SiLU activation
        swiglu = silu * val            # gated
        out    = swiglu @ self.W2      # (B, T, D)

        self._c = {
            'x': x, 'gate': gate, 'val': val,
            'sig': sig, 'silu': silu, 'swiglu': swiglu,
        }
        return out

    def backward(self, dout: np.ndarray):
        """
        dout : (B, T, D)

        Returns
        -------
        dx    : (B, T, D)
        grads : {'Wg': ..., 'Wv': ..., 'W2': ...}
        """
        x      = self._c['x']
        gate   = self._c['gate']
        val    = self._c['val']
        sig    = self._c['sig']
        silu   = self._c['silu']
        swiglu = self._c['swiglu']

        B, T, D = x.shape
        BT      = B * T
        inner   = self.inner

        # out = swiglu @ W2
        dW2      = swiglu.reshape(BT, inner).T @ dout.reshape(BT, D)
        d_swiglu = dout @ self.W2.T              # (B, T, inner)

        # swiglu = silu * val
        d_silu = d_swiglu * val
        d_val  = d_swiglu * silu

        # silu = gate * sig(gate)
        # dSiLU/dgate = sig(gate) * (1 + gate * (1 − sig(gate)))
        d_gate = d_silu * sig * (1.0 + gate * (1.0 - sig))

        dWg = x.reshape(BT, D).T @ d_gate.reshape(BT, inner)
        dWv = x.reshape(BT, D).T @ d_val.reshape(BT, inner)

        dx = d_gate @ self.Wg.T + d_val @ self.Wv.T   # (B, T, D)

        return dx, {'Wg': dWg, 'Wv': dWv, 'W2': dW2}

    # ── Introspection ─────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return self.Wg.size + self.Wv.size + self.W2.size

    def param_rows(self, prefix: str):
        D, inner = self.dim, self.inner
        return [
            (f'{prefix}.Wg', f'({D}×{inner})', self.Wg.size),
            (f'{prefix}.Wv', f'({D}×{inner})', self.Wv.size),
            (f'{prefix}.W2', f'({inner}×{D})', self.W2.size),
        ]
