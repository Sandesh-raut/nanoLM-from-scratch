"""
RMSNorm — Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

Drop-in replacement for LayerNorm with two simplifications:
  1. No mean subtraction (no re-centring).
  2. No additive bias β (no learnable shift).

Just divide by the RMS of the activation, then rescale with γ.

  rms = sqrt( mean(x²) + ε )
  out = γ · (x / rms)

Why simpler is better here
---------------------------
LayerNorm removes both the mean (1 DoF) and the variance (1 DoF).
Empirically, removing only variance (RMSNorm) performs equally well while:
  • Cutting the norm parameters by ~half (no β)
  • Being ~10% faster (no mean computation in the hot path)

Used in: LLaMA, Mistral, Falcon, Gemma, PaLM, T5 (as "T5LayerNorm").

Backward pass derivation
-------------------------
Let x̂ = x / rms, out = γ · x̂

  ∂L/∂γ = Σ (dout · x̂)               (sum over all non-last dims)
  ∂L/∂x̂ = dout · γ

  ∂L/∂x = ∂L/∂x̂ / rms
           − x · Σ(∂L/∂x̂ · x) / (D · rms³)

The second term propagates the constraint that x̂·x̂ = D (constant under
RMS normalisation), similar to the chain rule through batch norm.
"""

import numpy as np


class RMSNorm:
    """
    RMSNorm with a single learnable scale vector γ (no bias β).

    Interface matches LayerNorm so the two are interchangeable:
      backward() returns (dx, {'gamma': dgamma})
      — note: no 'beta' key, which _flat_grads uses to detect which norm is active.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim   = dim
        self.eps   = eps
        self.gamma = np.ones(dim, dtype=np.float64)
        self._c    = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x   : (..., dim)
        out : (..., dim)  — same shape
        """
        rms  = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)
        xhat = x / rms
        out  = self.gamma * xhat
        self._c = {'xhat': xhat, 'rms': rms, 'x': x}
        return out

    def backward(self, dout: np.ndarray):
        """
        dout : (..., dim)

        Returns
        -------
        dx    : (..., dim)
        grads : {'gamma': (dim,)}
        """
        xhat = self._c['xhat']
        rms  = self._c['rms']
        x    = self._c['x']
        D    = self.dim

        # Accumulate γ gradient over all leading dimensions
        axes   = tuple(range(dout.ndim - 1))
        dgamma = (dout * xhat).sum(axis=axes)

        dxhat = dout * self.gamma

        # Backprop through x / rms:
        #   dx_j = dxhat_j / rms  −  x_j · Σ_i(dxhat_i · x_i) / (D · rms³)
        dx = dxhat / rms - x * (dxhat * x).sum(axis=-1, keepdims=True) / (D * rms ** 3)

        return dx, {'gamma': dgamma}

    # ── Introspection ─────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return self.gamma.size

    def param_rows(self, prefix: str):
        D = self.dim
        return [(f'{prefix}.γ', f'({D},)', self.gamma.size)]
