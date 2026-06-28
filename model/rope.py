"""
RoPE — Rotary Positional Encoding (Su et al., 2021).

Key idea
--------
Instead of adding a positional embedding to token embeddings, rotate the
Query and Key vectors by an angle proportional to their position.

Why better than learned positional embeddings
---------------------------------------------
  1. Relative positions fall out naturally: Q_m · K_n depends only on
     the *difference* m - n, not the absolute positions.
  2. Generalises to sequences longer than seen during training.
  3. Zero extra parameters — no pos_embed table required.

Used in: LLaMA, Mistral, Gemma, Qwen, Falcon (all major open models since 2023).

Rotation formula (applied to consecutive pairs within each head dimension)
--------------------------------------------------------------------------
Given position m and dimension pair (2i, 2i+1):

  θ_i   = base ^ (−2i / head_dim)        ← fixed, decaying frequencies
  angle = m · θ_i

  out[2i]   = x[2i]   · cos(angle) − x[2i+1] · sin(angle)
  out[2i+1] = x[2i]   · sin(angle) + x[2i+1] · cos(angle)

This is a 2D rotation by angle m·θ_i in each consecutive dimension pair.
Because rotation is a linear operation, the inner product between a rotated
Q at position m and a rotated K at position n encodes only (m − n), not
m or n separately — that is the key property.

Backward pass
-------------
Rotation matrices are orthogonal: R^T = R^{-1}. So the backward pass is
just the transpose rotation (negate sin terms).
"""

import numpy as np


def rope_freqs(head_dim: int, seq_len: int, base: float = 10_000.0):
    """
    Pre-compute cosine / sine tables for all positions and dimension pairs.

    Parameters
    ----------
    head_dim : int   — must be even
    seq_len  : int   — number of positions to pre-compute
    base     : float — RoPE base (10,000 in the original paper; 500,000 in LLaMA 3)

    Returns
    -------
    cos : (seq_len, head_dim // 2) — float64
    sin : (seq_len, head_dim // 2) — float64
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    half = head_dim // 2

    # θ_i = 1 / (base ^ (2i / head_dim)),  i = 0, 1, ..., half-1
    theta = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))

    positions = np.arange(seq_len, dtype=np.float64)

    # Outer product: angles[pos, i] = pos * θ_i
    angles = np.outer(positions, theta)   # (seq_len, half)

    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply RoPE rotation to a Query or Key tensor.

    Parameters
    ----------
    x   : (B, H, T, Dh) — queries or keys, any floating dtype
    cos : (T, Dh // 2)
    sin : (T, Dh // 2)

    Returns
    -------
    rotated : (B, H, T, Dh) — same shape and dtype as x
    """
    # Split into consecutive pairs at the last dimension
    x1 = x[..., ::2]    # (B, H, T, Dh//2) — even indices
    x2 = x[..., 1::2]   # (B, H, T, Dh//2) — odd indices

    # Broadcast cos/sin: (T, Dh//2) → (1, 1, T, Dh//2)
    c = cos[np.newaxis, np.newaxis, :, :]
    s = sin[np.newaxis, np.newaxis, :, :]

    out = np.empty_like(x)
    out[..., ::2]  = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def apply_rope_backward(dout: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Backward pass of apply_rope.

    Derivation
    ----------
    Forward:  out1 = x1·c − x2·s,   out2 = x1·s + x2·c

    Jacobian:
        ∂out1/∂x1 = c,  ∂out1/∂x2 = −s
        ∂out2/∂x1 = s,  ∂out2/∂x2 =  c

    Chain rule:
        dx1 = dout1·c + dout2·s
        dx2 = −dout1·s + dout2·c

    This is the transpose (= inverse) rotation: R^T applied to d_out,
    which is equivalent to rotating backwards by the same angles.

    Parameters
    ----------
    dout : (B, H, T, Dh) — upstream gradient (or (B, Kv, T, Dh) for GQA keys)
    cos  : (T, Dh // 2)
    sin  : (T, Dh // 2)

    Returns
    -------
    dx : same shape as dout
    """
    d1 = dout[..., ::2]    # dout for even-index outputs
    d2 = dout[..., 1::2]   # dout for odd-index outputs

    c = cos[np.newaxis, np.newaxis, :, :]
    s = sin[np.newaxis, np.newaxis, :, :]

    dx = np.empty_like(dout)
    dx[..., ::2]  = d1 * c + d2 * s    # inverse: cos unchanged
    dx[..., 1::2] = -d1 * s + d2 * c   # inverse: sin negated
    return dx
