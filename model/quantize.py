"""
Post-training quantization — int8 weight compression.

What is quantization?
---------------------
Replace 64-bit (or 32-bit) floating-point weights with 8-bit integers.
Typical storage reduction: 8× from float64, 4× from float32.

Simple int8 scheme (per-tensor, symmetric)
------------------------------------------
For each weight matrix W:
  scale = max(|W|) / 127.0
  W_q   = round(W / scale).clip(-128, 127).astype(int8)

To use during forward pass:
  W_approx = W_q.astype(float64) * scale   (dequantise)

The quantisation error is bounded by ε = scale / 2 per element.
For large matrices the mean squared error is ≈ scale² / 12.

More advanced schemes (not implemented here, but named for reference):
  - Per-channel quantization: separate scale per output dim → lower error
  - NF4 (4-bit Normal Float): used in QLoRA — maps to 16 levels on N(0,1)
  - GPTQ: post-training quantization with second-order Hessian correction
  - AWQ: activation-aware weight quantization (scale weights by activation importance)

What changes at inference?
--------------------------
The model stores int8 weights and dequantises each matrix before the matmul.
Memory: 8× smaller (float64 → int8).
Speed: slightly slower on CPU (dequantise overhead) but faster on GPU with
       fused int8 kernels (e.g. bitsandbytes on CUDA).

Usage
-----
  from model.quantize import quantize_model, model_size_bytes, summary

  original_size = model_size_bytes(model)
  q_model       = quantize_model(model)
  quant_size    = model_size_bytes(q_model)
  print(summary(model, q_model))
"""

import copy
import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core quantization ops
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Quantize a float array to int8 with per-tensor symmetric scaling.

    Parameters
    ----------
    w : np.ndarray, any float dtype

    Returns
    -------
    w_q   : np.ndarray, dtype=int8
    scale : float  — multiply w_q by this to approximate w
    """
    max_val = float(np.abs(w).max())
    if max_val == 0.0:
        return np.zeros(w.shape, dtype=np.int8), 1.0
    scale = max_val / 127.0
    w_q   = np.round(w / scale).clip(-128, 127).astype(np.int8)
    return w_q, scale


def dequantize_int8(w_q: np.ndarray, scale: float) -> np.ndarray:
    """
    Dequantize int8 weights back to float64.

    Parameters
    ----------
    w_q   : np.ndarray, dtype=int8
    scale : float

    Returns
    -------
    w_approx : np.ndarray, dtype=float64
    """
    return w_q.astype(np.float64) * scale


def quantization_error(w: np.ndarray) -> dict:
    """
    Measure the error introduced by int8 quantization on a single weight matrix.

    Returns
    -------
    {
      'max_abs_error': float,
      'rms_error':     float,
      'max_rel_error': float,  # relative to max(|w|)
    }
    """
    w_q, scale  = quantize_int8(w)
    w_approx    = dequantize_int8(w_q, scale)
    error       = w - w_approx
    max_abs     = float(np.abs(error).max())
    rms         = float(np.sqrt((error ** 2).mean()))
    max_val     = float(np.abs(w).max())
    rel         = max_abs / max_val if max_val > 0 else 0.0
    return {'max_abs_error': max_abs, 'rms_error': rms, 'max_rel_error': rel}


# ─────────────────────────────────────────────────────────────────────────────
# Model-level quantization
# ─────────────────────────────────────────────────────────────────────────────

class QuantizedWeight:
    """
    A weight matrix stored as int8 + a float64 scale.

    This actually saves memory: the int8 array is 8× smaller than float64.
    It behaves like a 2-D array inside the forward pass — `x @ qw` and
    `qw @ x` dequantise lazily and do the matmul in float — so a model whose
    weight matrices are QuantizedWeight runs through the same forward() code
    with no special-casing. (Backward / training is not supported on a
    quantized model; quantization is a post-training, inference-only step.)
    """

    def __init__(self, w: np.ndarray):
        self.w_q, self.scale = quantize_int8(w)
        self.shape = w.shape

    # ── lazy dequantisation ────────────────────────────────────────────────
    @property
    def value(self) -> np.ndarray:
        """Float64 approximation of the original weight (reconstructed on demand)."""
        return dequantize_int8(self.w_q, self.scale)

    def __array__(self, dtype=None):
        v = self.value
        return v.astype(dtype) if dtype is not None else v

    # ── array-like surface used by forward() and the introspection helpers ──
    def __matmul__(self, other):      # qw @ x
        return self.value @ other

    def __rmatmul__(self, other):     # x @ qw   (the common case)
        return other @ self.value

    @property
    def T(self) -> np.ndarray:
        return self.value.T

    @property
    def ndim(self) -> int:
        return self.w_q.ndim

    @property
    def size(self) -> int:
        return self.w_q.size

    @property
    def nbytes(self) -> int:
        return self.w_q.nbytes + 8   # int8 array + one float64 scale


def model_size_bytes(model) -> int:
    """
    Total size of all model parameters in bytes. Counts QuantizedWeight at its
    real int8 footprint, so a quantized model reports a genuinely smaller size.
    """
    return sum(param.nbytes for _, param in model._flat_params())


def quantize_model(model):
    """
    Return a deep copy of the model with all 2-D weight matrices stored as int8
    (wrapped in QuantizedWeight). The copy is genuinely smaller in memory and is
    still runnable: forward() dequantises each matrix on the fly during its
    matmul.

    1-D parameters (embeddings, norm γ/β, biases) are left in float64 —
    quantizing them saves almost nothing and costs disproportionate accuracy.

    This is a simple educational implementation. Production quantization
    (bitsandbytes, GPTQ, AWQ) uses specialized int8/int4 CUDA kernels and
    per-channel or activation-aware scales.
    """
    q_model = copy.deepcopy(model)
    _quantize_params_inplace(q_model)
    return q_model


def _quantize_params_inplace(model):
    """Replace 2-D weight matrices with QuantizedWeight (int8) in-place."""
    for block in getattr(model, 'blocks', getattr(model, 'layers', [])):
        attn = block.attn
        for name in ('Wq', 'Wk', 'Wv', 'Wo'):
            w = getattr(attn, name)
            if getattr(w, 'ndim', 0) == 2:
                setattr(attn, name, QuantizedWeight(w))

        ffn = block.ffn
        for name in ('W1', 'W2', 'Wg', 'Wv'):
            w = getattr(ffn, name, None)
            if w is not None and getattr(w, 'ndim', 0) == 2:
                setattr(ffn, name, QuantizedWeight(w))

    if hasattr(model, 'proj') and getattr(model.proj, 'ndim', 0) == 2:
        model.proj = QuantizedWeight(model.proj)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def compression_stats(model) -> dict:
    """
    Compute int8 compression statistics for a model without modifying it.

    Returns
    -------
    {
      'float64_bytes': int,
      'int8_bytes':    int,   # estimate: 2D params at 1 byte, 1D at 8 bytes
      'compression_ratio': float,
      'avg_rms_error': float,
    }
    """
    float64_bytes = 0
    int8_bytes    = 0
    errors        = []

    for name, param in model._flat_params():
        float64_bytes += param.nbytes
        if param.ndim == 2:
            # Would be stored as int8 + one float64 scale
            int8_bytes += param.size * 1 + 8
            errs = quantization_error(param)
            errors.append(errs['rms_error'])
        else:
            # Keep in float64
            int8_bytes += param.nbytes

    ratio     = float64_bytes / max(int8_bytes, 1)
    avg_err   = float(np.mean(errors)) if errors else 0.0

    return {
        'float64_bytes':    float64_bytes,
        'int8_bytes':       int8_bytes,
        'compression_ratio': ratio,
        'avg_rms_error':     avg_err,
    }


def summary(model) -> str:
    """Return a human-readable quantization summary string."""
    stats = compression_stats(model)
    f64   = stats['float64_bytes']
    i8    = stats['int8_bytes']
    ratio = stats['compression_ratio']
    err   = stats['avg_rms_error']

    def _fmt(b: int) -> str:
        if b >= 1_000_000: return f"{b/1_000_000:.1f} MB"
        if b >= 1_000:     return f"{b/1_000:.1f} KB"
        return f"{b} B"

    return (
        f"Weights (float64) : {_fmt(f64)}\n"
        f"Weights (int8 est): {_fmt(i8)}\n"
        f"Compression ratio : {ratio:.1f}×\n"
        f"Avg RMS error     : {err:.2e}  (per-element, 2-D weight matrices)"
    )
