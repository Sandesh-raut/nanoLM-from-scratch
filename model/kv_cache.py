"""
KV-Cache — autoregressive inference acceleration.

Without cache
-------------
Generating token t requires a full forward pass over positions 0..t.
At each step the attention computes K and V for ALL previous positions.
Total attention cost: O(T²) per step × T steps = O(T³).

With cache
----------
On the PREFILL step (the prompt), K and V are computed for all L prompt
positions and stored in a list of per-layer dicts.
On each DECODE step, only the single new token is projected to K_t, V_t,
which are appended to the cache.  Attention then reads K_cache (length L+t)
vs Q_t (length 1).

Decode cost per step:
  • 1 token projected to Q, K, V
  • dot product Q_t (1×Dh) × K_cache (T_full×Dh)  → O(T_full)
Total: O(L²) prefill + O(T) per decode step ≈ O(T²) overall.

Practical speedup grows quadratically with context length.  At block_size=32
the advantage is modest; at block_size=4096 (LLaMA) it is the difference
between real-time and unusable.

Usage
-----
  from model.kv_cache import generate_cached

  model, tok = load('runs/latest.npz')
  out = generate_cached(model, tok, "Once upon a", max_new_tokens=40)
  print(tok.decode(out))
"""

import time
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core generation functions
# ─────────────────────────────────────────────────────────────────────────────

def _sample_token(logits_1d: np.ndarray, temperature: float = 1.0) -> int:
    """Sample one token from a (V,) logit vector."""
    if temperature == 0.0:
        return int(np.argmax(logits_1d))
    logits = logits_1d / temperature
    logits -= logits.max()
    probs   = np.exp(logits)
    probs  /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_no_cache(
    model,
    tokens:         list[int],
    max_new_tokens: int,
    temperature:    float = 1.0,
) -> list[int]:
    """
    Standard autoregressive generation WITHOUT KV cache.
    At each step, recomputes attention over the full context.
    Returns the list of generated token ids (excluding prompt).
    """
    generated = list(tokens)
    block_size = model.block_size

    for _ in range(max_new_tokens):
        ctx = generated[-block_size:]
        x   = np.array(ctx, dtype=np.int32).reshape(1, -1)
        logits = model.forward(x, training=False)          # (1, T, V)
        next_tok = _sample_token(logits[0, -1, :], temperature)
        generated.append(next_tok)

    return generated[len(tokens):]


def generate_cached(
    model,
    tokens:         list[int],
    max_new_tokens: int,
    temperature:    float = 1.0,
) -> list[int]:
    """
    Autoregressive generation WITH KV cache.
    Requires model to be a ModernTransformerLM (supports kv_caches parameter).

    Returns the list of generated token ids (excluding prompt).
    """
    block_size = model.block_size

    # PREFILL uses the last block_size prompt tokens; decode then appends one
    # position per new token. The cache has no sliding-window eviction, so the
    # whole (prompt + generated) run must fit inside the trained context window.
    # Guard explicitly instead of silently indexing pos_embed out of bounds
    # (learned pos_enc) or extrapolating past training length (rope).
    prompt_len = min(len(tokens), block_size)
    total_len  = prompt_len + max_new_tokens - 1
    if total_len > block_size:
        raise ValueError(
            f"KV-cache decode supports up to block_size={block_size} positions, "
            f"but prompt({prompt_len}) + new({max_new_tokens}) needs {total_len}. "
            f"Shorten the generation, raise block_size, or use generate_no_cache "
            f"(which slides a block_size window and has no length limit)."
        )

    # Initialise empty per-layer caches
    n_layers   = model.n_layers
    kv_caches  = [{} for _ in range(n_layers)]

    # PREFILL: process the full prompt in one forward pass
    ctx = tokens[-block_size:]
    x   = np.array(ctx, dtype=np.int32).reshape(1, -1)
    logits = model.forward(x, training=False, kv_caches=kv_caches)   # (1, T, V)

    # Sample first new token from last position of the prefill
    next_tok = _sample_token(logits[0, -1, :], temperature)
    generated = [next_tok]

    # DECODE: one token at a time, using the cache
    for _ in range(max_new_tokens - 1):
        x      = np.array([[next_tok]], dtype=np.int32)               # (1, 1)
        logits = model.forward(x, training=False, kv_caches=kv_caches) # (1, 1, V)
        next_tok = _sample_token(logits[0, -1, :], temperature)
        generated.append(next_tok)

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_kv_cache(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 30,
    temperature:    float = 1.0,
    n_trials:       int = 3,
) -> dict:
    """
    Time generate_no_cache vs generate_cached on the same prompt.

    Returns
    -------
    {
      'no_cache_ms': float,      # median ms for full generation
      'cached_ms':   float,      # median ms for full generation
      'speedup':     float,      # no_cache / cached
      'no_cache_ms_per_tok': float,
      'cached_ms_per_tok':   float,
      'output_no_cache': str,
      'output_cached':   str,
    }
    """
    tokens = tokenizer.encode(prompt)

    # Warm-up
    generate_no_cache(model, tokens, 2, temperature)
    try:
        generate_cached(model, tokens, 2, temperature)
    except (AttributeError, TypeError):
        # Model doesn't support kv_caches (e.g. base TransformerLM)
        return {'error': 'model does not support KV cache'}

    # Benchmark no-cache
    nc_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out_nc = generate_no_cache(model, tokens, max_new_tokens, temperature)
        nc_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark cached
    c_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out_c = generate_cached(model, tokens, max_new_tokens, temperature)
        c_times.append((time.perf_counter() - t0) * 1000)

    nc_ms = float(np.median(nc_times))
    c_ms  = float(np.median(c_times))

    return {
        'no_cache_ms':        nc_ms,
        'cached_ms':          c_ms,
        'speedup':            nc_ms / max(c_ms, 0.01),
        'no_cache_ms_per_tok': nc_ms / max_new_tokens,
        'cached_ms_per_tok':   c_ms  / max_new_tokens,
        'output_no_cache': tokenizer.decode(out_nc),
        'output_cached':   tokenizer.decode(out_c),
    }
