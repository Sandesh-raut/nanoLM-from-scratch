"""
Phase 6 — All sampling strategies in one place.

Sampling is a pure post-processing step: the model always outputs the same
logits for the same input.  How you turn those logits into the next token
is entirely up to you.  The same trained model can produce safe-and-boring
or wild-and-creative text by changing one number.

Strategies (applied in order)
------------------------------
1. Repetition penalty  — divide logits of already-seen tokens by rep_penalty (>1 = less repeat)
2. Greedy              — argmax; skip everything below
3. Temperature         — divide logits by T (<1 = sharper, >1 = flatter)
4. Top-k filter        — keep only the k highest-logit tokens; zero the rest
5. Top-p (nucleus)     — keep the smallest set of tokens summing to >= p probability
6. Softmax + sample    — convert to probabilities, draw one token

Maps to
-------
  The exact knobs every LLM API exposes:
    temperature, top_p, top_k in OpenAI / Anthropic APIs
    repetition_penalty in HuggingFace generation_config
  Greedy = what beam search starts from.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_repetition_penalty(logits: np.ndarray, ids: list, penalty: float) -> np.ndarray:
    """
    Penalise tokens already in `ids`.
    Positive logits are divided (making them less likely).
    Negative logits are multiplied (pushing them further down).
    penalty=1.0 → no change.
    """
    for tid in set(ids):
        if 0 <= tid < len(logits):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
    return logits


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k logits; set the rest to -inf.
    k=0 → disabled.
    """
    if k <= 0 or k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]
    out = logits.copy()
    out[out < threshold] = -1e9
    return out


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    softmax probability sums to >= p.  Zero out everything else.
    p=1.0 → disabled (keep all).

    Why this beats top-k:
    At a confident step, the top-1 token may cover 0.99 probability mass.
    top-k=10 still samples 10 tokens. top-p=0.9 keeps just that 1 token.
    Top-p is adaptive to the model's certainty at each step.
    """
    if p >= 1.0:
        return logits

    shifted = logits - logits.max()
    probs   = np.exp(shifted) / np.exp(shifted).sum()

    sorted_idx  = np.argsort(probs)[::-1]
    cumsum      = np.cumsum(probs[sorted_idx])

    cutoff = int(np.searchsorted(cumsum, p)) + 1
    cutoff = max(1, min(cutoff, len(logits)))

    out = logits.copy()
    out[sorted_idx[cutoff:]] = -1e9
    return out


def _softmax_sample(logits: np.ndarray) -> int:
    """Numerically stable softmax then multinomial sample."""
    logits = logits - logits.max()
    probs  = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    seed_text:      str,
    max_new_tokens: int,
    block_size:     int,
    temperature:    float = 1.0,
    top_k:          int   = 0,
    top_p:          float = 1.0,
    greedy:         bool  = False,
    rep_penalty:    float = 1.0,
) -> str:
    """
    Autoregressively generate `max_new_tokens` tokens from `model`.

    Parameters
    ----------
    seed_text      : string prompt used as initial context
    max_new_tokens : how many new tokens to produce
    block_size     : model's context window; input is cropped to this
    temperature    : logit scale factor. <1 = sharper, >1 = flatter, 1.0 = unchanged
    top_k          : sample only from the top-k tokens (0 = off)
    top_p          : nucleus threshold: keep tokens summing to >= p mass (1.0 = off)
    greedy         : always pick argmax; overrides temperature / top-k / top-p
    rep_penalty    : penalise already-seen tokens (1.0 = off, 1.3 = mild)
    """
    ids = tokenizer.encode(seed_text)
    if not ids:
        ids = [0]

    # Detect whether the model is a PyTorch nn.Module so we can feed it tensors.
    try:
        import torch
        import torch.nn as nn
        _is_torch = isinstance(model, nn.Module)
    except ImportError:
        _is_torch = False

    for _ in range(max_new_tokens):
        x_np = np.array(ids[-block_size:], dtype=np.int32).reshape(1, -1)

        if _is_torch:
            import torch
            x_in = torch.tensor(x_np, dtype=torch.long)
            with torch.no_grad():
                out = model.forward(x_in)
            # Convert output back to numpy for the rest of the pipeline
            logits = out[0, -1, :].cpu().numpy().astype(np.float64).copy()
        else:
            out    = model.forward(x_np)
            out    = out[0] if isinstance(out, tuple) else out   # (1, T, V)
            logits = out[0, -1, :].astype(np.float64).copy()  # last position

        # 1. Repetition penalty
        if rep_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, ids, rep_penalty)

        # 2. Greedy shortcut
        if greedy:
            ids.append(int(np.argmax(logits)))
            continue

        # 3. Temperature
        logits = logits / max(temperature, 1e-6)

        # 4. Top-k
        logits = _top_k_filter(logits, top_k)

        # 5. Top-p
        logits = _top_p_filter(logits, top_p)

        # 6. Sample
        ids.append(_softmax_sample(logits))

    return tokenizer.decode(ids)
