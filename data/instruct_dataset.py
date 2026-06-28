"""
Phase 8 — Instruction dataset.

Wraps raw text pairs into a consistent instruction template so the model
learns to *follow format*, not just predict next characters.

Template (Alpaca-style, kept simple for char-level tokenisation):

    ### System:
    {system}
    ### User:
    {user}
    ### Assistant:
    {assistant}
    ### End

Maps to
-------
  Real-world SFT datasets (Alpaca, OpenAssistant, ShareGPT) use the same
  idea: every example is (system, user, assistant) packed into a fixed
  template. The model trains on the formatted text; at inference time you
  stop generating after the assistant's turn.

Response masking
----------------
  A key insight: we should only back-propagate through the *response*
  tokens, not the instruction tokens. The instruction is a given condition,
  not something the model needs to predict.

  response_mask() returns a binary array aligned with the token sequence:
    0 = instruction / template prefix  (ignore in loss)
    1 = assistant response             (include in loss)
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Template
# ─────────────────────────────────────────────────────────────────────────────

# Full example (training)
TEMPLATE = (
    "### System:\n{system}\n"
    "### User:\n{user}\n"
    "### Assistant:\n{assistant}\n"
    "### End\n"
)

# Inference prompt: everything up to (but not including) the response
PROMPT_TEMPLATE = (
    "### System:\n{system}\n"
    "### User:\n{user}\n"
    "### Assistant:\n"
)

# Marker that separates instruction from response in the template
RESPONSE_START = "### Assistant:\n"
RESPONSE_END   = "\n### End\n"

DEFAULT_SYSTEM = "You are nanoLM, a tiny language model. Be helpful and concise."


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstructPair:
    system:    str
    user:      str
    assistant: str


# ─────────────────────────────────────────────────────────────────────────────
# Default pairs (used when no external dataset is provided)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PAIRS: list[InstructPair] = [
    InstructPair(DEFAULT_SYSTEM, "What is your name?",
                 "I am nanoLM, a tiny language model built from scratch."),
    InstructPair(DEFAULT_SYSTEM, "Say hello.",
                 "Hello! I am nanoLM. Nice to meet you."),
    InstructPair(DEFAULT_SYSTEM, "What can you do?",
                 "I can generate text one character at a time."),
    InstructPair(DEFAULT_SYSTEM, "Count to five.",
                 "One, two, three, four, five."),
    InstructPair(DEFAULT_SYSTEM, "What is two plus two?",
                 "Two plus two is four."),
    InstructPair(DEFAULT_SYSTEM, "Repeat after me: nanoLM.",
                 "nanoLM."),
    InstructPair(DEFAULT_SYSTEM, "Who made you?",
                 "I was built from scratch using NumPy and PyTorch."),
    InstructPair(DEFAULT_SYSTEM, "What is a language model?",
                 "A language model predicts the next token given a sequence of tokens."),
    InstructPair(DEFAULT_SYSTEM, "Tell me something interesting.",
                 "Every large language model started as a tiny one like me."),
    InstructPair(DEFAULT_SYSTEM, "How do you work?",
                 "I use a transformer with self-attention and feed-forward layers."),
    InstructPair(DEFAULT_SYSTEM, "What is attention?",
                 "Attention lets the model weigh which tokens are most relevant."),
    InstructPair(DEFAULT_SYSTEM, "What is a token?",
                 "A token is the smallest unit I process. For me each character is one token."),
    InstructPair(DEFAULT_SYSTEM, "Be brief: what is backpropagation?",
                 "Backpropagation computes gradients by applying the chain rule in reverse."),
    InstructPair(DEFAULT_SYSTEM, "What comes after learning?",
                 "After pre-training comes fine-tuning, then alignment."),
    InstructPair(DEFAULT_SYSTEM, "Say goodbye.",
                 "Goodbye! Thanks for chatting with nanoLM."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_pair(pair: InstructPair) -> str:
    """Return the full formatted training example (system + user + assistant)."""
    return TEMPLATE.format(
        system=pair.system,
        user=pair.user,
        assistant=pair.assistant,
    )


def format_prompt(system: str, user: str) -> str:
    """
    Return the inference-time prompt (no assistant response).
    Feed this to generate(); it will complete from '### Assistant:' onward.
    """
    return PROMPT_TEMPLATE.format(system=system, user=user)


def build_corpus(pairs: Sequence[InstructPair], repeat: int = 20) -> str:
    """
    Concatenate all formatted pairs, repeated `repeat` times so the
    model sees enough examples during the short SFT phase.
    """
    block = "\n".join(format_pair(p) for p in pairs) + "\n"
    return block * repeat


# ─────────────────────────────────────────────────────────────────────────────
# Response masking
# ─────────────────────────────────────────────────────────────────────────────

def response_mask(tokens: list[int], tokenizer, pairs: Sequence[InstructPair]) -> np.ndarray:
    """
    Build a binary mask (same length as `tokens`) where:
      1 = token belongs to an assistant response  (compute loss here)
      0 = token belongs to system/user/template   (skip in loss)

    Strategy: decode the token sequence back to text, find every
    '### Assistant:\\n' ... '\\n### End' span, mark those positions 1.

    This is an approximate character-level mask — works cleanly for
    char tokenisers where one token == one character.
    """
    text = tokenizer.decode(tokens)
    mask = np.zeros(len(tokens), dtype=np.float32)

    # Rebuild corpus to locate response spans
    pos = 0
    while True:
        start = text.find(RESPONSE_START, pos)
        if start == -1:
            break
        start += len(RESPONSE_START)   # first char of response
        end = text.find(RESPONSE_END, start)
        if end == -1:
            end = len(text)
        # For char tokeniser: token index == char index
        mask[start:end] = 1.0
        pos = end + len(RESPONSE_END)

    return mask


def masked_cross_entropy(logits: np.ndarray,
                         targets: np.ndarray,
                         mask: np.ndarray) -> float:
    """
    Cross-entropy loss computed only over tokens where mask == 1.

    logits  : (T, V) float64
    targets : (T,)   int32
    mask    : (T,)   float32  0 or 1

    Returns scalar loss (average over unmasked positions).
    """
    T, V = logits.shape
    # Numerically stable softmax
    shifted   = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    # Per-token NLL
    nll = -log_probs[np.arange(T), targets]   # (T,)
    # Apply mask and average
    total_mask = mask.sum()
    if total_mask == 0:
        return float(nll.mean())   # fallback if mask is all-zero
    return float((nll * mask).sum() / total_mask)
