"""
Phase 5 — Optimizers.

AdamW
-----
Decoupled weight decay regularisation (Loshchilov & Hutter, 2017).

SGD worked for Phases 2–4 but has no momentum and its LR must be tuned
carefully per problem.  AdamW fixes both:

  m_t = β₁·m_{t-1} + (1-β₁)·g_t           ← first moment (momentum)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          ← second moment (RMS)
  m̂   = m_t / (1-β₁^t)                     ← bias correction (near zero at t=1)
  v̂   = v_t / (1-β₂^t)
  θ   = θ - lr · (m̂/(√v̂+ε) + λ·θ)        ← update + weight decay

Key difference from Adam: weight decay λ is applied directly to the parameter
θ (decoupled), NOT folded into the gradient.  This makes the effective
regularisation independent of the adaptive learning rate.

Params excluded from weight decay: biases, LayerNorm γ/β, embeddings.
(Standard practice — these are scale/shift parameters, not weight matrices.)

SGD (wrapper)
-------------
Plain gradient descent, kept for Phase 2/3 backward compatibility and
for direct comparison with AdamW.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# No-decay parameter name suffixes
# ─────────────────────────────────────────────────────────────────────────────

_NO_DECAY = frozenset({
    'b', 'b1', 'b2',          # FFN biases
    'beta', 'gamma',           # LayerNorm shift and scale
    'token_embed', 'pos_embed' # embeddings (names, not suffixes — checked separately)
})

def _should_decay(name: str) -> bool:
    """True if this parameter should have weight decay applied."""
    base = name.split('.')[-1]                    # last component of dotted path
    full = name                                    # full path for embeddings
    return base not in _NO_DECAY and full not in _NO_DECAY


# ─────────────────────────────────────────────────────────────────────────────
# AdamW
# ─────────────────────────────────────────────────────────────────────────────

class AdamW:
    """
    AdamW optimizer.  Works with any model that implements:
      - _flat_params()  → generator of (name, param_array) pairs
      - _flat_grads(grads) → generator of (name, grad_array) pairs
        in the SAME order as _flat_params().

    Parameters
    ----------
    lr           : peak / initial learning rate (updated externally by scheduler)
    beta1        : momentum decay (default 0.9)
    beta2        : RMS decay (default 0.999)
    eps          : denominator stabiliser (default 1e-8)
    weight_decay : decoupled L2 coefficient (default 0.01)
    """

    def __init__(
        self,
        lr:           float = 3e-4,
        beta1:        float = 0.9,
        beta2:        float = 0.999,
        eps:          float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0       # step counter for bias correction
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def step(self, model, grads: dict) -> float:
        """
        Apply one AdamW update to all model parameters.
        Returns the global gradient L2 norm (computed before the update, after
        any external clipping the caller already applied to `grads`).

        Parameters
        ----------
        model  : model with _flat_params() and _flat_grads()
        grads  : gradient dict returned by model.loss_and_grads()
        """
        self.t += 1
        t = self.t
        b1, b2, eps = self.beta1, self.beta2, self.eps

        # Bias-correction factors — compensate for zero initialisation of moments
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        total_sq = 0.0

        for (name, param), (_, grad) in zip(
            model._flat_params(), model._flat_grads(grads)
        ):
            total_sq += float((grad ** 2).sum())
            # Lazy init of moment buffers on first encounter
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)

            m = self._m[name]
            v = self._v[name]

            # Update moments in-place (saves allocation)
            m *= b1;  m += (1.0 - b1) * grad
            v *= b2;  v += (1.0 - b2) * grad ** 2

            m_hat = m / bc1
            v_hat = v / bc2

            update = m_hat / (np.sqrt(v_hat) + eps)

            # Decoupled weight decay — only on true weight matrices
            if _should_decay(name):
                update += self.weight_decay * param

            param -= self.lr * update

        return total_sq ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SGD (thin wrapper — preserves Phase 2-4 compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class SGD:
    """Plain gradient descent.  Wraps model.sgd_step() so the Trainer can
    call optimizer.step(model, grads) regardless of which optimizer is active."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, model, grads: dict):
        model.sgd_step(grads, self.lr)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient clipping
# ─────────────────────────────────────────────────────────────────────────────

def clip_grad_norm(flat_grads: list, max_norm: float) -> float:
    """
    Clip the global L2 norm of all gradients to max_norm.
    Modifies gradients in-place.
    Returns the pre-clip norm so it can be logged.

    Parameters
    ----------
    flat_grads : list of (name, grad_array) from list(model._flat_grads(grads))
    max_norm   : clip threshold (common values: 1.0, 5.0)
    """
    total_sq = sum(float((g ** 2).sum()) for _, g in flat_grads)
    norm = total_sq ** 0.5
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for _, g in flat_grads:
            g *= scale           # in-place — affects the original grads dict
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(cfg: dict) -> 'AdamW | SGD':
    """Construct the right optimizer from config."""
    tcfg = cfg['training']
    kind = str(tcfg.get('optimizer', 'adamw')).lower()
    lr   = float(tcfg['lr'])
    if kind == 'adamw':
        return AdamW(
            lr=lr,
            beta1=float(tcfg.get('beta1', 0.9)),
            beta2=float(tcfg.get('beta2', 0.999)),
            weight_decay=float(tcfg.get('weight_decay', 0.01)),
        )
    return SGD(lr=lr)
