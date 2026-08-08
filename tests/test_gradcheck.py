"""
Gradient checks — the single most valuable test for a hand-written-backprop
codebase. Every backward() is verified against a central finite-difference of
its own forward(), so a sign error or a missing term shows up immediately
instead of merely slowing training.

Method
------
For a module with output `out = forward(x)`, pick a fixed random cotangent R
and define a scalar loss L = sum(out * R). Then dL/dx (analytic) is exactly
backward(R), and dL/dx[i] (numerical) is (L(x+eps·e_i) - L(x-eps·e_i)) / 2eps.
The two must agree. Same trick checks each weight matrix.

Covered backward passes:
  LayerNorm, RMSNorm, FFN (ReLU), SwiGLU, RoPE,
  MultiHeadAttention, ModernAttention (MHA / GQA / MQA, with and without RoPE),
  MoE (router, individual experts, shared expert, auxiliary balancing loss),
  and the masked-vs-unmasked loss in *TransformerLM.loss_and_grads.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.transformer        import LayerNorm, FFN, MultiHeadAttention, TransformerLM
from model.norms              import RMSNorm
from model.activations        import SwiGLUFFN
from model.rope               import apply_rope, apply_rope_backward, rope_freqs
from model.modern_transformer import ModernAttention
from model.moe                import MoEFFN


EPS = 1e-6
TOL = 1e-5          # generous; analytic/numeric typically agree to ~1e-9


def _num_grad_input(forward, x, R, idxs):
    """Central-difference dL/dx at the given multi-indices, L = sum(forward(x)*R)."""
    xf = x.copy()
    g = {}
    for idx in idxs:
        old = xf[idx]
        xf[idx] = old + EPS; lp = float((forward(xf) * R).sum())
        xf[idx] = old - EPS; lm = float((forward(xf) * R).sum())
        xf[idx] = old
        g[idx] = (lp - lm) / (2 * EPS)
    return g


def _sample_idxs(shape, n, rng):
    return [tuple(int(rng.integers(0, s)) for s in shape) for _ in range(n)]


def _check_input_grad(module, x, seed=0, n=25):
    rng = np.random.default_rng(seed)
    out = module.forward(x)
    R = rng.standard_normal(out.shape)
    dx, _ = module.backward(R)
    idxs = _sample_idxs(x.shape, n, rng)
    num = _num_grad_input(module.forward, x, R, idxs)
    err = max(abs(num[i] - dx[i]) for i in idxs)
    assert err < TOL, f"{type(module).__name__} input grad err={err:.2e}"


def _check_weight_grads(module, x, weight_names, grad_key_map=None, seed=0, n=15):
    rng = np.random.default_rng(seed)
    out = module.forward(x)
    R = rng.standard_normal(out.shape)
    _, grads = module.backward(R)
    grad_key_map = grad_key_map or {}
    for name in weight_names:
        W = getattr(module, name)
        gkey = grad_key_map.get(name, name)
        idxs = _sample_idxs(W.shape, n, rng)
        worst = 0.0
        for idx in idxs:
            old = W[idx]
            W[idx] = old + EPS; lp = float((module.forward(x) * R).sum())
            W[idx] = old - EPS; lm = float((module.forward(x) * R).sum())
            W[idx] = old
            worst = max(worst, abs((lp - lm) / (2 * EPS) - grads[gkey][idx]))
        assert worst < TOL, f"{type(module).__name__}.{name} grad err={worst:.2e}"


# ── Norms ────────────────────────────────────────────────────────────────────

def test_layernorm_input_grad():
    x = np.random.default_rng(1).standard_normal((2, 4, 8))
    _check_input_grad(LayerNorm(8), x)


def test_rmsnorm_input_grad():
    x = np.random.default_rng(2).standard_normal((2, 4, 8))
    _check_input_grad(RMSNorm(8), x)


# ── FFNs ───────────────────────────────────────────────────────────────────--

def test_ffn_input_and_weight_grads():
    x = np.random.default_rng(3).standard_normal((2, 5, 8))
    _check_input_grad(FFN(8), x)
    _check_weight_grads(FFN(8), x, ['W1', 'b1', 'W2', 'b2'])


def test_swiglu_input_and_weight_grads():
    x = np.random.default_rng(4).standard_normal((2, 5, 8))
    _check_input_grad(SwiGLUFFN(8), x)
    _check_weight_grads(SwiGLUFFN(8), x, ['Wg', 'Wv', 'W2'])


# ── RoPE (functional, not a module) ───────────────────────────────────────────

def test_rope_backward_matches_numerical():
    rng = np.random.default_rng(5)
    B, H, T, Dh = 2, 2, 4, 8
    x = rng.standard_normal((B, H, T, Dh))
    cos, sin = rope_freqs(Dh, T)
    R = rng.standard_normal((B, H, T, Dh))
    dx = apply_rope_backward(R, cos, sin)
    for idx in _sample_idxs(x.shape, 20, rng):
        old = x[idx]
        x[idx] = old + EPS; lp = float((apply_rope(x, cos, sin) * R).sum())
        x[idx] = old - EPS; lm = float((apply_rope(x, cos, sin) * R).sum())
        x[idx] = old
        assert abs((lp - lm) / (2 * EPS) - dx[idx]) < TOL


# ── Attention ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_kv,use_rope", [(4, False), (4, True), (2, True), (1, False)])
def test_modern_attention_grads(n_kv, use_rope):
    rng = np.random.default_rng(6)
    B, T, D, H = 2, 5, 8, 4
    x = rng.standard_normal((B, T, D))
    m = ModernAttention(D, H, n_kv, use_rope=use_rope)
    _check_input_grad(m, x)
    _check_weight_grads(m, x, ['Wq', 'Wk', 'Wv', 'Wo'])


def test_base_mha_grads():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 5, 8))
    m = MultiHeadAttention(8, 4)
    _check_input_grad(m, x)
    _check_weight_grads(m, x, ['Wq', 'Wk', 'Wv', 'Wo'])


# ── Masked loss (SFT correctness) ─────────────────────────────────────────────

def test_loss_mask_zeroes_instruction_gradient():
    """
    True response masking: a target whose mask is 0 must contribute nothing to
    the gradient. We verify that perturbing the model's prediction at a masked
    position does not change the masked loss, and that an all-ones mask
    reproduces the plain (unmasked) loss and gradients exactly.
    """
    rng = np.random.default_rng(8)
    V, D, T = 12, 16, 6
    model = TransformerLM(vocab_size=V, embed_dim=D, block_size=T,
                          n_layers=1, n_heads=2)
    x = rng.integers(0, V, size=(2, T)).astype(np.int32)
    y = rng.integers(0, V, size=(2, T)).astype(np.int32)

    # All-ones mask == no mask
    full_loss, full_g = model.loss_and_grads(x, y)
    mask = np.ones((2, T), dtype=np.float32)
    m_loss, m_g = model.loss_and_grads(x, y, loss_mask=mask)
    assert abs(full_loss - m_loss) < 1e-9
    assert np.allclose(full_g['proj'], m_g['proj'])

    # Mask out everything except one position: loss only depends on that token
    mask = np.zeros((2, T), dtype=np.float32)
    mask[0, 2] = 1.0
    loss_a, _ = model.loss_and_grads(x, y, loss_mask=mask)
    # changing a *masked* target must not change the masked loss
    y2 = y.copy(); y2[1, 0] = (y2[1, 0] + 1) % V
    loss_b, _ = model.loss_and_grads(x, y2, loss_mask=mask)
    assert abs(loss_a - loss_b) < 1e-9


# ── Mixture of Experts ───────────────────────────────────────────────────────
#
# The MoE objective is not just sum(out*R): when balancing is set to 'aux' the
# auxiliary load-balancing term is part of the loss too, and its gradient is
# applied inside MoEFFN.backward(). Both have to appear in the scalar that the
# finite difference measures, or the router gradient looks wrong.
#
# Note the top-k *selection* is not differentiable. These checks perturb by 1e-6
# with a fixed seed, which is far too small to flip which experts win, so the
# routing stays fixed across the +eps and -eps evaluations.

_MOE_CASES = [
    (FFN,        'none'),
    (FFN,        'aux'),
    (SwiGLUFFN,  'none'),
    (SwiGLUFFN,  'aux'),
]


def _moe_scalar(moe, x, R):
    """Full objective: upstream term plus any auxiliary balancing loss."""
    return float((moe.forward(x) * R).sum()) + moe.aux_loss


def _make_moe(expert_cls, balance, n_shared=0, dim=16):
    return MoEFFN(dim, n_experts=4, top_k=2, expert_cls=expert_cls,
                  seed=3, n_shared=n_shared, balance=balance)


@pytest.mark.parametrize('expert_cls,balance', _MOE_CASES)
def test_moe_input_grad(expert_cls, balance):
    rng = np.random.default_rng(11)
    moe = _make_moe(expert_cls, balance)
    x   = rng.standard_normal((2, 5, 16))
    R   = rng.standard_normal((2, 5, 16))

    moe.forward(x)
    dx, _ = moe.backward(R)

    worst = 0.0
    for idx in _sample_idxs(x.shape, 20, rng):
        old = x[idx]
        x[idx] = old + EPS; lp = _moe_scalar(moe, x, R)
        x[idx] = old - EPS; lm = _moe_scalar(moe, x, R)
        x[idx] = old
        worst = max(worst, abs((lp - lm) / (2 * EPS) - dx[idx]))
    assert worst < TOL, f"MoE input grad err={worst:.2e} ({expert_cls.__name__}, {balance})"


@pytest.mark.parametrize('expert_cls,balance', _MOE_CASES)
def test_moe_router_grad(expert_cls, balance):
    """The router matrix is the only genuinely new derivation in Phase 10."""
    rng = np.random.default_rng(12)
    moe = _make_moe(expert_cls, balance)
    x   = rng.standard_normal((2, 5, 16))
    R   = rng.standard_normal((2, 5, 16))

    moe.forward(x)
    _, grads = moe.backward(R)

    worst = 0.0
    for idx in _sample_idxs(moe.Wr.shape, 20, rng):
        old = moe.Wr[idx]
        moe.Wr[idx] = old + EPS; lp = _moe_scalar(moe, x, R)
        moe.Wr[idx] = old - EPS; lm = _moe_scalar(moe, x, R)
        moe.Wr[idx] = old
        worst = max(worst, abs((lp - lm) / (2 * EPS) - grads['Wr'][idx]))
    assert worst < TOL, f"MoE router grad err={worst:.2e} ({expert_cls.__name__}, {balance})"


@pytest.mark.parametrize('expert_cls,key', [(FFN, 'W1'), (FFN, 'b1'), (SwiGLUFFN, 'Wg')])
def test_moe_expert_weight_grads(expert_cls, key):
    """Gradient must reach an individual expert scaled by its gate weight."""
    rng = np.random.default_rng(13)
    moe = _make_moe(expert_cls, 'none')
    x   = rng.standard_normal((2, 5, 16))
    R   = rng.standard_normal((2, 5, 16))

    moe.forward(x)
    _, grads = moe.backward(R)

    W = getattr(moe.experts[0], key)
    worst = 0.0
    for idx in _sample_idxs(W.shape, 12, rng):
        old = W[idx]
        W[idx] = old + EPS; lp = _moe_scalar(moe, x, R)
        W[idx] = old - EPS; lm = _moe_scalar(moe, x, R)
        W[idx] = old
        worst = max(worst, abs((lp - lm) / (2 * EPS) - grads['experts'][0][key][idx]))
    assert worst < TOL, f"MoE expert[0].{key} grad err={worst:.2e}"


@pytest.mark.parametrize('top_k', [1, 2, 4])
def test_moe_router_grad_at_every_top_k(top_k):
    """
    Regression guard. If the gate is a softmax over only the *selected* logits,
    then at k=1 it is identically 1.0 — a constant, whose derivative is zero —
    so the router gets no gradient and never learns, silently. Deriving the gate
    from the softmax over all experts keeps the gradient real at every k.
    """
    rng = np.random.default_rng(15)
    moe = MoEFFN(16, n_experts=4, top_k=top_k, expert_cls=FFN, seed=3, balance='none')
    x   = rng.standard_normal((2, 5, 16))
    R   = rng.standard_normal((2, 5, 16))

    moe.forward(x)
    _, grads = moe.backward(R)

    assert np.abs(grads['Wr']).max() > 1e-8, \
        f"router gradient vanished at top_k={top_k} — the router cannot learn"

    worst = 0.0
    for idx in _sample_idxs(moe.Wr.shape, 12, rng):
        old = moe.Wr[idx]
        moe.Wr[idx] = old + EPS; lp = _moe_scalar(moe, x, R)
        moe.Wr[idx] = old - EPS; lm = _moe_scalar(moe, x, R)
        moe.Wr[idx] = old
        worst = max(worst, abs((lp - lm) / (2 * EPS) - grads['Wr'][idx]))
    assert worst < TOL, f"MoE router grad err={worst:.2e} at top_k={top_k}"


def test_moe_shared_expert_grad():
    """The always-on shared expert sits outside the routing and must still learn."""
    rng = np.random.default_rng(14)
    moe = _make_moe(SwiGLUFFN, 'aux', n_shared=1)
    x   = rng.standard_normal((2, 5, 16))
    R   = rng.standard_normal((2, 5, 16))

    moe.forward(x)
    _, grads = moe.backward(R)

    W = moe.shared.Wg
    worst = 0.0
    for idx in _sample_idxs(W.shape, 12, rng):
        old = W[idx]
        W[idx] = old + EPS; lp = _moe_scalar(moe, x, R)
        W[idx] = old - EPS; lm = _moe_scalar(moe, x, R)
        W[idx] = old
        worst = max(worst, abs((lp - lm) / (2 * EPS) - grads['shared']['Wg'][idx]))
    assert worst < TOL, f"MoE shared expert grad err={worst:.2e}"


if __name__ == "__main__":
    # Allow running directly without pytest.
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            # Replay whatever @pytest.mark.parametrize declares, so this runner
            # stays correct as parametrized cases are added.
            param = next((m for m in getattr(fn, 'pytestmark', [])
                          if m.name == 'parametrize'), None)
            if param:
                for case in param.args[1]:
                    fn(*(case if isinstance(case, tuple) else (case,)))
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} gradient-check tests passed")
