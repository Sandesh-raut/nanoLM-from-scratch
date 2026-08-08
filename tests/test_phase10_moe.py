"""
Phase 10 tests — Mixture of Experts.

The gradients are checked in tests/test_gradcheck.py. What is tested here is
behaviour, which is where MoE actually goes wrong:

  1. Routing         (5 tests) — shapes, gate weights, top-k, utilization
  2. Sparsity        (3 tests) — active vs total parameters, shared expert
  3. Load balancing  (4 tests) — collapse without it, balance restored with it
  4. Integration     (5 tests) — flat params/grads, training, checkpoint, aux loss

The balancing tests train small models for a few hundred steps, so this file is
slower than the rest of the suite. That is unavoidable: routing collapse is a
training dynamic and cannot be observed from a single forward pass.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.moe                import MoEFFN
from model.transformer        import FFN
from model.activations        import SwiGLUFFN
from model.modern_transformer import ModernTransformerLM
from data.tokenizer           import CharTokenizer
from data.loader              import BatchLoader
from train.optimizer          import build_optimizer, clip_grad_norm


CORPUS = ("the quick brown fox jumps over the lazy dog\n"
          "def train(model, data):\n    return model.step(data)\n") * 200


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tok():
    return CharTokenizer(CORPUS)


def _model(tok, balance='bias', top_k=2, n_experts=4, n_shared=0, seed=42):
    return ModernTransformerLM(
        tok.vocab_size, 32, 24, n_layers=2, n_heads=4, seed=seed,
        norm='rmsnorm', ffn='swiglu', pos_enc='rope',
        moe=dict(n_experts=n_experts, top_k=top_k,
                 n_shared=n_shared, balance=balance),
    )


def _train(model, tok, steps=400, lr=3e-3, seed=42):
    """Train briefly and return the final loss. Enough to expose routing drift."""
    np.random.seed(seed)
    data   = np.array(tok.encode(CORPUS), dtype=np.int32)
    loader = BatchLoader(data, 24, 16, seed=seed)
    opt    = build_optimizer({'training': {'optimizer': 'adamw', 'lr': lr}})
    loss   = float('nan')
    for _ in range(steps):
        x, y = loader.next_batch()
        loss, grads = model.loss_and_grads(x, y)
        clip_grad_norm(list(model._flat_grads(grads)), 1.0)
        opt.step(model, grads)
    return loss


# ─── Routing ─────────────────────────────────────────────────────────────────

class TestRouting:

    def test_output_shape_matches_dense_ffn(self):
        moe = MoEFFN(16, n_experts=4, top_k=2, expert_cls=FFN, seed=1)
        x   = np.random.default_rng(0).standard_normal((2, 5, 16))
        assert moe.forward(x).shape == x.shape

    @pytest.mark.parametrize('top_k', [1, 2, 4])
    def test_exactly_top_k_experts_per_token(self, top_k):
        moe = MoEFFN(16, n_experts=4, top_k=top_k, expert_cls=FFN, seed=1)
        moe.forward(np.random.default_rng(0).standard_normal((2, 5, 16)))
        idx = moe._c['idx']
        assert idx.shape[1] == top_k
        # a token must never be routed to the same expert twice
        for row in idx:
            assert len(set(row.tolist())) == top_k

    def test_gate_weights_sum_to_one_when_k_above_one(self):
        moe = MoEFFN(16, n_experts=4, top_k=2, expert_cls=FFN, seed=1)
        moe.forward(np.random.default_rng(0).standard_normal((2, 5, 16)))
        np.testing.assert_allclose(moe._c['g'].sum(axis=1), 1.0, rtol=1e-12)

    def test_gate_at_top_k_one_is_a_real_probability(self):
        """Switch-style: not renormalized to 1, or the router loses its gradient."""
        moe = MoEFFN(16, n_experts=8, top_k=1, expert_cls=FFN, seed=1)
        moe.forward(np.random.default_rng(0).standard_normal((2, 5, 16)))
        g = moe._c['g']
        assert (g > 0).all() and (g < 1).all(), \
            "top-1 gate collapsed to a constant — router gradient would vanish"

    def test_utilization_is_a_distribution(self):
        moe = MoEFFN(16, n_experts=4, top_k=2, expert_cls=FFN, seed=1)
        moe.forward(np.random.default_rng(0).standard_normal((4, 8, 16)))
        u = moe.utilization()
        assert u.shape == (4,)
        np.testing.assert_allclose(u.sum(), 1.0, rtol=1e-12)
        assert 0.0 <= moe.balance_entropy() <= 1.0


# ─── Sparsity ────────────────────────────────────────────────────────────────

class TestSparsity:

    def test_active_params_below_total(self, tok):
        m = _model(tok, n_experts=8, top_k=2)
        total, active = m.param_count()['total'], m.active_param_count()
        assert active < total, "MoE must use fewer parameters per token than it stores"

    def test_more_experts_grow_total_far_faster_than_active(self, tok):
        """
        Adding experts must not add expert work per token — only k of them ever
        run. The one thing that does grow is the router, which is dense and sees
        every token, so it costs embed_dim extra weights per added expert per
        layer. That term is negligible next to the experts it is choosing between.
        """
        small = _model(tok, n_experts=4,  top_k=2)
        big   = _model(tok, n_experts=16, top_k=2)

        d_total  = big.param_count()['total']   - small.param_count()['total']
        d_active = big.active_param_count()     - small.active_param_count()

        router_growth = big.n_layers * big.embed_dim * (16 - 4)
        assert d_active == router_growth, "expert cost per token must stay flat"
        assert d_total > 20 * d_active, \
            f"total grew {d_total} but active grew {d_active} — sparsity gap too small"

    def test_shared_expert_is_always_active(self, tok):
        without = _model(tok, n_experts=4, top_k=2, n_shared=0)
        with_   = _model(tok, n_experts=4, top_k=2, n_shared=1)
        assert with_.active_param_count() > without.active_param_count()


# ─── Load balancing ──────────────────────────────────────────────────────────

class TestLoadBalancing:
    """
    Top-1 routing with no balancing is the case that actually collapses: the
    gate cannot spread a token across experts, so early winners compound.
    """

    def test_top1_routing_collapses_without_balancing(self, tok):
        m = _model(tok, balance='none', top_k=1, n_experts=8)
        _train(m, tok)
        u = m.expert_utilization()[0]
        assert u.max() > 2.0 / 8, \
            f"expected an over-used expert without balancing, got max share {u.max():.3f}"
        assert m.balance_entropy() < 0.9, \
            f"expected uneven routing without balancing, entropy={m.balance_entropy():.3f}"

    @pytest.mark.parametrize('balance', ['aux', 'bias'])
    def test_balancing_keeps_every_expert_alive(self, tok, balance):
        m = _model(tok, balance=balance, top_k=1, n_experts=8)
        _train(m, tok)
        u = m.expert_utilization()[0]
        assert (u > 0.01).all(), f"{balance}: {int((u <= 0.01).sum())} experts died"
        assert m.balance_entropy() > 0.9, \
            f"{balance}: routing entropy only {m.balance_entropy():.3f}"

    def test_balancing_beats_no_balancing_on_evenness(self, tok):
        unbalanced = _model(tok, balance='none', top_k=1, n_experts=8)
        balanced   = _model(tok, balance='bias', top_k=1, n_experts=8)
        _train(unbalanced, tok)
        _train(balanced, tok)
        assert balanced.balance_entropy() > unbalanced.balance_entropy()

    def test_bias_moves_against_load_and_is_training_only(self, tok):
        m     = _model(tok, balance='bias', top_k=1, n_experts=8)
        layer = m._moe_layers()[0]
        before = layer.expert_bias.copy()

        x = np.random.randint(0, tok.vocab_size, (4, 24))
        m.forward(x, training=False)
        np.testing.assert_array_equal(layer.expert_bias, before,
                                      "bias must not drift during inference")

        m.forward(x, training=True)
        assert not np.array_equal(layer.expert_bias, before)
        # underused experts get a positive nudge, overused a negative one
        counts = layer.load_counts
        delta  = layer.expert_bias - before
        assert delta[counts.argmin()] > 0 >= delta[counts.argmax()]


# ─── Integration ─────────────────────────────────────────────────────────────

class TestIntegration:

    def test_flat_params_and_grads_stay_aligned(self, tok):
        """AdamW zips these two generators — a mismatch silently trains garbage."""
        m = _model(tok, n_experts=4, top_k=2, n_shared=1)
        x = np.random.randint(0, tok.vocab_size, (2, 16))
        _, grads = m.loss_and_grads(x, x)
        ps, gs = list(m._flat_params()), list(m._flat_grads(grads))
        assert len(ps) == len(gs)
        for (pn, p), (gn, g) in zip(ps, gs):
            assert pn == gn, f"name mismatch: {pn} vs {gn}"
            assert p.shape == g.shape, f"shape mismatch at {pn}"

    def test_training_reduces_loss(self, tok):
        m = _model(tok, n_experts=4, top_k=2)
        x = np.random.randint(0, tok.vocab_size, (4, 24))
        before, _ = m.loss_and_grads(x, x)
        after = _train(m, tok, steps=150)
        assert after < before

    def test_aux_loss_only_present_for_aux_strategy(self, tok):
        x = np.random.randint(0, tok.vocab_size, (2, 16))
        for balance, expect_positive in (('none', False), ('bias', False), ('aux', True)):
            m = _model(tok, balance=balance)
            m.forward(x, training=True)
            assert (m.aux_loss() > 0) is expect_positive, \
                f"{balance}: unexpected aux loss {m.aux_loss()}"

    @pytest.mark.parametrize('n_shared,balance', [(0, 'aux'), (1, 'bias')])
    def test_checkpoint_round_trip(self, tok, tmp_path, n_shared, balance):
        from sample.checkpoint import save, load

        m = _model(tok, balance=balance, n_experts=4, top_k=2, n_shared=n_shared)
        m._moe_layers()[0].expert_bias[:] = np.array([0.1, -0.2, 0.3, -0.4])

        path = str(tmp_path / 'moe.npz')
        save(m, tok, path)
        m2, _ = load(path)

        assert m2.moe == m.moe
        np.testing.assert_allclose(m2._moe_layers()[0].expert_bias,
                                   m._moe_layers()[0].expert_bias)
        x = np.random.randint(0, tok.vocab_size, (2, 16))
        np.testing.assert_allclose(m.forward(x), m2.forward(x), rtol=1e-9)

    def test_dense_model_reports_no_moe_state(self, tok):
        m = ModernTransformerLM(tok.vocab_size, 32, 24, n_layers=1, n_heads=4)
        assert m._moe_layers() == []
        assert m.aux_loss() == 0.0
        assert m.balance_entropy() == 1.0
        assert m.active_param_count() == m.param_count()['total']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
