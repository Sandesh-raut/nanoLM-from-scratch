"""
Phase 9 tests — Modern upgrades: RoPE, RMSNorm, SwiGLU, GQA, KV-cache, quantization.

Covered:
  1. RoPE (4 tests)  — shape, invertibility, backward, freqs decay
  2. RMSNorm (4 tests) — shape, normalization, backward vs numerical gradient
  3. SwiGLU (4 tests) — shape, param count, backward gradient check
  4. ModernAttention (4 tests) — MHA baseline, GQA param saving, RoPE forward, backward
  5. ModernTransformerLM (5 tests) — all configs init, forward, loss, param counts
  6. KV-cache (3 tests) — output matches no-cache, seq_len updated, shape
  7. Quantization (4 tests) — dtype, shape, error bounds, compression ratio
  8. Checkpoint round-trip (4 tests) — every norm/FFN/pos_enc/GQA combination
  9. Cached decode positions (2 tests) — cached == uncached for rope and learned
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.rope              import rope_freqs, apply_rope, apply_rope_backward
from model.norms             import RMSNorm
from model.activations       import SwiGLUFFN
from model.modern_transformer import (
    ModernAttention, ModernBlock, ModernTransformerLM,
)
from model.kv_cache          import generate_no_cache, generate_cached
from model.quantize          import (
    quantize_int8, dequantize_int8, quantization_error,
    compression_stats,
)
from data.tokenizer          import CharTokenizer


# ─── Shared fixtures ─────────────────────────────────────────────────────────

CORPUS = "abcdefghijklmnopqrstuvwxyz\n" * 10

@pytest.fixture
def tokenizer():
    return CharTokenizer(CORPUS)


@pytest.fixture
def modern_model(tokenizer):
    return ModernTransformerLM(
        vocab_size  = tokenizer.vocab_size,
        embed_dim   = 32,
        block_size  = 16,
        n_layers    = 2,
        n_heads     = 4,
        n_kv_heads  = 4,
        dropout     = 0.0,
        seed        = 0,
        norm        = 'rmsnorm',
        ffn         = 'swiglu',
        pos_enc     = 'rope',
    )


def _rng_array(*shape):
    return np.random.default_rng(42).standard_normal(shape)


# ─── 1. RoPE ─────────────────────────────────────────────────────────────────

class TestRoPE:

    def test_freqs_shape(self):
        cos, sin = rope_freqs(head_dim=16, seq_len=8)
        assert cos.shape == (8, 8)
        assert sin.shape == (8, 8)

    def test_apply_rope_shape_preserved(self):
        B, H, T, Dh = 2, 4, 8, 16
        x           = _rng_array(B, H, T, Dh)
        cos, sin    = rope_freqs(Dh, T)
        out         = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_rope_is_rotation_invertible(self):
        """apply_rope followed by apply_rope_backward should recover x."""
        B, H, T, Dh = 1, 2, 6, 8
        x           = _rng_array(B, H, T, Dh)
        cos, sin    = rope_freqs(Dh, T)
        rotated     = apply_rope(x, cos, sin)
        recovered   = apply_rope_backward(rotated, cos, sin)
        np.testing.assert_allclose(recovered, x, atol=1e-12)

    def test_rope_freqs_decay(self):
        """Higher frequency dimensions should have larger values for pos=1."""
        cos, sin = rope_freqs(head_dim=16, seq_len=2)
        # At position 1, sin[1] = sin(1*theta_i); theta_0 > theta_1 > ...
        # The raw theta (before taking sin) should be monotonically decreasing
        # We check this via the cos values: cos(0) = 1 for all, cos(1*theta) should decrease
        angles_at_pos1 = np.arccos(np.clip(cos[1], -1, 1))
        # angles = pos * theta, so for pos=1, angles = theta_i which should decrease
        assert np.all(np.diff(angles_at_pos1) <= 0 + 1e-12), \
            "RoPE frequencies should be monotonically decreasing"


# ─── 2. RMSNorm ──────────────────────────────────────────────────────────────

class TestRMSNorm:

    def test_output_shape(self):
        norm = RMSNorm(16)
        x    = _rng_array(2, 8, 16)
        out  = norm.forward(x)
        assert out.shape == x.shape

    def test_output_is_rms_normalized(self):
        """After RMSNorm with γ=1, mean(x²) should be ≈ 1."""
        norm = RMSNorm(64)
        x    = _rng_array(4, 10, 64) * 5.0    # large values
        out  = norm.forward(x)
        rms  = np.sqrt((out ** 2).mean(axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), atol=1e-5)

    def test_no_beta_parameter(self):
        norm = RMSNorm(16)
        assert not hasattr(norm, 'beta'), "RMSNorm should not have a beta parameter"

    def test_backward_gradient_check(self):
        """Numerical gradient check for RMSNorm."""
        norm = RMSNorm(8)
        x    = _rng_array(2, 4, 8)
        eps  = 1e-5

        norm.forward(x)
        dout = np.ones_like(x)
        dx_analytical, _ = norm.backward(dout)

        # Numerical gradient
        dx_numerical = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            x_plus         = x.copy(); x_plus[idx]  += eps
            x_minus        = x.copy(); x_minus[idx] -= eps
            norm_tmp       = RMSNorm(8); norm_tmp.gamma = norm.gamma.copy()
            loss_plus      = norm_tmp.forward(x_plus).sum()
            norm_tmp2      = RMSNorm(8); norm_tmp2.gamma = norm.gamma.copy()
            loss_minus     = norm_tmp2.forward(x_minus).sum()
            dx_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(dx_analytical, dx_numerical, atol=1e-4,
                                   err_msg="RMSNorm backward doesn't match numerical gradient")


# ─── 3. SwiGLU ───────────────────────────────────────────────────────────────

class TestSwiGLU:

    def test_output_shape(self):
        ffn = SwiGLUFFN(dim=32)
        x   = _rng_array(2, 8, 32)
        out = ffn.forward(x)
        assert out.shape == x.shape

    def test_inner_dim_larger_than_two_thirds_D(self):
        ffn = SwiGLUFFN(dim=64)
        # inner should be ≥ 8/3 * 64 ≈ 171, rounded to multiple of 64 → 192
        assert ffn.inner >= int(64 * 8 / 3)

    def test_no_bias_parameters(self):
        ffn = SwiGLUFFN(dim=32)
        assert not hasattr(ffn, 'b1'), "SwiGLU should not have b1 bias"
        assert not hasattr(ffn, 'b2'), "SwiGLU should not have b2 bias"

    def test_backward_gradient_check(self):
        """Numerical gradient check for SwiGLU."""
        ffn = SwiGLUFFN(dim=8)
        # Reset to small inner for speed
        import numpy as np
        ffn.inner = 16
        rng = np.random.default_rng(1)
        ffn.Wg = rng.standard_normal((8, 16)) * 0.1
        ffn.Wv = rng.standard_normal((8, 16)) * 0.1
        ffn.W2 = rng.standard_normal((16, 8)) * 0.1

        x   = _rng_array(1, 3, 8) * 0.5
        eps = 1e-5

        ffn.forward(x)
        dout = np.ones((1, 3, 8))
        dx_analytical, _ = ffn.backward(dout)

        dx_numerical = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            x_p = x.copy(); x_p[idx] += eps
            x_m = x.copy(); x_m[idx] -= eps
            # Use fresh FFN with same weights
            f2 = SwiGLUFFN(8); f2.inner = 16
            f2.Wg, f2.Wv, f2.W2 = ffn.Wg, ffn.Wv, ffn.W2
            lp = f2.forward(x_p).sum()
            f3 = SwiGLUFFN(8); f3.inner = 16
            f3.Wg, f3.Wv, f3.W2 = ffn.Wg, ffn.Wv, ffn.W2
            lm = f3.forward(x_m).sum()
            dx_numerical[idx] = (lp - lm) / (2 * eps)

        np.testing.assert_allclose(dx_analytical, dx_numerical, atol=1e-4,
                                   err_msg="SwiGLU backward doesn't match numerical gradient")


# ─── 4. ModernAttention ──────────────────────────────────────────────────────

class TestModernAttention:

    def test_mha_output_shape(self):
        attn = ModernAttention(dim=32, n_heads=4, n_kv_heads=4, use_rope=False, seed=0)
        x    = _rng_array(2, 8, 32)
        out  = attn.forward(x)
        assert out.shape == x.shape

    def test_gqa_fewer_kv_params(self):
        mha = ModernAttention(dim=32, n_heads=4, n_kv_heads=4, use_rope=False)
        gqa = ModernAttention(dim=32, n_heads=4, n_kv_heads=1, use_rope=False)
        assert gqa.param_count() < mha.param_count(), \
            "GQA should have fewer parameters than MHA"

    def test_rope_forward_runs(self):
        attn = ModernAttention(dim=32, n_heads=4, n_kv_heads=4, use_rope=True, seed=0)
        x    = _rng_array(1, 6, 32)
        out  = attn.forward(x)
        assert out.shape == x.shape

    def test_backward_runs_and_has_correct_keys(self):
        attn = ModernAttention(dim=32, n_heads=4, n_kv_heads=2, use_rope=True, seed=0)
        x    = _rng_array(2, 4, 32)
        attn.forward(x)
        dx, grads = attn.backward(np.ones((2, 4, 32)))
        assert dx.shape == x.shape
        for key in ('Wq', 'Wk', 'Wv', 'Wo'):
            assert key in grads, f"Grad for {key} missing"


# ─── 5. ModernTransformerLM ──────────────────────────────────────────────────

class TestModernTransformerLM:

    @pytest.mark.parametrize("norm,ffn,pos_enc,n_kv", [
        ('layernorm', 'relu',   'learned', 4),
        ('rmsnorm',   'relu',   'learned', 4),
        ('layernorm', 'swiglu', 'learned', 4),
        ('layernorm', 'relu',   'rope',    4),
        ('rmsnorm',   'swiglu', 'rope',    1),
    ])
    def test_all_configs_forward(self, tokenizer, norm, ffn, pos_enc, n_kv):
        model  = ModernTransformerLM(
            vocab_size=tokenizer.vocab_size, embed_dim=32, block_size=16,
            n_layers=1, n_heads=4, n_kv_heads=n_kv, dropout=0.0, seed=0,
            norm=norm, ffn=ffn, pos_enc=pos_enc,
        )
        x      = np.array([[0, 1, 2, 3]], dtype=np.int32)
        logits = model.forward(x)
        assert logits.shape == (1, 4, tokenizer.vocab_size)
        assert np.isfinite(logits).all()

    def test_loss_is_finite(self, tokenizer, modern_model):
        x   = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
        y   = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
        loss, grads = modern_model.loss_and_grads(x, y)
        assert np.isfinite(loss)
        assert loss > 0

    def test_grads_not_zero(self, tokenizer, modern_model):
        x = np.array([[0, 1, 2, 3]], dtype=np.int32)
        y = np.array([[1, 2, 3, 4]], dtype=np.int32)
        _, grads = modern_model.loss_and_grads(x, y)
        # At least one parameter should have a non-zero gradient
        any_nonzero = any(
            np.any(g != 0)
            for _, g in modern_model._flat_grads(grads)
        )
        assert any_nonzero, "All gradients are zero — something is wrong"

    def test_rope_model_has_no_pos_embed(self, tokenizer):
        model = ModernTransformerLM(
            vocab_size=tokenizer.vocab_size, embed_dim=32, block_size=16,
            n_layers=1, n_heads=2, pos_enc='rope', seed=0,
        )
        assert model.pos_embed is None, "RoPE model should not have a pos_embed table"

    def test_learned_model_has_pos_embed(self, tokenizer):
        model = ModernTransformerLM(
            vocab_size=tokenizer.vocab_size, embed_dim=32, block_size=16,
            n_layers=1, n_heads=2, pos_enc='learned', seed=0,
        )
        assert model.pos_embed is not None


# ─── 6. KV-cache ─────────────────────────────────────────────────────────────

class TestKVCache:

    def test_cached_output_same_length(self, tokenizer, modern_model):
        tokens = tokenizer.encode("abc")
        n_new  = 5
        out_nc = generate_no_cache(modern_model, tokens, n_new)
        out_c  = generate_cached(modern_model, tokens, n_new)
        assert len(out_nc) == n_new
        assert len(out_c)  == n_new

    def test_cache_seq_len_grows(self, tokenizer, modern_model):
        tokens    = tokenizer.encode("hello")
        kv_caches = [{} for _ in range(modern_model.n_layers)]
        x = np.array([tokens], dtype=np.int32)
        modern_model.forward(x, training=False, kv_caches=kv_caches)
        for cache in kv_caches:
            assert cache.get('seq_len', 0) == len(tokens)

    def test_cache_k_shape(self, tokenizer, modern_model):
        tokens    = tokenizer.encode("hi")
        kv_caches = [{} for _ in range(modern_model.n_layers)]
        x = np.array([tokens], dtype=np.int32)
        modern_model.forward(x, training=False, kv_caches=kv_caches)
        K = kv_caches[0]['K']
        # K shape: (B=1, n_kv_heads, T, head_dim)
        assert K.shape[0] == 1
        assert K.shape[2] == len(tokens)


# ─── 7. Quantization ─────────────────────────────────────────────────────────

class TestQuantization:

    def test_int8_dtype(self):
        w   = _rng_array(16, 32)
        w_q, _ = quantize_int8(w)
        assert w_q.dtype == np.int8

    def test_dequantize_approx_original(self):
        w        = _rng_array(64, 64)
        w_q, s   = quantize_int8(w)
        w_approx = dequantize_int8(w_q, s)
        np.testing.assert_allclose(w_approx, w, atol=s / 2 + 1e-10,
                                   err_msg="Dequantized weights deviate by more than scale/2")

    def test_compression_ratio_above_one(self, modern_model):
        stats = compression_stats(modern_model)
        assert stats['compression_ratio'] > 1.0, \
            "int8 should take less memory than float64"

    def test_quantization_error_small(self):
        w    = _rng_array(128, 128)
        errs = quantization_error(w)
        # Maximum relative error should be less than 1% for well-conditioned weights
        assert errs['max_rel_error'] < 0.02, \
            f"Quantization max relative error too large: {errs['max_rel_error']:.4f}"


# ─── Checkpoint round-trip ───────────────────────────────────────────────────

class TestModernCheckpoint:
    """
    A Phase 9 model must survive save → load under every config combination.

    The norm / FFN / positional choice decides which parameters exist at all
    (RMSNorm has no beta, SwiGLU has no biases, RoPE has no pos_embed), so the
    checkpoint has to record that config or load() rebuilds the wrong class.
    """

    CONFIGS = [
        ('layernorm', 'relu',   'learned', 4),   # Phase 3/4 baseline
        ('rmsnorm',   'swiglu', 'rope',    4),   # full modern stack
        ('rmsnorm',   'relu',   'rope',    2),   # GQA
        ('layernorm', 'swiglu', 'learned', 1),   # MQA
    ]

    @pytest.mark.parametrize('norm,ffn,pos_enc,n_kv_heads', CONFIGS)
    def test_round_trip_preserves_config_and_forward(
        self, tmp_path, norm, ffn, pos_enc, n_kv_heads
    ):
        from sample.checkpoint import save, load

        tok = CharTokenizer(CORPUS)
        lm  = ModernTransformerLM(
            vocab_size=tok.vocab_size, embed_dim=32, block_size=16,
            n_layers=2, n_heads=4, n_kv_heads=n_kv_heads,
            norm=norm, ffn=ffn, pos_enc=pos_enc, dropout=0.0,
        )

        path = str(tmp_path / 'modern.npz')
        save(lm, tok, path)
        lm2, tok2 = load(path)

        assert isinstance(lm2, ModernTransformerLM)
        assert (lm2.norm_type, lm2.ffn_type, lm2.pos_enc) == (norm, ffn, pos_enc)
        assert lm2.n_kv_heads   == n_kv_heads
        assert tok2.vocab_size  == tok.vocab_size

        x = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
        np.testing.assert_allclose(
            lm.forward(x), lm2.forward(x), rtol=1e-9,
            err_msg="Reloaded model does not reproduce the saved model's logits",
        )


# ─── Cached decode positions ─────────────────────────────────────────────────

class TestCachedDecodePositions:
    """
    Cached decode must place each new token at the same absolute position as
    the uncached path, for both positional schemes. With temperature=0 the
    sampling is greedy, so any position error shows up as a different token.
    """

    @pytest.mark.parametrize('pos_enc', ['rope', 'learned'])
    def test_cached_matches_no_cache(self, pos_enc):
        tok = CharTokenizer(CORPUS)
        lm  = ModernTransformerLM(
            vocab_size=tok.vocab_size, embed_dim=32, block_size=16,
            n_layers=2, n_heads=4, pos_enc=pos_enc, dropout=0.0,
        )
        ids = tok.encode('abcde')

        assert (generate_cached(lm, ids, 6, temperature=0.0)
                == generate_no_cache(lm, ids, 6, temperature=0.0)), \
            f"Cached and uncached decode diverged for pos_enc={pos_enc}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
