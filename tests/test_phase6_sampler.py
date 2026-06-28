"""
Phase 6 tests — sampling strategies and checkpoint round-trip.

Approach
--------
Use a tiny model (vocab=10, embed=8, block=4) so tests are instant.
For stochastic tests we check distributional properties (expected token in
top-k/top-p set, repetition-penalty effect) rather than exact outputs.
For greedy and checkpoint we check exact equality.
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sample.sampler import (
    _apply_repetition_penalty,
    _top_k_filter,
    _top_p_filter,
    _softmax_sample,
    generate,
)
from sample.checkpoint import save, load

# ─── tiny model stub ──────────────────────────────────────────────────────────

class TinyModel:
    """
    Minimal model that always returns a fixed logit vector.
    Makes sampling strategies fully predictable in tests.
    """
    def __init__(self, vocab_size=10, embed_dim=8, block_size=4, n_layers=2, n_heads=2, dropout=0.0):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.block_size = block_size
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.dropout    = dropout
        # Fixed logits: token 0 is highest, decreasing
        self._logits = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.1, -1.0, -2.0, -5.0, -10.0])

    def forward(self, x):
        B, T = x.shape
        # Return (B, T, V) logits
        logits = np.broadcast_to(
            self._logits[np.newaxis, np.newaxis, :], (B, T, self.vocab_size)
        ).copy()
        return logits

    def set_logits(self, logits: np.ndarray):
        """For parameterised tests, override the fixed logits."""
        self._logits = np.array(logits, dtype=float)

    def param_count(self) -> dict:
        return {'total': 1000}

    def _flat_params(self):
        # Expose one weight so checkpoint has something to save
        yield 'W', self._logits.reshape(1, 10)

    def _flat_grads(self, grads):
        yield 'W', grads.get('W', np.zeros((1, 10)))


class TinyTokenizer:
    """10-token char-like tokenizer."""
    def __init__(self):
        self.vocab_size = 10
        self.vocab      = [str(i) for i in range(10)]
        self.stoi       = {c: i for i, c in enumerate(self.vocab)}
        self.itos       = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids)


# ─── Repetition penalty ───────────────────────────────────────────────────────

class TestRepetitionPenalty:

    def test_positive_logit_reduced(self):
        logits = np.array([5.0, 3.0, 1.0])
        result = _apply_repetition_penalty(logits.copy(), [0], penalty=2.0)
        assert result[0] == pytest.approx(2.5)   # 5.0 / 2.0
        assert result[1] == pytest.approx(3.0)   # unchanged
        assert result[2] == pytest.approx(1.0)

    def test_negative_logit_pushed_down(self):
        logits = np.array([-5.0, 3.0])
        result = _apply_repetition_penalty(logits.copy(), [0], penalty=2.0)
        assert result[0] == pytest.approx(-10.0)  # -5.0 * 2.0

    def test_no_penalty(self):
        logits = np.array([5.0, 3.0, 1.0])
        result = _apply_repetition_penalty(logits.copy(), [0, 1], penalty=1.0)
        np.testing.assert_array_equal(result, logits)

    def test_deduplicated_ids(self):
        """Each token penalised once even if it appears multiple times in ids."""
        logits = np.array([10.0, 5.0])
        result = _apply_repetition_penalty(logits.copy(), [0, 0, 0], penalty=2.0)
        assert result[0] == pytest.approx(5.0)   # not 10 / 2 / 2 / 2

    def test_out_of_range_ids_ignored(self):
        logits = np.array([5.0, 3.0])
        result = _apply_repetition_penalty(logits.copy(), [99, -1], penalty=2.0)
        np.testing.assert_array_equal(result, logits)


# ─── Top-k filter ─────────────────────────────────────────────────────────────

class TestTopKFilter:

    def _top_logits(self, logits, k):
        """Return the set of token indices that survive the filter."""
        out = _top_k_filter(logits, k)
        return set(np.where(out > -1e8)[0].tolist())

    def test_k0_disabled(self):
        logits = np.array([5.0, 3.0, 1.0, -1.0])
        np.testing.assert_array_equal(_top_k_filter(logits, 0), logits)

    def test_k1_keeps_only_top(self):
        logits = np.array([5.0, 3.0, 1.0, -1.0])
        survivors = self._top_logits(logits, 1)
        assert survivors == {0}

    def test_k2_keeps_top2(self):
        logits = np.array([5.0, 3.0, 1.0, -1.0])
        survivors = self._top_logits(logits, 2)
        assert survivors == {0, 1}

    def test_k_ge_vocab_keeps_all(self):
        logits = np.array([5.0, 3.0, 1.0])
        result = _top_k_filter(logits, 100)
        np.testing.assert_array_equal(result, logits)

    def test_original_logits_for_survivors_unchanged(self):
        logits = np.array([5.0, 3.0, 1.0, -1.0])
        out = _top_k_filter(logits, 2)
        assert out[0] == pytest.approx(5.0)
        assert out[1] == pytest.approx(3.0)

    def test_does_not_mutate_input(self):
        logits = np.array([5.0, 3.0, 1.0])
        orig   = logits.copy()
        _top_k_filter(logits, 2)
        np.testing.assert_array_equal(logits, orig)


# ─── Top-p (nucleus) filter ───────────────────────────────────────────────────

class TestTopPFilter:

    def _survivors(self, logits, p):
        out = _top_p_filter(logits, p)
        return set(np.where(out > -1e8)[0].tolist())

    def test_p1_disabled(self):
        logits = np.array([5.0, 3.0, 1.0])
        np.testing.assert_array_equal(_top_p_filter(logits, 1.0), logits)

    def test_p_very_small_keeps_at_least_one(self):
        """Even p=0.0 should keep at least the top token."""
        logits = np.array([10.0, 1.0, -10.0])
        survivors = self._survivors(logits, 0.01)
        assert len(survivors) >= 1
        assert 0 in survivors

    def test_high_confidence_model_filtered_tightly(self):
        """If token 0 has 0.99 probability, p=0.9 keeps only token 0."""
        logits = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        survivors = self._survivors(logits, 0.9)
        assert survivors == {0}

    def test_uniform_model_p95_keeps_most_tokens(self):
        """Uniform logits → equal probs → p=0.95 keeps ≥ 95% of tokens."""
        n      = 20
        logits = np.zeros(n)
        out    = _top_p_filter(logits, 0.95)
        n_kept = int((out > -1e8).sum())
        assert n_kept >= int(0.95 * n)

    def test_does_not_mutate_input(self):
        logits = np.array([5.0, 3.0, 1.0])
        orig   = logits.copy()
        _top_p_filter(logits, 0.9)
        np.testing.assert_array_equal(logits, orig)


# ─── Softmax sample ───────────────────────────────────────────────────────────

class TestSoftmaxSample:

    def test_concentrated_logits_always_pick_argmax(self):
        """With very peaked logits, sampler should always pick token 0."""
        np.random.seed(42)
        logits = np.array([1000.0, -1000.0, -1000.0])
        for _ in range(20):
            assert _softmax_sample(logits) == 0

    def test_returns_valid_index(self):
        np.random.seed(0)
        logits = np.array([1.0, 2.0, 3.0])
        for _ in range(50):
            idx = _softmax_sample(logits)
            assert 0 <= idx < len(logits)

    def test_high_temperature_more_uniform(self):
        """
        Scale logits by 1/T. High T → uniform → each token chosen roughly equally.
        Run 500 draws and verify no token dominates more than 70% of the time.
        """
        np.random.seed(7)
        logits   = np.array([1.0, 0.0, 0.0]) / 1e6   # ~uniform after temp=high
        counts   = [0, 0, 0]
        n        = 500
        for _ in range(n):
            counts[_softmax_sample(logits)] += 1
        assert max(counts) / n < 0.70


# ─── Greedy generation ────────────────────────────────────────────────────────

class TestGreedyGenerate:

    def test_always_picks_argmax(self):
        """Greedy should deterministically pick the highest-logit token every step."""
        model = TinyModel()
        tok   = TinyTokenizer()
        text  = generate(model, tok, '0', max_new_tokens=10,
                         block_size=4, greedy=True)
        # Seed '0' + 10 tokens all = '0' (token index 0 = char '0')
        assert text == '0' * 11

    def test_greedy_is_deterministic(self):
        model = TinyModel()
        tok   = TinyTokenizer()
        t1 = generate(model, tok, '1', max_new_tokens=8, block_size=4, greedy=True)
        t2 = generate(model, tok, '1', max_new_tokens=8, block_size=4, greedy=True)
        assert t1 == t2


# ─── Temperature ─────────────────────────────────────────────────────────────

class TestTemperatureGenerate:

    def test_temp_near_zero_approaches_greedy(self):
        """Very low temperature should almost always pick the top token."""
        np.random.seed(0)
        model = TinyModel()
        tok   = TinyTokenizer()
        # logits: 10, 5, 3, 1, 0.5, 0.1, -1, -2, -5, -10 → token 0 dominates
        results = [
            generate(model, tok, '0', max_new_tokens=1, block_size=4, temperature=0.01)
            for _ in range(30)
        ]
        # Every run should produce '0' + '0' (seed + top token)
        assert all(r == '00' for r in results)

    def test_temp_high_more_diverse(self):
        """High temperature should produce more token diversity than low."""
        np.random.seed(42)
        model = TinyModel()
        tok   = TinyTokenizer()

        low_results  = {generate(model, tok, '0', max_new_tokens=1,
                                  block_size=4, temperature=0.1)
                        for _ in range(20)}
        high_results = {generate(model, tok, '0', max_new_tokens=1,
                                  block_size=4, temperature=5.0)
                        for _ in range(20)}
        assert len(high_results) >= len(low_results)


# ─── Top-k in generation ─────────────────────────────────────────────────────

class TestTopKGenerate:

    def test_k1_always_picks_argmax(self):
        """top_k=1 → only one candidate → deterministic like greedy."""
        np.random.seed(0)
        model = TinyModel()
        tok   = TinyTokenizer()
        results = {
            generate(model, tok, '0', max_new_tokens=5, block_size=4,
                     temperature=1.0, top_k=1)
            for _ in range(20)
        }
        assert len(results) == 1   # always the same

    def test_k5_never_picks_low_token(self):
        """Tokens outside top-5 should never be chosen."""
        np.random.seed(123)
        model = TinyModel()
        tok   = TinyTokenizer()
        generated_tokens: set[int] = set()
        for _ in range(200):
            text = generate(model, tok, '0', max_new_tokens=1, block_size=4,
                            temperature=1.0, top_k=5)
            # new token is at position 1 (index 1 in text)
            generated_tokens.add(int(tok.encode(text[1])[0]))

        # All generated tokens should be in the top 5 (indices 0–4)
        assert generated_tokens.issubset({0, 1, 2, 3, 4})


# ─── Top-p in generation ─────────────────────────────────────────────────────

class TestTopPGenerate:

    def test_p_very_small_always_argmax(self):
        """p≈0 → only the top token included → deterministic."""
        np.random.seed(0)
        model = TinyModel()
        tok   = TinyTokenizer()
        results = {
            generate(model, tok, '0', max_new_tokens=1, block_size=4,
                     temperature=1.0, top_p=1e-9)
            for _ in range(20)
        }
        assert len(results) == 1

    def test_p1_same_as_no_filter(self):
        """p=1.0 should not filter anything (permissive baseline)."""
        np.random.seed(55)
        model  = TinyModel()
        tok    = TinyTokenizer()
        # Just verify it runs and returns the right length
        result = generate(model, tok, '0', max_new_tokens=10,
                          block_size=4, temperature=1.0, top_p=1.0)
        assert len(result) == 11   # seed + 10 new tokens


# ─── Repetition penalty in generation ────────────────────────────────────────

class TestRepPenaltyGenerate:

    def test_strong_penalty_discourages_repeats(self):
        """
        With a very strong rep_penalty the model should eventually pick
        a non-argmax token.  We use 200 draws of 2 tokens (seed + 1 new).
        The initial token '0' is placed in ids, so next token should shift
        away from '0' under strong penalty.
        """
        np.random.seed(0)
        model = TinyModel()
        tok   = TinyTokenizer()

        # Without penalty: argmax always picks '0'
        no_pen = [generate(model, tok, '0', max_new_tokens=1, block_size=4,
                            temperature=1.0, rep_penalty=1.0)
                  for _ in range(50)]
        no_pen_tok0_count = sum(1 for t in no_pen if t[1] == '0')

        # With strong penalty: '0' should appear less often as second token
        pen = [generate(model, tok, '0', max_new_tokens=1, block_size=4,
                         temperature=1.0, rep_penalty=100.0)
               for _ in range(50)]
        pen_tok0_count = sum(1 for t in pen if t[1] == '0')

        assert pen_tok0_count < no_pen_tok0_count


# ─── Checkpoint round-trip ────────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_load_weights_match(self, tmp_path):
        """Loaded model should produce identical outputs to the saved one."""
        from sample.checkpoint import save, load

        model = TinyModel()
        tok   = TinyTokenizer()
        path  = str(tmp_path / 'test.npz')

        # Inject a real TransformerLM so checkpoint tests the full pipeline
        from model.transformer import TransformerLM
        from data.tokenizer import CharTokenizer

        lm = TransformerLM(vocab_size=10, embed_dim=8, block_size=4,
                           n_layers=1, n_heads=2, dropout=0.0)
        char_tok = CharTokenizer('0123456789')  # 10-char vocab

        save(lm, char_tok, path)
        lm2, tok2 = load(path)

        # Architecture preserved
        assert lm2.vocab_size == lm.vocab_size
        assert lm2.embed_dim  == lm.embed_dim
        assert lm2.block_size == lm.block_size
        assert lm2.n_layers   == lm.n_layers
        assert lm2.n_heads    == lm.n_heads
        assert tok2.vocab_size == char_tok.vocab_size

        # Weights match — compare forward pass
        x      = np.array([[0, 1, 2, 3]], dtype=np.int32)
        out1   = lm.forward(x)
        out2   = lm2.forward(x)
        np.testing.assert_allclose(out1, out2, rtol=1e-6)

    def test_load_produces_same_greedy_generation(self, tmp_path):
        """After round-trip, greedy generation is bit-for-bit identical."""
        from sample.checkpoint import save, load
        from model.transformer import TransformerLM
        from data.tokenizer import CharTokenizer

        lm  = TransformerLM(vocab_size=10, embed_dim=8, block_size=4,
                            n_layers=1, n_heads=2, dropout=0.0)
        tok = CharTokenizer('0123456789')

        path = str(tmp_path / 'ckpt.npz')
        save(lm, tok, path)
        lm2, tok2 = load(path)

        g1 = generate(lm,  tok,  '0', max_new_tokens=20, block_size=4, greedy=True)
        g2 = generate(lm2, tok2, '0', max_new_tokens=20, block_size=4, greedy=True)
        assert g1 == g2

    def test_checkpoint_file_created(self, tmp_path):
        from sample.checkpoint import save
        from model.transformer import TransformerLM
        from data.tokenizer import CharTokenizer

        lm  = TransformerLM(vocab_size=5, embed_dim=8, block_size=4,
                            n_layers=1, n_heads=2, dropout=0.0)
        tok = CharTokenizer('abcde')
        path = str(tmp_path / 'weights')  # no .npz extension
        result = save(lm, tok, path)
        assert result.endswith('.npz')
        assert Path(result).exists()

    def test_all_strategies_run_after_load(self, tmp_path):
        """Smoke test: all sampling strategies execute without error post-load."""
        from sample.checkpoint import save, load
        from model.transformer import TransformerLM
        from data.tokenizer import CharTokenizer

        lm  = TransformerLM(vocab_size=10, embed_dim=8, block_size=4,
                            n_layers=1, n_heads=2, dropout=0.0)
        tok = CharTokenizer('0123456789')
        save(lm, tok, str(tmp_path / 'x.npz'))
        lm2, tok2 = load(str(tmp_path / 'x.npz'))

        strategies = [
            dict(greedy=True),
            dict(temperature=0.2),
            dict(temperature=1.5),
            dict(top_k=3),
            dict(top_p=0.8),
            dict(rep_penalty=1.3),
            dict(top_k=5, top_p=0.9, rep_penalty=1.1),
        ]
        for kwargs in strategies:
            out = generate(lm2, tok2, '0', max_new_tokens=5, block_size=4, **kwargs)
            assert len(out) == 6   # seed + 5 tokens


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
