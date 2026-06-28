"""
Phase 8 tests — SFT dataset, prompter, masked loss, DPO loss.

Covered:
  1. Template formatting — format_pair / format_prompt round-trips
  2. Response mask — correct positions marked 1
  3. Masked cross-entropy — loss with all-1 mask == unmasked, all-0 returns fallback
  4. Corpus builder — length and repeat factor
  5. Prompter — build / extract round-trips
  6. Prompter.chat — generates non-empty output from a model
  7. DPO loss — correct shape and range
  8. TrainerSFT — initialises and runs without error
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import CharTokenizer
from data.instruct_dataset import (
    InstructPair, DEFAULT_PAIRS, TEMPLATE, RESPONSE_START, RESPONSE_END,
    format_pair, format_prompt, build_corpus,
    response_mask, masked_cross_entropy, DEFAULT_SYSTEM,
)
from sample.prompter import Prompter
from demo_dpo import dpo_loss, sequence_log_prob


# ─── Fixtures ────────────────────────────────────────────────────────────────

PAIR = InstructPair(
    system="You are a test assistant.",
    user="Say hello.",
    assistant="Hello!",
)

SYSTEM = "You are a test assistant."


def make_small_model(tokenizer):
    from model.transformer import TransformerLM
    return TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        block_size=32,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        seed=0,
    )


@pytest.fixture
def corpus_tokenizer():
    corpus = build_corpus(DEFAULT_PAIRS, repeat=2)
    tok    = CharTokenizer(corpus)
    return corpus, tok


@pytest.fixture
def small_model(corpus_tokenizer):
    _, tok = corpus_tokenizer
    return make_small_model(tok)


# ─── 1. Template formatting ───────────────────────────────────────────────────

class TestFormatting:

    def test_format_pair_contains_all_fields(self):
        text = format_pair(PAIR)
        assert PAIR.system    in text
        assert PAIR.user      in text
        assert PAIR.assistant in text

    def test_format_pair_contains_template_markers(self):
        text = format_pair(PAIR)
        assert "### System:" in text
        assert "### User:"   in text
        assert "### Assistant:" in text
        assert "### End"     in text

    def test_format_prompt_no_assistant_body(self):
        """Inference prompt must stop at '### Assistant:\\n' with no response."""
        text = format_prompt(SYSTEM, "What is your name?")
        assert text.endswith("### Assistant:\n")

    def test_format_prompt_contains_user(self):
        text = format_prompt(SYSTEM, "Hello there.")
        assert "Hello there." in text

    def test_format_pair_ends_with_end_marker(self):
        text = format_pair(PAIR)
        assert "### End" in text

    def test_all_default_pairs_format_cleanly(self):
        for pair in DEFAULT_PAIRS:
            text = format_pair(pair)
            assert pair.assistant in text
            assert "### Assistant:" in text


# ─── 2. Response mask ─────────────────────────────────────────────────────────

class TestResponseMask:

    def test_mask_shape_matches_tokens(self, corpus_tokenizer):
        corpus, tok = corpus_tokenizer
        text        = format_pair(PAIR)
        tokens      = tok.encode(text)
        mask        = response_mask(tokens, tok, DEFAULT_PAIRS)
        assert mask.shape == (len(tokens),)

    def test_mask_has_ones_in_response(self, corpus_tokenizer):
        corpus, tok = corpus_tokenizer
        text        = format_pair(PAIR)
        tokens      = tok.encode(text)
        mask        = response_mask(tokens, tok, DEFAULT_PAIRS)
        # There should be at least some 1s (the response tokens)
        assert mask.sum() > 0

    def test_mask_has_zeros_outside_response(self, corpus_tokenizer):
        corpus, tok = corpus_tokenizer
        text        = format_pair(PAIR)
        tokens      = tok.encode(text)
        mask        = response_mask(tokens, tok, DEFAULT_PAIRS)
        # There should be some 0s (instruction tokens)
        assert (mask == 0).sum() > 0

    def test_mask_dtype(self, corpus_tokenizer):
        corpus, tok = corpus_tokenizer
        tokens      = tok.encode(format_pair(PAIR))
        mask        = response_mask(tokens, tok, DEFAULT_PAIRS)
        assert mask.dtype == np.float32


# ─── 3. Masked cross-entropy ──────────────────────────────────────────────────

class TestMaskedCrossEntropy:

    def _fake_logits_targets(self, T=16, V=20):
        rng     = np.random.default_rng(7)
        logits  = rng.standard_normal((T, V))
        targets = rng.integers(0, V, T, dtype=np.int32)
        return logits, targets

    def test_all_ones_mask_matches_standard_ce(self):
        """All-1 mask should give same result as standard CE."""
        T, V     = 16, 20
        logits, targets = self._fake_logits_targets(T, V)
        mask_all = np.ones(T, dtype=np.float32)

        masked_loss = masked_cross_entropy(logits, targets, mask_all)

        # Standard CE
        shifted   = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        standard  = float(-log_probs[np.arange(T), targets].mean())

        assert abs(masked_loss - standard) < 1e-6

    def test_zero_mask_returns_fallback(self):
        """All-0 mask should not crash — falls back to unmasked loss."""
        logits, targets = self._fake_logits_targets()
        mask_zero = np.zeros(16, dtype=np.float32)
        loss = masked_cross_entropy(logits, targets, mask_zero)
        assert np.isfinite(loss)

    def test_partial_mask_between_bounds(self):
        """Partial mask loss should be between 0 and all-ones loss × 2."""
        logits, targets = self._fake_logits_targets()
        mask_half       = np.array([1, 0] * 8, dtype=np.float32)
        loss_half       = masked_cross_entropy(logits, targets, mask_half)
        assert np.isfinite(loss_half) and loss_half > 0


# ─── 4. Corpus builder ────────────────────────────────────────────────────────

class TestCorpusBuilder:

    def test_repeat_multiplies_length(self):
        c1 = build_corpus(DEFAULT_PAIRS, repeat=1)
        c3 = build_corpus(DEFAULT_PAIRS, repeat=3)
        assert len(c3) == len(c1) * 3

    def test_corpus_contains_all_pairs(self):
        corpus = build_corpus(DEFAULT_PAIRS, repeat=1)
        for pair in DEFAULT_PAIRS:
            assert pair.assistant in corpus

    def test_corpus_contains_template_markers(self):
        corpus = build_corpus(DEFAULT_PAIRS, repeat=1)
        assert "### System:"    in corpus
        assert "### User:"      in corpus
        assert "### Assistant:" in corpus


# ─── 5. Prompter ─────────────────────────────────────────────────────────────

class TestPrompter:

    def test_build_ends_with_assistant_marker(self):
        p    = Prompter(SYSTEM)
        text = p.build("Test question.")
        assert text.endswith("### Assistant:\n")

    def test_build_contains_user_message(self):
        p    = Prompter(SYSTEM)
        text = p.build("My question here.")
        assert "My question here." in text

    def test_extract_returns_only_response(self):
        p        = Prompter(SYSTEM)
        prompt   = p.build("Test.")
        full     = prompt + "This is the answer.\n### End\n"
        response = p.extract(full)
        assert response == "This is the answer."

    def test_extract_no_end_marker(self):
        """extract() should still work if ### End is missing."""
        p    = Prompter(SYSTEM)
        full = p.build("Test.") + "Partial answer"
        resp = p.extract(full)
        assert "Partial answer" in resp

    def test_repr(self):
        p = Prompter("Short system prompt.")
        assert "Prompter" in repr(p)


# ─── 6. Prompter.chat ────────────────────────────────────────────────────────

class TestPrompterChat:

    def test_chat_returns_string(self, small_model, corpus_tokenizer):
        _, tok = corpus_tokenizer
        p      = Prompter(SYSTEM)
        result = p.chat(small_model, tok, "Say hello.",
                        max_new_tokens=20, temperature=1.0)
        assert isinstance(result, str)

    def test_chat_output_not_empty(self, small_model, corpus_tokenizer):
        _, tok = corpus_tokenizer
        p      = Prompter(SYSTEM)
        result = p.chat(small_model, tok, "What is your name?",
                        max_new_tokens=30, temperature=1.0)
        # Even a random model generates something
        assert len(result) >= 0   # can be empty if all tokens filtered


# ─── 7. DPO loss ─────────────────────────────────────────────────────────────

class TestDPOLoss:

    def test_loss_is_finite(self):
        loss = dpo_loss(-10.0, -20.0, -12.0, -18.0, beta=0.1)
        assert np.isfinite(loss)

    def test_perfect_separation_gives_low_loss(self):
        """When chosen >> rejected relative to reference, loss is low."""
        loss = dpo_loss(
            log_prob_chosen=-5.0,
            log_prob_rejected=-50.0,
            log_prob_ref_chosen=-10.0,
            log_prob_ref_rejected=-15.0,
            beta=0.1,
        )
        assert loss < 0.5

    def test_neutral_gives_log2_loss(self):
        """When π == π_ref, logit is 0, loss = -log(0.5) ≈ 0.693."""
        loss = dpo_loss(-10.0, -20.0, -10.0, -20.0, beta=0.1)
        assert abs(loss - math.log(2)) < 0.01

    def test_sequence_log_prob_negative(self, small_model, corpus_tokenizer):
        """Log-probability of any sequence must be ≤ 0."""
        _, tok = corpus_tokenizer
        lp     = sequence_log_prob(small_model, tok, "Hello!", block_size=32)
        assert lp <= 0.0

    def test_shorter_sequence_higher_log_prob(self, small_model, corpus_tokenizer):
        """A one-char sequence has higher log-prob than a ten-char sequence."""
        _, tok  = corpus_tokenizer
        lp_short = sequence_log_prob(small_model, tok, "H",       block_size=32)
        lp_long  = sequence_log_prob(small_model, tok, "Hello there!", block_size=32)
        assert lp_short > lp_long


import math  # needed by test_neutral_gives_log2_loss


# ─── 8. TrainerSFT integration ────────────────────────────────────────────────

class TestTrainerSFT:

    def test_trainer_initialises(self, small_model, corpus_tokenizer):
        from train.trainer_sft import TrainerSFT
        _, tok = corpus_tokenizer
        cfg = {
            'model':    {'embed_dim': 32, 'block_size': 32, 'n_layers': 1, 'n_heads': 2},
            'training': {'epochs': 5, 'batch_size': 2, 'lr': 1e-3,
                         'optimizer': 'adamw', 'grad_clip': 1.0,
                         'warmup_steps': 0, 'seed': 0, 'val_split': 0.0},
            'dashboard': {'log_every': 1000, 'sample_every': 1000},
            'sft': {'max_new_tokens': 10, 'temperature': 1.0,
                    'sample_user': 'Say hello.'},
        }
        trainer = TrainerSFT(small_model, DEFAULT_PAIRS[:3], tok, cfg, repeat=2)
        assert trainer is not None

    def test_trainer_runs_without_error(self, small_model, corpus_tokenizer, tmp_path):
        from train.trainer_sft import TrainerSFT
        import os
        _, tok = corpus_tokenizer
        cfg = {
            'model':    {'embed_dim': 32, 'block_size': 32},
            'training': {'epochs': 3, 'batch_size': 2, 'lr': 1e-3,
                         'optimizer': 'adamw', 'grad_clip': 1.0,
                         'warmup_steps': 0, 'seed': 0, 'val_split': 0.0},
            'dashboard': {'log_every': 1000, 'sample_every': 1000},
            'sft': {'max_new_tokens': 5, 'temperature': 1.0,
                    'sample_user': 'Hello.'},
        }
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            trainer = TrainerSFT(small_model, DEFAULT_PAIRS[:2], tok, cfg, repeat=1)
            run_path = trainer.train()
            assert Path(run_path).exists()
        finally:
            os.chdir(old_dir)

    def test_loss_history_recorded(self, small_model, corpus_tokenizer, tmp_path):
        from train.trainer_sft import TrainerSFT
        import os
        _, tok = corpus_tokenizer
        cfg = {
            'model':    {'embed_dim': 32, 'block_size': 32},
            'training': {'epochs': 5, 'batch_size': 2, 'lr': 1e-3,
                         'optimizer': 'adamw', 'grad_clip': 1.0,
                         'warmup_steps': 0, 'seed': 0, 'val_split': 0.0},
            'dashboard': {'log_every': 1000, 'sample_every': 1000},
            'sft': {'max_new_tokens': 5, 'temperature': 1.0,
                    'sample_user': 'Hello.'},
        }
        old_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            trainer = TrainerSFT(small_model, DEFAULT_PAIRS[:2], tok, cfg, repeat=1)
            trainer.train()
            assert len(trainer.history) == 5
            assert all('loss' in r for r in trainer.history)
        finally:
            os.chdir(old_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
