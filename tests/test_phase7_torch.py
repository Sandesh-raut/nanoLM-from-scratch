"""
Phase 7 tests — PyTorch model architecture and weight transfer.

These tests require PyTorch (pip install torch).
They run on CPU only so they work without MPS/CUDA.

Covered:
  1. Param count: PyTorch model matches NumPy model exactly
  2. Weight transfer: logits match to float32 precision after copy
  3. Greedy generation: identical output after weight transfer
  4. No-grad inference: model.eval() + torch.no_grad()
  5. Architecture: correct n_layers, n_heads, block_size stored
  6. Device: model correctly reports device of parameters
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from model.transformer_torch import TransformerLMTorch, transfer_weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from model.transformer import TransformerLM
from sample.sampler import generate

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ─── Fixtures ────────────────────────────────────────────────────────────────

VOCAB  = 24
EMBED  = 32
BLOCK  = 8
LAYERS = 2
HEADS  = 4


def make_numpy(seed=42) -> TransformerLM:
    return TransformerLM(
        vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
        n_layers=LAYERS, n_heads=HEADS, dropout=0.0, seed=seed,
    )


def make_torch() -> 'TransformerLMTorch':
    return TransformerLMTorch(
        vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
        n_layers=LAYERS, n_heads=HEADS, dropout=0.0,
    )


# ─── 1. Param count ──────────────────────────────────────────────────────────

class TestParamCount:

    def test_total_params_match_numpy(self):
        np_model = make_numpy()
        pt_model = make_torch()
        assert np_model.param_count()['total'] == pt_model.param_count()['total']

    def test_param_count_scales_with_layers(self):
        """Doubling layers should roughly double per-layer params."""
        pt1 = TransformerLMTorch(vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
                                 n_layers=1, n_heads=HEADS)
        pt2 = TransformerLMTorch(vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
                                 n_layers=2, n_heads=HEADS)

        fixed  = VOCAB * EMBED + BLOCK * EMBED + 2 * EMBED + VOCAB * EMBED
        per_layer_approx = pt1.param_count()['total'] - fixed

        actual_diff = pt2.param_count()['total'] - pt1.param_count()['total']
        # Should be within 10% of one layer's param count
        assert abs(actual_diff - per_layer_approx) / per_layer_approx < 0.1

    def test_param_count_with_more_heads(self):
        """Heads don't change param count (D stays D — heads subdivide it)."""
        pt1 = TransformerLMTorch(vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
                                 n_layers=LAYERS, n_heads=1)
        pt4 = TransformerLMTorch(vocab_size=VOCAB, embed_dim=EMBED, block_size=BLOCK,
                                 n_layers=LAYERS, n_heads=HEADS)
        assert pt1.param_count()['total'] == pt4.param_count()['total']


# ─── 2. Weight transfer ───────────────────────────────────────────────────────

class TestWeightTransfer:

    def test_logits_match_after_transfer(self):
        """After transfer, forward pass logits should match to float32 precision."""
        np_model = make_numpy()
        pt_model = make_torch()
        transfer_weights(np_model, pt_model)

        np.random.seed(0)
        x_np = np.random.randint(0, VOCAB, (2, BLOCK), dtype=np.int32)
        x_pt = torch.tensor(x_np, dtype=torch.long)

        logits_np = np_model.forward(x_np, training=False)   # (2, T, V) float64

        pt_model.eval()
        with torch.no_grad():
            logits_pt = pt_model(x_pt).numpy().astype(np.float64)

        max_diff = float(np.abs(logits_np - logits_pt).max())
        assert max_diff < 1e-4, f"Max logit diff {max_diff:.2e} exceeds 1e-4"

    def test_probabilities_match_after_transfer(self):
        """Softmax probs should match, not just logits."""
        np_model = make_numpy()
        pt_model = make_torch()
        transfer_weights(np_model, pt_model)

        x_np = np.array([[1, 2, 3, 4, 5, 6, 7, 0]], dtype=np.int32)
        x_pt = torch.tensor(x_np, dtype=torch.long)

        logits_np = np_model.forward(x_np, training=False)[0, -1]  # (V,)
        logits_np -= logits_np.max()
        probs_np   = np.exp(logits_np) / np.exp(logits_np).sum()

        pt_model.eval()
        with torch.no_grad():
            logits_pt = pt_model(x_pt)[0, -1]
        probs_pt = F.softmax(logits_pt, dim=-1).numpy()

        max_diff = float(np.abs(probs_np - probs_pt).max())
        assert max_diff < 1e-4

    def test_top_token_matches_after_transfer(self):
        """Greedy next-token must be identical in both frameworks."""
        np_model = make_numpy()
        pt_model = make_torch()
        transfer_weights(np_model, pt_model)

        x_np = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        x_pt = torch.tensor(x_np, dtype=torch.long)

        np_top = int(np.argmax(np_model.forward(x_np, training=False)[0, -1]))

        pt_model.eval()
        with torch.no_grad():
            pt_top = int(pt_model(x_pt)[0, -1].argmax())

        assert np_top == pt_top

    def test_transfer_does_not_mutate_numpy_weights(self):
        """Transfer should be read-only on the NumPy model."""
        np_model = make_numpy()
        pt_model = make_torch()

        embed_before = np_model.token_embed.copy()
        transfer_weights(np_model, pt_model)
        embed_after  = np_model.token_embed

        np.testing.assert_array_equal(embed_before, embed_after)


# ─── 3. Greedy generation after transfer ─────────────────────────────────────

class TestGenerationAfterTransfer:

    def test_greedy_output_matches_numpy(self):
        """
        After weight transfer, greedy generation from the NumPy model
        and the PyTorch model should be identical.
        (Both run on CPU for this test.)
        """
        from data.tokenizer import CharTokenizer

        np_model = make_numpy()
        pt_model = make_torch()
        transfer_weights(np_model, pt_model)
        pt_model.eval()

        vocab_chars = [str(i % 10) for i in range(VOCAB)]
        tok = CharTokenizer(''.join(set(vocab_chars)))

        # Greedy: deterministic, same seed → must match exactly
        g_np = generate(np_model, tok, '0', max_new_tokens=10,
                        block_size=BLOCK, greedy=True)
        g_pt = generate(pt_model, tok, '0', max_new_tokens=10,
                        block_size=BLOCK, greedy=True)
        assert g_np == g_pt

    def test_generate_correct_length(self):
        """Generated text should have len(seed) + max_new_tokens characters."""
        from data.tokenizer import CharTokenizer

        pt_model = make_torch()
        pt_model.eval()
        tok = CharTokenizer('0123456789abcdefghijklmn')

        result = generate(pt_model, tok, '0', max_new_tokens=15,
                          block_size=BLOCK, temperature=1.0)
        assert len(result) == 16   # seed + 15 new


# ─── 4. Inference mode ────────────────────────────────────────────────────────

class TestInference:

    def test_eval_mode_is_deterministic(self):
        """In eval mode with same input, output is always identical."""
        pt_model = make_torch()
        pt_model.eval()

        x = torch.randint(0, VOCAB, (2, BLOCK))
        with torch.no_grad():
            out1 = pt_model(x)
            out2 = pt_model(x)

        assert torch.allclose(out1, out2)

    def test_output_shape(self):
        B, T = 3, BLOCK
        pt_model = make_torch()
        pt_model.eval()
        x = torch.randint(0, VOCAB, (B, T))
        with torch.no_grad():
            out = pt_model(x)
        assert out.shape == (B, T, VOCAB)

    def test_cross_entropy_loss_runs(self):
        """Full forward + loss should run without error."""
        pt_model = make_torch()
        x = torch.randint(0, VOCAB, (2, BLOCK))
        y = torch.randint(0, VOCAB, (2, BLOCK))
        logits = pt_model(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        assert loss.item() > 0


# ─── 5. Architecture attributes ──────────────────────────────────────────────

class TestArchitecture:

    def test_attributes_stored(self):
        m = make_torch()
        assert m.vocab_size == VOCAB
        assert m.embed_dim  == EMBED
        assert m.block_size == BLOCK
        assert m.n_layers   == LAYERS
        assert m.n_heads    == HEADS

    def test_n_blocks_correct(self):
        m = make_torch()
        assert len(m.layers) == LAYERS

    def test_attention_heads_divisible(self):
        with pytest.raises(AssertionError):
            TransformerLMTorch(vocab_size=10, embed_dim=5, block_size=4,
                               n_layers=1, n_heads=4)   # 5 not divisible by 4

    def test_description_string(self):
        m = make_torch()
        assert 'PyTorch' in m.description
        assert 'autograd' in m.description


# ─── 6. Autograd sanity ───────────────────────────────────────────────────────

class TestAutograd:

    def test_gradients_flow(self):
        """Loss.backward() should populate .grad on all parameters."""
        pt_model = make_torch()
        x = torch.randint(0, VOCAB, (2, BLOCK))
        y = torch.randint(0, VOCAB, (2, BLOCK))

        logits = pt_model(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()

        for name, param in pt_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_loss_decreases_after_step(self):
        """A single AdamW step should decrease the loss on the same batch."""
        pt_model = make_torch()
        optimizer = torch.optim.AdamW(pt_model.parameters(), lr=0.01)

        x = torch.randint(0, VOCAB, (4, BLOCK))
        y = torch.randint(0, VOCAB, (4, BLOCK))

        logits1 = pt_model(x)
        loss1   = F.cross_entropy(logits1.view(-1, VOCAB), y.view(-1))

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        pt_model.eval()
        with torch.no_grad():
            logits2 = pt_model(x)
            loss2   = F.cross_entropy(logits2.view(-1, VOCAB), y.view(-1))

        assert loss2.item() < loss1.item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
