"""
Phase 7 — PyTorch mirror of TransformerLM.

This is the SAME architecture as model/transformer.py, translated to PyTorch.
The point is not a new model — it is proof that the hand-written NumPy math
was correct, and a demonstration of what a framework buys you:
  - autograd replaces 300 lines of hand-written backward()
  - MPS backend runs the same forward pass on Apple Silicon GPU
  - identical param count (verifiable via verify_torch.py)

Architecture (unchanged from NumPy version):
  token_embed + pos_embed
  → N × (pre-LN → MultiHeadAttention → Dropout → residual
          pre-LN → FFN               → Dropout → residual)
  → LN_final → proj → logits

Shape conventions
-----------------
  B  = batch size
  T  = sequence length (≤ block_size)
  D  = embed_dim
  H  = n_heads
  Dh = D // H
  V  = vocab_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttentionTorch(nn.Module):
    """
    Matches NumPy MultiHeadAttention exactly:
      - Wq, Wk, Wv, Wo are (D, D) without bias  → nn.Linear(D, D, bias=False)
      - Causal upper-triangle mask (triu)
      - No dropout on attention weights (dropout is in the block, not here)
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        # Project and split into H heads — (B, H, T, Dh)
        Q = self.Wq(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.Wk(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product scores (B, H, T, T)
        scores = Q @ K.transpose(-2, -1) * (Dh ** -0.5)

        # Causal mask — upper triangle → -inf
        mask   = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Merge heads and project out
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FFNTorch(nn.Module):
    """
    D → 4D (ReLU) → D, with bias on both layers.
    Matches NumPy FFN: W1/b1/W2/b2.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlockTorch(nn.Module):
    """
    Pre-norm residual block, matching NumPy TransformerBlock.

    Pre-norm order:
      x + Dropout(Attn(LN(x)))
      x + Dropout(FFN(LN(x)))

    Dropout sits between the sub-layer and the residual add,
    exactly as in the NumPy version (attn_drop, ffn_drop in the block).
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1       = nn.LayerNorm(embed_dim)
        self.attn      = MultiHeadAttentionTorch(embed_dim, n_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.ln2       = nn.LayerNorm(embed_dim)
        self.ffn       = FFNTorch(embed_dim)
        self.ffn_drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.ln1(x)))
        x = x + self.ffn_drop(self.ffn(self.ln2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Language Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLMTorch(nn.Module):
    """
    PyTorch TransformerLM — same architecture as the NumPy version.

    Differences from the NumPy class:
      - No backward() methods — autograd handles all gradients
      - No _flat_params/_flat_grads — PyTorch optimizer uses .parameters()
      - device-aware: call .to('mps') or .to('cpu')

    Use verify_torch.py to confirm param count and forward pass match.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        block_size: int,
        n_layers:   int   = 1,
        n_heads:    int   = 1,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.block_size = block_size
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.dropout    = dropout

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(block_size, embed_dim)
        self.drop        = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlockTorch(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.proj     = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Match NumPy init: N(0, 0.02) for matrices, 0 for biases, 1/0 for LN."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) int64 → logits (B, T, V) float32."""
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        h    = self.drop(self.token_embed(x) + self.pos_embed(pos))
        for layer in self.layers:
            h = layer(h)
        return self.proj(self.ln_final(h))

    # ── Inspection (matching NumPy API) ──────────────────────────────────────

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}

    def param_table(self) -> list:
        """Returns (name, shape_str, count) rows — same format as NumPy version."""
        rows = []
        for name, p in self.named_parameters():
            rows.append((name, str(tuple(p.shape)), p.numel()))
        return rows

    @property
    def description(self) -> str:
        return (
            f"Phase 7 · PyTorch · {self.n_heads}-head · "
            f"{self.n_layers} layers · dropout={self.dropout} · autograd"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer: NumPy → PyTorch
# ─────────────────────────────────────────────────────────────────────────────

def transfer_weights(numpy_model, torch_model: TransformerLMTorch) -> TransformerLMTorch:
    """
    Copy weights from a NumPy TransformerLM into a PyTorch TransformerLMTorch.

    Shape note
    ----------
    NumPy:   x @ W  where W is (D_in, D_out)
    PyTorch: nn.Linear stores weight as (D_out, D_in), applied as x @ weight.T
    Therefore: torch_linear.weight = numpy_W.T  (transposed)

    Embeddings and LayerNorm params have the same shapes in both frameworks.

    Call this after building both models with the same architecture config,
    then run verify_torch.py to confirm the logits match.
    """
    import numpy as np

    def t(arr) -> torch.Tensor:
        """NumPy float64 → PyTorch float32 tensor."""
        return torch.tensor(arr, dtype=torch.float32)

    with torch.no_grad():
        # Embeddings — shape (V, D) and (T, D): same in both
        torch_model.token_embed.weight.copy_(t(numpy_model.token_embed))
        torch_model.pos_embed.weight.copy_(t(numpy_model.pos_embed))

        # Transformer blocks
        for i, (np_block, pt_block) in enumerate(
            zip(numpy_model.layers, torch_model.layers)
        ):
            # LayerNorm 1
            pt_block.ln1.weight.copy_(t(np_block.ln1.gamma))
            pt_block.ln1.bias.copy_(t(np_block.ln1.beta))

            # Multi-head attention weights (D, D) → transpose for nn.Linear
            pt_block.attn.Wq.weight.copy_(t(np_block.attn.Wq.T))
            pt_block.attn.Wk.weight.copy_(t(np_block.attn.Wk.T))
            pt_block.attn.Wv.weight.copy_(t(np_block.attn.Wv.T))
            pt_block.attn.Wo.weight.copy_(t(np_block.attn.Wo.T))

            # LayerNorm 2
            pt_block.ln2.weight.copy_(t(np_block.ln2.gamma))
            pt_block.ln2.bias.copy_(t(np_block.ln2.beta))

            # FFN — W1 (D, 4D) and W2 (4D, D): transpose; biases: same shape
            pt_block.ffn.fc1.weight.copy_(t(np_block.ffn.W1.T))
            pt_block.ffn.fc1.bias.copy_(t(np_block.ffn.b1))
            pt_block.ffn.fc2.weight.copy_(t(np_block.ffn.W2.T))
            pt_block.ffn.fc2.bias.copy_(t(np_block.ffn.b2))

        # Final LayerNorm
        torch_model.ln_final.weight.copy_(t(numpy_model.ln_final.gamma))
        torch_model.ln_final.bias.copy_(t(numpy_model.ln_final.beta))

        # Output projection (D, V) → transpose
        torch_model.proj.weight.copy_(t(numpy_model.proj.T))

    return torch_model
