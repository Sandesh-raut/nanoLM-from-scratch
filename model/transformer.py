"""
Phase 3 / Phase 4 — Transformer LM with full hand-written backprop.

Phase 3 (n_heads=1, n_layers=1, dropout=0.0):  single-head attention
Phase 4 (n_heads>1, n_layers>1, dropout>0.0):   multi-head, depth, regularisation

Architecture per layer (pre-norm, GPT-2 style):
  x → LN → MultiHeadAttention → Dropout → + residual
    → LN → FFN               → Dropout → + residual

Stack n_layers of the above, then:
  → LN_final → proj → logits

Every class stores its forward cache in self._c and exposes a backward() method.
No autograd is used — every gradient is derived by hand.

Shape convention
----------------
  B  = batch size
  T  = sequence length (≤ block_size)
  D  = embed_dim
  H  = n_heads
  Dh = D // H  (head dimension)
  V  = vocab_size
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# LayerNorm
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm:
    """
    Normalise the last dimension of x, then rescale with learned γ and β.

    Forward:  x̂ = (x - μ) / σ,  out = γ·x̂ + β
    Backward: compact formula avoids explicitly computing dσ and dμ.
              dx = (1 / D / σ) · (D·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂))
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim   = dim
        self.eps   = eps
        self.gamma = np.ones(dim,  dtype=np.float64)
        self.beta  = np.zeros(dim, dtype=np.float64)
        self._c    = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu   = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1,  keepdims=True)
        std  = np.sqrt(var + self.eps)
        xhat = (x - mu) / std
        out  = self.gamma * xhat + self.beta
        self._c = {'xhat': xhat, 'std': std}
        return out

    def backward(self, dout: np.ndarray):
        xhat, std = self._c['xhat'], self._c['std']
        D    = self.dim
        axes = tuple(range(dout.ndim - 1))

        dgamma = (dout * xhat).sum(axis=axes)
        dbeta  = dout.sum(axis=axes)

        dxhat = dout * self.gamma
        dx    = (1.0 / D / std) * (
            D * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx, {'gamma': dgamma, 'beta': dbeta}


# ─────────────────────────────────────────────────────────────────────────────
# Dropout
# ─────────────────────────────────────────────────────────────────────────────

class Dropout:
    """
    Randomly zero a fraction p of activations during training.
    Surviving values are scaled up by 1/(1-p) so expected value is unchanged.

    During generation (training=False): identity pass-through.
    Backward: same mask re-applied to upstream gradient.
    """

    def __init__(self, p: float = 0.1):
        self.p        = p
        self.training = True
        self._mask    = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = (np.random.random(x.shape) >= self.p).astype(np.float64) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return dout
        return dout * self._mask


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    Scaled dot-product attention with H parallel heads and a causal mask.

    Why multiple heads?
    -------------------
    A single head produces ONE type of token-to-token relationship.
    H heads each learn different D/H-dimensional subspaces, allowing
    simultaneous tracking of syntactic, semantic, positional patterns etc.
    Outputs are concatenated then re-projected to D.

    Implementation: Wq/Wk/Wv are (D, D) matrices. We project full D then
    reshape into H heads of dim Dh = D/H. Algebraically equivalent to H
    separate (Dh, Dh) matrices but more efficient (one big matmul).

    Causal mask: position t attends only to positions 0..t.
    Upper triangle of (T, T) scores is set to -inf before softmax.
    """

    def __init__(self, dim: int, n_heads: int, seed: int = 42):
        assert dim % n_heads == 0, f"embed_dim {dim} must be divisible by n_heads {n_heads}"
        self.dim      = dim
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        rng   = np.random.default_rng(seed)
        scale = 0.02
        self.Wq = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self.Wk = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self.Wv = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self.Wo = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self._c = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H  = self.n_heads
        Dh = self.head_dim

        # Project then split into H heads — shape (B, H, T, Dh)
        Q = (x @ self.Wq).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        K = (x @ self.Wk).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        V = (x @ self.Wv).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # Scaled dot-product attention scores (B, H, T, T)
        inv_sqrt_dh = Dh ** -0.5
        scores = Q @ K.transpose(0, 1, 3, 2) * inv_sqrt_dh

        # Causal mask: future positions → -inf
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, :, mask] = -1e9

        # Stable softmax
        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn  = exp_s / exp_s.sum(axis=-1, keepdims=True)   # (B, H, T, T)

        # Weighted values then merge heads
        out_heads  = attn @ V                                            # (B, H, T, Dh)
        out_merged = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)  # (B, T, D)
        out_proj   = out_merged @ self.Wo                               # (B, T, D)

        self._c = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'attn': attn, 'out_merged': out_merged, 'inv_sqrt_dh': inv_sqrt_dh,
        }
        return out_proj

    def backward(self, dout_proj: np.ndarray):
        x          = self._c['x']
        Q, K, V    = self._c['Q'], self._c['K'], self._c['V']
        attn       = self._c['attn']
        out_merged = self._c['out_merged']
        inv_sqrt   = self._c['inv_sqrt_dh']

        B, T, D = x.shape
        H  = self.n_heads
        Dh = self.head_dim
        BT = B * T

        # Output projection
        dWo  = out_merged.reshape(BT, D).T @ dout_proj.reshape(BT, D)
        dout = dout_proj @ self.Wo.T                                     # (B, T, D)

        # Un-merge heads: (B,T,D) → (B,H,T,Dh)
        dout_heads = dout.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # attn @ V
        dV    = attn.transpose(0, 1, 3, 2) @ dout_heads   # (B, H, T, Dh)
        dattn = dout_heads @ V.transpose(0, 1, 3, 2)      # (B, H, T, T)

        # Softmax backward
        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dscores *= inv_sqrt

        # Q @ K.T
        dQ = dscores @ K                         # (B, H, T, Dh)
        dK = dscores.transpose(0, 1, 3, 2) @ Q  # (B, H, T, Dh)

        # Un-split heads → (B, T, D)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Weight gradients
        dWq = x.reshape(BT, D).T @ dQ_flat.reshape(BT, D)
        dWk = x.reshape(BT, D).T @ dK_flat.reshape(BT, D)
        dWv = x.reshape(BT, D).T @ dV_flat.reshape(BT, D)

        # Input gradient
        dx = dQ_flat @ self.Wq.T + dK_flat @ self.Wk.T + dV_flat @ self.Wv.T

        return dx, {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo}

    def param_count(self) -> int:
        return self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size

    def param_rows(self, prefix: str):
        D = self.dim
        return [
            (f'{prefix}.Wq', f'({D}×{D})', self.Wq.size),
            (f'{prefix}.Wk', f'({D}×{D})', self.Wk.size),
            (f'{prefix}.Wv', f'({D}×{D})', self.Wv.size),
            (f'{prefix}.Wo', f'({D}×{D})', self.Wo.size),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FFN:
    """
    Two-layer MLP: D → 4D (ReLU) → D.

    Widens the representation 4× for per-token nonlinear computation
    after attention has mixed information across positions.
    """

    def __init__(self, dim: int, seed: int = 0):
        rng   = np.random.default_rng(seed)
        inner = dim * 4
        scale = 0.02
        self.W1 = rng.standard_normal((dim,   inner)).astype(np.float64) * scale
        self.b1 = np.zeros(inner, dtype=np.float64)
        self.W2 = rng.standard_normal((inner, dim)).astype(np.float64) * scale
        self.b2 = np.zeros(dim, dtype=np.float64)
        self._c = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        pre  = x @ self.W1 + self.b1
        act  = np.maximum(0, pre)           # ReLU
        out  = act @ self.W2 + self.b2
        self._c = {'x': x, 'pre': pre, 'act': act}
        return out

    def backward(self, dout: np.ndarray):
        x, pre, act = self._c['x'], self._c['pre'], self._c['act']
        B, T, D = x.shape
        BT    = B * T
        inner = self.W1.shape[1]

        dW2 = act.reshape(BT, inner).T @ dout.reshape(BT, D)
        db2 = dout.reshape(BT, D).sum(axis=0)

        dact = dout @ self.W2.T
        dpre = dact * (pre > 0)            # ReLU backward

        dW1 = x.reshape(BT, D).T @ dpre.reshape(BT, inner)
        db1 = dpre.reshape(BT, inner).sum(axis=0)
        dx  = dpre @ self.W1.T

        return dx, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def param_rows(self, prefix: str):
        D     = self.W1.shape[0]
        inner = self.W1.shape[1]
        return [
            (f'{prefix}.W1', f'({D}×{inner})', self.W1.size),
            (f'{prefix}.b1', f'({inner},)',     self.b1.size),
            (f'{prefix}.W2', f'({inner}×{D})', self.W2.size),
            (f'{prefix}.b2', f'({D},)',         self.b2.size),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock:
    """
    One repeating unit: pre-norm attention + pre-norm FFN with residuals.

    Pre-norm (LN before each sub-layer) is more stable than post-norm
    and is what GPT-2 onwards uses.
    Residuals let gradients bypass each sub-layer without vanishing.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, seed: int):
        self.ln1       = LayerNorm(dim)
        self.attn      = MultiHeadAttention(dim, n_heads, seed)
        self.attn_drop = Dropout(dropout)
        self.ln2       = LayerNorm(dim)
        self.ffn       = FFN(dim, seed + 1)
        self.ffn_drop  = Dropout(dropout)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.attn_drop.training = training
        self.ffn_drop.training  = training

        # Attention sub-block
        h = x + self.attn_drop.forward(self.attn.forward(self.ln1.forward(x)))
        # FFN sub-block
        out = h + self.ffn_drop.forward(self.ffn.forward(self.ln2.forward(h)))
        return out

    def backward(self, dout: np.ndarray):
        # FFN sub-block (reversed)
        dffn_drop          = self.ffn_drop.backward(dout)
        dffn_in, ffn_g     = self.ffn.backward(dffn_drop)
        dln2_in, ln2_g     = self.ln2.backward(dffn_in)
        dh                 = dout + dln2_in          # residual split

        # Attention sub-block (reversed)
        dattn_drop         = self.attn_drop.backward(dh)
        dattn_in, attn_g   = self.attn.backward(dattn_drop)
        dln1_in, ln1_g     = self.ln1.backward(dattn_in)
        dx                 = dh + dln1_in            # residual split

        return dx, {'ln1': ln1_g, 'attn': attn_g, 'ln2': ln2_g, 'ffn': ffn_g}


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Language Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM:
    """
    Full transformer language model — Phase 3 and Phase 4.

      token_embed + pos_embed
      → N × TransformerBlock
      → LayerNorm
      → proj to vocab

    Param scaling with depth and width:
      fixed  : token_embed(V×D) + pos_embed(T×D) + ln_final + proj(D×V)
      per-layer: 4D² (attn weights) + 8D² (FFN, 4D hidden) + 5D (norms + biases)
      total  : fixed + n_layers × per-layer
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        block_size: int,
        n_layers:   int   = 1,
        n_heads:    int   = 1,
        dropout:    float = 0.0,
        seed:       int   = 42,
    ):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.block_size = block_size
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.dropout    = dropout

        rng   = np.random.default_rng(seed)
        scale = 0.02

        self.token_embed = rng.standard_normal((vocab_size, embed_dim)).astype(np.float64) * scale
        self.pos_embed   = rng.standard_normal((block_size, embed_dim)).astype(np.float64) * scale

        self.layers = [
            TransformerBlock(embed_dim, n_heads, dropout, seed=seed + i * 10)
            for i in range(n_layers)
        ]

        self.ln_final = LayerNorm(embed_dim)
        self.proj     = rng.standard_normal((embed_dim, vocab_size)).astype(np.float64) * scale

        self._c = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """x: (B, T) → logits (B, T, V). training=False during generation."""
        B, T = x.shape
        h = self.token_embed[x] + self.pos_embed[:T]
        for layer in self.layers:
            h = layer.forward(h, training=training)
        h_norm = self.ln_final.forward(h)
        logits = h_norm @ self.proj
        self._c = {'x': x, 'h_norm': h_norm}
        return logits

    def loss_and_grads(
        self,
        x:         np.ndarray,
        targets:   np.ndarray,
        loss_mask: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """
        loss_mask : optional (B, T) array of 0/1 aligned with `targets`.
                    Where 0, that target contributes nothing to the loss OR the
                    gradient (true masking, not a global rescale). Used by SFT to
                    train on response tokens only. None → train on every token.
        """
        B, T = x.shape
        N    = B * T

        logits      = self.forward(x, training=True)
        logits_flat = logits.reshape(N, self.vocab_size)
        tgts_flat   = targets.reshape(N)

        shifted = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=1, keepdims=True)

        nll = -np.log(probs[np.arange(N), tgts_flat] + 1e-8)   # (N,)

        # Softmax + CE gradient: dlogits = probs - onehot(target)
        dlogits = probs.copy()
        dlogits[np.arange(N), tgts_flat] -= 1.0

        if loss_mask is not None:
            m     = loss_mask.reshape(N).astype(np.float64)
            denom = max(float(m.sum()), 1.0)         # # of response tokens
            loss  = float((nll * m).sum() / denom)
            dlogits *= m[:, None]                     # zero masked positions
            dlogits /= denom                          # average over response tokens
        else:
            loss = float(nll.mean())
            dlogits /= N

        dlogits = dlogits.reshape(B, T, self.vocab_size)

        # Output head backward
        h_norm = self._c['h_norm']
        dproj  = h_norm.reshape(N, self.embed_dim).T @ dlogits.reshape(N, self.vocab_size)
        dh     = dlogits @ self.proj.T

        # Final LN backward
        dh, ln_final_g = self.ln_final.backward(dh)

        # Blocks in reverse
        block_grads = []
        for layer in reversed(self.layers):
            dh, bg = layer.backward(dh)
            block_grads.insert(0, bg)

        # Embedding gradients
        dtoken_embed = np.zeros_like(self.token_embed)
        np.add.at(dtoken_embed, self._c['x'], dh)

        dpos_embed = np.zeros_like(self.pos_embed)
        dpos_embed[:T] = dh.sum(axis=0)

        grads = {
            'token_embed': dtoken_embed,
            'pos_embed':   dpos_embed,
            'blocks':      block_grads,
            'ln_final':    ln_final_g,
            'proj':        dproj,
        }
        return loss, grads

    def sgd_step(self, grads: dict, lr: float):
        self.token_embed -= lr * grads['token_embed']
        self.pos_embed   -= lr * grads['pos_embed']
        self.proj        -= lr * grads['proj']
        self.ln_final.gamma -= lr * grads['ln_final']['gamma']
        self.ln_final.beta  -= lr * grads['ln_final']['beta']

        for layer, bg in zip(self.layers, grads['blocks']):
            ag = bg['attn']
            layer.attn.Wq -= lr * ag['Wq']
            layer.attn.Wk -= lr * ag['Wk']
            layer.attn.Wv -= lr * ag['Wv']
            layer.attn.Wo -= lr * ag['Wo']
            fg = bg['ffn']
            layer.ffn.W1 -= lr * fg['W1']
            layer.ffn.b1 -= lr * fg['b1']
            layer.ffn.W2 -= lr * fg['W2']
            layer.ffn.b2 -= lr * fg['b2']
            layer.ln1.gamma -= lr * bg['ln1']['gamma']
            layer.ln1.beta  -= lr * bg['ln1']['beta']
            layer.ln2.gamma -= lr * bg['ln2']['gamma']
            layer.ln2.beta  -= lr * bg['ln2']['beta']

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def description(self) -> str:
        if self.n_layers == 1 and self.n_heads == 1:
            return "Phase 3 · Single-head Attention · hand-written backprop"
        return (
            f"Phase 4 · {self.n_heads}-head Attention · "
            f"{self.n_layers} layers · dropout={self.dropout}"
        )

    def param_table(self) -> list[tuple[str, str, int]]:
        V, D, T = self.vocab_size, self.embed_dim, self.block_size
        rows = [
            ('token_embed',  f'({V}×{D})',  self.token_embed.size),
            ('pos_embed',    f'({T}×{D})',  self.pos_embed.size),
        ]
        for i, layer in enumerate(self.layers):
            p = f'block[{i}]'
            rows += layer.attn.param_rows(f'{p}.attn')
            rows += [(f'{p}.ln1.γβ', f'({D},)×2', layer.ln1.gamma.size * 2)]
            rows += layer.ffn.param_rows(f'{p}.ffn')
            rows += [(f'{p}.ln2.γβ', f'({D},)×2', layer.ln2.gamma.size * 2)]
        rows += [
            ('ln_final.γβ', f'({D},)×2', self.ln_final.gamma.size * 2),
            ('proj',        f'({D}×{V})', self.proj.size),
        ]
        return rows

    def param_count(self) -> dict[str, int]:
        counts = {name: n for name, _, n in self.param_table()}
        counts['total'] = sum(n for _, _, n in self.param_table())
        return counts

    def tracked_weight(self, row: int = 0, col: int = 0) -> float:
        return float(self.token_embed[row, col])

    # ── Flat parameter / gradient iterators (used by AdamW) ──────────────────

    def _flat_params(self):
        """Yield (name, param_array) for every parameter, in a fixed order."""
        yield 'token_embed', self.token_embed
        yield 'pos_embed',   self.pos_embed
        for i, layer in enumerate(self.layers):
            p = f'block[{i}]'
            yield f'{p}.attn.Wq',    layer.attn.Wq
            yield f'{p}.attn.Wk',    layer.attn.Wk
            yield f'{p}.attn.Wv',    layer.attn.Wv
            yield f'{p}.attn.Wo',    layer.attn.Wo
            yield f'{p}.ln1.gamma',  layer.ln1.gamma
            yield f'{p}.ln1.beta',   layer.ln1.beta
            yield f'{p}.ffn.W1',     layer.ffn.W1
            yield f'{p}.ffn.b1',     layer.ffn.b1
            yield f'{p}.ffn.W2',     layer.ffn.W2
            yield f'{p}.ffn.b2',     layer.ffn.b2
            yield f'{p}.ln2.gamma',  layer.ln2.gamma
            yield f'{p}.ln2.beta',   layer.ln2.beta
        yield 'ln_final.gamma', self.ln_final.gamma
        yield 'ln_final.beta',  self.ln_final.beta
        yield 'proj',           self.proj

    def _flat_grads(self, grads: dict):
        """Yield (name, grad_array) in the same order as _flat_params()."""
        yield 'token_embed', grads['token_embed']
        yield 'pos_embed',   grads['pos_embed']
        for i, bg in enumerate(grads['blocks']):
            p = f'block[{i}]'
            yield f'{p}.attn.Wq',    bg['attn']['Wq']
            yield f'{p}.attn.Wk',    bg['attn']['Wk']
            yield f'{p}.attn.Wv',    bg['attn']['Wv']
            yield f'{p}.attn.Wo',    bg['attn']['Wo']
            yield f'{p}.ln1.gamma',  bg['ln1']['gamma']
            yield f'{p}.ln1.beta',   bg['ln1']['beta']
            yield f'{p}.ffn.W1',     bg['ffn']['W1']
            yield f'{p}.ffn.b1',     bg['ffn']['b1']
            yield f'{p}.ffn.W2',     bg['ffn']['W2']
            yield f'{p}.ffn.b2',     bg['ffn']['b2']
            yield f'{p}.ln2.gamma',  bg['ln2']['gamma']
            yield f'{p}.ln2.beta',   bg['ln2']['beta']
        yield 'ln_final.gamma', grads['ln_final']['gamma']
        yield 'ln_final.beta',  grads['ln_final']['beta']
        yield 'proj',           grads['proj']

    def compute_loss(self, x: np.ndarray, targets: np.ndarray) -> float:
        """Forward in inference mode (dropout off). Returns scalar loss only."""
        B, T = x.shape
        N    = B * T
        logits      = self.forward(x, training=False)
        logits_flat = logits.reshape(N, self.vocab_size)
        tgts_flat   = targets.reshape(N)
        shifted     = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l       = np.exp(shifted)
        probs       = exp_l / exp_l.sum(axis=1, keepdims=True)
        return float(-np.log(probs[np.arange(N), tgts_flat] + 1e-8).mean())
