"""
Phase 9 — Modern Transformer LM.

Drop-in upgrade of TransformerLM with four independently togglable improvements:

  norm     : 'layernorm' (Phase 3 default) | 'rmsnorm'  (LLaMA/Mistral)
  ffn      : 'relu'      (Phase 3 default) | 'swiglu'   (LLaMA/PaLM)
  pos_enc  : 'learned'   (Phase 3 default) | 'rope'     (LLaMA/Falcon)
  n_kv_heads : n_heads   (Phase 3 default) | 1..n_heads (GQA — LLaMA 2/3)

Each flag can be toggled independently so its effect is visible in isolation.
The same optimizer and training loop that works with TransformerLM works here.

Architecture summary
---------------------
  token_embed                           (always)
  pos_embed[:T]                         (learned only; omitted for rope)
  → N × ModernBlock:
      norm1 → ModernAttention → dropout → + residual
      norm2 → FFN             → dropout → + residual
  → norm_final → proj → logits

ModernAttention supports:
  • RoPE:   rotates Q and K in-place; no global pos_embed needed
  • GQA:    Wk/Wv are (D, n_kv_heads × Dh) instead of (D, D);
            K and V are repeated n_rep = n_heads // n_kv_heads times
            before the dot product

Backward pass
-------------
Every sub-module exposes a backward() method that returns (dx, grads_dict).
The grads_dict keys differ by config:
  norm     layernorm → {'gamma', 'beta'}   rmsnorm → {'gamma'}
  ffn      relu      → {'W1','b1','W2','b2'}  swiglu → {'Wg','Wv','W2'}

_flat_params() and _flat_grads() iterate the keys that are actually present,
so the AdamW optimizer does not need any changes.
"""

import numpy as np

from model.transformer import LayerNorm, Dropout, FFN
from model.norms      import RMSNorm
from model.activations import SwiGLUFFN
from model.rope        import rope_freqs, apply_rope, apply_rope_backward


# ─────────────────────────────────────────────────────────────────────────────
# Modern Multi-Head Attention (+ optional RoPE, + optional GQA)
# ─────────────────────────────────────────────────────────────────────────────

class ModernAttention:
    """
    Multi-Head Self-Attention with optional RoPE and GQA.

    Grouped Query Attention (GQA)
    -----------------------------
    When n_kv_heads < n_heads, each group of (n_heads // n_kv_heads) query
    heads shares one K head and one V head.
      n_kv_heads == n_heads → standard MHA (no sharing)
      n_kv_heads == 1       → MQA (one shared K/V pair for all queries)
    Saves K/V matrix parameters and reduces cache size at inference.

    KV-cache support (inference only)
    ----------------------------------
    Pass kv_cache=<dict> to forward() for autoregressive decode.
    The dict is updated in-place: {'K': ..., 'V': ..., 'seq_len': int}.
    Backward is NOT supported when kv_cache is active (training uses kv_cache=None).
    """

    def __init__(
        self,
        dim:        int,
        n_heads:    int,
        n_kv_heads: int,
        use_rope:   bool,
        seed:       int = 42,
    ):
        assert dim % n_heads == 0,    f"embed_dim {dim} must be divisible by n_heads {n_heads}"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}"

        self.dim        = dim
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repetitions for GQA expansion
        self.head_dim   = dim // n_heads
        self.use_rope   = use_rope

        rng    = np.random.default_rng(seed)
        scale  = 0.02
        kv_dim = n_kv_heads * self.head_dim       # may be < dim for GQA

        self.Wq = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self.Wk = rng.standard_normal((dim, kv_dim)).astype(np.float64) * scale
        self.Wv = rng.standard_normal((dim, kv_dim)).astype(np.float64) * scale
        self.Wo = rng.standard_normal((dim, dim)).astype(np.float64) * scale
        self._c = {}

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:        np.ndarray,
        training: bool = False,
        kv_cache: dict | None = None,
    ) -> np.ndarray:
        """
        x        : (B, T, D)
        kv_cache : optional dict, updated in-place (inference only)
        Returns  : (B, T, D)
        """
        B, T, D  = x.shape
        H        = self.n_heads
        Kv       = self.n_kv_heads
        Dh       = self.head_dim
        kv_dim   = Kv * Dh

        Q = (x @ self.Wq).reshape(B, T, H,  Dh).transpose(0, 2, 1, 3)   # (B, H,  T, Dh)
        K = (x @ self.Wk).reshape(B, T, Kv, Dh).transpose(0, 2, 1, 3)   # (B, Kv, T, Dh)
        V = (x @ self.Wv).reshape(B, T, Kv, Dh).transpose(0, 2, 1, 3)   # (B, Kv, T, Dh)

        # RoPE: rotate Q and K at their absolute position
        rope_cos = rope_sin = None
        if self.use_rope:
            offset = kv_cache.get('seq_len', 0) if kv_cache else 0
            cos_full, sin_full = rope_freqs(Dh, offset + T)
            rope_cos = cos_full[offset : offset + T]
            rope_sin = sin_full[offset : offset + T]
            Q = apply_rope(Q, rope_cos, rope_sin)
            K = apply_rope(K, rope_cos, rope_sin)

        # KV cache: append new K/V then use the full accumulated history
        if kv_cache is not None:
            if 'K' in kv_cache:
                K = np.concatenate([kv_cache['K'], K], axis=2)
                V = np.concatenate([kv_cache['V'], V], axis=2)
            kv_cache['K']       = K
            kv_cache['V']       = V
            kv_cache['seq_len'] = K.shape[2]

        T_full = K.shape[2]   # full context length (> T when using KV cache)

        # GQA: expand K and V to match n_heads
        if self.n_rep > 1:
            K_exp = np.repeat(K, self.n_rep, axis=1)   # (B, H, T_full, Dh)
            V_exp = np.repeat(V, self.n_rep, axis=1)   # (B, H, T_full, Dh)
        else:
            K_exp, V_exp = K, V

        # Scaled dot-product attention
        inv_sqrt = self.head_dim ** -0.5
        scores   = Q @ K_exp.transpose(0, 1, 3, 2) * inv_sqrt   # (B, H, T, T_full)

        # Causal mask: position t attends only to positions 0..t
        # When T=1 (KV-cache decode), the mask would be empty — handled correctly.
        if T > 1:
            # np.triu with k=T_full-T+1 gives the upper-right triangle to mask
            mask = np.triu(np.ones((T, T_full), dtype=bool), k=T_full - T + 1)
            scores[:, :, mask] = -1e9

        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn  = exp_s / exp_s.sum(axis=-1, keepdims=True)   # (B, H, T, T_full)

        out_heads  = attn @ V_exp                                           # (B, H, T, Dh)
        out_merged = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)
        out_proj   = out_merged @ self.Wo

        # Cache activations for backward (only meaningful when kv_cache is None)
        self._c = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'K_exp': K_exp, 'V_exp': V_exp,
            'attn': attn, 'out_merged': out_merged,
            'inv_sqrt': inv_sqrt, 'T_full': T_full,
            'rope_cos': rope_cos, 'rope_sin': rope_sin,
        }
        return out_proj

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, dout_proj: np.ndarray):
        """Assumes kv_cache was not active (training mode)."""
        x          = self._c['x']
        Q          = self._c['Q']
        K          = self._c['K']     # (B, Kv, T, Dh)
        V          = self._c['V']     # (B, Kv, T, Dh)
        K_exp      = self._c['K_exp'] # (B, H,  T, Dh)
        V_exp      = self._c['V_exp'] # (B, H,  T, Dh)
        attn       = self._c['attn']
        out_merged = self._c['out_merged']
        inv_sqrt   = self._c['inv_sqrt']
        rope_cos   = self._c['rope_cos']
        rope_sin   = self._c['rope_sin']

        B, T, D = x.shape
        H, Kv, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        kv_dim = Kv * Dh
        BT     = B * T

        # ── Output projection ─────────────────────────────────────────────
        dWo    = out_merged.reshape(BT, D).T @ dout_proj.reshape(BT, D)
        dout   = dout_proj @ self.Wo.T                                # (B, T, D)

        # Un-merge heads
        dout_h = dout.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)   # (B, H, T, Dh)

        # ── attn @ V_exp backward ─────────────────────────────────────────
        dV_exp = attn.transpose(0, 1, 3, 2) @ dout_h                # (B, H, T, Dh)
        dattn  = dout_h @ V_exp.transpose(0, 1, 3, 2)               # (B, H, T, T)

        # ── Softmax backward ──────────────────────────────────────────────
        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dscores *= inv_sqrt

        # ── Q @ K_exp.T backward ─────────────────────────────────────────
        dQ     = dscores @ K_exp                                     # (B, H, T, Dh)
        dK_exp = dscores.transpose(0, 1, 3, 2) @ Q                  # (B, H, T, Dh)

        # ── GQA backward: sum over repeated groups ────────────────────────
        if self.n_rep > 1:
            dK_kv = dK_exp.reshape(B, Kv, self.n_rep, T, Dh).sum(axis=2)
            dV_kv = dV_exp.reshape(B, Kv, self.n_rep, T, Dh).sum(axis=2)
        else:
            dK_kv = dK_exp   # (B, Kv, T, Dh) = (B, H, T, Dh) when Kv==H
            dV_kv = dV_exp

        # ── RoPE backward ─────────────────────────────────────────────────
        if self.use_rope:
            dQ    = apply_rope_backward(dQ,    rope_cos, rope_sin)   # (B, H,  T, Dh)
            dK_kv = apply_rope_backward(dK_kv, rope_cos, rope_sin)   # (B, Kv, T, Dh)

        # ── Un-transpose and flatten into (B, T, dim) ─────────────────────
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK_flat = dK_kv.transpose(0, 2, 1, 3).reshape(B, T, kv_dim)
        dV_flat = dV_kv.transpose(0, 2, 1, 3).reshape(B, T, kv_dim)

        # ── Weight gradients ──────────────────────────────────────────────
        dWq = x.reshape(BT, D).T @ dQ_flat.reshape(BT, D)
        dWk = x.reshape(BT, D).T @ dK_flat.reshape(BT, kv_dim)
        dWv = x.reshape(BT, D).T @ dV_flat.reshape(BT, kv_dim)

        # ── Input gradient ────────────────────────────────────────────────
        dx = (dQ_flat @ self.Wq.T
              + dK_flat @ self.Wk.T
              + dV_flat @ self.Wv.T)

        return dx, {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo}

    # ── Introspection ─────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size

    def param_rows(self, prefix: str):
        D, kv_dim = self.dim, self.n_kv_heads * self.head_dim
        return [
            (f'{prefix}.Wq', f'({D}×{D})',      self.Wq.size),
            (f'{prefix}.Wk', f'({D}×{kv_dim})', self.Wk.size),
            (f'{prefix}.Wv', f'({D}×{kv_dim})', self.Wv.size),
            (f'{prefix}.Wo', f'({D}×{D})',      self.Wo.size),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Modern Block
# ─────────────────────────────────────────────────────────────────────────────

class ModernBlock:
    """
    Transformer block with configurable norm and FFN type.

      norm1 → ModernAttention → dropout → + residual
      norm2 → FFN             → dropout → + residual

    norm_cls : LayerNorm | RMSNorm
    ffn_cls  : FFN       | SwiGLUFFN
    """

    def __init__(
        self,
        dim:        int,
        n_heads:    int,
        n_kv_heads: int,
        dropout:    float,
        use_rope:   bool,
        norm_cls,
        ffn_cls,
        seed:       int,
    ):
        self.norm1     = norm_cls(dim)
        self.attn      = ModernAttention(dim, n_heads, n_kv_heads, use_rope, seed)
        self.attn_drop = Dropout(dropout)
        self.norm2     = norm_cls(dim)
        self.ffn       = ffn_cls(dim, seed + 1)
        self.ffn_drop  = Dropout(dropout)

    def forward(
        self,
        x:        np.ndarray,
        training: bool = False,
        kv_cache: dict | None = None,
    ) -> np.ndarray:
        self.attn_drop.training = training
        self.ffn_drop.training  = training

        # Attention sub-block
        h = x + self.attn_drop.forward(
            self.attn.forward(self.norm1.forward(x), training=training, kv_cache=kv_cache)
        )
        # FFN sub-block
        out = h + self.ffn_drop.forward(self.ffn.forward(self.norm2.forward(h)))
        return out

    def backward(self, dout: np.ndarray):
        # FFN sub-block (reversed)
        dffn_drop         = self.ffn_drop.backward(dout)
        dffn_in, ffn_g    = self.ffn.backward(dffn_drop)
        dln2_in, norm2_g  = self.norm2.backward(dffn_in)
        dh                = dout + dln2_in       # residual

        # Attention sub-block (reversed)
        dattn_drop        = self.attn_drop.backward(dh)
        dattn_in, attn_g  = self.attn.backward(dattn_drop)
        dln1_in, norm1_g  = self.norm1.backward(dattn_in)
        dx                = dh + dln1_in         # residual

        return dx, {'norm1': norm1_g, 'attn': attn_g, 'norm2': norm2_g, 'ffn': ffn_g}


# ─────────────────────────────────────────────────────────────────────────────
# Modern Transformer Language Model
# ─────────────────────────────────────────────────────────────────────────────

class ModernTransformerLM:
    """
    Full language model with modern upgrades.

    Parameters
    ----------
    vocab_size  : int
    embed_dim   : int
    block_size  : int    — context window
    n_layers    : int
    n_heads     : int
    n_kv_heads  : int    — for GQA (default = n_heads → standard MHA)
    dropout     : float
    seed        : int
    norm        : 'layernorm' | 'rmsnorm'
    ffn         : 'relu'     | 'swiglu'
    pos_enc     : 'learned'  | 'rope'
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        block_size:  int,
        n_layers:    int   = 2,
        n_heads:     int   = 4,
        n_kv_heads:  int   = None,
        dropout:     float = 0.0,
        seed:        int   = 42,
        norm:        str   = 'layernorm',
        ffn:         str   = 'relu',
        pos_enc:     str   = 'learned',
    ):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.block_size  = block_size
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads if n_kv_heads is not None else n_heads
        self.dropout     = dropout
        self.norm_type   = norm
        self.ffn_type    = ffn
        self.pos_enc     = pos_enc

        # Select component classes
        norm_cls = RMSNorm   if norm    == 'rmsnorm' else LayerNorm
        ffn_cls  = SwiGLUFFN if ffn     == 'swiglu'  else FFN
        use_rope = (pos_enc == 'rope')

        rng   = np.random.default_rng(seed)
        scale = 0.02

        self.token_embed = rng.standard_normal((vocab_size, embed_dim)).astype(np.float64) * scale

        # Positional embedding only for 'learned' mode
        self.pos_embed = None
        if pos_enc == 'learned':
            self.pos_embed = rng.standard_normal((block_size, embed_dim)).astype(np.float64) * scale

        self.blocks = [
            ModernBlock(
                dim        = embed_dim,
                n_heads    = n_heads,
                n_kv_heads = self.n_kv_heads,
                dropout    = dropout,
                use_rope   = use_rope,
                norm_cls   = norm_cls,
                ffn_cls    = ffn_cls,
                seed       = seed + i * 10,
            )
            for i in range(n_layers)
        ]

        self.norm_final = norm_cls(embed_dim)
        self.proj       = rng.standard_normal((embed_dim, vocab_size)).astype(np.float64) * scale
        self._c         = {}

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          np.ndarray,
        training:   bool = False,
        kv_caches:  list | None = None,
    ) -> np.ndarray:
        """
        x         : (B, T) int32 token ids
        kv_caches : list of per-layer cache dicts (inference only)
        Returns   : (B, T, V) logits
        """
        B, T = x.shape
        h = self.token_embed[x]                    # (B, T, D)
        if self.pos_embed is not None:
            # New tokens sit after whatever is already cached. seq_len is read
            # before any block runs, so it is 0 on prefill and the running
            # context length on each decode step.
            offset = kv_caches[0].get('seq_len', 0) if kv_caches else 0
            h = h + self.pos_embed[offset : offset + T]

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            h     = block.forward(h, training=training, kv_cache=cache)

        h_norm = self.norm_final.forward(h)
        logits = h_norm @ self.proj
        self._c = {'x': x, 'h_norm': h_norm}
        return logits

    # ── Loss + Gradients ──────────────────────────────────────────────────────

    def loss_and_grads(
        self,
        x:         np.ndarray,
        targets:   np.ndarray,
        loss_mask: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """
        loss_mask : optional (B, T) array of 0/1 aligned with `targets`. Where 0,
                    that target contributes nothing to loss OR gradient (true
                    masking). None → train on every token. Used by SFT.
        """
        B, T = x.shape
        N    = B * T

        logits      = self.forward(x, training=True)
        logits_flat = logits.reshape(N, self.vocab_size)
        tgts_flat   = targets.reshape(N)

        shifted = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=1, keepdims=True)

        nll = -np.log(probs[np.arange(N), tgts_flat] + 1e-8)

        dlogits = probs.copy()
        dlogits[np.arange(N), tgts_flat] -= 1.0

        if loss_mask is not None:
            m     = loss_mask.reshape(N).astype(np.float64)
            denom = max(float(m.sum()), 1.0)
            loss  = float((nll * m).sum() / denom)
            dlogits *= m[:, None]
            dlogits /= denom
        else:
            loss = float(nll.mean())
            dlogits /= N

        dlogits = dlogits.reshape(B, T, self.vocab_size)

        h_norm = self._c['h_norm']
        dproj  = h_norm.reshape(N, self.embed_dim).T @ dlogits.reshape(N, self.vocab_size)
        dh     = dlogits @ self.proj.T

        dh, norm_final_g = self.norm_final.backward(dh)

        block_grads = []
        for block in reversed(self.blocks):
            dh, bg = block.backward(dh)
            block_grads.insert(0, bg)

        dtoken_embed = np.zeros_like(self.token_embed)
        np.add.at(dtoken_embed, self._c['x'], dh)

        grads: dict = {
            'token_embed': dtoken_embed,
            'blocks':      block_grads,
            'norm_final':  norm_final_g,
            'proj':        dproj,
        }

        if self.pos_embed is not None:
            dpos_embed = np.zeros_like(self.pos_embed)
            dpos_embed[:T] = dh.sum(axis=0)
            grads['pos_embed'] = dpos_embed

        return loss, grads

    def compute_loss(self, x: np.ndarray, targets: np.ndarray) -> float:
        B, T = x.shape
        N    = B * T
        logits      = self.forward(x, training=False)
        logits_flat = logits.reshape(N, self.vocab_size)
        tgts_flat   = targets.reshape(N)
        shifted     = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l       = np.exp(shifted)
        probs       = exp_l / exp_l.sum(axis=1, keepdims=True)
        return float(-np.log(probs[np.arange(N), tgts_flat] + 1e-8).mean())

    # ── Flat parameter / gradient iterators (AdamW-compatible) ───────────────

    def _flat_params(self):
        """Yield (name, array) for every trainable parameter, in a fixed order."""
        yield 'token_embed', self.token_embed
        if self.pos_embed is not None:
            yield 'pos_embed', self.pos_embed

        for i, block in enumerate(self.blocks):
            p = f'block[{i}]'
            yield f'{p}.attn.Wq', block.attn.Wq
            yield f'{p}.attn.Wk', block.attn.Wk
            yield f'{p}.attn.Wv', block.attn.Wv
            yield f'{p}.attn.Wo', block.attn.Wo
            yield f'{p}.norm1.gamma', block.norm1.gamma
            if hasattr(block.norm1, 'beta'):
                yield f'{p}.norm1.beta', block.norm1.beta
            for key in ('Wg', 'Wv', 'W2', 'W1', 'b1', 'b2'):
                arr = getattr(block.ffn, key, None)
                if arr is not None:
                    yield f'{p}.ffn.{key}', arr
            yield f'{p}.norm2.gamma', block.norm2.gamma
            if hasattr(block.norm2, 'beta'):
                yield f'{p}.norm2.beta', block.norm2.beta

        yield 'norm_final.gamma', self.norm_final.gamma
        if hasattr(self.norm_final, 'beta'):
            yield 'norm_final.beta', self.norm_final.beta
        yield 'proj', self.proj

    def _flat_grads(self, grads: dict):
        """Yield (name, array) in exactly the same order as _flat_params()."""
        yield 'token_embed', grads['token_embed']
        if self.pos_embed is not None:
            yield 'pos_embed', grads['pos_embed']

        for i, bg in enumerate(grads['blocks']):
            p  = f'block[{i}]'
            ag = bg['attn']
            yield f'{p}.attn.Wq', ag['Wq']
            yield f'{p}.attn.Wk', ag['Wk']
            yield f'{p}.attn.Wv', ag['Wv']
            yield f'{p}.attn.Wo', ag['Wo']
            yield f'{p}.norm1.gamma', bg['norm1']['gamma']
            if 'beta' in bg['norm1']:
                yield f'{p}.norm1.beta', bg['norm1']['beta']
            ffn_g = bg['ffn']
            for key in ('Wg', 'Wv', 'W2', 'W1', 'b1', 'b2'):
                if key in ffn_g:
                    yield f'{p}.ffn.{key}', ffn_g[key]
            yield f'{p}.norm2.gamma', bg['norm2']['gamma']
            if 'beta' in bg['norm2']:
                yield f'{p}.norm2.beta', bg['norm2']['beta']

        yield 'norm_final.gamma', grads['norm_final']['gamma']
        if 'beta' in grads['norm_final']:
            yield 'norm_final.beta', grads['norm_final']['beta']
        yield 'proj', grads['proj']

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def description(self) -> str:
        upgrades = []
        if self.norm_type  == 'rmsnorm': upgrades.append('RMSNorm')
        if self.ffn_type   == 'swiglu':  upgrades.append('SwiGLU')
        if self.pos_enc    == 'rope':     upgrades.append('RoPE')
        if self.n_kv_heads < self.n_heads: upgrades.append(f'GQA(kv={self.n_kv_heads})')
        tag = '+'.join(upgrades) if upgrades else 'baseline'
        return (
            f"Phase 9 · ModernTransformerLM [{tag}] · "
            f"{self.n_layers}L · {self.n_heads}H · D={self.embed_dim}"
        )

    def param_table(self) -> list[tuple[str, str, int]]:
        V, D, T = self.vocab_size, self.embed_dim, self.block_size
        rows: list[tuple[str, str, int]] = [
            ('token_embed', f'({V}×{D})', self.token_embed.size),
        ]
        if self.pos_embed is not None:
            rows.append(('pos_embed', f'({T}×{D})', self.pos_embed.size))

        def _norm_params(norm):
            # RMSNorm has param_count(); LayerNorm does not — compute manually
            if hasattr(norm, 'param_count'):
                return norm.param_count()
            # LayerNorm: gamma + beta
            return norm.gamma.size + norm.beta.size

        for i, block in enumerate(self.blocks):
            p = f'block[{i}]'
            rows += block.attn.param_rows(f'{p}.attn')
            n1 = _norm_params(block.norm1)
            rows.append((f'{p}.norm1', f'({D},)', n1))
            rows += block.ffn.param_rows(f'{p}.ffn')
            n2 = _norm_params(block.norm2)
            rows.append((f'{p}.norm2', f'({D},)', n2))

        rows += [
            ('norm_final', f'({D},)', _norm_params(self.norm_final)),
            ('proj',       f'({D}×{V})', self.proj.size),
        ]
        return rows

    def param_count(self) -> dict[str, int]:
        counts = {name: n for name, _, n in self.param_table()}
        counts['total'] = sum(n for _, _, n in self.param_table())
        return counts

    def tracked_weight(self, row: int = 0, col: int = 0) -> float:
        return float(self.token_embed[row, col])

    def sgd_step(self, grads: dict, lr: float):
        """Simple SGD update (used in tests; real training uses AdamW)."""
        for (_, param), (_, grad) in zip(self._flat_params(), self._flat_grads(grads)):
            param -= lr * grad
