"""
Phase 10 — Mixture of Experts.

Every dense FFN in a block is replaced by E expert FFNs and a router that sends
each token to only the top-k of them. Total parameters grow by roughly E×, but
the parameters *used* per token stay at k×. That gap — total vs active — is the
whole point, and it is why every serious open-weight model released since 2024
is sparse rather than dense.

    dense    : out = FFN(x)
    MoE      : out = sum_{e in topk(x)} g_e · FFN_e(x)          [+ shared expert]

Shapes
------
  N  = B·T   tokens in the batch
  D  = embed_dim
  E  = n_experts
  k  = top_k

Routing
-------
    logits = x · Wr                                  (N, E)
    idx    = indices of the k largest (logits + bias) per token
    g      = softmax(logits gathered at idx)         (N, k),  rows sum to 1
    out[n] = Σ_j g[n,j] · Expert_{idx[n,j]}(x[n])

`bias` steers *selection only* and never enters `g` — see "Load balancing".

Backward
--------
Let y[n] = Σ_j g[n,j]·o_j[n], where o_j[n] is expert idx[n,j] applied to x[n].

    ∂L/∂o_j[n] = g[n,j] · dy[n]        → goes into that expert's own backward
    ∂L/∂g[n,j] = dy[n] · o_j[n]        (a scalar per routed slot)

`g` is a softmax over the k gathered logits, so

    ∂L/∂sel_logits = g ⊙ (dg − Σ_j dg·g)

which is scattered back to the (N, E) logit grid at `idx` — every unselected
expert gets exactly zero, which is what makes the layer sparse in the backward
pass too. Then the usual linear-layer rules give

    dWr = xᵀ · dlogits
    dx  = dlogits · Wrᵀ  +  Σ_j g[n,j] · (dx from expert idx[n,j])

The top-k *selection* itself is not differentiable. Gradient flows through the
gate weights of the chosen experts and through the experts themselves, never
through the choice. That is standard, and it is why routing needs a balancing
mechanism rather than relying on the loss to spread tokens out.

Load balancing
--------------
Left alone, routers collapse: a few experts win early, receive most of the
gradient, get better, and win more. The rest die. Two fixes are implemented so
they can be compared directly.

`balance='aux'` — the classic auxiliary loss (Switch Transformer). With f_e the
fraction of routing slots that went to expert e, and P_e the mean router
probability for e,

    L_aux = α · E · Σ_e f_e · P_e

which is minimized when the load is uniform. It works, but it injects a second
gradient into the router that has nothing to do with predicting the next token.

`balance='bias'` — auxiliary-loss-free balancing (DeepSeek-V3, arXiv 2408.15664).
Each expert carries a scalar bias added to its score for the top-k comparison
and excluded from the gate weight. After every training step the bias moves
against recent load:

    bias_e ← bias_e + γ · sign(mean_load − load_e)

Overloaded experts become harder to select, underloaded ones easier. No extra
term in the loss, so no interference gradient — that is the paper's argument,
and at this scale it is measurable rather than merely citable.
"""

import numpy as np


class MoEFFN:
    """
    Sparse mixture-of-experts feed-forward layer.

    Drop-in for FFN / SwiGLUFFN: `forward(x)` and `backward(dout)` with the same
    shapes, so ModernBlock does not care which one it is holding. Experts are
    whatever `expert_cls` is (FFN or SwiGLUFFN), reused unchanged — their
    backward passes are already gradient-checked, so the only new derivation
    here is the router.

    Parameters
    ----------
    dim         : model width
    n_experts   : E, number of routed experts
    top_k       : k, experts consulted per token
    expert_cls  : FFN | SwiGLUFFN
    n_shared    : experts applied to every token, outside the routing (0 or 1).
                  DeepSeek-style: soaks up patterns common to all tokens so the
                  routed experts can specialize instead of each relearning them.
    balance     : 'none' | 'aux' | 'bias'
    aux_alpha   : α for the auxiliary loss
    bias_rate   : γ for the bias update
    """

    def __init__(
        self,
        dim:        int,
        n_experts:  int   = 4,
        top_k:      int   = 2,
        expert_cls        = None,
        seed:       int   = 0,
        n_shared:   int   = 0,
        balance:    str   = 'bias',
        aux_alpha:  float = 0.01,
        bias_rate:  float = 0.001,
    ):
        assert 1 <= top_k <= n_experts, f"top_k {top_k} must be in 1..n_experts {n_experts}"
        assert balance in ('none', 'aux', 'bias'), f"unknown balance strategy {balance!r}"

        if expert_cls is None:
            from model.transformer import FFN
            expert_cls = FFN

        rng = np.random.default_rng(seed)

        self.dim       = dim
        self.n_experts = n_experts
        self.top_k     = top_k
        self.balance   = balance
        self.aux_alpha = aux_alpha
        self.bias_rate = bias_rate

        # Router. Small init keeps the initial distribution near-uniform, which
        # matters: a router that starts opinionated collapses faster.
        self.Wr = rng.standard_normal((dim, n_experts)).astype(np.float64) * 0.02

        self.experts = [expert_cls(dim, seed + 100 + i) for i in range(n_experts)]
        self.shared  = expert_cls(dim, seed + 900) if n_shared else None

        # Selection bias — updated by a rule, never by a gradient.
        self.expert_bias = np.zeros(n_experts, dtype=np.float64)

        self.training    = False
        self.aux_loss    = 0.0                                  # last forward
        self.load_counts = np.zeros(n_experts, dtype=np.float64)  # last forward
        self._c          = {}

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        N  = B * T
        E  = self.n_experts
        k  = self.top_k
        xf = x.reshape(N, D)

        logits = xf @ self.Wr                              # (N, E)

        # Selection sees the bias; the gate weights below do not.
        sel_scores = logits + self.expert_bias
        idx = np.argpartition(-sel_scores, k - 1, axis=1)[:, :k]   # (N, k)

        # Gate weights come from a softmax over ALL experts, then gathered at the
        # chosen indices. Taking a softmax over only the selected logits would be
        # equivalent for k>1 after renormalizing, but at k=1 it collapses to a
        # constant 1.0 — and a constant has no derivative, so the router would
        # receive no gradient at all and never learn. Switch Transformer (k=1)
        # uses the un-renormalized probability for exactly this reason.
        shifted   = logits - logits.max(axis=1, keepdims=True)
        exp_s     = np.exp(shifted)
        probs_all = exp_s / exp_s.sum(axis=1, keepdims=True)       # (N, E)

        sel = np.take_along_axis(probs_all, idx, axis=1)           # (N, k)
        if k > 1:
            denom = sel.sum(axis=1, keepdims=True)
            g     = sel / denom                                    # rows sum to 1
        else:
            denom = None
            g     = sel                                            # Switch-style

        y = np.zeros((N, D), dtype=np.float64)
        per_expert = {}

        for e in range(E):
            hit = (idx == e)
            if not hit.any():
                continue
            tok, slot = np.nonzero(hit)          # token ids are unique within an expert
            xe  = xf[tok]                        # (M, D)
            oe  = self.experts[e].forward(xe.reshape(1, -1, D))[0]   # (M, D)
            y[tok] += g[tok, slot][:, None] * oe
            per_expert[e] = (tok, slot, oe)

        counts = np.bincount(idx.reshape(-1), minlength=E).astype(np.float64)
        self.load_counts = counts

        if self.balance == 'aux':
            f = counts / max(N * k, 1)                              # load fraction
            P = probs_all.mean(axis=0)                              # mean gate prob
            self.aux_loss = float(self.aux_alpha * E * (f * P).sum())
        else:
            self.aux_loss = 0.0

        # Bias balancing is a state update, not a gradient — training only.
        if self.training and self.balance == 'bias':
            self.expert_bias += self.bias_rate * np.sign(counts.mean() - counts)

        shared_out = None
        if self.shared is not None:
            shared_out = self.shared.forward(x)
            y += shared_out.reshape(N, D)

        self._c = {
            'x': x, 'xf': xf, 'idx': idx, 'g': g, 'N': N, 'denom': denom,
            'per_expert': per_expert, 'probs_all': probs_all,
            'counts': counts, 'has_shared': shared_out is not None,
        }
        return y.reshape(B, T, D)

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, dout: np.ndarray):
        """
        dout : (B, T, D)

        Returns (dx, grads). `grads` holds the router matrix plus one sub-dict
        per expert, so _flat_params / _flat_grads can walk them in a fixed order.
        Experts that received no tokens still get zero gradients of the right
        shape — the optimizer iterates every parameter every step.
        """
        x   = self._c['x']
        xf  = self._c['xf']
        idx = self._c['idx']
        g   = self._c['g']
        N   = self._c['N']
        per_expert = self._c['per_expert']

        B, T, D = x.shape
        E, k    = self.n_experts, self.top_k

        dyf = dout.reshape(N, D)
        dxf = np.zeros((N, D), dtype=np.float64)
        dg  = np.zeros((N, k),  dtype=np.float64)

        expert_grads = []
        for e in range(E):
            if e not in per_expert:
                expert_grads.append(self._zero_like_expert(self.experts[e]))
                continue
            tok, slot, oe = per_expert[e]
            w = g[tok, slot][:, None]                    # (M, 1)

            # ∂L/∂g for this slot is the dot product of upstream grad and output
            dg[tok, slot] = (dyf[tok] * oe).sum(axis=1)

            dxe, ge = self.experts[e].backward((dyf[tok] * w).reshape(1, -1, D))
            dxf[tok] += dxe[0]
            expert_grads.append(ge)

        # Back through the renormalization g = sel / Σsel (identity when k == 1,
        # where the un-renormalized probability is used directly).
        denom = self._c['denom']
        if denom is not None:
            d_sel = (dg - (dg * g).sum(axis=1, keepdims=True)) / denom
        else:
            d_sel = dg

        # Scatter onto the full expert grid: unselected experts get exactly zero,
        # which is what makes the backward pass sparse as well as the forward.
        probs = self._c['probs_all']
        dprobs = np.zeros((N, E), dtype=np.float64)
        np.put_along_axis(dprobs, idx, d_sel, axis=1)

        if self.balance == 'aux':
            dprobs = dprobs + self._aux_dprobs()

        # Back through the row-wise softmax over all E experts
        d_logits = probs * (dprobs - (dprobs * probs).sum(axis=1, keepdims=True))

        dWr  = xf.T @ d_logits
        dxf += d_logits @ self.Wr.T

        grads = {'Wr': dWr, 'experts': expert_grads}

        dx = dxf.reshape(B, T, D)
        if self.shared is not None:
            dxs, gs = self.shared.backward(dout)
            dx = dx + dxs
            grads['shared'] = gs

        return dx, grads

    # ── Auxiliary-loss gradient ──────────────────────────────────────────────

    def _aux_dprobs(self) -> np.ndarray:
        """
        ∂L_aux/∂probs, with the load fractions f held constant (they come from an
        argmax and carry no usable gradient). The caller pushes this through the
        same softmax backward as the routing term.

            L_aux    = α·E·Σ_e f_e·P_e,     P_e = (1/N)·Σ_n probs[n,e]
            ∂L/∂P_e  = α·E·f_e
            ∂L/∂prob = α·E·f_e / N
        """
        N = self._c['N']
        E = self.n_experts
        f = self._c['counts'] / max(N * self.top_k, 1)
        return (self.aux_alpha * E * f / N)[None, :]                   # (1, E)

    @staticmethod
    def _zero_like_expert(expert) -> dict:
        """Zero gradients matching an expert that received no tokens this step."""
        return {
            key: np.zeros_like(getattr(expert, key))
            for key in ('W1', 'b1', 'W2', 'b2', 'Wg', 'Wv')
            if getattr(expert, key, None) is not None
        }

    # ── Introspection ────────────────────────────────────────────────────────

    def utilization(self) -> np.ndarray:
        """Fraction of routing slots each expert received in the last forward."""
        total = self.load_counts.sum()
        if total == 0:
            return np.zeros(self.n_experts)
        return self.load_counts / total

    def balance_entropy(self) -> float:
        """
        Normalized entropy of the routing distribution, in [0, 1].
        1.0 = perfectly uniform, 0.0 = every token to one expert. This is the
        number that shows collapse happening.
        """
        u = self.utilization()
        nz = u[u > 0]
        if nz.size <= 1:
            return 0.0
        return float(-(nz * np.log(nz)).sum() / np.log(self.n_experts))

    def param_count(self) -> int:
        n = self.Wr.size + sum(e.param_count() for e in self.experts)
        if self.shared is not None:
            n += self.shared.param_count()
        return n

    def active_param_count(self) -> int:
        """Parameters actually used for a single token."""
        per_expert = self.experts[0].param_count()
        n = self.Wr.size + self.top_k * per_expert
        if self.shared is not None:
            n += self.shared.param_count()
        return n

    def param_rows(self, prefix: str):
        rows = [(f'{prefix}.Wr', f'({self.dim}×{self.n_experts})', self.Wr.size)]
        for i, e in enumerate(self.experts):
            rows += e.param_rows(f'{prefix}.expert[{i}]')
        if self.shared is not None:
            rows += self.shared.param_rows(f'{prefix}.shared')
        return rows
