#!/usr/bin/env python3
"""
Phase 10 — Mixture of Experts benchmark.

Three sections, each independent:

  Section 1: Sparsity accounting
    How total and active parameter counts diverge as experts are added.
    Active = what one token actually touches, which is the number that
    decides inference cost.

  Section 2: Load balancing
    Train the same model three ways — no balancing, auxiliary loss, and
    auxiliary-loss-free bias — and measure where the tokens actually go.
    Reports routing entropy, the busiest expert's share, and how many
    experts received essentially nothing.

  Section 3: Chart
    Writes docs/moe_expert_utilization.svg from the Section 2 results.

Run
---
  python bench_phase10.py
  python bench_phase10.py --steps 800 --n_experts 16 --top_k 1

Top-1 routing is the default because it is the case that actually misbehaves:
with a single expert per token the gate cannot hedge, so early winners compound.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table   import Table
from rich         import box

sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer           import CharTokenizer
from data.loader              import BatchLoader
from model.modern_transformer import ModernTransformerLM
from train.optimizer          import build_optimizer, clip_grad_norm

console = Console()

CORPUS_PATH = 'data/corpus_large.txt'
STRATEGIES  = [
    ('none', 'no balancing',        'red'),
    ('aux',  'auxiliary loss',      'yellow'),
    ('bias', 'aux-loss-free bias',  'green'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared training helper
# ─────────────────────────────────────────────────────────────────────────────

def build(tok, balance, args, seed=42):
    return ModernTransformerLM(
        tok.vocab_size, args.embed_dim, args.block_size,
        n_layers=args.n_layers, n_heads=4, seed=seed,
        norm='rmsnorm', ffn='swiglu', pos_enc='rope',
        moe=dict(n_experts=args.n_experts, top_k=args.top_k,
                 n_shared=0, balance=balance),
    )


def train(model, data, args, seed=42):
    """Train and record the routing-entropy trajectory."""
    np.random.seed(seed)
    loader = BatchLoader(data, args.block_size, args.batch_size, seed=seed)
    opt    = build_optimizer({'training': {'optimizer': 'adamw', 'lr': args.lr}})

    traj, loss, t0 = [], float('nan'), time.time()
    for step in range(args.steps):
        x, y = loader.next_batch()
        loss, grads = model.loss_and_grads(x, y)
        clip_grad_norm(list(model._flat_grads(grads)), 1.0)
        opt.step(model, grads)
        if step % max(args.steps // 20, 1) == 0:
            traj.append(model.balance_entropy())
    traj.append(model.balance_entropy())
    return loss, traj, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — sparsity accounting
# ─────────────────────────────────────────────────────────────────────────────

def section_sparsity(tok, args):
    console.rule("[bold cyan]Section 1 — total vs active parameters")
    console.print(
        "\nAdding experts grows what the model [bold]stores[/bold]. It does not grow what a "
        "single token\n[bold]uses[/bold] — only the router does, and only slightly. "
        "That gap is the entire point of MoE.\n"
    )

    tbl = Table(box=box.ROUNDED, show_header=True)
    for col in ("experts", "top-k", "total params", "active / token", "total ÷ active"):
        tbl.add_column(col, justify="right")

    for n_exp in (1, 2, 4, 8, 16):
        m = build(tok, 'bias', argparse.Namespace(**{**vars(args), 'n_experts': n_exp}))
        total, active = m.param_count()['total'], m.active_param_count()
        tbl.add_row(str(n_exp), str(min(args.top_k, n_exp)),
                    f"{total:,}", f"{active:,}", f"{total/active:.2f}×")
    console.print(tbl)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — load balancing
# ─────────────────────────────────────────────────────────────────────────────

def section_balancing(tok, data, args):
    console.rule("[bold cyan]Section 2 — where the tokens actually go")
    console.print(
        f"\nSame model, same seed, same data. Only the balancing strategy differs.\n"
        f"[dim]{args.n_experts} experts · top-{args.top_k} routing · {args.steps} steps · "
        f"uniform share would be {100/args.n_experts:.1f}%[/dim]\n"
    )

    results = []
    for balance, label, colour in STRATEGIES:
        model = build(tok, balance, args)
        loss, traj, secs = train(model, data, args)
        util = model.expert_utilization()[0]
        results.append({
            'balance': balance, 'label': label, 'colour': colour,
            'loss': loss, 'entropy': model.balance_entropy(),
            'util': util, 'traj': traj,
            'dead': int((util < 0.01).sum()), 'secs': secs,
        })
        console.print(f"  [{colour}]{label:22s}[/{colour}] "
                      f"loss={loss:.4f}  entropy={model.balance_entropy():.3f}  "
                      f"busiest={100*util.max():.1f}%  dead={results[-1]['dead']}/{args.n_experts}"
                      f"  [dim]({secs:.1f}s)[/dim]")

    console.print()
    tbl = Table(box=box.ROUNDED, show_header=True, title="Routing outcome")
    tbl.add_column("strategy");            tbl.add_column("final loss", justify="right")
    tbl.add_column("entropy", justify="right")
    tbl.add_column("busiest expert", justify="right")
    tbl.add_column("unused experts", justify="right")
    for r in results:
        tbl.add_row(f"[{r['colour']}]{r['label']}[/{r['colour']}]",
                    f"{r['loss']:.4f}", f"{r['entropy']:.3f}",
                    f"{100*r['util'].max():.1f}%",
                    f"{r['dead']}/{args.n_experts}")
    console.print(tbl)

    # Per-expert bars
    console.print("\n[bold]Expert utilization[/bold]  [dim](· marks an even split)[/dim]\n")
    uniform = 1.0 / args.n_experts
    width   = 44
    for r in results:
        console.print(f"  [{r['colour']}]{r['label']}[/{r['colour']}]")
        for i, share in enumerate(r['util']):
            filled = int(round(share / max(r['util'].max(), 1e-9) * width))
            marker = int(round(uniform / max(r['util'].max(), 1e-9) * width))
            bar    = ''.join('█' if j < filled else ('·' if j == marker else ' ')
                             for j in range(width))
            console.print(f"    e{i}  [{r['colour']}]{bar}[/{r['colour']}] {100*share:5.1f}%")
        console.print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — SVG chart (no plotting dependency)
# ─────────────────────────────────────────────────────────────────────────────

def section_chart(results, args, out='docs/moe_expert_utilization.svg'):
    console.rule("[bold cyan]Section 3 — chart")

    E        = args.n_experts
    uniform  = 1.0 / E
    peak     = max(max(r['util']) for r in results)
    W, H     = 860, 430
    pad_l, pad_t = 70, 70
    panel_w  = (W - pad_l - 40 - 40) / len(results)
    plot_h   = 200
    colours  = {'none': '#d1495b', 'aux': '#e0a458', 'bias': '#2a9d8f'}

    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{pad_l}" y="32" font-size="18" font-weight="600" fill="#1a1a1a">'
         f'Where {E} experts actually receive tokens</text>',
         f'<text x="{pad_l}" y="52" font-size="12.5" fill="#666">'
         f'top-{args.top_k} routing · {args.steps} steps · dashed line = even split '
         f'({100*uniform:.1f}%)</text>']

    for pi, r in enumerate(results):
        x0 = pad_l + pi * panel_w
        bw = (panel_w - 45) / E
        c  = colours.get(r['balance'], '#666')

        p.append(f'<text x="{x0}" y="{pad_t + 8}" font-size="13" font-weight="600" '
                 f'fill="{c}">{esc(r["label"])}</text>')
        p.append(f'<text x="{x0}" y="{pad_t + 26}" font-size="11.5" fill="#666">'
                 f'entropy {r["entropy"]:.2f} · loss {r["loss"]:.3f}'
                 + (f' · {r["dead"]} unused' if r['dead'] else '') + '</text>')

        base = pad_t + 40 + plot_h
        for i, share in enumerate(r['util']):
            h  = (share / peak) * plot_h
            bx = x0 + i * bw
            faded = share < 0.01
            p.append(f'<rect x="{bx:.1f}" y="{base - h:.1f}" width="{bw*0.78:.1f}" '
                     f'height="{max(h,0.6):.1f}" fill="{c}" '
                     f'opacity="{0.28 if faded else 0.88}"/>')

        p.append(f'<line x1="{x0}" y1="{base:.1f}" x2="{x0 + E*bw:.1f}" y2="{base:.1f}" '
                 f'stroke="#bbb" stroke-width="1"/>')
        uy = base - (uniform / peak) * plot_h
        p.append(f'<line x1="{x0}" y1="{uy:.1f}" x2="{x0 + E*bw:.1f}" y2="{uy:.1f}" '
                 f'stroke="#333" stroke-width="1" stroke-dasharray="4 3" opacity="0.55"/>')
        p.append(f'<text x="{x0}" y="{base + 16:.1f}" font-size="10.5" fill="#888">'
                 f'expert 0 … {E-1}</text>')
        p.append(f'<text x="{x0}" y="{base + 34:.1f}" font-size="11" fill="#444">'
                 f'busiest {100*r["util"].max():.1f}%</text>')

    p.append(f'<text x="{pad_l}" y="{H - 14}" font-size="11.5" fill="#666">'
             f'nanoLM Phase 10 — hand-written NumPy MoE. Without balancing the router '
             f'concentrates and some experts stop being used at all.</text>')
    p.append('</svg>')

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text('\n'.join(p))
    console.print(f"\n  Wrote [cyan]{out}[/cyan]\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='nanoLM Phase 10 — MoE benchmark')
    ap.add_argument('--steps',      type=int,   default=600)
    ap.add_argument('--n_experts',  type=int,   default=8)
    ap.add_argument('--top_k',      type=int,   default=1)
    ap.add_argument('--n_layers',   type=int,   default=2)
    ap.add_argument('--embed_dim',  type=int,   default=64)
    ap.add_argument('--block_size', type=int,   default=48)
    ap.add_argument('--batch_size', type=int,   default=16)
    ap.add_argument('--lr',         type=float, default=3e-3)
    ap.add_argument('--corpus',     type=str,   default=CORPUS_PATH)
    ap.add_argument('--skip_chart', action='store_true')
    args = ap.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        sys.exit(f"[error] corpus not found: {path}")

    text = path.read_text()
    tok  = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int32)

    console.print()
    console.rule("[bold]nanoLM Phase 10 — Mixture of Experts")
    console.print(f"\ncorpus [cyan]{path}[/cyan] · {len(text):,} chars · "
                  f"vocab {tok.vocab_size}\n")

    section_sparsity(tok, args)
    results = section_balancing(tok, data, args)
    if not args.skip_chart:
        section_chart(results, args)

    none = next(r for r in results if r['balance'] == 'none')
    bias = next(r for r in results if r['balance'] == 'bias')
    aux  = next(r for r in results if r['balance'] == 'aux')

    console.print("[bold green]Phase 10 benchmark complete.[/bold green]\n")
    console.print(
        "What the numbers show:\n"
        f"  • Unbalanced top-{args.top_k} routing concentrates: busiest expert "
        f"{100*none['util'].max():.1f}% vs {100/args.n_experts:.1f}% even, "
        f"{none['dead']} expert(s) unused\n"
        f"  • Both fixes restore balance — entropy {none['entropy']:.2f} → "
        f"{aux['entropy']:.2f} (aux) / {bias['entropy']:.2f} (bias)\n"
        f"  • The auxiliary loss is a second objective competing with next-token "
        f"prediction; the bias rule adds none\n"
        f"  • Total parameters grow with experts, active parameters per token do not\n"
    )


if __name__ == '__main__':
    main()
