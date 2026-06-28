#!/usr/bin/env python3
"""
Phase 6 — Standalone generation from a saved checkpoint.

This is the Phase 6 demo entry point: load a trained model from disk
and generate text using any combination of sampling strategies.
The --compare flag runs all strategies back to back on the SAME model,
making the effect of each knob immediately visible.

Usage
-----
  python sample_cli.py                          # default settings from config.yaml
  python sample_cli.py --greedy                 # argmax — deterministic, safe
  python sample_cli.py --temperature 0.2        # conservative / boring
  python sample_cli.py --temperature 1.5        # creative / random
  python sample_cli.py --top_k 5               # sample from top-5 tokens only
  python sample_cli.py --top_p 0.9             # nucleus: keep 90% of mass
  python sample_cli.py --rep_penalty 1.3       # mild repetition suppression
  python sample_cli.py --compare               # show all strategies side by side
  python sample_cli.py --checkpoint runs/latest.npz --seed "the"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from sample.checkpoint import load
from sample.sampler import generate

console = Console()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='nanoLM sampler — generate text from a saved checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint',   default='runs/latest.npz', help='path to .npz checkpoint')
    p.add_argument('--config',       default='config.yaml',     help='config file (for defaults)')
    p.add_argument('--seed',         default=None,              help='seed text / prompt')
    p.add_argument('--max_tokens',   type=int,   default=None,  help='tokens to generate')
    p.add_argument('--temperature',  type=float, default=None,  help='temperature (1.0 = unchanged)')
    p.add_argument('--top_k',        type=int,   default=None,  help='top-k filter (0 = off)')
    p.add_argument('--top_p',        type=float, default=None,  help='nucleus threshold (1.0 = off)')
    p.add_argument('--rep_penalty',  type=float, default=None,  help='repetition penalty (1.0 = off)')
    p.add_argument('--greedy',       action='store_true',       help='greedy / argmax decoding')
    p.add_argument('--compare',      action='store_true',       help='run all strategies and compare')
    p.add_argument('--seed_int',     type=int,   default=42,    help='numpy random seed')
    return p.parse_args()


# ── Load config defaults ───────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


# ── Single generation ─────────────────────────────────────────────────────────

def run_generate(model, tokenizer, seed_text, max_tokens, block_size, **kwargs) -> str:
    return generate(
        model=model,
        tokenizer=tokenizer,
        seed_text=seed_text,
        max_new_tokens=max_tokens,
        block_size=block_size,
        **kwargs,
    )


# ── Compare mode: all strategies ──────────────────────────────────────────────

STRATEGIES = [
    {
        'label':       'Greedy (argmax)',
        'settings':    'deterministic',
        'kwargs':      dict(greedy=True),
    },
    {
        'label':       'Temp = 0.2',
        'settings':    'conservative',
        'kwargs':      dict(temperature=0.2),
    },
    {
        'label':       'Temp = 1.0',
        'settings':    'default',
        'kwargs':      dict(temperature=1.0),
    },
    {
        'label':       'Temp = 1.5',
        'settings':    'creative',
        'kwargs':      dict(temperature=1.5),
    },
    {
        'label':       'Top-k  (k=5)',
        'settings':    'top-5 only',
        'kwargs':      dict(temperature=1.0, top_k=5),
    },
    {
        'label':       'Top-p  (p=0.9)',
        'settings':    'nucleus 90%',
        'kwargs':      dict(temperature=1.0, top_p=0.9),
    },
    {
        'label':       'Rep penalty 1.3',
        'settings':    'anti-repeat',
        'kwargs':      dict(temperature=1.0, rep_penalty=1.3),
    },
]


def compare_mode(model, tokenizer, seed_text: str, max_tokens: int, block_size: int):
    console.print()
    console.print(Panel(
        f"[bold cyan]Phase 6 Sampling Comparison[/bold cyan]\n"
        f"Same model · same seed '{seed_text}' · {max_tokens} tokens each",
        border_style="cyan",
    ))
    console.print()

    tbl = Table(box=box.ROUNDED, show_header=True, expand=True)
    tbl.add_column("Strategy",    style="bold cyan",  width=18)
    tbl.add_column("Settings",    style="dim",         width=14)
    tbl.add_column("Output",      style="italic",      ratio=1)

    for s in STRATEGIES:
        np.random.seed(42)   # same seed each time for fair comparison
        text = run_generate(model, tokenizer, seed_text, max_tokens, block_size, **s['kwargs'])
        # Show just the new tokens (strip seed)
        new_text = text[len(seed_text):]
        tbl.add_row(s['label'], s['settings'], new_text)

    console.print(tbl)
    console.print()

    console.print(Panel(
        "[bold]Key observations:[/bold]\n\n"
        "  [cyan]Greedy[/cyan]        — deterministic; often loops or gets stuck\n"
        "  [cyan]Temp < 1[/cyan]      — peaked distribution; safe but boring\n"
        "  [cyan]Temp > 1[/cyan]      — flat distribution; diverse but incoherent\n"
        "  [cyan]Top-k[/cyan]         — always considers exactly k candidates\n"
        "  [cyan]Top-p[/cyan]         — adapts to model confidence; sharper when certain\n"
        "  [cyan]Rep penalty[/cyan]   — penalises already-seen tokens; reduces loops",
        title="[bold]What the knobs do[/bold]",
        border_style="dim",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed_int)

    # ── Load checkpoint ──
    ckpt = args.checkpoint
    if not Path(ckpt).exists():
        console.print(f"[red]Checkpoint not found: {ckpt}[/red]")
        console.print("  Train first:  python main.py --epochs 500")
        sys.exit(1)

    model, tokenizer = load(ckpt)
    console.print(f"[green]Loaded[/green] {ckpt}  "
                  f"({model.param_count()['total']:,} params, "
                  f"vocab={tokenizer.vocab_size})")

    # ── Load config defaults ──
    cfg   = load_config(args.config)
    scfg  = cfg.get('sampling', {})
    mcfg  = cfg.get('model', {})

    seed_text   = args.seed       or scfg.get('seed_text', 't')
    max_tokens  = args.max_tokens or scfg.get('max_new_tokens', 120)
    block_size  = mcfg.get('block_size', getattr(model, 'block_size', 32))
    temperature = args.temperature if args.temperature is not None else float(scfg.get('temperature', 1.0))
    top_k       = args.top_k       if args.top_k       is not None else int(scfg.get('top_k', 0))
    top_p       = args.top_p       if args.top_p       is not None else float(scfg.get('top_p', 1.0))
    rep_penalty = args.rep_penalty if args.rep_penalty is not None else float(scfg.get('rep_penalty', 1.0))
    greedy      = args.greedy

    # ── Compare mode ──
    if args.compare:
        compare_mode(model, tokenizer, seed_text, max_tokens, block_size)
        return

    # ── Single generation ──
    settings = []
    if greedy:
        settings.append("greedy")
    else:
        settings.append(f"temp={temperature}")
        if top_k > 0:   settings.append(f"top_k={top_k}")
        if top_p < 1.0: settings.append(f"top_p={top_p}")
    if rep_penalty != 1.0:
        settings.append(f"rep_penalty={rep_penalty}")

    text = run_generate(
        model, tokenizer, seed_text, max_tokens, block_size,
        temperature=temperature, top_k=top_k, top_p=top_p,
        greedy=greedy, rep_penalty=rep_penalty,
    )

    console.print()
    console.print(Panel(
        Text(text, style="italic"),
        title=f"[yellow]Generated ({', '.join(settings)})[/yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))
    console.print()


if __name__ == '__main__':
    main()
