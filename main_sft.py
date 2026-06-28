#!/usr/bin/env python3
"""
nanoLM Phase 8 — Supervised Fine-Tuning entry point.

Trains the TransformerLM on instruction pairs and shows before/after
behaviour so the base-model vs aligned-model distinction is concrete.

Usage
-----
  # Train SFT from scratch on instruction data
  python main_sft.py

  # Fine-tune an existing base checkpoint
  python main_sft.py --base runs/latest.npz

  # More steps / custom question
  python main_sft.py --epochs 500 --sample_user "What is attention?"

  # Change system prompt
  python main_sft.py --system "You are a pirate. Respond in pirate speak."

Outputs
-------
  runs/<timestamp>_sft.json   — loss history
  runs/latest_sft.npz         — SFT checkpoint (loadable by sample_cli.py)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from data.tokenizer import CharTokenizer
from data.instruct_dataset import DEFAULT_PAIRS, build_corpus, InstructPair
from model.transformer import TransformerLM
from sample.prompter import Prompter
from sample.checkpoint import load as load_checkpoint
from train.trainer_sft import TrainerSFT

console = Console()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = 'config.yaml') -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def sft_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    """Merge SFT-specific settings into a copy of base config."""
    cfg = dict(base_cfg)
    cfg.setdefault('model', {})
    cfg.setdefault('training', {})
    cfg.setdefault('dashboard', {})
    cfg.setdefault('sft', {})

    # Model defaults (small — SFT is fast)
    cfg['model'].setdefault('embed_dim',  64)
    cfg['model'].setdefault('block_size', 64)
    cfg['model'].setdefault('n_layers',    2)
    cfg['model'].setdefault('n_heads',     4)
    cfg['model'].setdefault('dropout',   0.0)

    # Training defaults
    cfg['training'].setdefault('epochs',      300)
    cfg['training'].setdefault('batch_size',    4)
    cfg['training'].setdefault('lr',         3e-4)
    cfg['training'].setdefault('optimizer', 'adamw')
    cfg['training'].setdefault('grad_clip',   1.0)
    cfg['training'].setdefault('warmup_steps', 50)
    cfg['training'].setdefault('seed',         42)
    cfg['training'].setdefault('val_split',   0.0)

    # Dashboard
    cfg['dashboard'].setdefault('log_every',    10)
    cfg['dashboard'].setdefault('sample_every', 50)

    # SFT-specific
    cfg['sft']['max_new_tokens'] = args.max_new_tokens
    cfg['sft']['temperature']    = args.temperature
    cfg['sft']['sample_user']    = args.sample_user
    if args.system:
        cfg['sft']['system_prompt'] = args.system

    # CLI overrides
    if args.epochs:     cfg['training']['epochs']     = args.epochs
    if args.lr:         cfg['training']['lr']         = args.lr
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.embed_dim:  cfg['model']['embed_dim']     = args.embed_dim
    if args.n_layers:   cfg['model']['n_layers']      = args.n_layers
    if args.n_heads:    cfg['model']['n_heads']       = args.n_heads
    if args.block_size: cfg['model']['block_size']    = args.block_size

    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='nanoLM SFT — supervised fine-tuning on instruction pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',       default='config.yaml')
    p.add_argument('--base',         default=None,
                   help='path to base checkpoint .npz to fine-tune from')
    p.add_argument('--epochs',       type=int,   default=None)
    p.add_argument('--lr',           type=float, default=None)
    p.add_argument('--batch_size',   type=int,   default=None)
    p.add_argument('--embed_dim',    type=int,   default=None)
    p.add_argument('--n_layers',     type=int,   default=None)
    p.add_argument('--n_heads',      type=int,   default=None)
    p.add_argument('--block_size',   type=int,   default=None)
    p.add_argument('--system',       type=str,   default=None,
                   help='system prompt for all instruction pairs')
    p.add_argument('--sample_user',  type=str,   default='What is your name?',
                   help='user message shown as sample during training')
    p.add_argument('--max_new_tokens', type=int, default=80)
    p.add_argument('--temperature',  type=float, default=0.8)
    p.add_argument('--repeat',       type=int,   default=20,
                   help='how many times to repeat instruction pairs in corpus')
    p.add_argument('--skip_compare', action='store_true',
                   help='skip before/after comparison')
    return p.parse_args()


# ── Before / after comparison ─────────────────────────────────────────────────

def show_comparison(base_model, sft_model, tokenizer,
                    system: str, temperature: float, max_new: int):
    """Show the same questions answered by base model vs SFT model."""
    console.print()
    console.print(Panel(
        "[bold cyan]Before vs After SFT[/bold cyan]\n"
        "Same model architecture. Same questions. Different training data.",
        border_style="cyan",
    ))
    console.print()

    questions = [
        "What is your name?",
        "Say hello.",
        "What is a language model?",
    ]

    p_base = Prompter(system)
    p_sft  = Prompter(system)

    tbl = Table(box=box.ROUNDED, show_header=True, expand=True)
    tbl.add_column("Question",     style="bold", width=28)
    tbl.add_column("Base model",   style="dim",  width=36)
    tbl.add_column("SFT model",    style="green", width=36)

    for q in questions:
        base_resp = p_base.chat(base_model, tokenizer, q,
                                max_new_tokens=max_new, temperature=temperature)
        sft_resp  = p_sft.chat(sft_model,  tokenizer, q,
                                max_new_tokens=max_new, temperature=temperature)
        tbl.add_row(q,
                    base_resp[:120] or "(no response)",
                    sft_resp[:120]  or "(no response)")

    console.print(tbl)
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = sft_config(load_config(args.config), args)

    np.random.seed(cfg['training']['seed'])

    mcfg       = cfg['model']
    block_size = mcfg['block_size']
    system     = cfg.get('sft', {}).get('system_prompt', DEFAULT_PAIRS[0].system)

    # ── Build instruction corpus and tokenizer ──────────────────────────────
    pairs  = DEFAULT_PAIRS
    corpus = build_corpus(pairs, repeat=args.repeat)

    tokenizer = CharTokenizer(corpus)
    console.print(f"\n[cyan]Instruction vocab:[/cyan] {tokenizer.vocab_size} chars")
    console.print(f"[cyan]Corpus length:[/cyan]    {len(corpus):,} chars\n")

    # ── Base model ───────────────────────────────────────────────────────────
    if args.base and Path(args.base).exists():
        console.print(f"[cyan]Loading base checkpoint:[/cyan] {args.base}")
        base_model, base_tok = load_checkpoint(args.base)
        # For fine-tuning: if vocabs match, use base weights;
        # otherwise build fresh (vocab mismatch from different corpus)
        if base_tok.vocab_size == tokenizer.vocab_size:
            console.print("[green]Vocab matches — fine-tuning from base weights[/green]")
            model = base_model
        else:
            console.print(
                f"[yellow]Vocab mismatch (base={base_tok.vocab_size}, "
                f"sft={tokenizer.vocab_size}) — training fresh[/yellow]"
            )
            model = None
    else:
        base_model = None
        model      = None

    # Build fresh model if needed
    if model is None:
        model = TransformerLM(
            vocab_size = tokenizer.vocab_size,
            embed_dim  = mcfg['embed_dim'],
            block_size = block_size,
            n_layers   = mcfg.get('n_layers', 2),
            n_heads    = mcfg.get('n_heads',  4),
            dropout    = mcfg.get('dropout', 0.0),
            seed       = cfg['training']['seed'],
        )

    # Keep a copy of initial weights for before/after comparison
    if not args.skip_compare:
        import copy
        base_for_compare = copy.deepcopy(model)
    else:
        base_for_compare = None

    # ── SFT training ─────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold cyan]Phase 8 — Supervised Fine-Tuning[/bold cyan]\n"
        f"Pairs: {len(pairs)}  ·  Repeat: {args.repeat}  ·  "
        f"Steps: {cfg['training']['epochs']}",
        border_style="cyan",
    ))

    trainer = TrainerSFT(
        model     = model,
        pairs     = pairs,
        tokenizer = tokenizer,
        cfg       = cfg,
        repeat    = args.repeat,
    )
    trainer.train()

    # ── Before / after ────────────────────────────────────────────────────────
    if base_for_compare and not args.skip_compare:
        show_comparison(
            base_model = base_for_compare,
            sft_model  = model,
            tokenizer  = tokenizer,
            system     = system,
            temperature= args.temperature,
            max_new    = args.max_new_tokens,
        )

    console.print("[bold green]SFT complete.[/bold green] "
                  "Checkpoint saved to [cyan]runs/latest_sft.npz[/cyan]\n")
    console.print("Try it:\n  [dim]python sample_cli.py "
                  "--checkpoint runs/latest_sft.npz "
                  f"--seed_text '### System:\\n{system[:30]}...'[/dim]\n")


if __name__ == '__main__':
    main()
