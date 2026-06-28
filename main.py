#!/usr/bin/env python3
"""
nanoLM — entry point.

Usage
-----
  python main.py                              # use config.yaml defaults
  python main.py --epochs 500 --lr 0.1       # CLI overrides
  python main.py --corpus data/corpus.txt     # different corpus
  python main.py --embed_dim 128 --block_size 64

Every run is saved to runs/<timestamp>.json.
Compare runs with:  python run_history.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from data.tokenizer import CharTokenizer
from data.loader import BatchLoader
from model.bigram import BigramModel
from train.trainer import Trainer


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI flags over config dict — CLI always wins."""
    overrides = {
        'model.block_size':        args.block_size,
        'model.embed_dim':         args.embed_dim,
        'training.lr':             args.lr,
        'training.batch_size':     args.batch_size,
        'training.epochs':         args.epochs,
        'training.seed':           args.seed,
        'sampling.temperature':    args.temperature,
        'sampling.max_new_tokens': args.max_new_tokens,
        'data.corpus_path':        args.corpus,
        'model.type':              args.model,
        'model.n_layers':          args.n_layers,
        'model.n_heads':           args.n_heads,
        'model.dropout':           args.dropout,
        'training.optimizer':      args.optimizer,
        'training.grad_clip':      args.grad_clip,
        'training.warmup_steps':   args.warmup_steps,
        'training.val_split':      args.val_split,
    }
    for dotkey, value in overrides.items():
        if value is None:
            continue
        keys = dotkey.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='nanoLM — tiny language model from scratch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',          default='config.yaml',  help='path to config file')
    p.add_argument('--block_size',      type=int,   default=None, help='context window length')
    p.add_argument('--embed_dim',       type=int,   default=None, help='embedding dimension')
    p.add_argument('--lr',              type=float, default=None, help='learning rate')
    p.add_argument('--batch_size',      type=int,   default=None, help='batch size')
    p.add_argument('--epochs',          type=int,   default=None, help='number of gradient steps')
    p.add_argument('--seed',            type=int,   default=None, help='random seed')
    p.add_argument('--temperature',     type=float, default=None, help='sampling temperature')
    p.add_argument('--max_new_tokens',  type=int,   default=None, help='tokens to generate per sample')
    p.add_argument('--corpus',          type=str,   default=None, help='path to corpus text file')
    p.add_argument('--model',           type=str,   default=None, help='model type: bigram | transformer')
    p.add_argument('--n_layers',        type=int,   default=None, help='number of transformer blocks (Phase 4+)')
    p.add_argument('--n_heads',         type=int,   default=None, help='number of attention heads (Phase 4+)')
    p.add_argument('--dropout',         type=float, default=None, help='dropout rate (Phase 4+)')
    p.add_argument('--optimizer',       type=str,   default=None, help='optimizer: sgd | adamw (Phase 5+)')
    p.add_argument('--grad_clip',       type=float, default=None, help='gradient norm clip threshold (Phase 5+)')
    p.add_argument('--warmup_steps',    type=int,   default=None, help='LR warmup steps (Phase 5+)')
    p.add_argument('--val_split',       type=float, default=None, help='validation fraction 0-1 (Phase 5+)')
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = apply_cli(load_config(args.config), args)

    # ── Reproducibility ──
    seed = cfg['training']['seed']
    np.random.seed(seed)

    # ── Data ──
    corpus_path = Path(cfg['data']['corpus_path'])
    if not corpus_path.exists():
        sys.exit(f"[error] Corpus not found: {corpus_path}")

    text      = corpus_path.read_text()
    tokenizer = CharTokenizer(text)
    data      = np.array(tokenizer.encode(text), dtype=np.int32)

    block_size = cfg['model']['block_size']
    batch_size = cfg['training']['batch_size']

    if len(data) <= block_size:
        sys.exit(
            f"[error] Corpus has only {len(data)} tokens but block_size={block_size}. "
            "Use a larger corpus or reduce block_size."
        )

    loader = BatchLoader(data, block_size, batch_size, seed=seed)

    # ── Model ──
    model_type = cfg['model'].get('type', 'transformer')
    if model_type == 'bigram':
        model = BigramModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=cfg['model']['embed_dim'],
            seed=seed,
        )
    else:
        from model.transformer import TransformerLM
        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            embed_dim=cfg['model']['embed_dim'],
            block_size=block_size,
            n_layers=cfg['model'].get('n_layers', 1),
            n_heads=cfg['model'].get('n_heads', 1),
            dropout=cfg['model'].get('dropout', 0.0),
            seed=seed,
        )

    # ── Train ──
    trainer = Trainer(model, loader, tokenizer, cfg)
    trainer.train()


if __name__ == '__main__':
    main()
