#!/usr/bin/env python3
"""
Phase 7 — PyTorch training entry point.

Mirrors main.py but uses TransformerLMTorch + TrainerTorch.
Use verify_torch.py first to confirm the model matches the NumPy version.

Usage
-----
  python main_torch.py                        # default config.yaml
  python main_torch.py --epochs 500
  python main_torch.py --device mps          # force MPS
  python main_torch.py --device cpu          # force CPU (for comparison)
  python main_torch.py --n_layers 4 --n_heads 8 --embed_dim 128

Then compare to the NumPy run:
  python main.py --epochs 500        # NumPy baseline
  python main_torch.py --epochs 500  # PyTorch mirror

Both write to runs/ with timestamps. Loss curves should be similar
(different because of different random init, but converge at same rate).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from data.loader import BatchLoader
from data.tokenizer import CharTokenizer
from model.transformer_torch import TransformerLMTorch
from train.trainer_torch import TrainerTorch


def parse_args():
    p = argparse.ArgumentParser(
        description='nanoLM — PyTorch training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config
    p.add_argument('--config',       default='config.yaml')

    # Model
    p.add_argument('--block_size',   type=int,   default=None)
    p.add_argument('--embed_dim',    type=int,   default=None)
    p.add_argument('--n_layers',     type=int,   default=None)
    p.add_argument('--n_heads',      type=int,   default=None)
    p.add_argument('--dropout',      type=float, default=None)

    # Training
    p.add_argument('--epochs',       type=int,   default=None)
    p.add_argument('--batch_size',   type=int,   default=None)
    p.add_argument('--lr',           type=float, default=None)
    p.add_argument('--seed',         type=int,   default=None)
    p.add_argument('--device',       default=None,
                   choices=['auto', 'cpu', 'mps', 'cuda'],
                   help='compute device (default: auto-select)')

    # Data
    p.add_argument('--corpus_path',  default=None)

    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    """Merge CLI flags into config, CLI wins."""
    m = cfg.setdefault('model', {})
    t = cfg.setdefault('training', {})
    d = cfg.setdefault('data', {})

    if args.block_size:   m['block_size']  = args.block_size
    if args.embed_dim:    m['embed_dim']   = args.embed_dim
    if args.n_layers:     m['n_layers']    = args.n_layers
    if args.n_heads:      m['n_heads']     = args.n_heads
    if args.dropout is not None: m['dropout'] = args.dropout

    if args.epochs:       t['epochs']      = args.epochs
    if args.batch_size:   t['batch_size']  = args.batch_size
    if args.lr:           t['lr']          = args.lr
    if args.seed:         t['seed']        = args.seed
    if args.device:       t['device']      = args.device

    if args.corpus_path:  d['corpus_path'] = args.corpus_path

    return cfg


def main():
    args = parse_args()

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = apply_overrides(cfg, args)

    mcfg = cfg['model']
    tcfg = cfg['training']
    dcfg = cfg.get('data', {})

    seed = int(tcfg.get('seed', 42))
    np.random.seed(seed)

    # Load corpus + tokenizer
    corpus_path = Path(dcfg.get('corpus_path', 'data/corpus.txt'))
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    text      = corpus_path.read_text()
    tokenizer = CharTokenizer(text)

    block_size = int(mcfg.get('block_size', 32))
    batch_size = int(tcfg.get('batch_size', 16))

    loader = BatchLoader(
        text       = text,
        tokenizer  = tokenizer,
        block_size = block_size,
        batch_size = batch_size,
        seed       = seed,
    )

    # Build model
    model = TransformerLMTorch(
        vocab_size = tokenizer.vocab_size,
        embed_dim  = int(mcfg.get('embed_dim',  64)),
        block_size = block_size,
        n_layers   = int(mcfg.get('n_layers',    2)),
        n_heads    = int(mcfg.get('n_heads',     4)),
        dropout    = float(mcfg.get('dropout', 0.0)),
    )

    # Train
    trainer = TrainerTorch(model, loader, tokenizer, cfg)
    run_path = trainer.train()

    return run_path


if __name__ == '__main__':
    main()
