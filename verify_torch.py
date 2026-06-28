#!/usr/bin/env python3
"""
Phase 7 — Verify that the PyTorch mirror matches the NumPy model exactly.

Three checks:
  1. Param count — both models must have identical totals.
  2. Forward pass — transfer NumPy weights → PyTorch, run same input,
     compare logits.  Max absolute difference should be < 1e-5 (float32 limit).
  3. Speed — time 200 forward passes on NumPy, PyTorch-CPU, and PyTorch-MPS.

Usage
-----
  python verify_torch.py
  python verify_torch.py --n_layers 4 --n_heads 8 --embed_dim 128
"""

import argparse
import time

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from model.transformer import TransformerLM
from model.transformer_torch import TransformerLMTorch, transfer_weights

console = Console()

RULE = "─" * 56


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Param count comparison
# ─────────────────────────────────────────────────────────────────────────────

def check_param_count(np_model, pt_model) -> bool:
    np_total = np_model.param_count()['total']
    pt_total = pt_model.param_count()['total']

    match = np_total == pt_total
    icon  = "[green]✓ match[/green]" if match else "[red]✗ mismatch[/red]"

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Label", style="dim",        width=20)
    tbl.add_column("Value", style="bold cyan",  width=16)
    tbl.add_column("",      width=16)

    tbl.add_row("NumPy param count",   f"{np_total:,}", "")
    tbl.add_row("PyTorch param count", f"{pt_total:,}", icon)

    console.print(Panel(tbl, title="[bold]1 · Param count[/bold]", border_style="cyan"))
    return match


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forward pass comparison
# ─────────────────────────────────────────────────────────────────────────────

def check_forward(np_model, pt_model, vocab_size, block_size, batch=4) -> bool:
    rng = np.random.default_rng(99)
    x_np = rng.integers(0, vocab_size, (batch, block_size), dtype=np.int32)

    # NumPy forward (inference mode, dropout off)
    logits_np = np_model.forward(x_np, training=False)           # (B, T, V) float64

    # PyTorch forward
    pt_model.eval()
    x_pt = torch.tensor(x_np, dtype=torch.long)
    with torch.no_grad():
        logits_pt = pt_model(x_pt).cpu().numpy().astype(np.float64)  # (B, T, V)

    max_diff  = float(np.abs(logits_np - logits_pt).max())
    mean_diff = float(np.abs(logits_np - logits_pt).mean())
    match     = max_diff < 1e-4   # float32 transfer → some rounding expected

    icon = "[green]✓ match[/green]" if match else "[red]✗ mismatch[/red]"

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Label",  style="dim",       width=26)
    tbl.add_column("Value",  style="bold cyan", width=18)
    tbl.add_column("",       width=14)

    tbl.add_row("Input shape",    f"({batch}, {block_size})", "")
    tbl.add_row("Max  |Δ|",       f"{max_diff:.2e}",  icon)
    tbl.add_row("Mean |Δ|",       f"{mean_diff:.2e}", "")
    tbl.add_row("Threshold",      "< 1e-4",           "(float32 rounding)")

    console.print(Panel(tbl, title="[bold]2 · Forward pass comparison[/bold]", border_style="cyan"))
    return match


# ─────────────────────────────────────────────────────────────────────────────
# 3. Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(np_model, pt_model_cpu, vocab_size, block_size, batch=4, n_runs=200):
    import copy
    device = pick_device()
    rng    = np.random.default_rng(7)
    x_np   = rng.integers(0, vocab_size, (batch, block_size), dtype=np.int32)
    x_cpu  = torch.tensor(x_np, dtype=torch.long)
    x_dev  = x_cpu.to(device)

    # Deep-copy so the CPU benchmark model and device model are independent.
    # nn.Module.to() is in-place — without a copy, moving to MPS would leave
    # pt_model_cpu with parameters on MPS, causing failures with CPU tensors.
    pt_model_cpu = copy.deepcopy(pt_model_cpu)
    pt_model_cpu.eval()
    pt_model_dev = copy.deepcopy(pt_model_cpu).to(device)
    pt_model_dev.eval()

    # ── NumPy ──────────────────────────────────────────────────────────────
    # warm-up
    for _ in range(5):
        np_model.forward(x_np, training=False)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        np_model.forward(x_np, training=False)
    np_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ── PyTorch CPU ────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(5):
            pt_model_cpu(x_cpu)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            pt_model_cpu(x_cpu)
    pt_cpu_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ── PyTorch device (MPS / CUDA / CPU fallback) ─────────────────────────
    if device.type != 'cpu':
        with torch.no_grad():
            for _ in range(10):   # MPS warm-up is slower
                pt_model_dev(x_dev)
            if device.type == 'mps':
                torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                pt_model_dev(x_dev)
                if device.type == 'mps':
                    torch.mps.synchronize()
        pt_dev_ms = (time.perf_counter() - t0) / n_runs * 1000
    else:
        pt_dev_ms = pt_cpu_ms   # same device

    # ── Table ──────────────────────────────────────────────────────────────
    def speedup(baseline, candidate):
        if candidate > 0:
            return f"×{baseline/candidate:.1f}"
        return "—"

    tbl = Table(box=box.ROUNDED, show_header=True)
    tbl.add_column("Backend",        style="bold cyan", width=22)
    tbl.add_column("ms / step",      style="bold",      width=12, justify="right")
    tbl.add_column("vs NumPy",       style="green",     width=12, justify="right")

    tbl.add_row("NumPy (CPU)",           f"{np_ms:.2f}",     "baseline")
    tbl.add_row("PyTorch CPU",           f"{pt_cpu_ms:.2f}", speedup(np_ms, pt_cpu_ms))
    tbl.add_row(f"PyTorch {device.type.upper()}",
                f"{pt_dev_ms:.2f}", speedup(np_ms, pt_dev_ms))

    console.print(Panel(
        tbl,
        title=f"[bold]3 · Speed ({n_runs} forward passes, batch={batch}, seq={block_size})[/bold]",
        border_style="cyan",
    ))
    console.print(f"  Device: [bold]{device}[/bold]  ({_device_name(device)})\n")


def _device_name(device: torch.device) -> str:
    if device.type == 'mps':
        return "Apple Silicon (MPS)"
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    return "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Verify NumPy ↔ PyTorch parity")
    p.add_argument('--config',     default='config.yaml')
    p.add_argument('--vocab_size', type=int,   default=None)
    p.add_argument('--embed_dim',  type=int,   default=None)
    p.add_argument('--block_size', type=int,   default=None)
    p.add_argument('--n_layers',   type=int,   default=None)
    p.add_argument('--n_heads',    type=int,   default=None)
    p.add_argument('--n_runs',     type=int,   default=200)
    p.add_argument('--batch',      type=int,   default=4)
    p.add_argument('--skip_bench', action='store_true', help='skip speed benchmark')
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}

    mcfg       = cfg.get('model', {})
    vocab_size = args.vocab_size or 24          # default corpus vocab
    embed_dim  = args.embed_dim  or int(mcfg.get('embed_dim',  64))
    block_size = args.block_size or int(mcfg.get('block_size', 32))
    n_layers   = args.n_layers   or int(mcfg.get('n_layers',    2))
    n_heads    = args.n_heads    or int(mcfg.get('n_heads',     4))

    console.print()
    console.print(Panel(
        f"[bold cyan]Phase 7 — NumPy ↔ PyTorch Verification[/bold cyan]\n"
        f"vocab={vocab_size}  embed={embed_dim}  block={block_size}  "
        f"layers={n_layers}  heads={n_heads}",
        border_style="cyan",
    ))
    console.print()

    # Build NumPy model
    np_model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        seed=42,
    )

    # Build PyTorch model and transfer weights
    pt_model = TransformerLMTorch(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
    )
    console.print(f"  Transferring {n_layers * 12 + 5} parameter tensors from NumPy → PyTorch...")
    transfer_weights(np_model, pt_model)
    console.print("  Done.\n")

    # Run checks
    ok_count  = check_param_count(np_model, pt_model)
    console.print()
    ok_forward = check_forward(np_model, pt_model, vocab_size, block_size, args.batch)
    console.print()

    if not args.skip_bench:
        benchmark(np_model, pt_model, vocab_size, block_size, args.batch, args.n_runs)

    # Summary
    all_ok = ok_count and ok_forward
    style  = "green" if all_ok else "red"
    msg    = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"
    console.print(Panel(f"[bold {style}]{msg}[/bold {style}]", border_style=style))
    console.print()


if __name__ == '__main__':
    main()
