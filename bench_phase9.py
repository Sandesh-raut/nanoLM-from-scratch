#!/usr/bin/env python3
"""
Phase 9 — Modern Upgrades Benchmark.

Runs three independent sections:

  Section 1: Training comparison
    Train a ModernTransformerLM under 6 configurations (baseline + one
    upgrade at a time + all together) for N steps on the same corpus.
    Report: final loss, ms/step, total param count.

  Section 2: KV-cache inference speedup
    Generate max_new_tokens characters with and without KV cache.
    Report: ms total, ms/token, speedup ratio.

  Section 3: Quantization compression
    Quantize a trained model to int8 and report size reduction and error.

Run
---
  python bench_phase9.py
  python bench_phase9.py --steps 300 --max_new_tokens 50

Each section is independent — you can read them out of order.
"""

import argparse
import math
import time
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table   import Table
from rich         import box

sys.path.insert(0, str(Path(__file__).parent))

from data.tokenizer         import CharTokenizer
from data.loader            import BatchLoader
from model.modern_transformer import ModernTransformerLM
from model.kv_cache         import generate_no_cache, generate_cached, benchmark_kv_cache
from model.quantize         import compression_stats, summary as quant_summary
from train.optimizer        import build_optimizer
from train.scheduler        import build_scheduler

console = Console()

# ── Tiny training corpus (same as Phase 2 default) ────────────────────────────
CORPUS = (
    "alice\nbob\ncharlie\ndiana\neve\nfrank\ngrace\nhector\niris\njack\n"
    "kate\nliam\nmia\nnolan\nolive\npeter\nquinn\nrosa\nsam\ntara\n"
    "uma\nvictor\nwendy\nxander\nyara\nzoe\n"
) * 20


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Training comparison
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS = [
    ('Baseline',          dict(norm='layernorm', ffn='relu',   pos_enc='learned', n_kv_heads=4)),
    ('+ RMSNorm',         dict(norm='rmsnorm',   ffn='relu',   pos_enc='learned', n_kv_heads=4)),
    ('+ SwiGLU',          dict(norm='layernorm', ffn='swiglu', pos_enc='learned', n_kv_heads=4)),
    ('+ RoPE',            dict(norm='layernorm', ffn='relu',   pos_enc='rope',    n_kv_heads=4)),
    ('+ GQA (kv=1)',      dict(norm='layernorm', ffn='relu',   pos_enc='learned', n_kv_heads=1)),
    ('All upgrades',      dict(norm='rmsnorm',   ffn='swiglu', pos_enc='rope',    n_kv_heads=1)),
]


def _make_cfg(steps: int) -> dict:
    return {
        'model': {'embed_dim': 64, 'block_size': 32, 'n_layers': 2, 'n_heads': 4},
        'training': {
            'optimizer': 'adamw', 'lr': 3e-4, 'min_lr': 1e-5,
            'warmup_steps': 20, 'beta1': 0.9, 'beta2': 0.999,
            'weight_decay': 0.01, 'grad_clip': 1.0,
            'batch_size': 8, 'epochs': steps, 'seed': 42, 'val_split': 0.0,
        },
    }


def _train_steps(label: str, kwargs: dict, n_steps: int, tokenizer: CharTokenizer,
                 data: np.ndarray) -> dict:
    cfg   = _make_cfg(n_steps)
    mcfg  = cfg['model']
    model = ModernTransformerLM(
        vocab_size  = tokenizer.vocab_size,
        embed_dim   = mcfg['embed_dim'],
        block_size  = mcfg['block_size'],
        n_layers    = mcfg['n_layers'],
        n_heads     = mcfg['n_heads'],
        dropout     = 0.0,
        seed        = 42,
        **kwargs,
    )
    opt   = build_optimizer(cfg)
    sched = build_scheduler(opt, cfg, n_steps)

    loader = BatchLoader(data, mcfg['block_size'], cfg['training']['batch_size'], seed=42)

    losses, step_times = [], []
    for step in range(n_steps):
        if sched:
            sched.update(step)
        x, y = loader.next_batch()

        t0 = time.perf_counter()
        loss, grads = model.loss_and_grads(x, y)
        opt.step(model, grads)
        step_times.append((time.perf_counter() - t0) * 1000)

        losses.append(loss)

    final_loss   = float(np.mean(losses[-20:]))
    ms_per_step  = float(np.median(step_times))
    params       = model.param_count()['total']

    return {
        'label':       label,
        'final_loss':  final_loss,
        'ppl':         math.exp(min(final_loss, 20)),
        'ms_per_step': ms_per_step,
        'params':      params,
        'model':       model,
    }


def section_training(n_steps: int):
    console.print()
    console.print(f"[bold cyan]Section 1 — Training comparison[/bold cyan]  "
                  f"({n_steps} steps, same corpus & seed per config)")

    tok    = CharTokenizer(CORPUS)
    data   = np.array(tok.encode(CORPUS), dtype=np.int32)
    results = []

    for label, kwargs in BENCH_CONFIGS:
        console.print(f"  Training [{label}]...", end='')
        r = _train_steps(label, kwargs, n_steps, tok, data)
        results.append(r)
        console.print(f"  loss={r['final_loss']:.4f}  {r['ms_per_step']:.1f}ms/step")

    tbl = Table(box=box.ROUNDED, show_header=True, title="Training comparison")
    tbl.add_column("Config",         style="bold",  width=20)
    tbl.add_column("Params",         style="cyan",  justify="right", width=10)
    tbl.add_column("Final loss",     style="green", justify="right", width=12)
    tbl.add_column("Perplexity",     style="green", justify="right", width=12)
    tbl.add_column("ms / step",      style="yellow",justify="right", width=12)

    baseline_loss = results[0]['final_loss']
    for r in results:
        delta = r['final_loss'] - baseline_loss
        delta_str = f" ({delta:+.4f})" if r['label'] != 'Baseline' else ""
        tbl.add_row(
            r['label'],
            f"{r['params']:,}",
            f"{r['final_loss']:.4f}{delta_str}",
            f"{r['ppl']:.2f}",
            f"{r['ms_per_step']:.1f}",
        )

    console.print(tbl)
    console.print(
        "  [dim]Each upgrade is tested in isolation vs the baseline (except 'All upgrades').\n"
        "  Loss difference of ±0.05 may be noise at this data scale.[/dim]\n"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — KV-cache speedup
# ─────────────────────────────────────────────────────────────────────────────

def section_kv_cache(model, tokenizer, max_new_tokens: int):
    console.print(f"[bold cyan]Section 2 — KV-cache speedup[/bold cyan]  "
                  f"(generating {max_new_tokens} tokens)")

    prompt = "alice"
    tokens = tokenizer.encode(prompt)

    console.print("  Without cache...", end='')
    nc_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        generate_no_cache(model, tokens, max_new_tokens)
        nc_times.append((time.perf_counter() - t0) * 1000)
    nc_ms = float(np.median(nc_times))
    console.print(f"  {nc_ms:.1f}ms total  ({nc_ms/max_new_tokens:.2f}ms/token)")

    console.print("  With cache...",    end='')
    c_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        generate_cached(model, tokens, max_new_tokens)
        c_times.append((time.perf_counter() - t0) * 1000)
    c_ms = float(np.median(c_times))
    console.print(f"  {c_ms:.1f}ms total  ({c_ms/max_new_tokens:.2f}ms/token)")

    speedup = nc_ms / max(c_ms, 0.01)

    tbl = Table(box=box.ROUNDED, show_header=True, title="KV-cache benchmark")
    tbl.add_column("Mode",          style="bold",   width=20)
    tbl.add_column("Total ms",      style="cyan",   justify="right", width=12)
    tbl.add_column("ms / token",    style="cyan",   justify="right", width=12)
    tbl.add_column("Speedup",       style="green",  justify="right", width=10)

    tbl.add_row("No cache",  f"{nc_ms:.1f}",   f"{nc_ms/max_new_tokens:.2f}", "1.0×")
    tbl.add_row("KV cache",  f"{c_ms:.1f}",    f"{c_ms/max_new_tokens:.2f}",  f"{speedup:.1f}×")

    console.print(tbl)
    console.print(
        f"  [dim]Without cache: O(T²) attention per step × T steps = O(T³) total.\n"
        f"  With cache: O(1) new token projection + O(T) attention per step = O(T²) total.\n"
        f"  Speedup grows quadratically with context length.\n"
        f"  At block_size=32 the advantage is {speedup:.1f}×; at block_size=4096 it is dramatic.[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Quantization
# ─────────────────────────────────────────────────────────────────────────────

def section_quantization(model):
    console.print("[bold cyan]Section 3 — int8 quantization[/bold cyan]")

    stats = compression_stats(model)
    f64   = stats['float64_bytes']
    i8    = stats['int8_bytes']
    ratio = stats['compression_ratio']
    err   = stats['avg_rms_error']

    def _fmt(b: int) -> str:
        if b >= 1_000_000: return f"{b/1_000_000:.1f} MB"
        if b >= 1_000:     return f"{b/1_000:.1f} KB"
        return f"{b} B"

    tbl = Table(box=box.ROUNDED, show_header=True, title="int8 quantization")
    tbl.add_column("Format",      style="bold",  width=22)
    tbl.add_column("Size",        style="cyan",  justify="right", width=12)
    tbl.add_column("Compression", style="green", justify="right", width=14)
    tbl.add_column("Avg RMS err", style="yellow",justify="right", width=14)

    tbl.add_row("float64 (current)",  _fmt(f64), "1.0×",    "—")
    tbl.add_row("int8 (quantized)", _fmt(i8),  f"{ratio:.1f}×", f"{err:.2e}")

    console.print(tbl)
    console.print(
        "  [dim]2-D weight matrices quantized to int8 (1 byte/element + float64 scale).\n"
        "  1-D arrays (embeddings, norm scales) kept at float64.\n"
        "  The RMS error per element is small because scale = max(|W|) / 127.\n"
        "  Production: bitsandbytes int8, GPTQ, AWQ, NF4 achieve better quality-per-bit.[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Phase 9 — Modern upgrades benchmark')
    p.add_argument('--steps',          type=int, default=300,
                   help='training steps per config (default 300)')
    p.add_argument('--max_new_tokens', type=int, default=30,
                   help='tokens to generate for KV-cache benchmark (default 30)')
    return p.parse_args()


def main():
    args = parse_args()

    console.print()
    console.print("[bold]Phase 9 — Modern Upgrades Benchmark[/bold]")
    console.print("Six upgrades: RMSNorm · SwiGLU · RoPE · GQA · KV-cache · int8 quantization\n")

    # Section 1 — training
    results = section_training(args.steps)

    # Use the 'All upgrades' model for Sections 2 and 3
    all_model = results[-1]['model']
    tok       = CharTokenizer(CORPUS)

    # Section 2 — KV cache
    section_kv_cache(all_model, tok, args.max_new_tokens)

    # Section 3 — quantization
    section_quantization(all_model)

    console.print("[bold green]Phase 9 benchmark complete.[/bold green]\n")
    console.print(
        "Key takeaways:\n"
        "  • RMSNorm, SwiGLU, RoPE give small but consistent loss improvements\n"
        "  • GQA reduces KV parameter count with minimal loss degradation\n"
        "  • KV-cache speedup grows with context length (quadratic scaling)\n"
        "  • int8 quantization cuts weight storage by ~8× with small error\n"
        "  • Together, these six upgrades are the gap between GPT-2 (2019) and LLaMA 3 (2024)\n"
    )


if __name__ == '__main__':
    main()
