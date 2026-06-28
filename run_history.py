#!/usr/bin/env python3
"""
Run history viewer — compare past training runs side by side.

Usage
-----
  python run_history.py              # show all runs
  python run_history.py --last 5     # show last 5 runs
  python run_history.py --run 20240101_120000   # show loss curve for one run
"""
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def sparkline(values: list[float], width: int = 20) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not values or len(values) < 2:
        return "─" * width
    mn, mx = min(values), max(values)
    if mx == mn:
        return blocks[1] * width
    step    = (mx - mn) / (len(blocks) - 1)
    sampled = values[:: max(1, len(values) // width)][:width]
    return ''.join(blocks[min(int((v - mn) / step), len(blocks) - 1)] for v in sampled)


def load_runs(runs_dir: Path, last: int | None = None) -> list[dict]:
    paths = sorted(runs_dir.glob('*.json'))
    if last:
        paths = paths[-last:]
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def show_table(runs: list[dict]):
    tbl = Table(title="nanoLM Run History", box=box.SIMPLE_HEAD)
    tbl.add_column("Run ID",     style="cyan",  no_wrap=True)
    tbl.add_column("Steps",      justify="right")
    tbl.add_column("LR",         justify="right")
    tbl.add_column("Embed",      justify="right")
    tbl.add_column("Params",     justify="right", style="dim")
    tbl.add_column("Loss start", justify="right")
    tbl.add_column("Loss end",   justify="right", style="green")
    tbl.add_column("Δ",          justify="right")
    tbl.add_column("Curve",      style="dim")

    for run in runs:
        steps  = run.get('steps', [])
        cfg    = run.get('config', {})
        pcnt   = run.get('param_count', {})
        losses = [s['loss'] for s in steps]

        l0 = losses[0]  if losses else None
        l1 = losses[-1] if losses else None
        dl = (l1 - l0) if (l0 is not None and l1 is not None) else None

        tbl.add_row(
            run.get('run_id', '?'),
            str(cfg.get('training', {}).get('epochs', '?')),
            str(cfg.get('training', {}).get('lr', '?')),
            str(cfg.get('model',    {}).get('embed_dim', '?')),
            f"{pcnt.get('total', '?'):,}" if isinstance(pcnt.get('total'), int) else '?',
            f"{l0:.4f}" if l0 is not None else "?",
            f"{l1:.4f}" if l1 is not None else "?",
            (f"[green]{dl:+.4f}[/green]" if dl is not None and dl < 0
             else f"[red]{dl:+.4f}[/red]" if dl is not None else "?"),
            sparkline(losses),
        )

    console.print()
    console.print(tbl)


def show_run(run: dict):
    steps  = run.get('steps', [])
    losses = [s['loss'] for s in steps]
    console.print(f"\n[cyan]Run:[/cyan] {run['run_id']}")
    console.print(f"[dim]Config:[/dim] {run.get('config', {}).get('training', {})}")
    console.print(f"[dim]Params:[/dim] {run.get('param_count', {})}")
    console.print(f"\nLoss curve:\n  {sparkline(losses, width=60)}")
    if steps:
        console.print(f"\n  Steps: {len(steps)}  |  {losses[0]:.4f} → {losses[-1]:.4f}")
    # Last sample
    for s in reversed(steps):
        if 'sample' in s:
            console.print(f"\nLast sample (step {s['step']}):\n  {s['sample']!r}")
            break
    console.print()


def main():
    p = argparse.ArgumentParser(description='nanoLM run history')
    p.add_argument('--last', type=int, default=None, help='show last N runs')
    p.add_argument('--run',  type=str, default=None, help='show detail for run_id')
    args = p.parse_args()

    runs_dir = Path('runs')
    if not runs_dir.exists() or not list(runs_dir.glob('*.json')):
        console.print("[yellow]No runs found. Train first with: python main.py[/yellow]")
        return

    runs = load_runs(runs_dir, last=args.last)

    if args.run:
        matched = [r for r in runs if args.run in r.get('run_id', '')]
        if not matched:
            console.print(f"[red]No run matching '{args.run}'[/red]")
        for r in matched:
            show_run(r)
    else:
        show_table(runs)


if __name__ == '__main__':
    main()
