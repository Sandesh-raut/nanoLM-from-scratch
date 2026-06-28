"""
Terminal dashboard using Rich.

Displays during training:
  - Run header:  config summary + parameter breakdown by layer
  - Step rows:   step | loss | tracked weight E[0,0] | loss bar
  - Sample panels: generated text at sample_every intervals
  - Footer:      timing + loss delta

Maps to: the role Weights & Biases plays in real labs, but terminal-native.
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class Dashboard:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.loss_history: list[float] = []

    # ── Header ───────────────────────────────────────────────────────────────

    def print_header(self, model, tokenizer):
        counts = model.param_count()

        # Config summary panel — reads description from model
        tcfg = self.cfg['training']
        mcfg = self.cfg['model']
        desc = getattr(model, 'description', 'nanoLM')
        console.print()
        console.print(Panel(
            f"[bold cyan]nanoLM[/bold cyan]  ·  {desc}\n\n"
            f"  vocab={tokenizer.vocab_size}  "
            f"block_size={mcfg['block_size']}  "
            f"embed_dim={mcfg['embed_dim']}  "
            f"lr={tcfg['lr']}  "
            f"batch={tcfg['batch_size']}  "
            f"steps={tcfg['epochs']}",
            title="[bold]Run Config[/bold]",
            border_style="cyan",
        ))

        # Parameter breakdown — generic, reads from model.param_table()
        tbl = Table(title="Parameter Breakdown", box=box.SIMPLE, show_header=True)
        tbl.add_column("Layer",  style="cyan")
        tbl.add_column("Shape",  style="dim")
        tbl.add_column("Params", style="green", justify="right")
        for name, shape, n in model.param_table():
            tbl.add_row(name, shape, f"{n:,}")
        tbl.add_section()
        tbl.add_row("[bold]TOTAL[/bold]", "", f"[bold green]{counts['total']:,}[/bold green]")
        console.print(tbl)

        # Column header for step rows
        console.print()
        console.print(
            f"  {'step':>5}  {'train':>8}  {'val':>8}  {'ppl':>7}  {'lr':>9}  {'‖g‖':>6}  {'E[0,0]':>10}  loss",
            style="bold dim",
        )
        console.print("  " + "─" * 78)

    # ── Step row ─────────────────────────────────────────────────────────────

    def print_step(
        self,
        step:      int,
        total:     int,
        loss:      float,
        tracked:   float,
        lr:        float = 0.0,
        val_loss:  float | None = None,
        grad_norm: float = 0.0,
    ):
        import math
        self.loss_history.append(loss)
        bar      = self._loss_bar(loss)
        color    = "green" if loss < 2.0 else "yellow" if loss < 3.0 else "red"
        ppl      = math.exp(min(loss, 20))
        val_str  = f"{val_loss:>8.4f}" if val_loss is not None else f"{'—':>8}"
        lr_str   = f"{lr:.2e}" if lr > 0 else f"{'—':>9}"
        norm_str = f"{grad_norm:>6.2f}" if grad_norm > 0 else f"{'—':>6}"
        console.print(
            f"  {step:>5}  [{color}]{loss:>8.4f}[/{color}]  "
            f"[dim]{val_str}[/dim]  "
            f"[cyan]{ppl:>7.2f}[/cyan]  "
            f"[dim]{lr_str:>9}[/dim]  "
            f"[dim]{norm_str}[/dim]  "
            f"{tracked:>+10.6f}  {bar}"
        )

    # ── Sample panel ─────────────────────────────────────────────────────────

    def print_sample(self, step: int, sample: str):
        console.print()
        console.print(Panel(
            Text(sample, style="italic"),
            title=f"[yellow]⟩ sample @ step {step}[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        ))
        console.print()

    # ── Footer ───────────────────────────────────────────────────────────────

    def print_footer(self, elapsed: float, run_path: str = ""):
        if len(self.loss_history) >= 2:
            spark = self._sparkline(self.loss_history)
            delta = self.loss_history[-1] - self.loss_history[0]
            delta_str = f"[green]{delta:+.4f}[/green]" if delta < 0 else f"[red]{delta:+.4f}[/red]"
        else:
            spark, delta_str = "", ""

        body = (
            f"  Elapsed   : [cyan]{elapsed:.1f}s[/cyan]\n"
            f"  Loss curve: {spark}\n"
            f"  Δ loss     : {delta_str}  "
            f"({self.loss_history[0]:.4f} → {self.loss_history[-1]:.4f})"
        )
        if run_path:
            body += f"\n  Run saved : [dim]{run_path}[/dim]"

        console.print()
        console.print(Panel(body, title="[bold green]Training complete[/bold green]",
                            border_style="green"))
        console.print()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _loss_bar(self, loss: float, max_loss: float = 4.0, width: int = 24) -> str:
        pct    = min(loss / max_loss, 1.0)
        filled = int(pct * width)
        bar    = "█" * filled + "░" * (width - filled)
        color  = "green" if loss < 1.5 else "yellow" if loss < 2.5 else "red"
        return f"[{color}]{bar}[/{color}]"

    def _sparkline(self, values: list[float], width: int = 30) -> str:
        """ASCII sparkline of the loss curve."""
        blocks = " ▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        if mx == mn:
            return blocks[1] * width
        step    = (mx - mn) / (len(blocks) - 1)
        sampled = values[:: max(1, len(values) // width)][:width]
        return ''.join(blocks[min(int((v - mn) / step), len(blocks) - 1)] for v in sampled)
