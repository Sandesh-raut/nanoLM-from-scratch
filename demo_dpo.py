#!/usr/bin/env python3
"""
Phase 8 — DPO (Direct Preference Optimisation) conceptual demo.

This is an educational demo, not a full training loop.
It makes DPO concrete by:

  1. Showing what a preference pair looks like (chosen vs rejected response)
  2. Computing log-probabilities for both under the current model
  3. Showing the DPO loss formula and computing it
  4. Comparing DPO to RLHF conceptually

Run
---
  python demo_dpo.py                  # use random init model
  python demo_dpo.py --checkpoint runs/latest_sft.npz

Maps to
-------
  Paper: "Direct Preference Optimization: Your Language Model is Secretly a
  Reward Model" (Rafailov et al., 2023)

  Used in: LLaMA 3, Mistral, Gemma, Claude, GPT-4 alignment pipelines.
"""

import argparse
import math
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box

console = Console()

RULE = "─" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Preference pairs
# ─────────────────────────────────────────────────────────────────────────────

PREFERENCE_PAIRS = [
    {
        "prompt":   "What is your name?",
        "chosen":   "I am nanoLM, a tiny language model built from scratch.",
        "rejected": "asdfgh jkl zxcv qwerty random noise...",
        "reason":   "Coherent, on-topic response vs gibberish",
    },
    {
        "prompt":   "Count to three.",
        "chosen":   "One, two, three.",
        "rejected": "Seven, potato, blue.",
        "reason":   "Correct sequence vs incorrect/nonsensical",
    },
    {
        "prompt":   "Say hello politely.",
        "chosen":   "Hello! Nice to meet you.",
        "rejected": "HELLO!!!!!!! YO YO YO",
        "reason":   "Polite tone vs aggressive/informal",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability computation
# ─────────────────────────────────────────────────────────────────────────────

def sequence_log_prob(model, tokenizer, text: str, block_size: int) -> float:
    """
    Compute log P(text) = sum of per-token log-probs under the model.

    This is what RLHF and DPO both measure — how likely the model thinks
    a given response is.
    """
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return 0.0

    total_log_prob = 0.0
    for i in range(1, len(ids)):
        context = ids[max(0, i - block_size):i]
        x = np.array(context, dtype=np.int32).reshape(1, -1)
        out = model.forward(x, training=False)
        logits = out[0, -1, :]   # last position

        # Numerically stable log-softmax
        logits = logits - logits.max()
        log_probs = logits - np.log(np.exp(logits).sum())
        total_log_prob += float(log_probs[ids[i]])

    return total_log_prob


# ─────────────────────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(
    log_prob_chosen:   float,
    log_prob_rejected: float,
    log_prob_ref_chosen:   float,
    log_prob_ref_rejected: float,
    beta: float = 0.1,
) -> float:
    """
    DPO loss for a single preference pair.

    L_DPO = -log σ( β · (log π(y_w|x) - log π_ref(y_w|x))
                      - β · (log π(y_l|x) - log π_ref(y_l|x)) )

    where:
      π     = current (fine-tuned) policy
      π_ref = reference policy (base model, frozen)
      y_w   = chosen (winning) response
      y_l   = rejected (losing) response
      β     = temperature controlling KL divergence strength
      σ     = sigmoid

    Intuition: push the model to assign higher probability to chosen
    responses *relative to* the reference, and lower probability to
    rejected responses *relative to* the reference.
    This is more stable than RLHF because it avoids training a reward model.
    """
    delta_chosen   = log_prob_chosen   - log_prob_ref_chosen
    delta_rejected = log_prob_rejected - log_prob_ref_rejected
    logit = beta * (delta_chosen - delta_rejected)
    loss  = -math.log(1 / (1 + math.exp(-logit)) + 1e-9)   # -log σ(logit)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────

def panel_what_is_dpo():
    console.print()
    console.print(Panel(
        "[bold cyan]What is DPO?[/bold cyan]\n\n"
        "RLHF (Reinforcement Learning from Human Feedback) trains a reward\n"
        "model on human preferences, then uses RL to maximise it. This is:\n"
        "  • Unstable (RL training is finicky)\n"
        "  • Expensive (reward model + policy model + PPO loop)\n\n"
        "DPO solves the same problem with a much simpler objective:\n"
        "  Given a preference pair (chosen, rejected), update the model\n"
        "  directly so that chosen becomes more likely [bold]relative to[/bold] a\n"
        "  frozen reference copy of the model.\n\n"
        "No reward model. No RL loop. One loss function.",
        border_style="cyan",
    ))


def panel_preference_pairs(pairs: list):
    console.print()
    console.print(Panel("[bold]1 · Preference pairs[/bold]", border_style="cyan"))

    tbl = Table(box=box.ROUNDED, show_header=True)
    tbl.add_column("Prompt",   style="bold",  width=22)
    tbl.add_column("Chosen ✓", style="green", width=28)
    tbl.add_column("Rejected ✗", style="red", width=28)
    tbl.add_column("Why",      style="dim",   width=26)

    for pair in pairs:
        tbl.add_row(
            pair["prompt"],
            pair["chosen"],
            pair["rejected"],
            pair["reason"],
        )
    console.print(tbl)
    console.print(
        "  [dim]Human annotators (or a strong AI judge) label which response\n"
        "  they prefer. This creates the (prompt, chosen, rejected) dataset.[/dim]\n"
    )


def panel_log_probs(model, tokenizer, block_size: int, pairs: list):
    console.print(Panel("[bold]2 · Log-probabilities under the model[/bold]",
                        border_style="cyan"))
    console.print(
        "  [dim]log P(response) = sum of per-token log-probs. Higher = model\n"
        "  thinks this response is more likely given the prompt.[/dim]\n"
    )

    tbl = Table(box=box.ROUNDED, show_header=True)
    tbl.add_column("Prompt",        style="bold",  width=22)
    tbl.add_column("log P(chosen)", style="green", width=16, justify="right")
    tbl.add_column("log P(rejected)", style="red", width=18, justify="right")
    tbl.add_column("Δ (chosen - rejected)", style="cyan", width=22, justify="right")

    results = []
    for pair in pairs:
        lp_w = sequence_log_prob(model, tokenizer, pair["chosen"],   block_size)
        lp_l = sequence_log_prob(model, tokenizer, pair["rejected"], block_size)
        delta = lp_w - lp_l
        results.append((lp_w, lp_l))
        sign = "[green]✓[/green]" if delta > 0 else "[red]✗[/red]"
        tbl.add_row(
            pair["prompt"],
            f"{lp_w:.2f}",
            f"{lp_l:.2f}",
            f"{delta:+.2f} {sign}",
        )

    console.print(tbl)
    console.print(
        "  [dim]A freshly initialised (or base) model may not assign higher\n"
        "  log-prob to the chosen response. DPO training fixes this.[/dim]\n"
    )
    return results


def panel_dpo_formula():
    console.print(Panel("[bold]3 · The DPO loss formula[/bold]", border_style="cyan"))

    code = (
        "# DPO loss for one preference pair\n"
        "# π  = current policy (model being trained)\n"
        "# π_ref = frozen reference policy (base model)\n\n"
        "delta_w = log_π(chosen)   - log_π_ref(chosen)    # how much π diverges on chosen\n"
        "delta_l = log_π(rejected) - log_π_ref(rejected)  # how much π diverges on rejected\n\n"
        "logit = β * (delta_w - delta_l)   # β controls KL-divergence strength\n"
        "loss  = -log(sigmoid(logit))       # push logit positive → chosen preferred\n\n"
        "# When loss decreases:\n"
        "#   π(chosen)   increases  (relative to reference)\n"
        "#   π(rejected) decreases  (relative to reference)\n"
        "# Without the reference: the model could just boost both — reference prevents that."
    )
    console.print(Syntax(code, "python", theme="monokai", line_numbers=False))
    console.print()


def panel_dpo_loss(model, tokenizer, block_size: int, pairs: list, log_prob_results: list):
    console.print(Panel("[bold]4 · DPO loss values[/bold]", border_style="cyan"))
    console.print(
        "  [dim]Using the same model as both π and π_ref (reference = current).\n"
        "  In real DPO: π_ref is the frozen base model; π is the model being trained.[/dim]\n"
    )

    beta = 0.1
    tbl  = Table(box=box.ROUNDED, show_header=True)
    tbl.add_column("Prompt",       style="bold",  width=22)
    tbl.add_column("delta_chosen", style="green", width=14, justify="right")
    tbl.add_column("delta_reject", style="red",   width=14, justify="right")
    tbl.add_column("DPO loss",     style="cyan",  width=12, justify="right")

    for i, pair in enumerate(pairs):
        lp_w, lp_l = log_prob_results[i]
        # Reference = same model (frozen at current state)
        loss = dpo_loss(lp_w, lp_l, lp_w, lp_l, beta=beta)
        # With same ref, deltas are zero so loss is -log(0.5) ≈ 0.693
        # Show conceptual deltas assuming ref was different
        tbl.add_row(
            pair["prompt"],
            f"{0.0:+.3f}",    # delta_chosen  (ref == current → 0)
            f"{0.0:+.3f}",    # delta_rejected
            f"{loss:.4f}",
        )

    console.print(tbl)
    console.print(
        "  [dim]With ref == current model, deltas are 0 → loss = -log(σ(0)) = 0.693.\n"
        "  In a real DPO run, π trains while π_ref stays frozen — deltas grow\n"
        "  as the policy shifts toward preferred responses.[/dim]\n"
    )


def panel_vs_rlhf():
    console.print(Panel("[bold]5 · DPO vs RLHF[/bold]", border_style="cyan"))

    tbl = Table(box=box.SIMPLE, show_header=True)
    tbl.add_column("Aspect",          style="bold", width=24)
    tbl.add_column("RLHF",            style="dim",  width=28)
    tbl.add_column("DPO",             style="green", width=28)

    rows = [
        ("Reward model",    "Separate model trained on preferences", "Not needed — baked into loss"),
        ("Training loop",   "PPO (reinforcement learning)",          "Supervised loss like SFT"),
        ("Stability",       "Finicky — reward hacking common",       "More stable"),
        ("Compute",         "3× model copies in memory",             "2× model copies (ref + policy)"),
        ("Used by",         "InstructGPT, early ChatGPT",            "LLaMA 3, Mistral, Gemma"),
        ("Math insight",    "Indirect via reward maximisation",       "Direct from Bradley-Terry model"),
    ]
    for row in rows:
        tbl.add_row(*row)

    console.print(tbl)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='nanoLM DPO conceptual demo')
    p.add_argument('--checkpoint', default=None,
                   help='path to .npz checkpoint (default: random init)')
    p.add_argument('--embed_dim',  type=int, default=32)
    p.add_argument('--block_size', type=int, default=32)
    p.add_argument('--n_layers',   type=int, default=1)
    p.add_argument('--n_heads',    type=int, default=2)
    p.add_argument('--beta',       type=float, default=0.1,
                   help='DPO temperature (controls KL strength)')
    return p.parse_args()


def main():
    args = parse_args()

    console.print()
    console.print(Panel(
        "[bold cyan]Phase 8 — DPO: Direct Preference Optimisation[/bold cyan]\n"
        "Conceptual demo: preference pairs → log-probs → DPO loss",
        border_style="cyan",
    ))

    # Load or build model
    if args.checkpoint:
        from sample.checkpoint import load as load_checkpoint
        model, tokenizer = load_checkpoint(args.checkpoint)
        console.print(f"\n  Loaded checkpoint: [cyan]{args.checkpoint}[/cyan]")
        block_size = model.block_size
    else:
        from data.instruct_dataset import DEFAULT_PAIRS, build_corpus
        from data.tokenizer import CharTokenizer
        from model.transformer import TransformerLM

        corpus    = build_corpus(DEFAULT_PAIRS, repeat=1)
        tokenizer = CharTokenizer(corpus)
        block_size = args.block_size

        model = TransformerLM(
            vocab_size = tokenizer.vocab_size,
            embed_dim  = args.embed_dim,
            block_size = block_size,
            n_layers   = args.n_layers,
            n_heads    = args.n_heads,
            dropout    = 0.0,
            seed       = 42,
        )
        console.print(f"\n  Using random-init model ({model.param_count()['total']:,} params)")

    # Run demo panels
    panel_what_is_dpo()
    panel_preference_pairs(PREFERENCE_PAIRS)
    log_prob_results = panel_log_probs(model, tokenizer, block_size, PREFERENCE_PAIRS)
    panel_dpo_formula()
    panel_dpo_loss(model, tokenizer, block_size, PREFERENCE_PAIRS, log_prob_results)
    panel_vs_rlhf()

    console.print(Panel(
        "[bold green]Key takeaway[/bold green]\n\n"
        "DPO turns human preferences into a supervised loss.\n"
        "No reward model. No RL. The same gradient descent\n"
        "we used in Phase 2 — just with a different loss function.",
        border_style="green",
    ))
    console.print()


if __name__ == '__main__':
    main()
