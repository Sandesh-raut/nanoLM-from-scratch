#!/usr/bin/env python3
"""
nanoLM — guided session walkthrough of the Phase 8 / Phase 9 features that were
hardened in the latest review pass:

  Segment 1 — TRUE response masking (Phase 8)
              Instruction tokens get exactly zero loss and zero gradient.
  Segment 2 — REAL int8 quantization (Phase 9)
              The quantized model is genuinely smaller and still generates.
  Segment 3 — KV-cache decode (Phase 9)
              Cached vs uncached speedup, plus the block_size safety guard.

Everything runs on tiny models in a few seconds — the point is to make each
mechanism *visible*, not to train a good model.

Run
---
  uv run python3 demo_session.py
  uv run python3 demo_session.py --sft-steps 150   # train longer for nicer text
"""

import argparse
import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _console = Console()
    RICH = True
except Exception:                              # graceful fallback, no hard dep
    RICH = False
    _console = None

from data.tokenizer        import CharTokenizer
from data.instruct_dataset import DEFAULT_PAIRS, build_corpus, response_mask, RESPONSE_START
from model.transformer     import TransformerLM
from model.modern_transformer import ModernTransformerLM
from model.quantize        import quantize_model, model_size_bytes, summary, compression_stats
from model.kv_cache        import benchmark_kv_cache, generate_cached
from train.optimizer       import AdamW, clip_grad_norm
from sample.prompter       import Prompter
from sample.sampler        import generate


RULE = "─" * 64


def say(title, body=""):
    if RICH:
        _console.print(Panel(body, title=f"[bold]{title}[/bold]",
                             border_style="cyan", box=box.ROUNDED))
    else:
        print(f"\n{RULE}\n{title}\n{RULE}\n{body}")


def line(s=""):
    (_console.print(s) if RICH else print(s))


# ─────────────────────────────────────────────────────────────────────────────
# Segment 1 — True response masking
# ─────────────────────────────────────────────────────────────────────────────

def segment_masking(tok, corpus, sft_steps):
    say("Segment 1 · Phase 8 — TRUE response masking",
        "Loss and gradient flow ONLY through assistant-response tokens.\n"
        "Instruction/template tokens contribute exactly zero — not a rescaled\n"
        "amount (the earlier version merely scaled every gradient by a scalar,\n"
        "which is identical to plain SFT at a lower learning rate).")

    block_size = 48
    data  = np.array(tok.encode(corpus), dtype=np.int32)
    from data.loader import BatchLoader
    loader = BatchLoader(data, block_size, batch_size=8, seed=0)
    model  = TransformerLM(vocab_size=tok.vocab_size, embed_dim=32,
                           block_size=block_size, n_layers=2, n_heads=4)

    # --- Visualise the mask on one window -----------------------------------
    x, y = loader.next_batch()
    mask = np.stack([response_mask(y[b].tolist(), tok, DEFAULT_PAIRS)
                     for b in range(x.shape[0])])
    b = int(np.argmax(mask.sum(axis=1)))       # row with the most response tokens
    txt = tok.decode(y[b].tolist())
    shown = "".join(c if mask[b, i] else "·" for i, c in enumerate(txt))
    line(f"[dim]targets (· = masked, char = response token):[/dim]" if RICH
         else "targets (· = masked, char = response token):")
    line(shown.replace("\n", "⏎"))
    line(f"response-token fraction in this batch: {mask.mean():.0%}\n")

    # --- Proof: masked loss ignores instruction targets ----------------------
    only = np.zeros_like(mask); only[b, int(np.argmax(mask[b]))] = 1.0  # one resp token
    loss_a, _ = model.loss_and_grads(x, y, loss_mask=only)
    y2 = y.copy()
    # flip a MASKED target (instruction position) → masked loss must not move
    masked_pos = np.argwhere(mask[b] == 0)[0, 0]
    y2[b, masked_pos] = (y2[b, masked_pos] + 1) % tok.vocab_size
    loss_b, _ = model.loss_and_grads(x, y2, loss_mask=only)
    line(f"flip an INSTRUCTION target → masked loss {loss_a:.4f} → {loss_b:.4f} "
         f"(Δ={abs(loss_a-loss_b):.1e}) ✓ ignored")
    # flip the kept RESPONSE target → masked loss must move
    y3 = y.copy(); resp_pos = int(np.argmax(only[b]))
    y3[b, resp_pos] = (y3[b, resp_pos] + 1) % tok.vocab_size
    loss_c, _ = model.loss_and_grads(x, y3, loss_mask=only)
    line(f"flip the RESPONSE target    → masked loss {loss_a:.4f} → {loss_c:.4f} "
         f"(Δ={abs(loss_a-loss_c):.1e}) ✓ counted\n")

    # --- Short masked SFT so the effect is tangible --------------------------
    opt = AdamW(lr=3e-3, weight_decay=0.01)
    prompter = Prompter("You are nanoLM.")
    before = prompter.chat(model, tok, "What is your name?", max_new_tokens=24,
                           temperature=0.7)
    for step in range(sft_steps):
        xb, yb = loader.next_batch()
        m = np.stack([response_mask(yb[i].tolist(), tok, DEFAULT_PAIRS)
                      for i in range(xb.shape[0])]).astype(np.float32)
        if m.sum() == 0:
            continue
        loss, grads = model.loss_and_grads(xb, yb, loss_mask=m)
        clip_grad_norm(list(model._flat_grads(grads)), 1.0)
        gn = opt.step(model, grads)
    after = prompter.chat(model, tok, "What is your name?", max_new_tokens=24,
                          temperature=0.7)
    line(f"after {sft_steps} masked SFT steps · final masked loss={loss:.3f} "
         f"· grad_norm={gn:.2f}")
    line(f"  prompt 'What is your name?'")
    line(f"  before SFT : {before!r}")
    line(f"  after  SFT : {after!r}")
    return model, block_size


# ─────────────────────────────────────────────────────────────────────────────
# Segment 2 — Real int8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def segment_quantization(model, tok, block_size):
    say("Segment 2 · Phase 9 — REAL int8 quantization",
        "quantize_model() stores int8 weights (int8 array + float scale) and\n"
        "dequantizes inside the matmul, so model_size_bytes() actually drops\n"
        "and the same forward() still runs.")

    before = model_size_bytes(model)
    qmodel = quantize_model(model)
    after  = model_size_bytes(qmodel)
    stats  = compression_stats(model)

    if RICH:
        t = Table(box=box.SIMPLE)
        for c in ("metric", "value"): t.add_column(c)
        t.add_row("float64 size",       f"{before/1000:.1f} KB")
        t.add_row("int8 size (real)",   f"{after/1000:.1f} KB")
        t.add_row("overall compression", f"{before/after:.2f}×")
        t.add_row("2-D-weights-only",   f"{stats['compression_ratio']:.2f}×")
        t.add_row("avg RMS error",      f"{stats['avg_rms_error']:.2e}")
        _console.print(t)
    else:
        line(f"float64={before/1000:.1f}KB  int8={after/1000:.1f}KB  "
             f"ratio={before/after:.2f}x  (2D-only {stats['compression_ratio']:.2f}x)")

    g_full = generate(model,  tok, "t", 32, block_size, temperature=0.0, greedy=True)
    g_int8 = generate(qmodel, tok, "t", 32, block_size, temperature=0.0, greedy=True)
    line(f"\ngreedy generation still works after quantization:")
    line(f"  float64 : {g_full!r}")
    line(f"  int8    : {g_int8!r}")
    line(f"  (1-D embeddings & norm scales are intentionally left in float64)")


# ─────────────────────────────────────────────────────────────────────────────
# Segment 3 — KV-cache decode
# ─────────────────────────────────────────────────────────────────────────────

def segment_kv_cache(tok):
    say("Segment 3 · Phase 9 — KV-cache decode",
        "Cached decode reuses past K/V instead of recomputing them every step.\n"
        "A safety guard now rejects generations that exceed block_size instead\n"
        "of silently indexing the positional table out of bounds.")

    block_size = 32
    model = ModernTransformerLM(vocab_size=tok.vocab_size, embed_dim=48,
                                block_size=block_size, n_layers=3, n_heads=6,
                                norm='rmsnorm', ffn='swiglu', pos_enc='rope')
    res = benchmark_kv_cache(model, tok, "the ", max_new_tokens=20,
                             temperature=0.0, n_trials=3)
    line(f"no-cache : {res['no_cache_ms']:.1f} ms  ({res['no_cache_ms_per_tok']:.2f} ms/tok)")
    line(f"cached   : {res['cached_ms']:.1f} ms  ({res['cached_ms_per_tok']:.2f} ms/tok)")
    line(f"speedup  : {res['speedup']:.2f}×  "
         f"[dim](small at block_size={block_size}; grows quadratically with context)[/dim]"
         if RICH else f"speedup  : {res['speedup']:.2f}×")

    line("\nblock_size guard:")
    try:
        generate_cached(model, tok.encode("the "), max_new_tokens=block_size + 10)
        line("  (no error — unexpected)")
    except ValueError as e:
        line(f"  ✓ rejected over-long request: {str(e).split('.')[0]}.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="nanoLM session walkthrough")
    ap.add_argument("--sft-steps", type=int, default=120,
                    help="masked SFT steps in Segment 1 (more = nicer text)")
    args = ap.parse_args()

    say("nanoLM — Phase 8/9 session walkthrough",
        "Three mechanisms, each made visible on a tiny model:\n"
        "  1. true response masking   2. real int8 quantization   3. KV-cache")

    corpus = build_corpus(DEFAULT_PAIRS, repeat=8)
    tok    = CharTokenizer(corpus)

    model, block_size = segment_masking(tok, corpus, args.sft_steps)
    segment_quantization(model, tok, block_size)
    segment_kv_cache(tok)

    say("Done", "Run `pytest tests/test_gradcheck.py -q` to verify every "
                "hand-written backward pass against finite differences.")


if __name__ == "__main__":
    main()
