# nanoLM

A tiny character-level language model implemented from scratch in NumPy, with
every gradient hand-derived — no autograd. A PyTorch mirror exists to verify
the NumPy math and to run on Apple Silicon (MPS).

Background reading: [I Built a Language Model From Scratch in NumPy](docs/article_building_nanoLM.md).

## Concept map

The codebase is organized into phases; each phase implements one concept from
modern LLMs as runnable code:

| Phase | Module(s) | LLM concept |
|-------|-----------|-------------|
| 0 | `config.yaml`, `run_history.py` | Experiment tracking |
| 1 | `data/tokenizer.py`, `data/loader.py` | Tokenization, batching |
| 2 | `model/bigram.py` | Embeddings, cross-entropy, hand-written backprop |
| 3 | `model/transformer.py` | Attention, FFN, LayerNorm — the transformer block |
| 4 | `model/transformer.py` | Multi-head attention, depth, dropout |
| 5 | `train/optimizer.py`, `train/scheduler.py` | AdamW, warmup + cosine LR, grad clipping, val loss |
| 6 | `sample/sampler.py`, `sample_cli.py` | Greedy, temperature, top-k, top-p, repetition penalty |
| 7 | `model/transformer_torch.py`, `main_torch.py` | Autograd, GPU backends |
| 8 | `train/trainer_sft.py`, `demo_dpo.py` | SFT with response masking, DPO |
| 9 | `model/modern_transformer.py`, `model/kv_cache.py`, `model/quantize.py` | RoPE, RMSNorm, SwiGLU, GQA, KV-cache, int8 quantization |

All hand-written backward passes are verified against finite differences in
`tests/test_gradcheck.py`.

## Setup

Requires Python 3.11+.

```bash
uv venv
uv pip install -r requirements.txt
```

## Usage

### Train (NumPy)

```bash
uv run python3 main.py                          # defaults from config.yaml
uv run python3 main.py --epochs 1000 --lr 0.1   # CLI overrides
```

All hyperparameters live in `config.yaml`; any of them can be overridden at
the command line:

```bash
uv run python3 main.py \
  --epochs 500        \   # gradient steps
  --lr 0.05           \   # learning rate
  --batch_size 32     \   # windows per step
  --block_size 64     \   # context length
  --embed_dim 128     \   # embedding dimension
  --n_layers 4        \   # transformer blocks
  --n_heads 8         \   # attention heads
  --temperature 0.8   \   # sampling temperature
  --corpus data/corpus.txt
```

Each run writes `runs/<timestamp>.json` (loss history + config) and
`runs/<timestamp>.npz` (checkpoint), plus a `runs/latest.npz` alias.

### Train (PyTorch / MPS)

```bash
uv run python3 main_torch.py       # same recipe, autograd + MPS/CUDA/CPU
uv run python3 verify_torch.py     # confirm NumPy and PyTorch models match
```

### Generate from a checkpoint

```bash
uv run python3 sample_cli.py --checkpoint runs/latest.npz --seed "the"
uv run python3 sample_cli.py --compare     # all sampling strategies side by side
```

### Instruction fine-tuning (SFT)

```bash
uv run python3 main_sft.py                          # train on built-in instruct pairs
uv run python3 main_sft.py --base runs/latest.npz   # fine-tune an existing checkpoint
uv run python3 main_sft.py --system "You are a pirate."
```

Loss is computed only over assistant-response tokens (response masking).
Prints a before/after comparison table and saves `runs/latest_sft.npz`.

### Demos and benchmarks

```bash
uv run python3 demo_session.py    # guided walkthrough: SFT masking, int8, KV-cache
uv run python3 demo_dpo.py        # DPO loss computed on preference pairs (conceptual)
uv run python3 bench_phase9.py    # modern-upgrade ablations, KV-cache speedup, int8 stats
uv run python3 run_history.py     # compare past runs side by side
```

## Training output

The terminal dashboard (`dashboard/display.py`, built on Rich) prints:

- a parameter breakdown by layer at startup
- per-step rows: loss, val loss, perplexity, LR, grad norm, a tracked
  weight (`E[0,0]` by default), and a loss bar
- generated samples every `sample_every` steps
- a footer with elapsed time and a loss sparkline

## Repo layout

```
nanoLM/
├── config.yaml             ← all hyperparams
├── main.py                 ← training entry point (NumPy)
├── main_torch.py           ← training entry point (PyTorch/MPS)
├── main_sft.py             ← SFT entry point
├── sample_cli.py           ← generation from a saved checkpoint
├── demo_session.py         ← guided walkthrough of SFT/int8/KV-cache
├── demo_dpo.py             ← DPO conceptual demo
├── bench_phase9.py         ← ablation + inference benchmarks
├── run_history.py          ← compare runs
├── data/
│   ├── corpus.txt          ← training text (swappable)
│   ├── tokenizer.py        ← char tokenizer (encode / decode / save / load)
│   ├── loader.py           ← random batch sampler + train/val split
│   └── instruct_dataset.py ← instruct template, pairs, response_mask()
├── model/
│   ├── bigram.py           ← E → W → logits, backprop by hand
│   ├── transformer.py      ← LayerNorm, MHA, FFN, TransformerLM
│   ├── modern_transformer.py ← RoPE/RMSNorm/SwiGLU/GQA model
│   ├── rope.py · norms.py · activations.py ← modern components
│   ├── kv_cache.py         ← cached autoregressive decode
│   ├── quantize.py         ← int8 weight quantization
│   └── transformer_torch.py ← PyTorch mirror
├── train/
│   ├── trainer.py          ← AdamW loop, LR schedule, grad clip, val
│   ├── trainer_sft.py      ← SFT with response masking
│   ├── trainer_torch.py    ← PyTorch training loop
│   └── optimizer.py · scheduler.py
├── sample/                 ← sampler, prompter, checkpoint I/O
├── dashboard/display.py    ← Rich terminal dashboard
├── tests/                  ← incl. test_gradcheck.py (finite-diff checks)
└── runs/                   ← one JSON + npz checkpoint per training run
```

## Design notes

- **Model interface.** All three models (`BigramModel`, `TransformerLM`,
  `ModernTransformerLM`) expose the same surface — `loss_and_grads()`,
  `compute_loss()`, `_flat_params()`, `_flat_grads()`, `param_table()`,
  `tracked_weight()` — so the trainer, optimizer, and checkpoint code are
  model-agnostic.
- **Hand-written backprop.** Every module caches its forward activations in
  `self._c` and implements `backward(dout) → (dx, grads)`. NumPy is the
  engine for everything except the Phase 7 mirror.
- **Checkpoints** are compressed `.npz` files containing weights, architecture
  metadata, and the tokenizer vocab — no pickle, no framework dependency.

## Tests

```bash
uv run python3 -m pytest -q                       # full suite
uv run python3 -m pytest tests/test_gradcheck.py  # finite-diff gradient checks only
```
