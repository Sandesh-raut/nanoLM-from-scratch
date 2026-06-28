"""
Training loop — Phase 5 upgrade.

New in Phase 5:
  - AdamW optimizer (replaces plain SGD)
  - LR warmup + cosine decay via CosineScheduler
  - Gradient clipping (global norm)
  - Train / val split → validation loss evaluated every val_every steps
  - Perplexity = exp(loss) logged alongside loss

The trainer remains model-agnostic: it works with BigramModel (Phase 2)
or TransformerLM (Phase 3/4) as long as the model exposes:
  loss_and_grads(), _flat_params(), _flat_grads(), compute_loss(),
  tracked_weight(), param_count(), param_table(), description.

Run JSON schema (Phase 5+)
--------------------------
{
  "run_id":      "20240101_120000",
  "config":      { ... },
  "param_count": { ... },
  "steps": [
    { "step": 0,  "loss": 3.19, "ppl": 24.3, "lr": 1e-5,
      "val_loss": 3.21, "val_ppl": 24.8,
      "grad_norm": 1.4, "tracked_weight": 0.006, "elapsed": 0.01 },
    ...
  ]
}
"""
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data.loader import BatchLoader
from data.tokenizer import CharTokenizer
from sample.sampler import generate
from dashboard.display import Dashboard
from train.optimizer import build_optimizer, clip_grad_norm
from train.scheduler import build_scheduler
from sample.checkpoint import save as save_checkpoint


class Trainer:
    def __init__(self, model, loader: BatchLoader, tokenizer: CharTokenizer, cfg: dict):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.dashboard = Dashboard(cfg)
        self.run_id    = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.history: list[dict] = []

        tcfg = cfg['training']
        n_steps = tcfg['epochs']

        # ── Train / val split ────────────────────────────────────────────────
        val_split = tcfg.get('val_split', 0.0)
        if val_split > 0 and len(loader.data) > loader.block_size * 2:
            self.train_loader, self.val_loader = loader.split(val_split)
        else:
            self.train_loader = loader
            self.val_loader   = None

        # ── Optimizer + scheduler ────────────────────────────────────────────
        self.optimizer = build_optimizer(cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, n_steps)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> str:
        tcfg         = self.cfg['training']
        scfg         = self.cfg.get('sampling', {})
        dcfg         = self.cfg.get('dashboard', {})

        n_steps      = tcfg['epochs']
        max_norm     = tcfg.get('grad_clip', 1.0)
        val_every    = tcfg.get('val_every', 50)
        log_every    = dcfg.get('log_every', 10)
        sample_every = dcfg.get('sample_every', 50)
        seed_text    = scfg.get('seed_text', 't')
        max_new      = scfg.get('max_new_tokens', 120)
        temp         = float(scfg.get('temperature', 1.0))
        top_k        = int(scfg.get('top_k', 0))
        top_p        = float(scfg.get('top_p', 1.0))
        rep_penalty  = float(scfg.get('rep_penalty', 1.0))
        greedy       = bool(scfg.get('greedy', False))
        block_size   = self.cfg['model']['block_size']

        self.dashboard.print_header(self.model, self.tokenizer)

        t0       = time.time()
        val_loss = None
        val_ppl  = None

        for step in range(n_steps):

            # ── LR schedule ──────────────────────────────────────────────────
            if self.scheduler:
                current_lr = self.scheduler.update(step)
            else:
                current_lr = self.optimizer.lr

            # ── Forward + backward ───────────────────────────────────────────
            x, y = self.train_loader.next_batch()
            loss, grads = self.model.loss_and_grads(x, y)

            # ── Gradient clipping ─────────────────────────────────────────────
            flat_g    = list(self.model._flat_grads(grads))
            grad_norm = clip_grad_norm(flat_g, max_norm)

            # ── Optimizer step ────────────────────────────────────────────────
            self.optimizer.step(self.model, grads)

            ppl     = math.exp(min(loss, 20))   # cap to avoid overflow display
            tracked = self.model.tracked_weight()
            elapsed = round(time.time() - t0, 3)

            # ── Val loss ─────────────────────────────────────────────────────
            if self.val_loader and step % val_every == 0:
                val_loss = self._val_loss(n_batches=4)
                val_ppl  = math.exp(min(val_loss, 20))

            # ── Sample ───────────────────────────────────────────────────────
            record: dict = {
                'step':           step,
                'loss':           round(loss, 6),
                'ppl':            round(ppl, 4),
                'lr':             round(current_lr, 8),
                'grad_norm':      round(grad_norm, 4),
                'tracked_weight': round(tracked, 6),
                'elapsed':        elapsed,
            }
            if val_loss is not None:
                record['val_loss'] = round(val_loss, 6)
                record['val_ppl']  = round(val_ppl, 4)

            if step % sample_every == 0:
                sample = generate(self.model, self.tokenizer, seed_text,
                                  max_new, block_size,
                                  temperature=temp, top_k=top_k, top_p=top_p,
                                  greedy=greedy, rep_penalty=rep_penalty)
                record['sample'] = sample
                self.dashboard.print_sample(step, sample)

            self.history.append(record)

            if step % log_every == 0:
                self.dashboard.print_step(
                    step, n_steps, loss, tracked,
                    lr=current_lr, val_loss=val_loss, grad_norm=grad_norm,
                )

        # Final sample
        if (n_steps - 1) % sample_every != 0:
            sample = generate(self.model, self.tokenizer, seed_text,
                              max_new, block_size,
                              temperature=temp, top_k=top_k, top_p=top_p,
                              greedy=greedy, rep_penalty=rep_penalty)
            self.dashboard.print_sample(n_steps - 1, sample)

        run_path  = self._save_run()
        ckpt_path = save_checkpoint(self.model, self.tokenizer, f'runs/{self.run_id}.npz')
        # Also keep a "latest" alias so sample_cli.py can find it without a timestamp
        save_checkpoint(self.model, self.tokenizer, 'runs/latest.npz')
        self.dashboard.print_footer(time.time() - t0, str(run_path))
        return str(run_path)

    # ── Val loss helper ───────────────────────────────────────────────────────

    def _val_loss(self, n_batches: int = 4) -> float:
        losses = []
        for _ in range(n_batches):
            xv, yv = self.val_loader.next_batch()
            losses.append(self.model.compute_loss(xv, yv))
        return float(np.mean(losses))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_run(self) -> Path:
        runs_dir = Path('runs')
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f'{self.run_id}.json'
        payload = {
            'run_id':      self.run_id,
            'config':      self.cfg,
            'param_count': self.model.param_count(),
            'steps':       self.history,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        return path
