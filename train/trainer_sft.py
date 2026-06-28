"""
Phase 8 — Supervised Fine-Tuning (SFT) trainer.

Extends the base Trainer with one key difference:
  Loss is computed ONLY over response tokens (response masking).

Why this matters
----------------
Without masking, the model would try to predict the instruction tokens too
(the "### System:", "### User:" parts). But those are given conditions, not
things the model should learn to generate. Masking forces the model to focus
its gradient signal on the response.

In practice at scale: HuggingFace TRL's SFTTrainer does exactly this via
DataCollatorForCompletionOnlyLM. We implement it from scratch here.

Response masking pipeline
--------------------------
  tokens  = [t0, t1, ..., tN]
  mask    = [0,  0,  1,  1, ..., 0, 0, 1, 1, ...]
              ↑ instruction      ↑ response

  loss = mean( cross_entropy(logits[i], targets[i]) for i where mask[i]==1 )
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from data.tokenizer import CharTokenizer
from data.loader import BatchLoader
from data.instruct_dataset import (
    InstructPair, build_corpus, response_mask,
)
from sample.sampler import generate
from sample.prompter import Prompter
from sample.checkpoint import save as save_checkpoint
from dashboard.display import Dashboard
from train.optimizer import build_optimizer, clip_grad_norm
from train.scheduler import build_scheduler


class TrainerSFT:
    """
    Supervised Fine-Tuning trainer with response masking.

    Differences from base Trainer:
      1. Corpus is built from InstructPairs, not a raw text file.
      2. Loss is masked to response tokens only.
      3. Samples are generated via Prompter (system + user → response).
      4. Checkpoint saved to runs/latest_sft.npz to keep separate from base.
    """

    def __init__(
        self,
        model,
        pairs:     Sequence[InstructPair],
        tokenizer: CharTokenizer,
        cfg:       dict,
        repeat:    int = 20,
    ):
        self.model     = model
        self.pairs     = list(pairs)
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.dashboard = Dashboard(cfg)
        self.run_id    = datetime.now().strftime('%Y%m%d_%H%M%S') + '_sft'
        self.history:  list[dict] = []

        # Build instruction corpus and loader
        corpus = build_corpus(pairs, repeat=repeat)
        data   = np.array(tokenizer.encode(corpus), dtype=np.int32)

        block_size = cfg['model']['block_size']
        batch_size = cfg['training']['batch_size']
        seed       = cfg['training']['seed']

        self.loader     = BatchLoader(data, block_size, batch_size, seed=seed)
        self.mask_text  = corpus          # kept for mask computation

        # Build optimizer and scheduler
        n_steps          = cfg['training']['epochs']
        self.optimizer   = build_optimizer(cfg)
        self.scheduler   = build_scheduler(self.optimizer, cfg, n_steps)

        # Prompter for sample generation
        system = cfg.get('sft', {}).get('system_prompt',
                 pairs[0].system if pairs else "You are nanoLM.")
        self.prompter = Prompter(system)

    # ── Masked loss ───────────────────────────────────────────────────────────

    def _masked_loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        """
        Forward + backward with TRUE response masking.

        x : (B, T) input tokens
        y : (B, T) target tokens (x shifted by 1)

        The mask is built on the TARGETS (y): a position is "active" iff the
        token being predicted belongs to an assistant response. The mask is
        passed straight into the model's loss_and_grads, which zeroes both the
        loss and the gradient at instruction positions and averages only over
        response tokens. Instruction tokens contribute exactly zero — not a
        rescaled amount (the previous version merely scaled every gradient by a
        scalar, which is identical to plain SFT at a lower learning rate).
        """
        B, T = x.shape

        # Mask aligned with the predicted token y[t], not the input x[t]:
        # the loss at position t scores the prediction of y[t], so we keep it
        # iff y[t] is a response token.
        batch_mask = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            batch_mask[b] = response_mask(y[b].tolist(), self.tokenizer, self.pairs)

        # If the whole batch landed in an instruction-only region, fall back to
        # an unmasked step so training doesn't stall on a zero-gradient batch.
        if batch_mask.sum() == 0:
            return self.model.loss_and_grads(x, y)

        return self.model.loss_and_grads(x, y, loss_mask=batch_mask)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> str:
        tcfg         = self.cfg['training']
        dcfg         = self.cfg.get('dashboard', {})
        scfg         = self.cfg.get('sft', {})

        n_steps      = tcfg['epochs']
        max_norm     = tcfg.get('grad_clip', 1.0)
        log_every    = dcfg.get('log_every', 10)
        sample_every = dcfg.get('sample_every', 50)
        block_size   = self.cfg['model']['block_size']
        max_new      = scfg.get('max_new_tokens', 80)
        temperature  = float(scfg.get('temperature', 0.8))
        sample_user  = scfg.get('sample_user', 'What is your name?')

        self.dashboard.print_header(self.model, self.tokenizer)

        t0 = time.time()

        for step in range(n_steps):
            if self.scheduler:
                current_lr = self.scheduler.update(step)
            else:
                current_lr = self.optimizer.lr

            x, y = self.loader.next_batch()
            loss, grads = self._masked_loss_and_grads(x, y)

            flat_g    = list(self.model._flat_grads(grads))
            grad_norm = clip_grad_norm(flat_g, max_norm)

            self.optimizer.step(self.model, grads)

            ppl     = math.exp(min(loss, 20))
            tracked = self.model.tracked_weight()
            elapsed = round(time.time() - t0, 3)

            record: dict = {
                'step':           step,
                'loss':           round(loss, 6),
                'ppl':            round(ppl, 4),
                'lr':             round(current_lr, 8),
                'grad_norm':      round(grad_norm, 4),
                'tracked_weight': round(tracked, 6),
                'elapsed':        elapsed,
            }

            if step % sample_every == 0:
                response = self.prompter.chat(
                    self.model, self.tokenizer, sample_user,
                    max_new_tokens=max_new, temperature=temperature,
                )
                sample = f"[USR] {sample_user}\n[AST] {response}"
                record['sample'] = sample
                self.dashboard.print_sample(step, sample)

            self.history.append(record)

            if step % log_every == 0:
                self.dashboard.print_step(
                    step, n_steps, loss, tracked,
                    lr=current_lr, grad_norm=grad_norm,
                )

        # Final sample
        if (n_steps - 1) % sample_every != 0:
            response = self.prompter.chat(
                self.model, self.tokenizer, sample_user,
                max_new_tokens=max_new, temperature=temperature,
            )
            self.dashboard.print_sample(n_steps - 1,
                                        f"[USR] {sample_user}\n[AST] {response}")

        run_path = self._save_run()
        save_checkpoint(self.model, self.tokenizer, f'runs/{self.run_id}.npz')
        save_checkpoint(self.model, self.tokenizer, 'runs/latest_sft.npz')
        self.dashboard.print_footer(time.time() - t0, str(run_path))
        return str(run_path)

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
            'mode':        'sft',
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        return path
