"""
Phase 7 — PyTorch training loop.

Same training recipe as Phase 5 (trainer.py) but using:
  - torch.autograd  (replaces hand-written backward)
  - torch.optim.AdamW with param groups for weight-decay exclusions
  - torch.optim.lr_scheduler.CosineAnnealingLR (replaces CosineScheduler)
  - MPS / CUDA / CPU device selection
  - Same dashboard, same run-JSON format, same checkpoint save

The corpus, tokenizer, and batch loader are identical to the NumPy path —
they return plain numpy arrays; the trainer converts them to tensors.

Why keep the same dashboard/run format?
Because Phase 7 is a mirror, not a replacement.  A single compare_runs.py
script (or just reading the JSON) can show NumPy loss vs PyTorch loss
on the same corpus from the same starting weights.
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.loader import BatchLoader
from data.tokenizer import CharTokenizer
from sample.sampler import generate
from dashboard.display import Dashboard
from model.transformer_torch import TransformerLMTorch

_NO_DECAY_SUFFIXES = {'bias', 'ln_final.weight', 'ln_final.bias'}
_NO_DECAY_TYPES    = (nn.LayerNorm, nn.Embedding)


def _build_adamw(model: TransformerLMTorch, cfg: dict) -> torch.optim.AdamW:
    """
    Create AdamW with two param groups:
      - matrices  → weight_decay applied
      - biases, LN, embeddings → no weight decay

    Mirrors the logic in train/optimizer.py (_should_decay).
    """
    tcfg = cfg['training']
    lr   = float(tcfg.get('lr',           0.0003))
    b1   = float(tcfg.get('beta1',        0.9))
    b2   = float(tcfg.get('beta2',        0.999))
    wd   = float(tcfg.get('weight_decay', 0.01))

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, LN params, and embedding weights skip weight decay
        is_no_decay = (
            name.endswith('.bias')
            or 'ln' in name        # ln1, ln2, ln_final
            or 'embed' in name     # token_embed, pos_embed
        )
        (no_decay if is_no_decay else decay).append(param)

    param_groups = [
        {'params': decay,    'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(b1, b2), eps=1e-8)


def _pick_device(cfg: dict) -> torch.device:
    requested = cfg.get('training', {}).get('device', 'auto')
    if requested == 'mps' or (requested == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')


class TrainerTorch:
    """
    PyTorch training loop for TransformerLMTorch.

    Exposes the same .train() → run_path interface as Trainer (NumPy).
    """

    def __init__(
        self,
        model:     TransformerLMTorch,
        loader:    BatchLoader,
        tokenizer: CharTokenizer,
        cfg:       dict,
    ):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.run_id    = datetime.now().strftime('%Y%m%d_%H%M%S') + '_torch'
        self.history:  list[dict] = []
        self.dashboard = Dashboard(cfg)

        # Device
        self.device = _pick_device(cfg)
        self.model  = model.to(self.device)

        # Train / val split
        tcfg      = cfg['training']
        val_split = float(tcfg.get('val_split', 0.0))
        if val_split > 0 and len(loader.data) > loader.block_size * 2:
            self.train_loader, self.val_loader = loader.split(val_split)
        else:
            self.train_loader = loader
            self.val_loader   = None

        # Optimizer and LR scheduler
        n_steps         = int(tcfg['epochs'])
        warmup_steps    = int(tcfg.get('warmup_steps', 0))
        min_lr          = float(tcfg.get('min_lr', 0.0))
        lr              = float(tcfg.get('lr', 0.0003))

        self.optimizer  = _build_adamw(model, cfg)
        self.max_norm   = float(tcfg.get('grad_clip', 1.0))

        # Cosine schedule with linear warmup (manual lambda)
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / lr + (1.0 - min_lr / lr) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self) -> str:
        tcfg         = self.cfg['training']
        scfg         = self.cfg.get('sampling', {})
        dcfg         = self.cfg.get('dashboard', {})

        n_steps      = int(tcfg['epochs'])
        val_every    = int(tcfg.get('val_every', 50))
        log_every    = int(dcfg.get('log_every', 10))
        sample_every = int(dcfg.get('sample_every', 50))
        seed_text    = scfg.get('seed_text', 't')
        max_new      = int(scfg.get('max_new_tokens', 120))
        temp         = float(scfg.get('temperature', 1.0))
        top_k        = int(scfg.get('top_k', 0))
        top_p        = float(scfg.get('top_p', 1.0))
        rep_penalty  = float(scfg.get('rep_penalty', 1.0))
        greedy       = bool(scfg.get('greedy', False))
        block_size   = self.cfg['model']['block_size']

        self.dashboard.print_header(self.model, self.tokenizer)
        console_extra = f"  [dim]device: {self.device}[/dim]"

        try:
            from rich.console import Console
            Console().print(console_extra)
        except Exception:
            pass

        t0       = time.time()
        val_loss = None
        val_ppl  = None

        for step in range(n_steps):
            # Forward + loss
            x_np, y_np = self.train_loader.next_batch()
            x = torch.tensor(x_np, dtype=torch.long,  device=self.device)
            y = torch.tensor(y_np, dtype=torch.long,  device=self.device)

            self.model.train()
            logits = self.model(x)                              # (B, T, V)
            loss   = F.cross_entropy(
                logits.view(-1, self.model.vocab_size), y.view(-1)
            )

            # Backward + clip + step
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            self.scheduler.step()

            loss_val  = loss.item()
            ppl       = math.exp(min(loss_val, 20))
            current_lr = self.optimizer.param_groups[0]['lr']
            tracked   = float(self.model.token_embed.weight[0, 0].item())
            elapsed   = round(time.time() - t0, 3)

            # Compute grad norm for logging
            grad_norm = float(sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.model.parameters() if p.grad is not None
            ) ** 0.5)

            # Val loss
            if self.val_loader and step % val_every == 0:
                val_loss = self._val_loss(n_batches=4)
                val_ppl  = math.exp(min(val_loss, 20))

            record: dict = {
                'step':           step,
                'loss':           round(loss_val, 6),
                'ppl':            round(ppl, 4),
                'lr':             round(current_lr, 8),
                'grad_norm':      round(grad_norm, 4),
                'tracked_weight': round(tracked, 6),
                'elapsed':        elapsed,
            }
            if val_loss is not None:
                record['val_loss'] = round(val_loss, 6)
                record['val_ppl']  = round(val_ppl, 4)

            # Sample
            if step % sample_every == 0:
                self.model.eval()
                sample = self._generate(
                    seed_text, max_new, block_size, temp, top_k, top_p, greedy, rep_penalty
                )
                record['sample'] = sample
                self.dashboard.print_sample(step, sample)

            self.history.append(record)

            if step % log_every == 0:
                self.dashboard.print_step(
                    step, n_steps, loss_val, tracked,
                    lr=current_lr, val_loss=val_loss, grad_norm=grad_norm,
                )

        # Final sample
        if (n_steps - 1) % sample_every != 0:
            self.model.eval()
            sample = self._generate(
                seed_text, max_new, block_size, temp, top_k, top_p, greedy, rep_penalty
            )
            self.dashboard.print_sample(n_steps - 1, sample)

        run_path = self._save_run()
        self.dashboard.print_footer(time.time() - t0, str(run_path))
        return str(run_path)

    # ── Val loss ─────────────────────────────────────────────────────────────

    def _val_loss(self, n_batches: int = 4) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                xv, yv = self.val_loader.next_batch()
                x = torch.tensor(xv, dtype=torch.long, device=self.device)
                y = torch.tensor(yv, dtype=torch.long, device=self.device)
                logits = self.model(x)
                loss   = F.cross_entropy(logits.view(-1, self.model.vocab_size), y.view(-1))
                losses.append(loss.item())
        return float(np.mean(losses))

    # ── Generation (wraps CPU-side generate()) ─────────────────────────────

    def _generate(self, seed_text, max_new, block_size, temp, top_k, top_p, greedy, rep_penalty) -> str:
        """
        Move model to CPU for generation (MPS tensors and numpy interop is simpler this way).
        Move back to device after.
        """
        self.model.eval()
        cpu_model = self.model.cpu()
        result = generate(
            cpu_model, self.tokenizer, seed_text, max_new, block_size,
            temperature=temp, top_k=top_k, top_p=top_p,
            greedy=greedy, rep_penalty=rep_penalty,
        )
        self.model.to(self.device)
        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_run(self) -> Path:
        runs_dir = Path('runs')
        runs_dir.mkdir(exist_ok=True)
        path = runs_dir / f'{self.run_id}.json'
        with open(path, 'w') as f:
            json.dump({
                'run_id':      self.run_id,
                'backend':     'pytorch',
                'device':      str(self.device),
                'config':      self.cfg,
                'param_count': self.model.param_count(),
                'steps':       self.history,
            }, f, indent=2)
        return path
