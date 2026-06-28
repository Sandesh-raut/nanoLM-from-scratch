"""
Phase 5 — Learning rate scheduler: linear warmup + cosine decay.

Why not a fixed LR?
-------------------
At step 0 the weights are random and the gradients are noisy.  A large LR at
this point causes chaotic early steps.  Warmup solves this by ramping from
near-zero to max_lr over the first `warmup_steps` steps.

After warmup, cosine decay slowly reduces the LR so the model can fine-tune
into a sharper minimum without overshooting.

Schedule shape
--------------
  Steps 0 → warmup_steps :  LR rises linearly from 0 to max_lr
  Steps warmup_steps → T :  LR falls along a cosine from max_lr to min_lr

  LR(step) =
    max_lr * step / warmup_steps                          if step ≤ warmup_steps
    min_lr + 0.5*(max_lr-min_lr)*(1+cos(π·progress))    otherwise
      where progress = (step - warmup) / (total - warmup)

Typical values for small models:
  max_lr       : 3e-4 (AdamW);  5e-2 (SGD)
  min_lr       : 1e-5  (≈ max_lr / 30)
  warmup_steps : 5–10% of total steps
"""

import math


class CosineScheduler:
    """
    Warmup then cosine decay, compatible with any optimizer that exposes `.lr`.

    Usage
    -----
    scheduler = CosineScheduler(optimizer, warmup_steps=50, total_steps=500,
                                 max_lr=3e-4, min_lr=1e-5)
    for step in range(total_steps):
        scheduler.update(step)          # sets optimizer.lr
        optimizer.step(model, grads)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps:  int,
        max_lr:       float,
        min_lr:       float = 0.0,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr

    def get_lr(self, step: int) -> float:
        """Compute the learning rate for `step` (0-indexed)."""
        ws = self.warmup_steps
        T  = self.total_steps

        if ws > 0 and step < ws:
            # Linear warmup: 0 → max_lr
            return self.max_lr * (step + 1) / ws

        # Cosine decay: max_lr → min_lr
        progress = (step - ws) / max(T - ws, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def update(self, step: int) -> float:
        """Set optimizer.lr for this step. Returns the new LR."""
        lr = self.get_lr(step)
        self.optimizer.lr = lr
        return lr


def build_scheduler(optimizer, cfg: dict, total_steps: int) -> 'CosineScheduler | None':
    """Construct scheduler from config, or return None if not configured."""
    tcfg         = cfg['training']
    warmup_steps = int(tcfg.get('warmup_steps', 0))
    min_lr       = float(tcfg.get('min_lr', 0.0))
    max_lr       = float(tcfg['lr'])
    if warmup_steps == 0 and min_lr == 0.0:
        return None
    return CosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )
