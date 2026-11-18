"""Recompute a training sub-step whose loss or gradients came out non-finite (used by run_train_nanguard.py).

Four trainings stopped on a sub-step with a finite loss and NaN gradients that the guard's exact replay (same weights,
inputs and RNG state, same process) did not reproduce (README, step 13 amendment; tests/test_nanguard_retry.py).  The
recomputation starts from the gradients accumulated before the sub-step and from its recorded RNG state, so a finite
result is exactly what an unfaulted run computes; a fault that reproduces is returned as such.  No import side
effects (no default-dtype change, no patching).
"""
import math

import torch


def nonfinite(named):
    """[(name, n_nan, n_inf)] for the tensors in ``named`` that are not finite."""
    out = []
    for name, value in named:
        if value is not None and not torch.isfinite(value).all():
            out.append((name, int(torch.isnan(value).sum()), int(torch.isinf(value).sum())))
    return out


def grads(model):
    return [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]


def restore_grads(model, before):
    """The accumulated gradients as recorded before the sub-step (``None`` where there was none)."""
    for name, p in model.named_parameters():
        p.grad = before[name].clone() if name in before else None


def recompute_nonfinite(step, model, args, kwargs, rng, before, retries):
    """``(loss, attempts)``: the first recomputation with a finite loss and finite gradients, or ``(None, attempts)``
    if none of ``retries`` is (``retries = 0``: no recomputation, the guard stops as before)."""
    for attempt in range(1, int(retries) + 1):
        restore_grads(model, before)
        torch.set_rng_state(rng)
        loss = step(*args, **kwargs)
        value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        if math.isfinite(value) and not nonfinite(grads(model)):
            return loss, attempt
    return None, int(retries)
