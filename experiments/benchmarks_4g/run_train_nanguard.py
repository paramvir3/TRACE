"""Reproduce a non-finite training step: run_train_dump.py plus a finiteness guard.

Usage: NANGUARD_SKIP_EPOCHS=K python run_train_nanguard.py CONFIG
       (CONFIG resumes from the checkpoint written after epoch K, with
       resume_load_optimizer and resume_load_scheduler true)

What is reproduced, and what is not:
* the batch order: exactly.  train.py seeds the training loader's generator once, at
  construction, and every epoch consumes it (one base seed, then two permutations; torch
  2.2's RandomSampler draws its second permutation when the epoch is exhausted).  On its
  first use the guard runs the shuffled loader through K complete epochs, so the resumed
  run sees the order that the uninterrupted run saw in epoch K+1;
* the parameters and the optimiser and scheduler state at the start of epoch K+1: exactly
  (the checkpoint);
* the attention-dropout masks: NOT reproduced (the global RNG state after K epochs is not
  in the checkpoint).  The guard records the RNG state before every training sub-step, so
  whatever failure does occur here is itself exactly replayable.

Checks:
* after every training sub-step (one structure at batch_size 1): the returned loss and
  every accumulated gradient are finite; the structure's own gradient contribution
  ||g_after - g_before|| and its dataset index are recorded;
* at every optimiser step: one log line with the gradient norm before clipping and the
  window's per-structure (index, loss, contribution); after the step, every parameter is
  finite (parameters and optimiser state are snapshotted before the step).

A sub-step with a non-finite loss or gradient is first recomputed from the gradients accumulated before it and
its RNG state (NANGUARD_RETRIES, default 1; nanguard_retry.py); a finite recomputation is kept and logged as
TRANSIENT, with the first pass's gradients in DIR(CONFIG)/nanguard_transientN.pt.

On the first violation that reproduces: DIR(CONFIG)/nanguard.pt (parameters -- unchanged since the
window's step, so the pre-step values --, optimiser state, the sub-step's arguments and
items, its RNG state, the window record), then the failing sub-step is replayed from the
recorded RNG state, (1) plainly, to confirm the reproduction, and (2) under
torch.autograd.detect_anomaly(), to name the backward operation that produced the NaN,
and the run stops.
"""
import copy
import math
import os
import sys
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
torch.set_default_dtype(torch.float64)
# the NaN-localising gradient hooks of trace_redox.nanprobe (read when trace_redox is first imported)
os.environ.setdefault("TRACE_NANPROBE", "1")
# NANGUARD_PREPEND_PATH: a directory holding another copy of trace_redox (e.g. the code as it
# was when a failure happened), imported first and cached, so the later sys.path inserts of
# run_train_dump.py / run_train.py cannot replace it.  Refused if a different copy loads.
_PREPEND = os.environ.get("NANGUARD_PREPEND_PATH")
if _PREPEND:
    sys.path.insert(0, os.path.abspath(_PREPEND))
    import trace_redox  # noqa: E402

    if not os.path.abspath(trace_redox.__file__).startswith(os.path.abspath(_PREPEND)):
        raise SystemExit(f"NANGUARD_PREPEND_PATH: trace_redox loaded from {trace_redox.__file__}")
    print(f"[nanguard] trace_redox from {os.path.dirname(trace_redox.__file__)}", flush=True)

from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler  # noqa: E402
from nanguard_retry import recompute_nonfinite  # noqa: E402
from torch.utils.data import dataloader as _dataloader  # noqa: E402

STATE = {"epoch": 0, "substep": 0, "step": 0, "window": [], "last_index": None, "last_norm": float("nan"),
         "snapshot": None}


class NonFiniteTraining(RuntimeError):
    """Raised by the guard at the first non-finite loss, gradient or parameter."""


def say(message):
    print(f"[nanguard] {message}", flush=True)


def _nonfinite(named):
    """[(name, n_nan, n_inf)] for the tensors in ``named`` that are not finite."""
    out = []
    for name, value in named:
        if value is not None and not torch.isfinite(value).all():
            out.append((name, int(torch.isnan(value).sum()), int(torch.isinf(value).sum())))
    return out


def _grads(model):
    return [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]


def _portable(value):
    """Tensors, numbers, strings and containers of them; anything else (the model, a
    closure such as train.main's baseline_energy, the metric trackers) becomes its repr,
    so that saving the failure can never fail."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_portable(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


FIXED_SOLVER_NAMES = ("differentiable_solution", "differentiable_constrained_solution")


def use_fixed_solver(path):
    """Rebind the free-solver entry points to the copy at ``path`` in every trace_redox
    module that imported them by name; returns a function that restores the originals."""
    import importlib.util

    import trace_redox

    spec = importlib.util.spec_from_file_location("trace_redox._fixed_free_solver", path,
                                                  submodule_search_locations=None)
    fixed = importlib.util.module_from_spec(spec)
    fixed.__package__ = "trace_redox"
    sys.modules[spec.name] = fixed                     # dataclasses resolve their module while executing
    spec.loader.exec_module(fixed)
    saved = []
    for name in ("trace_redox", "trace_redox.free_solver", "trace_redox.module", "trace_redox.fragments"):
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr in FIXED_SOLVER_NAMES:
            if hasattr(module, attr):
                saved.append((module, attr, getattr(module, attr)))
                setattr(module, attr, getattr(fixed, attr))

    def restore():
        for module, attr, value in saved:
            setattr(module, attr, value)

    return restore


def install(skip_epochs, dump_path, train_module):
    """Patch the loader, the training sub-step, gradient clipping and the optimiser step."""
    original_iter = DataLoader.__iter__

    def guarded_iter(self):
        if isinstance(self.sampler, RandomSampler):
            if not getattr(self, "_nanguard_advanced", False):
                self._nanguard_advanced = True
                for _ in range(skip_epochs):
                    for _ in original_iter(self):
                        pass
                say(f"training loader advanced by {skip_epochs} complete epochs")
            STATE["epoch"] = skip_epochs + 1 if STATE["epoch"] == 0 else STATE["epoch"] + 1
            STATE["substep"] = 0
        return original_iter(self)

    DataLoader.__iter__ = guarded_iter

    original_next_index = _dataloader._BaseDataLoaderIter._next_index

    def recording_next_index(self):
        index = original_next_index(self)
        STATE["last_index"] = list(index) if isinstance(index, (list, tuple)) else index
        return index

    _dataloader._BaseDataLoaderIter._next_index = recording_next_index

    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        norm = original_clip(parameters, max_norm, *args, **kwargs)
        STATE["last_norm"] = float(norm)
        return norm

    torch.nn.utils.clip_grad_norm_ = recording_clip

    original_step = train_module._batched_train_step

    def replay(model, args, kwargs, rng):
        """Re-run the failing sub-step from its RNG state: plainly, then under anomaly mode."""
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        loss = original_step(*args, **kwargs)
        bad = _nonfinite(_grads(model))
        say(f"replay: loss {float(loss):.6g}; non-finite gradients in {len(bad)} tensors "
            f"({'reproduced' if bad or not math.isfinite(float(loss)) else 'NOT reproduced'})")
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with torch.autograd.detect_anomaly(check_nan=True):
                    original_step(*args, **kwargs)
                say("anomaly replay: no backward operation returned NaN")
            except RuntimeError as error:
                say(f"anomaly replay: {error}")
        for w in caught:
            text = str(w.message)
            if "Error detected" in text or "Traceback of forward call" in text:
                say("forward traceback of the failing operation:\n" + text)
        fixed = os.environ.get("NANGUARD_FIXED_SOLVER")
        if fixed:
            restore = use_fixed_solver(fixed)
            try:
                model.zero_grad(set_to_none=True)
                torch.set_rng_state(rng)
                loss = original_step(*args, **kwargs)
                bad = _nonfinite(_grads(model))
                say(f"replay with the fixed solver ({fixed}): loss {float(loss):.6g}; "
                    f"non-finite gradients in {len(bad)} tensors")
            except Exception as error:
                say(f"replay with the fixed solver raised {error!r}")
            finally:
                restore()

    def fail(where, model, detail, args=None, kwargs=None, rng=None):
        say(f"FIRST NON-FINITE VALUE ({where}) at epoch {STATE['epoch']}, sub-step {STATE['substep']}, "
            f"optimiser step {STATE['step']}: {detail[:12]}{' ...' if len(detail) > 12 else ''}")
        for entry in STATE["window"]:
            say(f"  window: {entry}")
        try:
            torch.save({"where": where, "detail": detail, "epoch": STATE["epoch"], "substep": STATE["substep"],
                        "step": STATE["step"], "window": list(STATE["window"]),
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "grads": {n: g.detach().cpu().clone() for n, g in _grads(model)},
                        "snapshot": STATE["snapshot"], "args": _portable(args[1:] if args else None),
                        "kwargs": _portable(kwargs), "rng": rng}, dump_path)
            say(f"saved {dump_path}")
        except Exception as error:                      # the replay below must still run
            say(f"could not save the failure: {error!r}")
        if args is not None:
            replay(model, args, kwargs, rng)
        raise NonFiniteTraining(f"nanguard: first non-finite value ({where}); see {dump_path}")

    def guarded_step(*args, **kwargs):
        from trace_redox import nanprobe
        model = args[0]
        rng = torch.get_rng_state()
        nanprobe.drain()
        before = {n: g.detach().clone() for n, g in _grads(model)}
        try:
            loss = original_step(*args, **kwargs)
        except RuntimeError as error:
            # 2026-10-31: an autograd-engine error ("dependency not found for ViewBackward0", step 22 s43, epoch 31)
            # left no reproducer; save the sub-step's inputs, RNG state and model state, then re-raise
            crash = os.path.join(os.path.dirname(dump_path), "nanguard_exception.pt")
            try:
                torch.save({"error": repr(error), "epoch": STATE["epoch"], "substep": STATE["substep"] + 1,
                            "index": STATE["last_index"], "step": STATE["step"],
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "args": _portable(args[1:]), "kwargs": _portable(kwargs), "rng": rng}, crash)
                say(f"EXCEPTION in training sub-step {STATE['substep'] + 1} (epoch {STATE['epoch']}, index "
                    f"{STATE['last_index']}): {error!r}; saved {crash}")
            except Exception as save_error:
                say(f"EXCEPTION {error!r}; could not save the reproducer: {save_error!r}")
            raise
        STATE["substep"] += 1
        after = _grads(model)
        contribution = math.sqrt(sum(float(((g - before[n]) if n in before else g).pow(2).sum()) for n, g in after))
        loss_value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        STATE["window"].append({"substep": STATE["substep"], "index": STATE["last_index"],
                                "loss": loss_value, "grad": contribution})
        bad = _nonfinite(after)
        if bad or not math.isfinite(loss_value):
            # A fault the exact recomputation does not reproduce is transient (nanguard_retry.py): keep the
            # recomputed sub-step, which is what an unfaulted run computes; one that reproduces stops the run.
            retries = int(os.environ.get("NANGUARD_RETRIES", "1"))
            first = bad or [("loss", loss_value)]
            events = nanprobe.drain()
            say(f"nanprobe: first non-finite watched gradient bracketed by {nanprobe.bracket(events)}; "
                f"events {events[:40]}")
            STATE["transient"] = STATE.get("transient", 0) + 1
            transient = os.path.join(os.path.dirname(dump_path), f"nanguard_transient{STATE['transient']}.pt")
            try:
                torch.save({"epoch": STATE["epoch"], "substep": STATE["substep"], "index": STATE["last_index"],
                            "loss": loss_value, "grads": {n: g.detach().cpu().clone() for n, g in after},
                            "nanprobe": events}, transient)
            except Exception as error:
                say(f"could not save {transient}: {error!r}")
            recomputed, attempts = recompute_nonfinite(original_step, model, args, kwargs, rng, before, retries)
            nanprobe.drain()
            if recomputed is None:
                fail("training sub-step", model, first, args, kwargs, rng)
            loss = recomputed
            new_value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            after = _grads(model)
            contribution = math.sqrt(sum(float(((g - before[n]) if n in before else g).pow(2).sum()) for n, g in after))
            say(f"TRANSIENT non-finite value at epoch {STATE['epoch']}, sub-step {STATE['substep']}, index "
                f"{STATE['last_index']}: {first[:6]}; recomputed finite on attempt {attempts} (loss {loss_value!r} -> "
                f"{new_value!r}, {'bitwise equal' if new_value == loss_value else 'DIFFERENT'}); first pass saved to "
                f"{transient}; continuing")
            STATE["window"][-1].update(loss=new_value, grad=contribution, transient=True)
        return loss

    train_module._batched_train_step = guarded_step

    def pre_step(optimizer, args, kwargs):
        STATE["snapshot"] = {"params": [p.detach().cpu().clone() for g in optimizer.param_groups for p in g["params"]],
                             "optimizer": copy.deepcopy(optimizer.state_dict())}

    def post_step(optimizer, args, kwargs):
        STATE["step"] += 1
        window = STATE["window"]
        say(f"epoch {STATE['epoch']} step {STATE['step']} sub-steps {window[0]['substep'] if window else '-'}"
            f"-{STATE['substep']}: |g| before clipping {STATE['last_norm']:.4g}; per structure "
            + " ".join(f"{e['index']}:{e['loss']:.3g}/{e['grad']:.3g}" for e in window))
        named = [(f"group{i}.param{j}", p) for i, g in enumerate(optimizer.param_groups) for j, p in enumerate(g["params"])]
        bad = _nonfinite(named)
        if bad:
            model = _MODEL.get("model")
            if model is None:
                raise NonFiniteTraining(f"nanguard: non-finite parameters after optimiser step: {bad[:12]}")
            fail("optimiser step", model, bad)
        STATE["window"] = []

    register_optimizer_step_pre_hook(pre_step)
    register_optimizer_step_post_hook(post_step)

    # the model, for a failure detected after an optimiser step (set at the first sub-step)
    def remember_model(*args, **kwargs):
        _MODEL["model"] = args[0]
        return guarded_step(*args, **kwargs)

    train_module._batched_train_step = remember_model


_MODEL = {}


if __name__ == "__main__":
    import train  # noqa: E402  (after the float64 default, as run_train.py does)

    config = os.path.abspath(sys.argv[1])
    # under torchrun every rank runs its own guard, so each dumps to a file of its own
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dump = "nanguard.pt" if world <= 1 else f"nanguard_rank{int(os.environ['RANK'])}.pt"
    install(int(os.environ.get("NANGUARD_SKIP_EPOCHS", "0")), os.path.join(os.path.dirname(config), dump), train)
    say(f"guard installed; torch {torch.__version__}; started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.argv = [os.path.join(HERE, "run_train_dump.py"), config]
    with open(sys.argv[0]) as handle:
        code = compile(handle.read(), sys.argv[0], "exec")
    exec(code, {"__name__": "__main__", "__file__": sys.argv[0]})
