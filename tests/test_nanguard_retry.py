"""The training guard's recomputation of a non-finite sub-step (experiments/benchmarks_4g/nanguard_retry.py).

Four AuMgO trainings (step 11 seed 42, step 12 seed 43, step 13 seed 42; 2026-10-16..18) stopped on a sub-step whose
loss was finite but whose gradients were NaN, and the guard's exact replay -- same weights, inputs and RNG state, in
the same process -- was finite each time.  The saved gradients localise the entry point: in all three dumps
the free-charge widths' gradient (electrostatics.width.log_factor) is NaN on exactly the elements present; in step 11
nothing else is, so the NaN arose inside a backward fed by finite gradients (hypothesis: a transient fault of the
BLAS under oversubscription; not reproduced at load 7-10, 18 000 bitwise-identical GEMMs).

A sub-step whose result is not reproducible is recomputed from the gradients accumulated before it and its RNG state:
the recomputation is exactly what an unfaulted run computes.  A fault that reproduces still stops the run.
"""
import importlib.util
import math
import os

import torch

PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "benchmarks_4g", "nanguard_retry.py")


def _module():
    spec = importlib.util.spec_from_file_location("nanguard_retry", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))


def _step_factory(faults):
    """A sub-step: loss = sum(a^2) * b + noise from the global RNG; its first ``faults`` calls return NaN gradients."""
    calls = {"n": 0}

    def step(model, scale):
        calls["n"] += 1
        noise = torch.rand((), dtype=torch.float64)
        loss = scale * (model.a * model.a).sum() * model.b + noise
        loss.backward()
        if calls["n"] <= faults:
            model.a.grad.fill_(float("nan"))
        return loss.detach()

    return step, calls


def _run(faults, retries):
    nr = _module()
    model = _Model()
    model.a.grad = torch.tensor([0.5, -0.5], dtype=torch.float64)       # accumulated by earlier sub-steps
    before = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    rng = torch.get_rng_state()
    step, calls = _step_factory(faults)
    first = step(model, 2.0)
    loss, attempts = nr.recompute_nonfinite(step, model, (model, 2.0), {}, rng, before, retries)
    return nr, model, first, loss, attempts, calls


def test_a_transient_fault_is_recomputed_exactly():
    _, model, first, loss, attempts, calls = _run(faults=1, retries=1)
    assert attempts == 1 and calls["n"] == 2
    assert float(loss) == float(first)                                  # the same RNG state: the same loss, bitwise
    clean = _Model()
    clean.a.grad = torch.tensor([0.5, -0.5], dtype=torch.float64)
    (2.0 * (clean.a * clean.a).sum() * clean.b).backward()
    assert torch.equal(model.a.grad, clean.a.grad)                     # earlier sub-steps' gradient kept, once
    assert torch.equal(model.b.grad, clean.b.grad)                     # b had no gradient before: no double count


def test_a_reproducible_fault_is_reported():
    _, model, _, loss, attempts, calls = _run(faults=5, retries=2)
    assert loss is None and attempts == 2 and calls["n"] == 3
    assert not bool(torch.isfinite(model.a.grad).all())


def test_no_retries_means_the_old_behaviour():
    _, _, _, loss, attempts, calls = _run(faults=1, retries=0)
    assert loss is None and attempts == 0 and calls["n"] == 1


def test_a_finite_recomputation_needs_a_finite_loss():
    nr = _module()
    model = _Model()
    before = {}
    rng = torch.get_rng_state()

    def step(model):
        loss = (model.a * model.a).sum() * model.b
        loss.backward()
        return loss.detach() * float("inf")

    loss, attempts = nr.recompute_nonfinite(step, model, (model,), {}, rng, before, 1)
    assert loss is None and attempts == 1 and math.isinf(float(step(model)))
