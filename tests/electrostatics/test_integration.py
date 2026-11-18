"""Opt-in integration with flashace -- SPEC section 5.

    Electrostatics must be **opt-in** and default OFF so every existing test
    passes unchanged.  `trace_electrostatics` must NOT import `flashace`.

Two separable claims, both tested here:

  * the **one-way dependency** -- checked in a fresh interpreter, because an
    import that happened earlier in this process would mask it; and
  * **default off** -- a TRACE model that nobody has called
    ``attach_electrostatics`` on must produce the same energies, forces and
    stress as the model that existed before this package did.

The second claim is what makes "every existing test passes unchanged" a fact
rather than a hope.  It is asserted against the potential's *own* measured
process-to-process spread rather than bitwise, because the potential is not
bitwise reproducible across processes to begin with (1 ULP of threaded
reduction order; see the test for the measurement).
"""

import subprocess
import sys

import pytest
import torch

from conftest import require


def _model():
    """The same small V3 model ``tests/test_architecture_v3.py`` uses."""
    TransformersACEV3 = pytest.importorskip("flashace.model").TransformersACEV3
    torch.manual_seed(23)
    return TransformersACEV3(
        r_max=4.5,
        l_max=2,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_order=4,
        correlation_channels=4,
        attention_num_heads=2,
    ).eval()


def _data():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
    )
    n_atoms = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(n_atoms):
        for sender in range(n_atoms):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return {
        "z": torch.tensor([55, 82, 53, 53], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3) * 9.0,
        "edge_index": torch.tensor([senders, receivers], dtype=torch.long),
        "edge_shift": torch.zeros((len(senders), 3)),
        "volume": torch.tensor(729.0),
    }


# ---------------------------------------------------------------------------
# SPEC 5: the one-way dependency
# ---------------------------------------------------------------------------
def test_trace_electrostatics_does_not_import_flashace():
    """SPEC 5: "``trace_electrostatics`` must NOT import ``flashace``".

    Run in a subprocess: this process has already imported flashace for the
    tests below, so an in-process ``sys.modules`` check would pass vacuously.
    """
    script = (
        "import sys, torch; torch.set_default_dtype(torch.float64);"
        "import trace_electrostatics;"
        "print(sorted({m.split('.')[0] for m in sys.modules}"
        " & {'flashace', 'transformers_ace'}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", (
        "importing trace_electrostatics pulled in the potential: "
        f"{completed.stdout.strip()}"
    )


def test_importing_the_package_does_not_change_the_default_dtype():
    """A package that called ``set_default_dtype`` at import time would change
    the numerics of every existing test just by being importable.

    ``trace_electrostatics`` documents that the *caller* must set float64; it
    must not do so itself, and it must not set anything else either.
    """
    script = (
        "import torch; before = torch.get_default_dtype();"
        "import trace_electrostatics;"
        "print(before is torch.get_default_dtype())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"


# ---------------------------------------------------------------------------
# SPEC 5: default off
# ---------------------------------------------------------------------------
def test_a_fresh_model_has_no_electrostatics():
    """SPEC 5: "A TRACE model has no electrostatics until
    ``attach_electrostatics`` is called on it"."""
    model = _model()
    assert getattr(model, "electrostatics", None) is None


@pytest.mark.slow
def test_energy_forces_and_stress_are_unchanged_with_the_package_imported():
    """SPEC 5: "so every existing test passes unchanged".

    The comparison is against subprocesses that never import
    ``trace_electrostatics`` at all, so this catches an import-time side effect
    (a dtype change, a seeded RNG draw, a monkeypatch) that an in-process
    comparison could not.

    It cannot be a *bitwise* comparison, because the potential is not bitwise
    reproducible across processes in the first place: running the clean script
    below three times in a row gives 0x1.c3705b92f2ab7p+3, ...ab6p+3,
    ...ab7p+3 -- a 1 ULP spread from threaded reduction order, with no
    electrostatics anywhere in the process.  The test therefore measures that
    run-to-run spread itself and requires the electrostatics-imported run to
    fall inside it (with a 4 ULP floor, so a run where the two clean processes
    happen to agree exactly does not make the assertion infinitely strict).
    """
    script = r"""
import torch
torch.set_default_dtype(torch.float64)
from flashace.model import TransformersACEV3
torch.manual_seed(23)
model = TransformersACEV3(r_max=4.5, l_max=2, num_radial=4, hidden_dim=8,
                          num_layers=1, correlation_order=4,
                          correlation_channels=4, attention_num_heads=2).eval()
pos = torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.7],
                    [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]])
n = pos.shape[0]
s, r = [], []
for rec in range(n):
    for snd in range(n):
        if snd != rec:
            s.append(snd); r.append(rec)
data = {"z": torch.tensor([55, 82, 53, 53], dtype=torch.long), "pos": pos,
        "cell": torch.eye(3) * 9.0,
        "edge_index": torch.tensor([s, r], dtype=torch.long),
        "edge_shift": torch.zeros((len(s), 3)),
        "volume": torch.tensor(729.0)}
energy, forces, stress, _ = model(data, training=False, compute_stress=True)
print(repr([float(energy.detach())]
           + forces.detach().reshape(-1).tolist()
           + stress.detach().reshape(-1).tolist()))
"""

    def clean_run():
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        return torch.tensor(eval(completed.stdout.strip()))  # noqa: S307

    first, second = clean_run(), clean_run()
    # The potential's own process-to-process spread, measured not assumed.
    reproducibility = (first - second).abs()

    require("energy", "attach_electrostatics")  # the package is imported here
    model = _model()
    energy, forces, stress, _ = model(_data(), training=False, compute_stress=True)
    observed = torch.tensor(
        [float(energy.detach())]
        + forces.detach().reshape(-1).tolist()
        + stress.detach().reshape(-1).tolist()
    )

    # Floor under the measured spread, for the case where the two clean runs
    # happen to agree exactly and `reproducibility` collapses to zero.
    #
    # 64 ULP, not 4.  Two samples badly underestimate the tail of the threaded
    # reduction wobble: a 4 ULP floor failed on roughly half of runs, and the
    # failure was reproduced between two clean subprocesses that never import
    # trace_electrostatics (worst excess 3.564e-18 on a near-zero stress
    # component, about 12 ULP there).  64 ULP is ~1.4e-14 relative -- orders
    # below any change the electrostatics could physically make, so the test
    # still catches what it is for.
    floor = 64.0 * torch.finfo(torch.float64).eps * first.abs().clamp(min=1.0e-12)
    budget = torch.maximum(reproducibility, floor)
    excess = (observed - first).abs() - budget
    assert float(excess.max()) <= 0.0, (
        "importing trace_electrostatics moved the potential by more than its "
        f"own run-to-run spread: worst excess {float(excess.max()):.3e}"
    )


def test_attach_and_detach_round_trip():
    """SPEC 5 / ``attach_electrostatics``: "The only supported way to do so"."""
    attach, detach = require(
        "energy", "attach_electrostatics", "detach_electrostatics"
    )
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticModule = require("energy", "ElectrostaticModule")

    model = _model()
    heads = ElectrostaticHeads(irreps_in=str(model.attention_irreps), r_max=model.r_max)
    module = attach(model, heads)
    assert isinstance(module, ElectrostaticModule)
    assert getattr(model, "electrostatics", None) is module

    detach(model)
    assert getattr(model, "electrostatics", None) is None
    detach(model)  # idempotent
    assert getattr(model, "electrostatics", None) is None


def test_attach_returns_a_module_whose_parameters_are_separable():
    """``attach_electrostatics`` "returns the module it attached, so the caller
    can put its parameters in a separate optimiser group".

    That only works if the returned module actually owns parameters, and if
    they are disjoint from the potential's.
    """
    attach = require("energy", "attach_electrostatics")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    model = _model()
    potential_parameters = {id(p) for p in model.parameters()}
    module = attach(model, ElectrostaticHeads(irreps_in=str(model.attention_irreps), r_max=model.r_max))
    electrostatic_parameters = {id(p) for p in module.parameters()}
    assert electrostatic_parameters, "the electrostatics owns no parameters"
    assert not (electrostatic_parameters & potential_parameters)


def test_the_attached_module_evaluates_at_the_documented_call_site():
    """docs/electrostatics.md section 1 puts the call immediately after
    ``E = torch.sum(self.readout(h))``, with ``h``, ``pos``, ``edge_index``,
    ``edge_vec`` and ``edge_len`` already in scope.

    This reproduces that call site against a real model's features, so the
    documented three-line diff is known to typecheck against the real shapes
    rather than only against the SPEC.
    """
    attach = require("energy", "attach_electrostatics")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")

    model = _model()
    data = _data()
    module = attach(model, ElectrostaticHeads(irreps_in=str(model.attention_irreps), r_max=model.r_max))

    pos = data["pos"].clone().requires_grad_(True)
    edge_index = data["edge_index"]
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_len = edge_vec.norm(dim=1)
    features = model.ace(model.emb(data["z"]), edge_index, edge_vec, edge_len)[0]
    for layer_index, layer in enumerate(model.layers):
        _ = layer_index  # the first layer's output is enough for a shape check
        break
    assert features.shape[1] == model.attention_irreps.dim

    energy, result = module(
        features, pos, edge_index, edge_vec, edge_len, data=None
    )
    assert torch.isfinite(energy)
    # SPEC 5: forces follow automatically, so the term must be differentiable
    # with respect to the same ``pos`` the potential differentiates.
    (gradient,) = torch.autograd.grad(energy, pos)
    assert torch.isfinite(gradient).all()
    # SPEC 2.1 item 5: the solve returns a feasible point.
    assert abs(float(result.q.sum())) < 1.0e-10


@pytest.mark.slow
def test_the_existing_flashace_architecture_suite_still_passes():
    """SPEC 5: "so every existing test passes unchanged".

    Runs the repo's own V3 architecture suite in a subprocess.  It is the suite
    most likely to notice a perturbation of the potential, and it is cheap.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_architecture_v3.py",
            "tests/test_stress_finite_difference.py",
            "-q",
            "-p",
            "no:randomly",
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    assert completed.returncode == 0, (
        "the pre-existing flashace suite does not pass:\n"
        + completed.stdout[-4000:]
    )


def _repo_root():
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2])
