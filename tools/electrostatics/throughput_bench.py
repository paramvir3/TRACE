"""Training-step cost of the electrostatics at foundation-dataset sizes (CPU).

What it answers: for a TRACE model of the size we train (water config: hidden
64, l_max 2, r_max 6, v2 attention), how much does one TRAINING step cost with
and without the electrostatics (default heads, SPEC 23 contact law on), and
where does the time go -- for

* molecules of 12-348 atoms (OMol25's range), neutral and charged;
* periodic water boxes of 96 and 192 atoms (OMat24-sized cells);
* a batch of 32 small molecules through ``forward_batched`` (one solve per
  structure in a Python loop) vs the same structures one at a time.

A training step = forward with forces (create_graph), a squared loss on energy
and forces, and backward.  Median of a few repeats after a warm-up.  float64
for both arms (the electrostatics requires it); the base model is also timed
in float32, its production precision.  Geometries are synthetic but physical:
water molecules at 0.96 A / 104.5 deg on a jittered 3.1 A grid.

Usage: python tools/electrostatics/throughput_bench.py [THREADS]
"""
from __future__ import annotations

import math
import statistics
import sys
import time
import warnings

import numpy as np
import torch

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from ase import Atoms  # noqa: E402

MODEL = dict(r_max=6.0, l_max=2, num_radial=12, hidden_dim=64, num_layers=1, correlation_order=4,
             correlation_channels=16, attention_num_heads=2, radial_mlp_hidden=32, radial_mlp_layers=2)


def water_cluster(n_molecules, seed=0, box=None):
    rng = np.random.default_rng(seed)
    half = math.radians(104.52) / 2.0
    mol = np.array([[0, 0, 0], [0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)],
                    [-0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)]])
    side = math.ceil(n_molecules ** (1 / 3))
    spacing = 3.1 if box is None else box / side
    pos, k = [], 0
    for i in range(side):
        for j in range(side):
            for l in range(side):
                if k == n_molecules:
                    break
                R = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed + k)))[0].numpy()
                centre = np.array([i, j, l]) * spacing + 0.15 * rng.standard_normal(3)
                pos.append(mol @ R.T + centre)
                k += 1
    atoms = Atoms("OH2" * n_molecules, positions=np.concatenate(pos))
    if box is not None:
        atoms.cell = [box] * 3
        atoms.pbc = True
        atoms.wrap()
    return atoms


def build(electrostatics, dtype):
    import train
    from flashace.model import TransformersACE

    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    model = TransformersACE(**MODEL)
    if electrostatics:
        settings = train.parse_electrostatics_config(
            {"r_max": MODEL["r_max"], "electrostatics": {"enabled": True}})
        train.attach_electrostatics_from_settings(model, settings)
    return model.train()


def item(atoms, dtype, charge=0.0):
    import train

    atoms = atoms.copy()
    if charge:
        atoms.info["charge"] = charge
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((len(atoms), 3)))
    ds = train.AtomisticDataset([atoms], MODEL["r_max"], dtype=dtype, electrostatics=True)
    return ds[0]


def step(model, data):
    E, F, _, _ = model(data, training=True, compute_stress=False)
    loss = (E ** 2).sum() + (F ** 2).sum()
    loss.backward()


def batched_step(model, items):
    import train

    batch = train.AtomisticDataset.collate_batched(items)
    E, F, _, _ = model.forward_batched(batch, training=True, compute_stress=False)
    ((E ** 2).sum() + (F ** 2).sum()).backward()


def timed(fn, repeats=3):
    fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def iterations(model, data):
    model.eval()
    _, _, _, aux = model(data, training=False, return_electrostatics=True)
    model.train()
    return int(aux["electrostatics"][0].solve_result.diagnostics.n_iter)


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    print(f"threads {torch.get_num_threads()}, torch {torch.__version__}")
    systems = [(f"(H2O)_{n} molecule", water_cluster(n), 0.0) for n in (4, 16, 40, 80, 116)]
    systems.append(("(H2O)_40 + charge -1 (as anion)", water_cluster(40), -1.0))
    systems += [(f"water box {3 * n} atoms (PBC)", water_cluster(n, box=box), 0.0)
                for n, box in ((32, 9.86), (64, 12.42))]

    base64 = build(False, torch.float64)
    elec = build(True, torch.float64)
    base32 = build(False, torch.float32)
    torch.set_default_dtype(torch.float64)
    print(f"{'system':34s} {'atoms':>5s} {'base fp32':>10s} {'base fp64':>10s} {'+elec fp64':>11s} "
          f"{'elec share':>10s} {'CG it':>6s}")
    for name, atoms, charge in systems:
        d64 = item(atoms, torch.float64, charge)
        d32 = item(atoms, torch.float32, 0.0)
        t32 = timed(lambda: step(base32, d32))
        t64 = timed(lambda: step(base64, d64))
        te = timed(lambda: step(elec, d64))
        print(f"{name:34s} {len(atoms):5d} {t32:9.3f}s {t64:9.3f}s {te:10.3f}s "
              f"{(te - t64) / te:9.0%} {iterations(elec, d64):6d}", flush=True)

    small = [item(water_cluster(4, seed=s), torch.float64) for s in range(32)]
    t_loop = timed(lambda: [step(elec, d) for d in small], repeats=2)
    t_batch = timed(lambda: batched_step(elec, small), repeats=2)
    t_base = timed(lambda: batched_step(base64, small), repeats=2)
    print(f"\n32 x (H2O)_4 batched: base fp64 {t_base:.3f}s, +elec {t_batch:.3f}s "
          f"(one-at-a-time +elec {t_loop:.3f}s) -> elec share {(t_batch - t_base) / t_batch:.0%}")


if __name__ == "__main__":
    main()


def build_route(solver, dtype=torch.float64):
    """The electrostatics model with a chosen solver route (same seed, same weights)."""
    import train
    from flashace.model import TransformersACE

    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    model = TransformersACE(**MODEL)
    settings = train.parse_electrostatics_config(
        {"r_max": MODEL["r_max"], "electrostatics": {"enabled": True, "config": {"solver": solver}}})
    train.attach_electrostatics_from_settings(model, settings)
    return model.train()


def omat24(path, sizes=(10, 15, 30, 60, 80)):
    """Training-step cost on real OMat24 cells closest to the given sizes."""
    from ase.io import read

    frames = read(path, ":")
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    base = build(False, torch.float64)
    dense = build_route("dense")
    free = build_route("matrix_free")
    torch.set_default_dtype(torch.float64)
    print(f"{'cell':28s} {'atoms':>5s} {'volume':>7s} {'base':>7s} {'dense':>8s} {'matrix-free':>11s}")
    for n in sizes:
        atoms = min(frames, key=lambda a: abs(len(a) - n))
        d = item(atoms, torch.float64)
        row = [timed(lambda: step(m, d)) for m in (base, dense, free)]
        print(f"{atoms.get_chemical_formula():28s} {len(atoms):5d} {atoms.get_volume():7.1f} "
              f"{row[0]:6.2f}s {row[1]:7.2f}s {row[2]:10.2f}s", flush=True)


if __name__ == "__omat24__":
    pass
