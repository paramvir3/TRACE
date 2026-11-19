"""Robustness and exactness of the electrostatics on real periodic cells (OMat24 sample).

For every cell of an extxyz sample (``sample_aselmdb.py``): one forward of a
TRACE model with the electrostatics attached exactly as a NEW training run
builds it (``solver='auto'``, SPEC 23 contact law), with energy, forces and
stress.  Reports failures by exception type, CG iterations, the route taken,
and time per cell.  On a few cells (the most triclinic ones) the total forces
and stress are checked against central finite differences of the energy --
under strain for the stress (sigma = (1/V) dE/d(epsilon), symmetric strain).

Untrained weights: this measures whether the physics runs, converges and is
differentiated exactly on 82 elements and arbitrary cells -- not accuracy.

Usage: python tools/electrostatics/omat24_sweep.py SAMPLE.xyz [N_FD_CELLS]
"""
import collections
import os
import sys
import time
import traceback
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import throughput_bench as tb  # noqa: E402


def forward(model, atoms, stress=True):
    data = tb.item(atoms, torch.float64)
    E, F, S, aux = model(data, training=False, compute_stress=stress, return_electrostatics=True)
    out = aux["electrostatics"][0]
    return float(E), F.detach().numpy(), S.detach().numpy(), out


def fd_check(model, atoms, h=1e-4, n_force=3):
    """Central differences: forces on a few components, the full stress tensor."""
    E0, F, S, _ = forward(model, atoms)
    rng = np.random.default_rng(0)
    force_err = 0.0
    for _ in range(n_force):
        i, a = int(rng.integers(len(atoms))), int(rng.integers(3))
        up, down = atoms.copy(), atoms.copy()
        up.positions[i, a] += h
        down.positions[i, a] -= h
        fd = -(forward(model, up, False)[0] - forward(model, down, False)[0]) / (2 * h)
        force_err = max(force_err, abs(fd - F[i, a]))
    volume = atoms.get_volume()
    stress_fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            eps = np.zeros((3, 3))
            eps[a, b] = eps[b, a] = h / (1.0 if a == b else 2.0)
            values = []
            for sign in (1, -1):
                strained = atoms.copy()
                deformation = np.eye(3) + sign * eps
                strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
                values.append(forward(model, strained, False)[0])
            # dE/d(eps_ab) for a symmetric strain; off-diagonals carry both ab and ba.
            stress_fd[a, b] = stress_fd[b, a] = (values[0] - values[1]) / (2 * h) / volume
    return force_err, float(np.abs(stress_fd - S).max()), float(np.abs(S).max())


def main():
    from ase.io import read

    frames = read(sys.argv[1], ":")
    n_fd = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    model = tb.build(True, torch.float64).eval()
    torch.set_default_dtype(torch.float64)
    failures = collections.Counter()
    examples = {}
    iterations, routes, times = [], collections.Counter(), []
    for k, atoms in enumerate(frames):
        t0 = time.perf_counter()
        try:
            E, F, S, out = forward(model, atoms)
            assert np.isfinite(E) and np.all(np.isfinite(F)) and np.all(np.isfinite(S))
            iterations.append(int(out.solve_result.diagnostics.n_iter))
            routes["matrix_free" if out.state().assembly.lattice_operator is not None else "dense"] += 1
        except Exception as error:  # noqa: BLE001 - classify every failure
            key = f"{type(error).__name__}: {str(error).splitlines()[0][:110]}"
            failures[key] += 1
            examples.setdefault(key, (k, atoms.get_chemical_formula(), traceback.format_exc(limit=3)))
        times.append(time.perf_counter() - t0)
    n_ok = len(frames) - sum(failures.values())
    print(f"{len(frames)} cells: {n_ok} ok, {sum(failures.values())} failed; routes {dict(routes)}")
    if iterations:
        print(f"CG iterations: median {int(np.median(iterations))}, max {max(iterations)}; "
              f"time per cell median {np.median(times):.2f}s, max {max(times):.2f}s")
    for key, count in failures.most_common():
        k, formula, tb_text = examples[key]
        print(f"  {count:4d} x {key}   (first: cell {k}, {formula})")
    # The most triclinic cells: largest deviation of the cell angles from 90 degrees.
    skew = [max(abs(x - 90.0) for x in a.cell.cellpar()[3:]) for a in frames]
    for k in np.argsort(skew)[::-1][:n_fd]:
        atoms = frames[int(k)]
        try:
            f_err, s_err, s_max = fd_check(model, atoms)
            print(f"FD {atoms.get_chemical_formula():20s} {len(atoms):3d} atoms, angles "
                  f"{np.round(atoms.cell.cellpar()[3:], 1).tolist()}: |dF| {f_err:.1e} eV/A, "
                  f"|dS| {s_err:.1e} eV/A^3 (|S| {s_max:.1e})", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"FD {atoms.get_chemical_formula()}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
