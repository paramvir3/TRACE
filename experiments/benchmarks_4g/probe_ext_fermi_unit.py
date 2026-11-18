"""C4 stage 2 scored on the benchmark cells (2026-11-11): the Fermi band unit, ``band_form='fermi'``, frozen weights.

    python probe_ext_fermi_unit.py MODEL.pt [U1] [U2] [U3] [U4] [nx ny ...]

U1  the doped 1x1 cells at Gamma, 'fermi' against the legacy 'phi1' (pre-registered |dE| <= 1e-4 eV).
U2  the doped 2x1 and 2x2 cells evaluate; forces against a fourth-order central difference of the energy (step 5e-3 A,
    the atomic-energy baseline left out of the differenced energies) on three components fixed before the run: the
    first two Au atoms along z and the first Al along z (pre-registered 1e-6 relative).
U3  ``bvk_length = 18.0`` A: the 1x1 cell (mesh (2,2,1)) against its own 2x2 replication (mesh (1,1,1)) on both doped
    geometries (pre-registered |E(2x2)/4 - E(1x1)| <= 1e-3 meV, max|dF| <= 1e-6 eV/A); the unit mesh beside it.
U4  the gradient of sum |F|^2 with respect to the band's hopping parameter on a doped cell (1x1, then 2x1 with four
    band electrons): a double backward against a fourth-order central difference of the loss (step 1e-4).

``state_polarizability_stiffness = 'onsite'`` and ``diabatic_space = 'clusters'`` as in R16-R18.  Reference geometries,
unrelaxed.  Writes nothing into the run directory."""
import dataclasses
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
flags = [a for a in sys.argv[2:] if a.startswith("U")]
numbers = [int(a) for a in sys.argv[2:] if a.isdigit()]
reps = [(numbers[i], numbers[i + 1], 1) for i in range(0, len(numbers) - 1, 2)] or [(2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
base = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
fermi = dataclasses.replace(base, band_form="fermi")
REC = {}
mod.register_forward_hook(lambda module, inputs, output: REC.update(record=output[1]))
refs = references()
STEP = 5.0e-3                     # A, the displacement of U2's fourth-order central difference (fixed before the run)
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space; frozen weights; kT {mod.redox.kT} eV")


def single(key, rep, redox, forces=True):
    mod.redox = redox
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); REC.clear(); t0 = time.time()
    e = a.get_potential_energy()
    f = a.get_forces() if forces else None
    return a, e, f, time.time() - t0


def band_line(a):
    rec, last = REC["record"], mod._band_last
    z = a.get_atomic_numbers()
    au = " ".join(f"{float(rec.free_charge[i]):+.3f}" for i in range(len(a)) if z[i] == 79)
    line = f"band holds {-float(rec.unit_charge[-1]):.4f} e; Au free charges {au}; configurations {rec.iterations}"
    if last is not None and "reference" in last:
        pops = " ".join(f"{float(x):.4f}" for x in last["populations"])
        line += f"; N0 {last['reference']} in {last['passes']} pass(es); band-state populations {pops}"
    return line


if "U1" in flags:
    print("U1  doped 1x1 at Gamma, 'fermi' against 'phi1'")
    for geometry in ("wetting", "non-wetting"):
        out = {}
        for name, redox in (("phi1", base), ("fermi", fermi)):
            a, e, f, dt = single(("doped", geometry), (1, 1, 1), redox)
            out[name] = (e, f)
            print(f"   {geometry:11s} {name:5s}: E {e:.8f} eV; {band_line(a)}  ({dt:.0f} s)", flush=True)
        print(f"== {geometry}: E(fermi) - E(phi1) = {out['fermi'][0] - out['phi1'][0]:+.3e} eV; "
              f"max|dF| = {np.abs(out['fermi'][1] - out['phi1'][1]).max():.3e} eV/A", flush=True)

if "U2" in flags:
    print("U2  the replicated doped cells with 'fermi': evaluation, and forces against differences of the energy")
    for geometry in ("wetting", "non-wetting"):
        _, e1, f1, _ = single(("doped", geometry), (1, 1, 1), fermi)
        for rep in reps:
            copies = rep[0] * rep[1]
            a, e, f, dt = single(("doped", geometry), rep, fermi)
            z = a.get_atomic_numbers()
            print(f"   {geometry:11s} {rep[0]}x{rep[1]}: E/cell - E(1x1) {1e3 * (e / copies - e1):+.4f} meV; "
                  f"max|F - F(1x1)| {np.abs(f - np.tile(f1, (copies, 1))).max():.3e} eV/A; {band_line(a)}  ({dt:.0f} s)", flush=True)
            gold = [i for i in range(len(a)) if z[i] == 79][:2]
            al = [i for i in range(len(a)) if z[i] == 13][0]
            keep = calc.atomic_energy_tensor
            calc.atomic_energy_tensor = torch.zeros_like(keep)                 # differences without the -1.5e6 eV baseline
            try:
                for i, axis in ((gold[0], 2), (gold[1], 2), (al, 2)):
                    es = []
                    for s in (-2, -1, 1, 2):
                        b = a.copy(); b.info = {"charge": 0.0}; b.positions[i, axis] += s * STEP
                        mod.redox = fermi; b.calc = calc; calc.reset()
                        es.append(b.get_potential_energy())
                    fd = -(es[0] - 8 * es[1] + 8 * es[2] - es[3]) / (12 * STEP)
                    fd2 = -(es[2] - es[1]) / (2 * STEP)
                    print(f"      atom {i} (Z {z[i]}) axis {axis}: F {f[i, axis]:+.10f}  -dE/dR {fd:+.10f}  difference {f[i, axis] - fd:+.2e} "
                          f"eV/A  relative {abs(f[i, axis] - fd) / abs(fd):.1e}  (second-order stencil {fd2:+.10f})", flush=True)
            finally:
                calc.atomic_energy_tensor = keep

if "U3" in flags:
    print("U3  extensivity at a fixed sampling length, doped reference geometries, 'fermi'")
    for label, redox in (("bvk_length = 18.0", dataclasses.replace(fermi, bvk_length=18.0)), ("unit mesh", fermi)):
        for geometry in ("wetting", "non-wetting"):
            a1, e1, f1, t1 = single(("doped", geometry), (1, 1, 1), redox); m1 = tuple(REC["record"].bvk_mesh)
            line1 = band_line(a1)
            a4, e4, f4, t4 = single(("doped", geometry), (2, 2, 1), redox); m4 = tuple(REC["record"].bvk_mesh)
            df = np.abs(f4 - np.tile(f1, (4, 1))).max()
            print(f"   {label:18s} {geometry:11s}: mesh 1x1 {m1}, 2x2 {m4};  E(2x2)/4 - E(1x1) = {1e3 * (e4 / 4 - e1):+.6f} meV;  "
                  f"max|dF| = {df:.2e} eV/A;  E(1x1) {e1:.6f} eV  ({t1:.0f} s, {t4:.0f} s)\n      1x1: {line1}\n      2x2: {band_line(a4)}", flush=True)

if "U4" in flags:
    print("U4  d sum|F|^2 / d(parameter) on a doped cell, 'fermi': autograd through the forces against central differences")
    DATA = {}
    handle = calc.model.register_forward_pre_hook(lambda module, args, kwargs: DATA.update(data=args[0]), with_kwargs=True)
    targets = [("band.log_factor", mod.band.log_factor)]
    for geometry, rep in ((("wetting"), (1, 1, 1)), (("wetting"), (2, 1, 1))):
        a, e, f, dt = single(("doped", geometry), rep, fermi)
        data = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in DATA["data"].items()}
        print(f"   {geometry} {rep[0]}x{rep[1]}: {band_line(a)}", flush=True)

        def loss(training):
            mod.redox = fermi
            _, F, _, _ = calc.model(dict(data), training=training, compute_stress=False)
            return (F ** 2).sum()

        t0 = time.time()
        L = loss(True)
        grads = torch.autograd.grad(L, [p for _, p in targets], allow_unused=True)
        t1 = time.time() - t0
        for (name, p), g in zip(targets, grads):
            d = float(os.environ.get("U4_STEP", "1.0e-4"))      # the stencil's step in the parameter
            def shifted(delta, p=p):                     # the forces need autograd: only the shift is under no_grad
                with torch.no_grad():
                    p.add_(delta)
                try:
                    return float(loss(False))
                finally:
                    with torch.no_grad():
                        p.sub_(delta)
            Lp, Lm, Lpp, Lmm = shifted(d), shifted(-d), shifted(2 * d), shifted(-2 * d)
            fd = (Lmm - 8 * Lm + 8 * Lp - Lpp) / (12 * d)
            g = float("nan") if g is None else float(g)
            print(f"      {name}: sum|F|^2 {float(L):.8f}; autograd {g:+.10e}  differences {fd:+.10e}  relative {abs(g - fd) / abs(fd):.1e}  "
                  f"(double backward {t1:.0f} s)", flush=True)
    handle.remove()
