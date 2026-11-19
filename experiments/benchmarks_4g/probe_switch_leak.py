# Diagnostic probe (2026-10-19; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_switch_leak.py CKPT [...]
"""How much of a model's finite-field polarisability is the SPEC 21 switch rather than the physics.

The whole model's alpha = -d2E/dF2 (finite fields of the total energy) includes the field dependence of the
amplitude switch sigma(e(F)) (e depends on the configuration energies, which shift with F).  The physical response of
the selected ensemble at fixed H is alpha_b + alpha_ens (trace_redox/response.py, SPEC 30).  Isolated Au2 in three
charge states at 2.47 A: both, and their difference (the switch's leak)."""
import sys, warnings
import numpy as np, torch
warnings.simplefilter("ignore"); sys.path.insert(0, "."); torch.set_default_dtype(torch.float64)
from ase import Atoms
from flashace.calculator import TransformersACECalculator
from trace_redox.response import ensemble_polarizability
K_E = 14.399645
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    cap = []
    calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    for q in (0.0, -1.0, 1.0):
        def energy(field):
            a = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.47]]); a.info = {"charge": q, "external_field": list(field)}
            a.calc = calc; calc.reset(); cap.clear(); return a.get_potential_energy()
        e0 = energy([0, 0, 0]); ens = cap[-1].ensemble
        out = []
        for ax in (2, 0):
            f = np.zeros(3); hh = 1e-3; f[ax] = hh
            ff = -(energy(f) - 2 * e0 + energy(-f)) / hh ** 2 * K_E
            if ens is not None and ens.get("exact"):
                dec = float((ens["alpha_bound"] + ensemble_polarizability(ens["H"], ens["M"], ens["kT"]))[ax, ax]) * K_E
            else:
                dec = float("nan")
            out.append((ff, dec))
        n = 1 if ens is None else int(ens["H"].shape[0]) if ens.get("exact") else -1
        print(f"{ck.split('/')[-2][-26:]} Au2 q={q:+.0f} ({n} configs): alpha_par whole-model {out[0][0]:8.2f} vs fixed-H {out[0][1]:8.2f} A^3; "
              f"alpha_perp {out[1][0]:6.2f} vs {out[1][1]:6.2f}", flush=True)
