"""The model's ionisation potential on the clean MgO slab, split exactly into the configuration-energy terms
(2026-10-23; a probe, no model change).

Usage (repo root): python experiments/benchmarks_4g/probe_hole_terms.py MODEL.pt

The clean six-layer 3x3 MgO(100) slab of probe_spectrum.py (the undoped reference with Au2 removed) is evaluated at
total charge 0 and +1.  IP = E0(+1) - E0(0), with E0 the ground state of H.  The ground configuration at +1 is one
O- (measured, probe_spectrum: 54 configurations, ground 'O+1'); its energy minus the frame's at charge 0 is split
into the terms of probe_dump_terms.config_terms (ladder, shifts, self, screened, overlap, ind26, alpha28, band), each
checked to sum to H's diagonal (1e-9), and the coherent part (E0 - the ground configuration's diagonal) is reported.
Compared with PBE: IP(Gamma) = 4.796 eV on the same slab (tools/electrostatics/mgo_pbe.py); the model gives 7.21.
"""
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import probe_dump_terms as PD  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402


def terms_at(calc, atoms, charge):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.info["charge"] = float(charge)
    a.calc = calc
    calc.reset()
    for k in ("O", "own", "ind", "aug", "excl", "aug_in"):
        PD.STORE.pop(k, None)
    cap = []
    hook = calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    a.get_potential_energy()
    hook.remove()
    mod = calc.model.electrostatics
    z = torch.as_tensor(a.numbers)
    out, configs, H, kT = PD.config_terms(mod, cap[-1], z)
    diag = torch.diagonal(H)
    c = int(torch.argmin(diag))
    E0 = float(torch.linalg.eigvalsh(H)[0])
    return {k: float(v[c]) for k, v in out.items()}, float(diag[c]), E0, configs[c], z


def main():
    PD._install()
    base = references()[("undoped", "wetting")]
    clean = base.copy()
    del clean[[i for i in AU]]
    clean.info = dict(base.info)
    clean.set_constraint()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        import dataclasses
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        t0, d0, E00, c0, z = terms_at(calc, clean, 0)
        t1, d1, E01, c1, _ = terms_at(calc, clean, 1)
        n = int(z.numel())
        from trace_redox.constants import formal_frame
        fr = formal_frame(z)
        hole = (c1[:n] != fr).nonzero().flatten().tolist()
        print(f"=== {os.path.basename(os.path.dirname(model))}: IP = E0(+1) - E0(0) = {E01 - E00:+.3f} eV "
              f"(PBE slab IP(Gamma) +4.796); ground configuration at +1 changes atoms {hole} "
              f"(Z {[int(z[j]) for j in hole]}, layer z {[round(float(clean.positions[j, 2]), 2) for j in hole]})")
        for k in sorted(set(t0) | set(t1)):
            print(f"  {k:12s} {t1.get(k, 0.0) - t0.get(k, 0.0):+9.3f} eV")
        print(f"  {'diag sum':12s} {d1 - d0:+9.3f} eV;  coherent (E0 - diag) at +1 {E01 - d1:+.3f}, at 0 {E00 - d0:+.3f}")


if __name__ == "__main__":
    main()
