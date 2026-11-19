"""C4 stage 2, the doped wet_d shift between the unit mesh and the 2x2 Born-von Karman supercell: which part of the
energy carries it.  Frozen weights; 'onsite' stiffness, clusters space (closure on), band_form 'fermi'; the doped
reference geometries (not relaxed).  Nothing in the package or a run directory changes.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_fermi_split.py MODEL.pt [cap=N]

cap=N sets RedoxConfig.fragment_max_configurations for this measurement only (default: the checkpoint's, 512);
the cap refuses, it never truncates, so a result that the default cap allowed does not depend on it.

Per geometry and cell (1x1 and 2x2, both at Gamma; E(2x2)/4 is what bvk_mesh = (2, 2, 1) returns, SPEC 37 B1):
    E        the total energy per cell
    rest     E - F, everything outside the ensemble
    F_b      the band unit's ladder entry at the reference filling N0 (SPEC 38: F_b(N0))
    E0 - F_b the reference configuration's energy without it (atoms, and the band-atom term at n0)
    clusters sum_c (F_c - E0)                (SPEC 36)
    pairs    F - E0 - sum_c (F_c - E0)       (SPEC 36's pair increments)
"""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
from trace_redox.diabatic import canonical_free_energy

model_path = sys.argv[1]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
cap = next((int(x.split("=")[1]) for x in sys.argv[2:] if x.startswith("cap=")), None)
if cap is not None:
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=cap)
refs = references()
CAP = {}
original = mod._cluster_solution


def capture(e_hat, states, *args, **kwargs):
    sol = original(e_hat, states, *args, **kwargs)
    CAP.update(e_hat=e_hat.detach(), states=states.detach(), F=sol.energy.detach(), calls=CAP.get("calls", 0) + 1)
    return sol


mod._cluster_solution = capture
NAMES = ("E", "rest", "F_b", "E0 - F_b", "clusters", "pairs")


def split(key, rep):
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); mod._ensemble_last = None; CAP.clear(); t0 = time.time()
    e = a.get_potential_energy()
    last = mod._ensemble_last
    kT = float(last["kT"]); copies = rep[0] * rep[1]
    with torch.no_grad():
        n0, Hs = last["reference"], [H.detach() for H in last["hamiltonians"]]
        E0 = Hs[0][0, 0]
        assert all(abs(float(H[0, 0] - E0)) < 1e-9 for H in Hs)            # every cluster starts at the reference
        assert all(bool((S[0] == n0).all()) for S in last["clusters"])
        cl = sum(canonical_free_energy(H, kT) - E0 for H in Hs)
        F = CAP["F"]
        column = int(torch.round(n0[-1].to(CAP["states"].dtype) - CAP["states"][0]))
        Fb = CAP["e_hat"][-1, column]                                       # the band unit is the last unit
        assert bool(torch.isfinite(Fb))
        out = dict(zip(NAMES, (e, e - float(F), float(Fb), float(E0 - Fb), float(cl), float(F - E0 - cl))))
    out = {k: v / copies for k, v in out.items()}
    print(f"   {key[1]:11s} {rep[0]}x{rep[1]}: clusters {len(Hs)} (sizes {[int(H.shape[0]) for H in Hs]}); band state at n0 {int(n0[-1]):+d}; "
          f"solver calls {CAP['calls']}; per cell (eV): " + "  ".join(f"{k} {v:+.6f}" for k, v in out.items())
          + f"  ({time.time() - t0:.0f} s)", flush=True)
    return out


print(f"model {model_path.split('/')[-2]}; doped reference geometries; 'onsite', clusters, 'fermi'; frozen weights; "
      f"cap {mod.redox.fragment_max_configurations}")
shift = {}
for geometry in ("wetting", "non-wetting"):
    one = split(("doped", geometry), (1, 1, 1))
    four = split(("doped", geometry), (2, 2, 1))
    shift[geometry] = {k: 1e3 * (four[k] - one[k]) for k in NAMES}
    print(f"== {geometry}: E(2x2)/4 - E(1x1), meV: " + "  ".join(f"{k} {shift[geometry][k]:+.4f}" for k in NAMES)
          + f"  (parts sum to {sum(shift[geometry][k] for k in NAMES[1:]):+.4f})", flush=True)
print("== wetting - non-wetting (the shift of wet_d at fixed geometry), meV: "
      + "  ".join(f"{k} {shift['wetting'][k] - shift['non-wetting'][k]:+.4f}" for k in NAMES))
print(time.strftime("done %Y-%m-%d %H:%M %Z"))
