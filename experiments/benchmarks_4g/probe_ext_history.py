"""Is the doped 2x2 energy at Gamma a function of the structure alone?  The same cell evaluated in different orders
in one process (frozen weights; 'onsite', clusters with the closure, 'fermi'; reference geometry, not relaxed).
Found because U3's unit-mesh control and U2 gave E(2x2)/4 - E(1x1) = +1791.512941 and +1791.5162 meV for the same
non-wetting cell.  Nothing in the package or a run directory changes.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_history.py MODEL.pt [geometry | u3]

"threads" evaluates the non-wetting 2x2 cell at Gamma with 1, 2, 3, 4, 6 and 8 torch threads (different
round-off in the threaded linear algebra): a discrete selection sitting on a tie would show as a jump.
"u3" repeats the order of probe_ext_fermi_unit.py's U3 block (both geometries at bvk_length 18, then both at the
unit mesh), forces included as there.
"""
import dataclasses
import hashlib
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
geometry = sys.argv[2] if len(sys.argv) > 2 else "non-wetting"
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
fermi = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
bvk = dataclasses.replace(fermi, bvk_length=18.0)
refs = references()


def tag(t):
    return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()[:8]


def run(label, rep, redox, geometry=geometry, forces=False):
    mod.redox = redox
    a = refs[("doped", geometry)].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset()
    e = a.get_potential_energy()
    if forces:
        a.get_forces()
    last = mod._ensemble_last
    sizes = [int(S.shape[0]) for S in last["clusters"]]
    print(f"   {label:34s} E/copy {e / (rep[0] * rep[1]):.9f} eV   n0 {tag(last['reference'])}  clusters {sizes} "
          f"{' '.join(tag(S) for S in last['clusters'])}", flush=True)


print(f"model {model_path.split('/')[-2]}; doped {geometry}; 'onsite', clusters, 'fermi'; frozen weights")
if geometry == "threads":
    for k in (1, 2, 3, 4, 6, 8, 4):
        torch.set_num_threads(k)
        run(f"non-wetting 2x2 Gamma, {k} thread(s)", (2, 2, 1), fermi, "non-wetting", True)
    sys.exit(0)
if geometry == "u3":
    for name, redox in (("bvk_length 18", bvk), ("unit mesh", fermi)):
        for g in ("wetting", "non-wetting"):
            run(f"{name}, {g} 1x1", (1, 1, 1), redox, g, True)
            run(f"{name}, {g} 2x2", (2, 2, 1), redox, g, True)
    sys.exit(0)
run("2x2 Gamma, first call", (2, 2, 1), fermi)
run("2x2 Gamma, repeated", (2, 2, 1), fermi)
run("1x1 Gamma", (1, 1, 1), fermi)
run("2x2 Gamma, after the 1x1", (2, 2, 1), fermi)
run("1x1 at bvk_length 18 (mesh 2,2,1)", (1, 1, 1), bvk)
run("2x2 at bvk_length 18 (mesh 1,1,1)", (2, 2, 1), bvk)
run("2x2 Gamma, after the bvk calls", (2, 2, 1), fermi)
run("1x1 Gamma, again", (1, 1, 1), fermi)
run("2x2 Gamma, last", (2, 2, 1), fermi)
