"""R26: the sizes of the learned one-electron couplings (SPEC 17.6) against Harrison's ceiling (SPEC 27's B(r)).

For a frozen model and each structure: the couplings on every directed edge by element pair (range, and the largest
ratio |tau| / B(r), B = 3.24 hbar^2 / (m r^2)), and the largest off-diagonal element of the configuration
Hamiltonian(s) the model actually used.  Settings as stored in the checkpoint.  Nothing is changed.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_coupling_sizes.py MODEL.pt
"""
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from ase.data import chemical_symbols
from ase.io import read
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

HBAR2_OVER_ME = 7.61996                                               # eV A^2
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
CAP = {}
orig = mod.coupling.edge_values


def spy(inv, ei, el, species=None):
    out = orig(inv, ei, el, species=species)
    CAP.update(ei=ei.detach().numpy(), el=el.detach().numpy(), tau=out.detach().numpy())
    return out


mod.coupling.edge_values = spy
structures = {f"{d} {g}": a for (d, g), a in references().items()}
structures["MgO bulk prior (8 atoms)"] = read("experiments/benchmarks_4g/data/MgO_bulk_eps_prior.xyz")
print(f"model {sys.argv[1].split('/')[-2]}; coupling bound {mod.coupling.bound!r}; space {mod.redox.diabatic_space!r}")
for name, atoms in structures.items():
    a = atoms.copy(); a.set_constraint()
    charge = float(atoms.info.get("charge", atoms.info.get("total_charge", 0.0)))
    a.info = {"charge": charge}; a.calc = calc; calc.reset(); mod._ensemble_last = None
    try:
        e = a.get_potential_energy()
    except Exception as err:
        print(f"== {name}: REFUSED {type(err).__name__}: {str(err)[:160]}"); continue
    z, ei, el, tau = np.array(a.numbers), CAP["ei"], CAP["el"], CAP["tau"]
    last = mod._ensemble_last or {}
    hams = [last["H"]] if last.get("H") is not None else list(last.get("hamiltonians", []))
    off = max([float((H - torch.diag(torch.diagonal(H))).detach().abs().max()) for H in hams if H.shape[0] > 1] or [0.0])
    # step 32's Z1: how far the lowest state of any Hamiltonian lies below its reference configuration (row 0)
    low = min([float(torch.linalg.eigvalsh(H.detach())[0] - H.detach()[0, 0]) for H in hams] or [0.0])
    print(f"== {name}: charge {charge:+.0f}; configurations {[int(H.shape[0]) for H in hams]}; largest |off-diagonal| of H {off:.4f} eV; "
          f"lowest state below the reference {low:+.4f} eV")
    for za, zb in sorted({tuple(sorted((int(z[i]), int(z[j])))) for i, j in zip(ei[0], ei[1])}):
        m = ((z[ei[0]] == za) & (z[ei[1]] == zb)) | ((z[ei[0]] == zb) & (z[ei[1]] == za))
        if not m.any():
            continue
        ratio = np.abs(tau[m]) / (3.24 * HBAR2_OVER_ME / el[m] ** 2)
        k = int(np.argmax(ratio))
        print(f"   {chemical_symbols[za]:>2s}-{chemical_symbols[zb]:<2s}: {m.sum():5d} edges; tau in [{tau[m].min():+9.4f}, {tau[m].max():+9.4f}] eV; "
              f"largest |tau| / B(r) = {ratio[k]:7.3f} at r = {el[m][k]:.3f} A (tau {tau[m][k]:+.4f} eV); edges above B: {(ratio > 1).sum()}")
