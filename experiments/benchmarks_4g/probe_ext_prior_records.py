"""Do step 31's response-prior structures get an exact ensemble response record in the clusters space (SPEC 36 L5)?

train.py's dielectric and polarizability priors need ``RedoxResult.ensemble['exact']`` whenever a structure has more
than one configuration.  SPEC 36 refuses that record on several clusters.  This probe evaluates the two prior
structures of step 31 with the frozen model and prints the configuration count, the cluster sizes and the record.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_ext_prior_records.py MODEL.pt
"""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from ase.io import read
from flashace.calculator import TransformersACECalculator

D = "experiments/benchmarks_4g/data/"
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
stored = mod.redox
SEEN = {}


def record(module, inputs, output):
    SEEN["record"] = output[1]


mod.register_forward_hook(record)
mod.record_ensemble = True                                           # as train.py's priors do around their forward
settings = {
    "as stored": stored,
    "'onsite'": dataclasses.replace(stored, state_polarizability_stiffness="onsite"),
    "'onsite', clusters": dataclasses.replace(stored, state_polarizability_stiffness="onsite", diabatic_space="clusters"),
    "'onsite', clusters, 'fermi'": dataclasses.replace(stored, state_polarizability_stiffness="onsite",
                                                       diabatic_space="clusters", band_form="fermi"),
    "'onsite', clusters, 'fermi', (2,2,2)": dataclasses.replace(stored, state_polarizability_stiffness="onsite",
                                                                diabatic_space="clusters", band_form="fermi", bvk_mesh=(2, 2, 2)),
}
print(f"model {sys.argv[1].split('/')[-2]}")
for name, file in (("MgO bulk (dielectric prior)", "MgO_bulk_eps_prior.xyz"), ("Au atom (polarizability prior)", "Au_atom_alpha_prior.xyz")):
    for label, redox in settings.items():
        a = read(D + file); a.info = {"charge": 0.0}
        if "(2,2,2)" in label and not a.pbc.any():
            continue
        mod.redox = redox; a.calc = calc; calc.reset(); mod._ensemble_last = None; SEEN.clear()
        try:
            e = a.get_potential_energy()
        except Exception as err:
            print(f"   {name:32s} {label:38s}: REFUSED {type(err).__name__}: {str(err)[:150]}"); continue
        res = SEEN["record"]; last = mod._ensemble_last or {}
        sizes = sorted(int(H.shape[0]) for H in last["hamiltonians"]) if "hamiltonians" in last else None
        ens = res.ensemble
        note = "none (one configuration)" if ens is None else f"exact {ens['exact']}" + (f" ({ens['reason'][:70]})" if ens.get("reason") else "")
        if ens is not None and ens.get("exact"):                     # SPEC 36.10: the record's own polarisability (e^2 A^2 / eV)
            from trace_redox.response import record_polarizability
            a_b, a_e = ens["alpha_bound"].detach(), record_polarizability(ens).detach()
            note += (f"; terms {len(ens['terms']) if ens.get('terms') else 1}; alpha_bound (xx, yy, zz) "
                     + ", ".join(f"{float(a_b[i, i]):+.4f}" for i in range(3)) + "; alpha_ens "
                     + ", ".join(f"{float(a_e[i, i]):+.4f}" for i in range(3)))
        print(f"   {name:32s} {label:38s}: configurations {int(res.iterations):4d}; cluster sizes {sizes}; record: {note}; E {e:.6f} eV")
