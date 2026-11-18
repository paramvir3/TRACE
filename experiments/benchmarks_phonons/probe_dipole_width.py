"""Frozen-weights probe (2026-11-07 04:30): the induced-dipole width as the Born-charge lever at fixed eps_inf.
The gaussian_a scan showed Z* and eps_inf falling together as the SPEC 12 widths widen (both are the Lorentz
enhancement of the smeared dipoles); the physical width of the O2- dipole is its density width (~1.2-1.4 A), not the
stiffness-derived 0.52 A.  Here sigma_mu(O) is set directly (sigma_mu(Mg) kept at its SPEC 12 value), and for each
width the polarizability scale s (A_mu -> A_mu / s on every atom) is bisected so that eps_inf = 3.235 (PBE); Z*(Mg) at
that point is the prediction for a model trained under the ε∞ prior with that width.  Pre-registered reading: the width
is the lever if some sigma_mu(O) in 0.9-1.5 A gives Z*(Mg) in 1.90-2.10 at eps_inf 3.235.
Usage: python experiments/benchmarks_phonons/probe_dipole_width.py MODEL.pt [--a 4.265]"""
import argparse, dataclasses, sys, warnings
sys.path.insert(0, "."); sys.path.insert(0, "tools/electrostatics"); sys.path.insert(0, "experiments/benchmarks_phonons"); warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
from ase.build import bulk
from flashace.calculator import TransformersACECalculator
import trace_electrostatics.solver as SOL
import trace_electrostatics.energy as EN
from trace_electrostatics.solver import gaussian_multipole_width
from probe_sigma_scan import z_and_eps
from mgo_phonons import _edges

ORIG = SOL.gaussian_widths
OVERRIDE = {"sigma_mu_O": None, "sigma_mu_Mg": None, "species": None}


def patched_gaussian_widths(params, gaussian_a):
    smu, sth = ORIG(params, gaussian_a)
    if OVERRIDE["sigma_mu_O"] is not None:
        smu = torch.where(OVERRIDE["species"] == 8, torch.full_like(smu, OVERRIDE["sigma_mu_O"]), smu)
    if OVERRIDE["sigma_mu_Mg"] is not None:
        smu = torch.where(OVERRIDE["species"] == 12, torch.full_like(smu, OVERRIDE["sigma_mu_Mg"]), smu)
    return smu, sth


for modname in (SOL, EN):
    if hasattr(modname, "gaussian_widths"):
        modname.gaussian_widths = patched_gaussian_widths


CONFIG_PATCH = {}


def evaluate(mod, h, pos0, ei, keys, data, species, sigma_mu_O, alpha_scale, sigma_mu_Mg=None, species_alpha=None):
    """species_alpha: {Z: factor} multiplying that species' polarizability before the global scale.  CONFIG_PATCH
    (module level) is applied to the electrostatic config (e.g. multipole_kernel='thole', thole_a)."""
    OVERRIDE["sigma_mu_O"], OVERRIDE["sigma_mu_Mg"], OVERRIDE["species"] = sigma_mu_O, sigma_mu_Mg, species
    heads = mod.heads; orig = heads.forward
    f = torch.ones(len(species), dtype=torch.float64)
    for zz, fac in (species_alpha or {}).items():
        f = torch.where(species == zz, torch.full_like(f, fac), f)

    def scaled(*a, **k):
        out = orig(*a, **k)
        return out._replace(params=dataclasses.replace(out.params, A_mu=out.params.A_mu / (alpha_scale * f.view(-1, 1, 1))))
    heads.forward = scaled
    try:
        pos = pos0.detach().clone().requires_grad_(True); vec, L = _edges(pos, ei, keys)
        orig_config = mod.config
        if CONFIG_PATCH:
            mod.config = dataclasses.replace(mod.config, **CONFIG_PATCH)
        try:
            Z, eps, _, w = z_and_eps(mod, h.detach(), pos, ei, vec, L, keys, data, {})
        finally:
            mod.config = orig_config
    finally:
        heads.forward = orig; OVERRIDE["sigma_mu_O"] = None; OVERRIDE["sigma_mu_Mg"] = None
    z = species
    return float(torch.diagonal(Z[z == 12].mean(0)).mean()), float(torch.diagonal(eps).mean()), w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--a", type=float, default=4.265); ap.add_argument("--eps", type=float, default=3.235)
    args = ap.parse_args()
    calc = TransformersACECalculator(model_path=args.model, device="cpu"); mod = calc.model.electrostatics
    cell = bulk("MgO", "rocksalt", a=args.a, cubic=True).repeat((2, 2, 2)); cell.info = {"charge": 0.0}; cell.calc = calc
    cap = []; hk = mod.register_forward_pre_hook(lambda m, a, k: cap.append((a, k)) or None, with_kwargs=True); cell.get_potential_energy(); hk.remove()
    (h, pos0, ei, vec0, L0), kw = cap[0][0][:5], cap[0][1]; keys = {k: kw[k] for k in ("pbc", "cell", "edge_shift") if k in kw}; data = kw["data"]
    species = torch.as_tensor(cell.numbers)
    print(f"=== {args.model.split('/')[-2]}, a = {args.a}, 64 atoms; target eps_inf {args.eps}")
    zmg, e, w = evaluate(mod, h, pos0, ei, keys, data, species, None, 1.0)
    print(f"as trained: sigma_mu O {w[0]:.3f}, Mg {w[1]:.3f} A; Z*(Mg) {zmg:+.4f}, eps_inf {e:.4f}", flush=True)
    import os
    from trace_electrostatics.constants import K_E
    # the species polarizabilities as trained: alpha = K_E ell^2 / A_mu (isotropic part)
    mod_heads = mod.heads
    pos = pos0.detach().clone(); vec, L = _edges(pos, ei, keys)
    from trace_redox.bound import assemble_neutral
    inputs = mod._inputs(pos, data, keys.get("pbc"))
    asm = assemble_neutral(h.detach(), pos, ei, vec, L, mod_heads, mod.config, pbc=inputs["pbc"], cell=keys.get("cell"), edge_shift=keys.get("edge_shift"), quadrupoles=inputs["quadrupoles"], species=inputs["species"], external_field=None)
    Amu = asm.params.A_mu; ell = asm.params.ell
    alpha = K_E * ell ** 2 / torch.diagonal(Amu, dim1=-2, dim2=-1).mean(-1)
    print(f"trained polarizabilities: alpha_O {float(alpha[species == 8].mean()):.3f} A^3, alpha_Mg {float(alpha[species == 12].mean()):.3f} A^3 (Clausius-Mossotti O2- in MgO ~1.7-1.9)", flush=True)
    aO, aMg = float(alpha[species == 8].mean()), float(alpha[species == 12].mean())
    scan = os.environ.get("SCAN", "alpha")
    if scan == "o":
        cases = [(f"O width {smu}", smu, None, None) for smu in (0.524, 0.9, 1.1, 1.3, 1.5)]
    elif scan == "mg":
        cases = [(f"Mg width {smg}", None, smg, None) for smg in (1.103, 1.6, 2.0, 2.6, 3.2)] + [("both widths 1.3 / 2.6", 1.3, 2.6, None)]
    elif scan == "thole":   # SPEC 1.6 Thole damping (distance dependent, u = r / (alpha_i alpha_j)^(1/6)) on the periodic path
        cases = [(f"thole a = {a}", None, None, None, {"multipole_kernel": "thole", "thole_a": a}) for a in (0.39, 0.2, 0.1, 0.05, 0.6, 0.8)]
        cases = [("gaussian (as trained)", None, None, None, {})] + cases
        for label, smu, smg, sa, patch in cases:
            CONFIG_PATCH.clear(); CONFIG_PATCH.update(patch)
            lo, hi = 0.2, 8.0; zmg = e = float("nan"); w = (float("nan"), float("nan"))
            for it in range(16):
                sc = 0.5 * (lo + hi)
                try:
                    zmg, e, w = evaluate(mod, h, pos0, ei, keys, data, species, smu, sc, sigma_mu_Mg=smg, species_alpha=sa)
                except Exception as err:
                    print(f"{label}: refused at scale {sc:.3f} ({type(err).__name__}: {str(err)[:110]})", flush=True); zmg = e = float("nan"); break
                if e < args.eps: lo = sc
                else: hi = sc
                if abs(e - args.eps) < 2e-3: break
            print(f"{label}: sigma_mu O {w[0]:.3f}, Mg {w[1]:.3f} A; global alpha scale {sc:.3f} -> eps_inf {e:.4f}, Z*(Mg) {zmg:+.4f}   | PBE 1.992", flush=True)
        CONFIG_PATCH.clear(); return
    else:   # the species split of the polarizability, at the trained widths and at the ion-sized O width
        cases = [("as trained (alpha_O 0.2, alpha_Mg 1.9)", None, None, None),
                 ("physical split (alpha_O 1.9, alpha_Mg 0.2)", None, None, {8: 1.9 / aO, 12: 0.2 / aMg}),
                 ("all on O (alpha_O 2.1, alpha_Mg 0.0001)", None, None, {8: 2.1 / aO, 12: 1e-4 / aMg}),
                 ("even (1.05 / 1.05)", None, None, {8: 1.05 / aO, 12: 1.05 / aMg}),
                 ("physical split + O width 1.3 A", 1.3, None, {8: 1.9 / aO, 12: 0.2 / aMg}),
                 ("all on O + O width 1.3 A", 1.3, None, {8: 2.1 / aO, 12: 1e-4 / aMg})]
    for label, smu, smg, sa in cases:
        lo, hi = 0.2, 8.0
        for it in range(16):                                       # bisection on the global polarizability scale for eps_inf
            s = 0.5 * (lo + hi)
            try:
                zmg, e, w = evaluate(mod, h, pos0, ei, keys, data, species, smu, s, sigma_mu_Mg=smg, species_alpha=sa)
            except Exception as err:
                print(f"{label}: refused at scale {s:.3f} ({type(err).__name__}: {str(err)[:90]})", flush=True); zmg = e = float("nan"); break
            if e < args.eps: lo = s
            else: hi = s
            if abs(e - args.eps) < 2e-3: break
        print(f"{label}: sigma_mu O {w[0]:.3f}, Mg {w[1]:.3f} A; global alpha scale {s:.3f} -> eps_inf {e:.4f}, Z*(Mg) {zmg:+.4f}   | PBE 1.992", flush=True)


if __name__ == "__main__":
    main()
