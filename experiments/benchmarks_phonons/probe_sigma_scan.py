"""Frozen-weights probe (2026-11-07): is the dipole smearing the lever for Z*?  The model's +0.43 e above the formal
charge is the collective (Lorentz) local-field enhancement of its Gaussian-smeared induced dipoles (widths sigma_i of
the bound sector; the point-dipole limit would give Z (eps_inf + 2)/3 = 3.3).  Scaling sigma_O and sigma_Mg post hoc on
step 20's weights, Z* (frame charge + dP_bound/dr, as mgo_phonons.py) and eps_inf (dielectric_tensor of the same
assembly) are recomputed together on the 64-atom cell: the near-field dipole sums that make Z*'s enhancement are
damped by sigma, the q -> 0 response that makes eps_inf much less, so the two can separate.  A sigma consistent with the
ion density (O2- ~1.3-1.4 A) that gives Z* 1.9-2.1 at eps_inf 2.9-3.5 would be a physical prior; a sigma that does not
exist rules the lever out.  Usage: python experiments/benchmarks_phonons/probe_sigma_scan.py MODEL.pt [--a 4.265]"""
import argparse, dataclasses, sys, warnings
sys.path.insert(0, "."); sys.path.insert(0, "tools/electrostatics"); warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
from ase.build import bulk
from flashace.calculator import TransformersACECalculator
from trace_electrostatics.energy import state_from_assembly
from trace_electrostatics.observables import dielectric_tensor
from trace_redox.bound import assemble_neutral, eliminate_bound_sector, eliminate_core_carrier
from trace_redox.constants import formal_frame
from trace_redox.module import carrier_widths, without_bound_drive
from mgo_phonons import _edges, _jacobian


def z_and_eps(mod, h, pos, ei, vec, L, keys, data, scale, gaussian_a=None):
    inputs = mod._inputs(pos, data, keys.get("pbc")); r, species = mod.redox, inputs["species"]
    base_config = mod.config if gaussian_a is None else dataclasses.replace(mod.config, gaussian_a=gaussian_a)
    # scale sigma where the heads emit it, BEFORE the assembly builds its lattice operator (replacing params afterwards
    # leaves the operator untouched: the first scan of 2026-11-07 03:30 was inert for that reason)
    f = torch.ones(len(species), dtype=pos.dtype)
    for zz, sc in scale.items():
        f = torch.where(species == zz, torch.full_like(f, sc), f)
    heads = mod.heads; orig = heads.forward

    def scaled_forward(*a, **k):
        out = orig(*a, **k)
        if hasattr(out, "sigma"):
            return dataclasses.replace(out, sigma=out.sigma * f)
        if hasattr(out, "params") and hasattr(out.params, "sigma"):
            new_params = dataclasses.replace(out.params, sigma=out.params.sigma * f)
            if dataclasses.is_dataclass(out):
                return dataclasses.replace(out, params=new_params)
            if hasattr(out, "_replace"):                      # a NamedTuple readout
                return out._replace(params=new_params)
            out.params = new_params                           # a plain object
            return out
        raise TypeError(f"heads returned {type(out)} without sigma")
    heads.forward = scaled_forward
    try:
        asm = assemble_neutral(h, pos, ei, vec, L, heads, base_config, pbc=inputs["pbc"], cell=keys.get("cell"), edge_shift=keys.get("edge_shift"),
                               quadrupoles=inputs["quadrupoles"], species=species, external_field=None)
    finally:
        heads.forward = orig
    if r.bound_electronegativity == "none":
        asm = without_bound_drive(asm)
    sig = asm.params.sigma.detach(); z8 = species == 8
    scale_check = float(sig[z8].mean()) / 1.4421
    asm = mod._partition_bonds(asm, species, ei, L); asm = mod._with_exclusion_anchors(asm, species, ei, L, vec)
    dtype = asm.pos.dtype; sf = mod.free_widths(species).to(dtype) if mod.width is not None else asm.params.sigma; frames = formal_frame(species)
    if r.free_charge_model == "core_carrier":
        elim = eliminate_core_carrier(asm, sf, carrier_widths(species, sf, int(r.max_charge)), frames, method=r.bound_elimination)
    else:
        elim = eliminate_bound_sector(asm, sf, method=r.bound_elimination, sigma_self=sf if r.anchor == "formal" else None)
    t, mu_hat, _ = elim.split_moments(elim.bound_moments(frames.to(dtype))); A = elim.assembly
    P = t @ A.bond_dipole + A.dipoles(mu_hat).sum(0)
    Zb = _jacobian(P, pos); Z = frames.to(dtype)[:, None, None] * torch.eye(3, dtype=dtype) + Zb
    config = base_config if base_config.solver == "dense" else dataclasses.replace(base_config, solver="dense")
    asm_d = dataclasses.replace(asm, pos=asm.pos.detach(), bond_dipole=None if asm.bond_dipole is None else asm.bond_dipole.detach())
    st = state_from_assembly(asm_d, config); eps = dielectric_tensor(st.solve_result, st.assembly).detach()
    from trace_electrostatics.solver import gaussian_widths
    smu, _ = gaussian_widths(asm.params, base_config.gaussian_a)
    widths = (float(smu[species == 8].mean()), float(smu[species == 12].mean()))
    return Z.detach(), eps, scale_check, widths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--a", type=float, default=4.265)
    args = ap.parse_args()
    calc = TransformersACECalculator(model_path=args.model, device="cpu"); mod = calc.model.electrostatics
    cell = bulk("MgO", "rocksalt", a=args.a, cubic=True).repeat((2, 2, 2)); cell.info = {"charge": 0.0}; cell.calc = calc
    cap = []; hk = mod.register_forward_pre_hook(lambda m, a, k: cap.append((a, k)) or None, with_kwargs=True); cell.get_potential_energy(); hk.remove()
    (h, pos0, ei, vec0, L0), kw = cap[0][0][:5], cap[0][1]; keys = {k: kw[k] for k in ("pbc", "cell", "edge_shift") if k in kw}; data = kw["data"]
    z = cell.numbers
    print(f"=== {args.model.split('/')[-2]}, a = {args.a}, 64 atoms; step-20 bound widths sigma_O 1.442, sigma_Mg 1.499 A (measured 2026-11-07)")
    # only widths >= the trained ones: shrinking sigma raises the hardness floor eta >= sqrt(2) K_E / (sigma sqrt(pi)) above
    # eta_O = 9.69 eV (the assembly refuses it; measured 2026-11-07), and eta is the charge-transfer response, not the lever
    import os
    if os.environ.get("SCAN", "a") == "sigma":
        for sO, sMg in ((1.0, 1.0), (1.4, 1.0), (2.0, 2.0)):
            pos = pos0.detach().clone().requires_grad_(True); vec, L = _edges(pos, ei, keys)
            Z, eps, chk, w = z_and_eps(mod, h.detach(), pos, ei, vec, L, keys, data, {8: sO, 12: sMg})
            zmg = float(torch.diagonal(Z[z == 12].mean(0)).mean()); e = float(torch.diagonal(eps).mean())
            print(f"charge sigma scale O x{sO:.1f}, Mg x{sMg:.1f}: Z*(Mg) {zmg:+.4f}, eps_inf {e:.4f}; dipole widths sigma_mu O {w[0]:.3f}, Mg {w[1]:.3f} A", flush=True)
    for a in (0.5, 0.35, 0.25, 0.15, 0.08, 0.04, 0.7, 0.9):
        pos = pos0.detach().clone().requires_grad_(True); vec, L = _edges(pos, ei, keys)
        try:
            Z, eps, chk, w = z_and_eps(mod, h.detach(), pos, ei, vec, L, keys, data, {}, gaussian_a=a)
        except Exception as err:
            print(f"gaussian_a {a:.2f}: refused ({type(err).__name__}: {str(err)[:100]})", flush=True); continue
        zmg = float(torch.diagonal(Z[z == 12].mean(0)).mean()); e = float(torch.diagonal(eps).mean())
        print(f"gaussian_a {a:.2f}: dipole widths sigma_mu O {w[0]:.3f}, Mg {w[1]:.3f} A; Z*(Mg) {zmg:+.4f}, eps_inf {e:.4f}   | targets PBE Z* 1.992, eps_inf 3.235", flush=True)


if __name__ == "__main__":
    main()
