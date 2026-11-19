"""Frozen-weights probe (2026-11-07 06:20, pre-registered in the theory note section 4e): the overlap damping of the
frame-charge -> bound-dipole coupling.  On step 20's bound elimination, A[i, mu_j] is multiplied by s(R_ij) for the
nearest-neighbour frame pairs (minimum image; R < 2.7 A), s(R) = s0 + s1 (R - 2.128 A) clamped to [0.2, 1], with the ALMO
pair's law s0 = 0.61, s1 = 0.37 /A as the measured case and controls (no slope; half and double slope).  Z* = frame
charge + dP_bound/dr with the damped coupling (autograd through s(R) and A); eps_inf from the same assembly (untouched by
construction: the field couples to the bound dipoles directly).  Reported at the trained polarizability and at the
global scale 1.095 that puts eps_inf at 3.235.  Pass: Z*(Mg) in 1.90-2.10 for the measured law.
Usage: python experiments/benchmarks_phonons/probe_overlap_damping.py MODEL.pt [--a 4.265]"""
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
R0 = 4.255656884758435 / 2


def damping_matrix(pos, cell, species, n_bonds, m, s0, s1, r_cut=2.7):
    """[n, m] factors: s(R_ij) on the dipole columns of atom j for the frame pairs (i, j) within r_cut (minimum image)."""
    n = pos.shape[0]; inv = torch.linalg.inv(cell)
    d = pos.unsqueeze(1) - pos.unsqueeze(0)                                   # r_i - r_j
    frac = d @ inv; frac = frac - torch.round(frac); d = frac @ cell
    R = torch.linalg.norm(d, dim=-1) + torch.eye(n, dtype=pos.dtype) * 1e6
    f = formal_frame(species)
    pair = (f.unsqueeze(1) * f.unsqueeze(0) < 0) & (R < r_cut)                # cation-anion nearest neighbours
    s = torch.clamp(s0 + s1 * (R - R0), 0.2, 1.0)
    S_atoms = torch.where(pair, s, torch.ones_like(s))                       # [n_i, n_j]
    S = torch.ones(n, m, dtype=pos.dtype)
    S[:, n_bonds:n_bonds + 3 * n] = S_atoms.repeat_interleave(3, dim=1)
    return S, int(pair.sum() // 2)


def run(mod, h, pos0, ei, keys, data, species, s0, s1, alpha_scale, species_alpha=None):
    """species_alpha: {Z: factor} on that species' polarizability (before the global scale)."""
    pos = pos0.detach().clone().requires_grad_(True); vec, L = _edges(pos, ei, keys)
    inputs = mod._inputs(pos, data, keys.get("pbc")); r = mod.redox
    heads = mod.heads; orig = heads.forward
    fac = torch.ones(len(species), dtype=torch.float64)
    for zz, v in (species_alpha or {}).items():
        fac = torch.where(species == zz, torch.full_like(fac, v), fac)
    heads.forward = lambda *a, **k: (lambda out: out._replace(params=dataclasses.replace(out.params, A_mu=out.params.A_mu / (alpha_scale * fac.view(-1, 1, 1)))))(orig(*a, **k))
    try:
        asm = assemble_neutral(h, pos, ei, vec, L, heads, mod.config, pbc=inputs["pbc"], cell=keys.get("cell"), edge_shift=keys.get("edge_shift"),
                               quadrupoles=inputs["quadrupoles"], species=species, external_field=None)
    finally:
        heads.forward = orig
    if r.bound_electronegativity == "none":
        asm = without_bound_drive(asm)
    asm = mod._partition_bonds(asm, species, ei, L); asm = mod._with_exclusion_anchors(asm, species, ei, L, vec)
    dtype = asm.pos.dtype; sf = mod.free_widths(species).to(dtype) if mod.width is not None else asm.params.sigma; frames = formal_frame(species)
    if r.free_charge_model == "core_carrier":
        elim = eliminate_core_carrier(asm, sf, carrier_widths(species, sf, int(r.max_charge)), frames, method=r.bound_elimination)
    else:
        elim = eliminate_bound_sector(asm, sf, method=r.bound_elimination, sigma_self=sf if r.anchor == "formal" else None)
    A = elim.A; m = A.shape[1]
    S, n_pairs = damping_matrix(pos, keys["cell"], species, int(elim.n_bonds), m, s0, s1)
    A_damped = A * S
    f = frames.to(dtype)
    y = elim.y0 - elim.operator.solve((A_damped.transpose(0, 1) @ f).unsqueeze(1)).squeeze(1)   # bound_moments with the damped coupling
    t, mu_hat, _ = elim.split_moments(y); Aasm = elim.assembly
    P = t @ Aasm.bond_dipole + Aasm.dipoles(mu_hat).sum(0)
    Zb = _jacobian(P, pos); Z = f[:, None, None] * torch.eye(3, dtype=dtype) + Zb
    config = mod.config if mod.config.solver == "dense" else dataclasses.replace(mod.config, solver="dense")
    asm_d = dataclasses.replace(asm, pos=asm.pos.detach(), bond_dipole=None if asm.bond_dipole is None else asm.bond_dipole.detach())
    st = state_from_assembly(asm_d, config); eps = dielectric_tensor(st.solve_result, st.assembly).detach()
    z = species
    return float(torch.diagonal(Z[z == 12].mean(0)).mean()), float(torch.diagonal(eps).mean()), n_pairs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--a", type=float, default=4.265)
    args = ap.parse_args()
    calc = TransformersACECalculator(model_path=args.model, device="cpu"); mod = calc.model.electrostatics
    cell = bulk("MgO", "rocksalt", a=args.a, cubic=True).repeat((2, 2, 2)); cell.info = {"charge": 0.0}; cell.calc = calc
    cap = []; hk = mod.register_forward_pre_hook(lambda m_, a_, k_: cap.append((a_, k_)) or None, with_kwargs=True); cell.get_potential_energy(); hk.remove()
    (h, pos0, ei, vec0, L0), kw = cap[0][0][:5], cap[0][1]; keys = {k: kw[k] for k in ("pbc", "cell", "edge_shift") if k in kw}; data = kw["data"]
    species = torch.as_tensor(cell.numbers)
    print(f"=== {args.model.split('/')[-2]}, a = {args.a}, 64 atoms; s(R) = s0 + s1 (R - {R0:.3f}) on nearest-neighbour frame pairs")
    import os
    if os.environ.get("SCAN", "law") == "law":
        cases = (("undamped (control: must reproduce 2.4291 / 3.0094)", 1.0, 0.0), ("measured law s0 0.61, s1 0.37", 0.61, 0.37),
                 ("no slope (s = 0.61)", 0.61, 0.0), ("half slope (0.61, 0.185)", 0.61, 0.185), ("double slope (0.61, 0.74)", 0.61, 0.74),
                 ("weaker damping (0.8, 0.37)", 0.8, 0.37))
        for label, s0, s1 in cases:
            for scale in (1.0, 1.095):
                zmg, e, n_pairs = run(mod, h.detach(), pos0, ei, keys, data, species, s0, s1, scale)
                print(f"{label}: alpha scale {scale:.3f}: Z*(Mg) {zmg:+.4f}, eps_inf {e:.4f} ({n_pairs} frame pairs damped)   | PBE 1.992 / 3.235", flush=True)
        return
    # the combined test: the physical species split of alpha (first-principles table: O2- 2.44, Mg2+ 0.07 A^3; or the
    # Clausius-Mossotti in-crystal 1.9 / 0.2) with the measured damping law, alpha rescaled globally to eps_inf = 3.235
    aO, aMg = 0.205, 1.916                                                   # the trained values (measured 2026-11-07)
    for label, sa, s0, s1 in (("trained split, measured law", None, 0.61, 0.37),
                              ("table split O 2.44 / Mg 0.07, undamped", {8: 2.44 / aO, 12: 0.07 / aMg}, 1.0, 0.0),
                              ("table split O 2.44 / Mg 0.07, measured law", {8: 2.44 / aO, 12: 0.07 / aMg}, 0.61, 0.37),
                              ("CM split O 1.9 / Mg 0.2, measured law", {8: 1.9 / aO, 12: 0.2 / aMg}, 0.61, 0.37),
                              ("table split, half slope", {8: 2.44 / aO, 12: 0.07 / aMg}, 0.61, 0.185)):
        lo, hi = 0.2, 8.0; zmg = e = float("nan"); sc = float("nan")
        for it in range(16):
            sc = 0.5 * (lo + hi)
            try:
                zmg, e, n_pairs = run(mod, h.detach(), pos0, ei, keys, data, species, s0, s1, sc, species_alpha=sa)
            except Exception as err:
                print(f"{label}: refused at scale {sc:.3f} ({type(err).__name__}: {str(err)[:100]})", flush=True); break
            if e < 3.235: lo = sc
            else: hi = sc
            if abs(e - 3.235) < 2e-3: break
        print(f"{label}: global alpha scale {sc:.3f} -> eps_inf {e:.4f}, Z*(Mg) {zmg:+.4f}   | PBE 1.992", flush=True)


if __name__ == "__main__":
    main()
