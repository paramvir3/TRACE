"""Bulk MgO phonons from a TRACE checkpoint, with and without the non-analytic correction (2026-11-06): where the
electrostatics shows.

    python tools/electrostatics/mgo_phonons.py MODEL.pt [--a 4.2650] [--out DIR] [--born BORN_FILE]

The LO-TO splitting of a polar crystal (MgO: TO 401, LO 718 cm^-1 at Gamma, experiment) exists only through the
long-range dipole field: the squared splitting is proportional to Z*^2 / eps_inf.  A finite supercell cannot hold the
q -> 0 macroscopic field, so finite-displacement force constants -- from DFT or from any model whose energy is periodic
in the supercell, the Ewald sum included -- give LO = TO at Gamma; the non-analytic correction (Gonze and Lee, Phys. Rev.
B 55, 10355 (1997); ESTABLISHED, as implemented in phonopy) restores it from the Born effective charges and the electronic
dielectric tensor.  Here both come from the model itself: Z* = -d2U/dE dr (trace_electrostatics.observables.
born_charges_from_energy) and eps_inf (dielectric_tensor), evaluated on the bound sector of the formal crystal (no free
carriers: the ensemble holds the frame alone).  Reported:
  (1) Z*(Mg), Z*(O), eps_inf of the model against PBE / experiment;
  (2) phonopy frequencies at Gamma without and with NAC (the model's own Z*, eps_inf; optionally a reference BORN file's),
      TO and LO, and the dispersion along Gamma-X-K-Gamma-L written to DIR/band_{nonac,nac}.yaml;
  (3) eps_0 by the Lyddane-Sachs-Teller relation eps_0 = eps_inf (w_LO / w_TO)^2 against experiment (9.8);
  (4) the same force constants with the electrostatic term removed from the model's energy (the backbone's part alone):
      no Z*, so no NAC and no splitting -- what a local potential gives.
Reference values (provenance stated): experiment TO 401 / LO 718 cm^-1 (Jasperse et al., Phys. Rev. 146, 526 (1966);
Sangster et al., J. Phys. C 3, 1026 (1970)), eps_inf 2.95-3.0, eps_0 9.8; PBE (GGA, Togo-style supercell + DFPT NAC)
Z* ~ 1.96-1.99, eps_inf 2.75-3.2 (basis / pseudopotential dependent), TO ~ 395, LO ~ 695 cm^-1.  The 4G data's lattice
constant 4.2650 A (PBE, FHI-aims) is the default; experiment 4.212 A.
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from phonopy import Phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics.observables import born_charges_from_energy, dielectric_tensor  # noqa: E402

K_E = 14.399645
THZ_TO_CM = 33.35641
EXPERIMENT = {"TO": 401.0, "LO": 718.0, "eps_inf": 2.95, "eps_0": 9.8}


def _edges(pos, ei, keys):
    """Edge vectors from ``pos`` in the model's own convention (flashace.model: pos[j] - pos[i] + S cell)."""
    vec = pos[ei[0]] - pos[ei[1]]
    if keys.get("edge_shift") is not None and keys.get("cell") is not None:
        vec = vec + keys["edge_shift"].to(pos.dtype) @ keys["cell"]
    return vec, torch.norm(vec, dim=1)


def _frame_bound_dipole(mod, h, pos, ei, vec, L, keys, data):
    """The bound sector's dipole ``P_b = sum_b t_b d_b + sum_i mu_i`` (e A) at the moments the formal frame drives,
    ``y_f = y0 - K^-1 A_f^T f`` (trace_redox.bound.BoundElimination.bound_moments), built exactly as
    RedoxElectrostatics.forward builds its elimination; and the frame charges ``f`` [n]."""
    from trace_redox.bound import assemble_neutral, eliminate_bound_sector, eliminate_core_carrier
    from trace_redox.constants import formal_frame
    from trace_redox.module import carrier_widths, without_bound_drive
    inputs = mod._inputs(pos, data, keys.get("pbc"))
    r, species = mod.redox, inputs["species"]
    heads = mod._heads_for(species) if hasattr(mod, "_heads_for") else mod.heads     # SPEC 35's anchored readout when on
    assembly = assemble_neutral(h, pos, ei, vec, L, heads, mod.config, pbc=inputs["pbc"], cell=keys.get("cell"),
                                edge_shift=keys.get("edge_shift"), quadrupoles=inputs["quadrupoles"], species=species,
                                external_field=None)
    if r.bound_electronegativity == "none":
        assembly = without_bound_drive(assembly)
    nb_before = int(assembly.t_seed.numel())
    assembly = mod._partition_bonds(assembly, species, ei, L)
    assembly = mod._with_exclusion_anchors(assembly, species, ei, L, vec)
    dtype, device = assembly.pos.dtype, assembly.pos.device
    sigma_free = mod.free_widths(species) if mod.width is not None else assembly.params.sigma
    frames = formal_frame(species).to(device)
    if r.free_charge_model == "core_carrier":
        sigma_carrier = carrier_widths(species, sigma_free.to(dtype), int(r.max_charge))
        elim = eliminate_core_carrier(assembly, sigma_free.to(dtype), sigma_carrier, frames, method=r.bound_elimination)
    else:
        elim = eliminate_bound_sector(assembly, sigma_free.to(dtype), method=r.bound_elimination,
                                      sigma_self=sigma_free.to(dtype) if r.anchor == "formal" else None)
    if r.frame_induction_damping:                                              # SPEC 34: the same damped drive as forward
        elim = mod._damp_frame_induction(elim, assembly, species, sigma_free.to(dtype), pos, keys.get("cell"), inputs["pbc"])
    t, mu_hat, _ = elim.split_moments(elim.bound_moments(frames.to(dtype)))
    A = elim.assembly
    P_t, P_mu = t @ A.bond_dipole, A.dipoles(mu_hat).sum(0)        # the bond-charge and the induced-dipole channels
    return P_t + P_mu, frames.to(dtype), t, (P_t, P_mu, nb_before, int(elim.n_bonds))


def _jacobian(P, pos):
    """``[n, a, b] = dP_a / dr_{u b}``."""
    return torch.stack([torch.autograd.grad(P[a], pos, retain_graph=True)[0] for a in range(3)], dim=1)


def model_born_and_eps(calc, atoms, fd_atom=None, fd_step=1e-4):
    """Z* [n, 3, 3] (e), eps_inf [3, 3] and the checks, from the model on ``atoms`` (the formal crystal).

    TRACE-Redox refuses an applied field under periodic boundaries for its free charges (trace_redox/bound.py: the
    free charges' -sum c_u r_u . E is branch dependent), so Z* = -d2E/dE dr is assembled from its two field-dependent
    terms in the frame configuration n = f (its population is reported):
      * the frame's integer charge, Z*_u = f_u delta_ab exactly (d/dr_u of -E . sum f_u r_u; the branch of the sum is
        position independent, the derivative is not);
      * the bound sector's dynamic part, dP_b / dr_u, with P_b the SPEC 15 bond-current dipole of the moments the frame
        drives (the sector is quadratic, so the energy and moment routes coincide at the dense exact solve).
    Computed inside the module's forward pre-hook, where the node features h still carry their graph from the
    positions: the derivative includes the backbone's dependence (chi, eta, alpha, widths), as the forces do.  The
    mean-field ElectrostaticModule (no frame) takes born_charges_from_energy on its own state.
    Checks: the recomputed edge vectors against the model's; the translational sum rule sum_u Z*_u = 0; a central
    difference of P_b at fixed h on ``fd_atom`` against the fixed-h autograd column.
    """
    mod = calc.model.electrostatics
    out = {}

    def hook(m, args, kwargs):
        h, pos, ei, vec, L = args[:5]
        keys = {k: kwargs[k] for k in ("pbc", "cell", "edge_shift") if k in kwargs}
        data = kwargs.get("data", {})
        assert pos.requires_grad, "the module's positions carry no graph"
        vec2, L2 = _edges(pos, ei, keys)
        out["edge_check"] = float((vec2.detach() - vec.detach()).abs().max())
        if hasattr(m, "bound_state"):
            P, frames, t, (P_t, P_mu, nb0, nb1) = _frame_bound_dipole(m, h, pos, ei, vec, L, keys, data)
            out["n_bonds"] = (nb0, nb1)
            Zb = _jacobian(P, pos)
            out["Zb_t"], out["Zb_mu"] = _jacobian(P_t, pos).detach(), _jacobian(P_mu, pos).detach()
            gh = torch.autograd.grad(P.sum(), h, retain_graph=True, allow_unused=True)[0]
            out["h_used"] = (h.grad_fn is not None, None if gh is None else float(gh.abs().max()))
            eye = torch.eye(3, dtype=P.dtype, device=P.device)
            out["Z"] = (frames[:, None, None] * eye + Zb).detach(); out["Zb"] = Zb.detach(); out["frames"] = frames
            out["P_bound"] = P.detach(); out["t_max"] = float(t.detach().abs().max())
            # fixed-h route (the electrostatics alone) and its central difference
            hd, pd = h.detach(), pos.detach().clone().requires_grad_(True)
            vd, Ld = _edges(pd, ei, keys)
            Pd = _frame_bound_dipole(m, hd, pd, ei, vd, Ld, keys, data)[0]
            out["Zb_fixed_h"] = _jacobian(Pd, pd).detach()
            if fd_atom is not None:
                fd = torch.zeros(3, 3, dtype=P.dtype)
                for b in range(3):
                    Ppm = []
                    for sgn in (+1.0, -1.0):
                        pp = pos.detach().clone(); pp[fd_atom, b] += sgn * fd_step
                        vp, Lp = _edges(pp, ei, keys)
                        Ppm.append(_frame_bound_dipole(m, hd, pp, ei, vp, Lp, keys, data)[0].detach())
                    fd[:, b] = (Ppm[0] - Ppm[1]) / (2 * fd_step)
                out["fd"] = fd; out["fd_atom"] = fd_atom
            # eps_inf: the bound sector's response (drive independent), on detached inputs as probe_eps_bound.py does
            st = m.bound_state(hd, pd, ei, vd, Ld, data=data, **keys)
            out["eps"] = dielectric_tensor(st.solve_result, st.assembly).detach()
        else:
            from trace_electrostatics.energy import electrostatic_state
            st = electrostatic_state(h, pos, ei, vec, L, m.heads, m.config, species=data.get("z"), **keys)
            out["Z"] = born_charges_from_energy(st.assembly).detach(); out["frames"] = None
            out["eps"] = dielectric_tensor(st.solve_result, st.assembly).detach()
        return None

    hk = mod.register_forward_pre_hook(hook, with_kwargs=True)
    hk2 = mod.register_forward_hook(lambda m, args, output: out.__setitem__("result", output[1]) or None)
    try:
        a = atoms.copy(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset(); a.get_potential_energy()
    finally:
        hk.remove(); hk2.remove()
    if hasattr(mod, "bound_state") and getattr(mod, "_ensemble_last", None):
        last = mod._ensemble_last
        H, cfgs, kT = last["H"].detach(), last["configs"], float(last["kT"])
        n = int(out["frames"].numel())                     # configs are integer charges per unit: the atoms, then any
        f = out["frames"].to(cfgs.dtype)                       # extra unit (the band unit of SPEC 20: 0 in the frame)
        rows = ((cfgs[:, :n] == f).all(dim=1) & (cfgs[:, n:] == 0).all(dim=1)).nonzero().flatten()
        assert rows.numel() == 1, (rows, cfgs.shape)
        idx = int(rows[0]); out["frame_row_extra"] = cfgs[idx, n:].tolist(); out["n_configs"] = int(cfgs.shape[0])
        out["coupling_max"] = float((H - torch.diag(torch.diagonal(H))).abs().max())
        w, V = torch.linalg.eigh(H)
        p = torch.softmax(-(w - w.min()) / kT, dim=0)
        out["p_frame"] = float((p * V[idx] ** 2).sum()); out["eig_min_rel"] = float(w.min() - H[idx, idx])
        res = out["result"]
        out["energies"] = {k: float(getattr(res, k)) for k in ("energy", "energy_bound", "energy_free") if hasattr(res, k)}
        others = torch.diagonal(H)[torch.arange(H.shape[0]) != idx]
        out["gap"] = float(others.min() - H[idx, idx]) if others.numel() else float("inf"); out["kT"] = kT
    return out


def report_born(out, numbers):
    Z = out["Z"].numpy(); eps = out["eps"].numpy()
    z_mg = Z[numbers == 12].mean(0); z_o = Z[numbers == 8].mean(0)
    print(f"(1) model: Z*(Mg) diag {np.round(np.diagonal(z_mg), 3).tolist()}, Z*(O) diag {np.round(np.diagonal(z_o), 3).tolist()}, "
          f"off-diagonal max {np.abs(Z - np.array([np.diag(np.diagonal(z)) for z in Z])).max():.4f}; eps_inf diag "
          f"{np.round(np.diagonal(eps), 3).tolist()} | PBE Z* 1.96-1.99, eps_inf 2.75-3.2; experiment eps_inf 2.95", flush=True)
    print(f"    checks: edge vectors {out['edge_check']:.1e}; sum rule max |sum_u Z*_u| {np.abs(Z.sum(0)).max():.2e}", flush=True)
    if out.get("frames") is not None:
        Zb = out["Zb"].numpy(); Zf = out["Zb_fixed_h"].numpy()
        print(f"    frame charges f(Mg) {float(out['frames'][numbers == 12][0]):+.0f}, f(O) {float(out['frames'][numbers == 8][0]):+.0f}; bound sector's dynamic part: "
              f"Mg {np.round(np.diagonal(Zb[numbers == 12].mean(0)), 4).tolist()}, O {np.round(np.diagonal(Zb[numbers == 8].mean(0)), 4).tolist()} "
              f"(of which through the backbone's h: Mg {np.trace(Zb[numbers == 12].mean(0) - Zf[numbers == 12].mean(0)) / 3:+.4f}, "
              f"O {np.trace(Zb[numbers == 8].mean(0) - Zf[numbers == 8].mean(0)) / 3:+.4f}); |t|max {out['t_max']:.3e}, "
              f"|P_bound| {np.linalg.norm(out['P_bound'].numpy()):.2e} e A", flush=True)
        if "fd" in out:
            u = out["fd_atom"]
            print(f"    central difference of P_b (atom {u}, Z {int(numbers[u])}, fixed h): max |fd - autograd| {np.abs(out['fd'].numpy() - Zf[u]).max():.2e} "
                  f"(fd diag {np.round(np.diagonal(out['fd'].numpy()), 4).tolist()})", flush=True)
        Zt, Zm = out["Zb_t"].numpy(), out["Zb_mu"].numpy()
        print(f"    channels of the dynamic part (Mg / O, isotropic): bond charge {np.trace(Zt[numbers == 12].mean(0)) / 3:+.4f} / {np.trace(Zt[numbers == 8].mean(0)) / 3:+.4f}, "
              f"induced dipole {np.trace(Zm[numbers == 12].mean(0)) / 3:+.4f} / {np.trace(Zm[numbers == 8].mean(0)) / 3:+.4f}; "
              f"h carries a graph: {out['h_used'][0]}, max |dP/dh| {out['h_used'][1]}; bound-sector bonds before / after the partition "
              f"{out['n_bonds'][0]} / {out['n_bonds'][1]}", flush=True)
        print(f"    ensemble: {out['n_configs']} configurations, frame row's extra units {out['frame_row_extra']}, frame population {out['p_frame']:.6f}, "
              f"lowest other configuration {out['gap']:.3f} eV above it, largest coupling {out['coupling_max']:.2f} eV, lowest eigenvalue "
              f"{out['eig_min_rel']:+.2f} eV relative to the frame, kT {out['kT']:.4f} eV; energies {out['energies']}", flush=True)
    return z_mg, z_o, eps


def symmetrized_born(z_mg, z_o):
    """Cubic site symmetry (isotropic) and the acoustic sum rule; the raw violation is returned for the record."""
    zm, zo = np.trace(z_mg) / 3, np.trace(z_o) / 3
    asr = zm + zo
    return (zm - asr / 2) * np.eye(3), (zo - asr / 2) * np.eye(3), asr


GUARD = []   # per force evaluation of a TRACE-Redox model: (F - E_frame, frame population, configurations, max coupling)


def forces_of(calc, cell, positions, numbers, pbc=(True, True, True)):
    a = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc); a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
    mod = calc.model.electrostatics
    redox = mod is not None and hasattr(mod, "bound_state")
    cap = []
    hk = mod.register_forward_hook(lambda m, args, output: cap.append(output[1]) or None) if redox else None
    F = a.get_forces()
    if redox:
        hk.remove()
        from trace_redox.constants import formal_frame
        last = mod._ensemble_last; H = last["H"].detach(); cfgs = last["configs"]; n = len(numbers)
        f = formal_frame(torch.as_tensor(numbers)).to(cfgs.dtype)
        rows = ((cfgs[:, :n] == f).all(1) & (cfgs[:, n:] == 0).all(1)).nonzero().flatten()
        w, V = torch.linalg.eigh(H); p = torch.softmax(-(w - w.min()) / float(last["kT"]), 0)
        idx = int(rows[0]) if rows.numel() else None
        GUARD.append((float(cap[0].energy_free - H[idx, idx]) if idx is not None else float("nan"),
                      float((p * V[idx] ** 2).sum()) if idx is not None else 0.0, int(cfgs.shape[0]),
                      float((H - torch.diag(torch.diagonal(H))).abs().max())))
    return F


def guard_summary(label):
    if GUARD:
        print(f"    ensemble guard ({label}, {len(GUARD)} supercells): max |F - E_frame| {max(abs(g[0]) for g in GUARD):.4f} eV, min frame population "
              f"{min(g[1] for g in GUARD):.6f}, configurations {min(g[2] for g in GUARD)}-{max(g[2] for g in GUARD)}, max coupling {max(g[3] for g in GUARD):.3f} eV", flush=True)
    GUARD.clear()


def phonons(calc, unit, supercell_matrix, nac=None, label=""):
    ph = Phonopy(PhonopyAtoms(symbols=unit.get_chemical_symbols(), cell=unit.cell.array, scaled_positions=unit.get_scaled_positions()),
                 supercell_matrix=supercell_matrix, primitive_matrix="auto")
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        forces.append(forces_of(calc, sc.cell, sc.positions, sc.numbers))
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    if nac is not None:
        ph.nac_params = nac
    return ph


Q_LIMIT = 1.0e-3   # the q -> 0 limit is taken at |q| = 1e-3 (reduced) along the direction: phonopy 2.17.2's
                   # nac_q_direction has no effect in run_qpoints at q = 0 exactly (measured 2026-11-06)


def gamma_frequencies(ph, q_direction):
    """Frequencies (cm^-1) at Gamma (``q_direction`` None) or in the q -> 0 limit along ``q_direction`` (reduced
    coordinates of the primitive reciprocal cell), where the non-analytic term applies.  In a cubic crystal the LO-TO
    splitting at q -> 0 does not depend on the direction (Born-Huang; ESTABLISHED); the direction decides which
    eigenvector is longitudinal."""
    q = [0.0, 0.0, 0.0] if q_direction is None else (Q_LIMIT * np.asarray(q_direction, float) / np.linalg.norm(q_direction)).tolist()
    ph.run_qpoints([q])
    return np.sort(ph.get_qpoints_dict()["frequencies"][0]) * THZ_TO_CM


def band_gamma(ph):
    """The band path's own first point (Gamma) frequencies (cm^-1), as phonopy's band routine evaluates it with NAC."""
    d = ph.get_band_structure_dict()
    return np.sort(d["frequencies"][0][0]) * THZ_TO_CM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--a", type=float, default=4.2650); ap.add_argument("--out", default=None, help="output directory (default $TRACE_FP_CACHE/mgo_phonons, TRACE_FP_CACHE defaulting to ./fp_cache)")
    ap.add_argument("--born", default=None, help="a phonopy BORN file (reference Z*, eps_inf) for a second NAC")
    ap.add_argument("--cell", type=int, default=2, help="conventional-cell repeats for Z* / eps_inf (2: 64 atoms)")
    ap.add_argument("--supercell", type=int, default=2, help="conventional-cell repeats of the force-constant supercell (2: 64 atoms)")
    ap.add_argument("--no-phonons", action="store_true", help="stop after Z* / eps_inf and their checks")
    args = ap.parse_args()
    outdir = args.out or os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "mgo_phonons"); os.makedirs(outdir, exist_ok=True)
    calc = TransformersACECalculator(model_path=args.model, device="cpu")
    torch.set_num_threads(4)
    conv = bulk("MgO", "rocksalt", a=args.a, cubic=True)                    # 8 atoms
    # (1) Z* and eps_inf on the conventional cell repeated --cell times (2: the 64-atom T6 cell)
    cell = conv.repeat((args.cell,) * 3)
    numbers = cell.numbers
    print(f"=== {os.path.basename(os.path.dirname(args.model))} at a = {args.a} A, {len(cell)} atoms for Z* / eps_inf")
    out = model_born_and_eps(calc, cell, fd_atom=int(np.nonzero(numbers == 8)[0][0]))
    z_mg, z_o, eps = report_born(out, numbers)
    zs_mg, zs_o, asr = symmetrized_born(z_mg, z_o)
    print(f"    NAC uses the isotropic, sum-rule-corrected Z*: Mg {zs_mg[0, 0]:+.4f}, O {zs_o[0, 0]:+.4f} (raw Z*_Mg + Z*_O = {asr:+.2e})", flush=True)
    if args.no_phonons:
        return
    # (2) phonons: --supercell^3 conventional supercell (2: 64 atoms); one set of force constants, NAC switched on afterwards
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]   # Gamma-X-K-Gamma-L (fcc primitive)
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
    labels = ["G", "X", "K", "G", "L"]
    ph = phonons(calc, conv, np.diag([args.supercell] * 3))
    guard_summary("force constants")
    f0 = gamma_frequencies(ph, None)
    ph.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=False)
    ph.write_yaml_band_structure(filename=os.path.join(outdir, "band_nonac.yaml"))
    born_prim = np.array([zs_mg if sym == "Mg" else zs_o for sym in ph.primitive.symbols])
    ph.nac_params = {"born": born_prim, "dielectric": eps, "factor": K_E}
    res = {name: gamma_frequencies(ph, qdir) for name, qdir in (("[100]", [1, 0, 0]), ("[110]", [1, 1, 0]), ("[111]", [1, 1, 1]))}   # reduced directions
    print(f"(2) Gamma without NAC (cm^-1): {np.round(f0, 1).tolist()}", flush=True)
    for name, f in res.items():
        print(f"    with NAC, q -> 0 along {name} (reduced, |q| = {Q_LIMIT}): {np.round(f, 1).tolist()}", flush=True)
    to, lo = float(res["[100]"][3]), float(res["[100]"][5])
    print(f"    TO {to:.1f}, LO {lo:.1f} cm^-1 (experiment {EXPERIMENT['TO']:.0f} / {EXPERIMENT['LO']:.0f}; PBE ~395 / ~695); splitting {lo - to:.1f} (experiment 317)", flush=True)
    mu = 24.305 * 15.999 / (24.305 + 15.999); omega = float(np.linalg.det(ph.primitive.cell))
    to_s2 = 1.602176634e-19 / 1e-20 / 1.66053907e-27                   # eV / (A^2 amu) -> s^-2
    bh = np.sqrt(4 * np.pi * K_E * zs_mg[0, 0] ** 2 / (omega * mu * float(np.trace(eps) / 3)) * to_s2) / (2 * np.pi) / 1e12 * THZ_TO_CM
    print(f"    Born-Huang check: sqrt(LO^2 - TO^2) = {np.sqrt(lo * lo - to * to):.1f} cm^-1 from phonopy against "
          f"{bh:.1f} from 4 pi K_E Z*^2 / (Omega mu eps_inf) (ESTABLISHED, cubic diatomic)", flush=True)
    eps0 = float(np.trace(eps) / 3) * (lo / to) ** 2
    print(f"(3) LST eps_0 = eps_inf (LO/TO)^2 = {eps0:.2f} (experiment {EXPERIMENT['eps_0']})", flush=True)
    ph.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=False)
    ph.write_yaml_band_structure(filename=os.path.join(outdir, "band_nac.yaml"))
    print(f"    band path's Gamma with NAC (phonopy's own direction handling): {np.round(band_gamma(ph), 1).tolist()}", flush=True)
    ph.save(os.path.join(outdir, "phonopy_params.yaml"), settings={"force_constants": True})
    if args.born:
        from phonopy.file_IO import parse_BORN
        nac_ref = parse_BORN(ph.primitive, filename=args.born)
        ph.nac_params = nac_ref
        f2 = gamma_frequencies(ph, [1, 0, 0])
        print(f"    with the reference BORN (Z* {np.round(np.diagonal(nac_ref['born'][0]), 3).tolist()}, eps "
              f"{np.round(np.diagonal(nac_ref['dielectric']), 3).tolist()}): Gamma {np.round(f2, 1).tolist()}", flush=True)
        ph.run_band_structure(qpoints, path_connections=connections, labels=labels)
        ph.write_yaml_band_structure(filename=os.path.join(outdir, "band_nac_refborn.yaml"))
    # (4) the backbone's part alone: the electrostatic term removed from the energy
    elec = calc.model.electrostatics
    calc.model.electrostatics = None
    try:
        ph3 = phonons(calc, conv, np.diag([args.supercell] * 3))
        f3 = gamma_frequencies(ph3, None)
        print(f"(4) backbone alone (no electrostatic term): Gamma {np.round(f3, 1).tolist()} -- no Z*, so no NAC and no splitting", flush=True)
        ph3.run_band_structure(qpoints, path_connections=connections, labels=labels)
        ph3.write_yaml_band_structure(filename=os.path.join(outdir, "band_backbone.yaml"))
    finally:
        calc.model.electrostatics = elec
    print(f"band structures written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
