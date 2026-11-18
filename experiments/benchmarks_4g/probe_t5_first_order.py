"""The model's wetting energies split as the first-principles decomposition splits them (2026-10-27; a probe, frozen
weights, no model change).

Usage (repo root): python experiments/benchmarks_4g/probe_t5_first_order.py MODEL.pt [...]

The first-principles side (tools/electrostatics/wetting_eda.py, embedded MgO cluster, PBE; theory note 2026-10-18
section 4c) splits each geometry's interaction energy dE = E_AB - E_A - E_B into the frozen-density first order E1 and
the relaxation (polarisation + charge transfer), and the wetting energy wet = [dE + E_A](flat) - [dE + E_A](upright)
into E1, relax and the dimer's deformation E_A(flat) - E_A(upright).  At 10 ions (doped with total-ion
pseudopotentials) T5 = wet_d - wet_u splits into E1 -903.9, relax -281.9, deformation -33.0 meV: three quarters of
what doping does to the wetting is first order.  This probe puts the model through the same split (NEW-TO-TRACE: the
first-order state of the configuration ensemble):

    A = Au2 alone (in AB's cell, so its image interaction cancels; Q_A = 0 undoped, -1 doped), B = the slab alone (the reference cell, Q_B = 0 / +1), each
    with its own canonical ensemble rho_A, rho_B.  The first-order state of AB is the product rho_A (x) rho_B mapped
    into AB's units, with AB's Hamiltonian in which the band unit's orbital is frozen at B's own (the doped slab
    carries two band electrons, B1; their orbital's response to Au2 is B's polarisation, so it belongs to the
    relaxation):

        F1_AB = Tr(rho_A (x) rho_B  H_AB) - kT (S_A + S_B)

    the analogue of E_AB[P_HL] (each fragment frozen in its isolated state, the other present).  Gibbs' inequality
    gives F_AB <= F1_AB.  Then, exactly,

        dE = d(backbone) + d(e_b0) + [F1_AB - F_A - F_B] + [F_AB - F1_AB],

    and every free-sector bracket splits into the configuration-energy terms of probe_t5_terms.py, with the screened
    Coulomb term split further: n.b + 1/2 m^T J m (bare: the point Coulomb of the configuration, with the bound
    sector's ground-state field) + 1/2 m^T (W - J) m (response: the bound sector's induction by the free charges),
    m = omega^T n the atoms' charges.

Assignment to the first-principles pieces (the model's terms are not the functional's, so this is a mapping, stated):
    model E1     = the first-order bracket without its response terms (response, ind26, alpha28)
    model POL    = those response terms in the first-order bracket + d(e_b0)
    model CT     = F_AB - F1_AB (the configurations' own relaxation: charge transfer and the dimer's internal
                   redistribution, with every term's change)
    backbone     = d(backbone), reported apart (learned, short-range; it has no first-principles counterpart)
    deformation  = E_tot(A, flat geometry) - E_tot(A, upright geometry) (A in the cell; the isolated dimer's is
                   reported too, against PBE's isolated-dimer values)

Checks (assertions): (k1) every term table sums to the configuration energies used by the solver (1e-9);
(k2) the module's ensemble is reproduced (F from H, 1e-9) and the totals add up (backbone + E_q, 1e-8 eV);
(k3) W = omega W_atoms omega^T and b = omega u0, so the bare / response split is exact (1e-9);
(k4) separated limit: the undoped wetting geometry with Au2 moved to mid-vacuum (8 A from either slab face) has
|F_AB - F1_AB| < 1e-4 eV.  (k5) Gibbs: F_AB <= F1_AB + 1e-9 (reported if violated, with the reason: the product
configurations outside AB's selection).
"""
import dataclasses
import os
import sys
import warnings

import numpy as np
import torch
from ase import Atoms

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import probe_t5_terms as PT  # noqa: E402
import trace_redox.module as M  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics.kernels import gram_self_energy  # noqa: E402
from trace_redox.diabatic import (canonical_free_energy, configuration_energies, density_matrix,  # noqa: E402
                                  diabatic_hamiltonian, formal_ladder, switched_hamiltonian)
from eval_aumgo import AU, references  # noqa: E402

ARGS = {}
FROZEN = {"phi": None}
TERMS = ("ladder", "shifts", "self", "bare", "response", "exclusion18", "overlap", "ind26", "alpha28", "band")
RESPONSE = ("response", "ind26", "alpha28")
DFT = {"T5": -931.6, "wet_d": -2.7, "wet_u": 928.9}
#: tools/electrostatics/wetting_eda.py, meV: (E1, relax, deformation) shares.  wet_u: 28 ions (converged: 34 ions give
#: -275.7 / +1242.2 / +10.3).  wet_d: the one-Al 10-ion cluster with 26 TIPs; it is not the 4G doped cell, which holds
#: three Al and two band electrons (erratum 2026-10-27).  T5 = wet_d(10, one Al) - wet_u(10) = -903.9 / -281.9 / -33.0.
CLUSTER = {"wet_u": (-272.2, 1238.6, 10.3), "wet_d": (-1189.2, 1111.9, -22.8), "T5": (-903.9, -281.9, -33.0)}


def _install():
    PT._install()
    inner = M.RedoxElectrostatics._diabatic_solution

    def full(self, e_hat, states, b, W, L_units, K_units, Q, T=None, **kwargs):
        ARGS.clear()
        ARGS.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), L=L_units, K=K_units, Q=Q,
                    T=None if T is None else T.detach(), **kwargs)
        return inner(self, e_hat, states, b, W, L_units, K_units, Q, T=T, **kwargs)

    M.RedoxElectrostatics._diabatic_solution = full
    import trace_redox.band as BD
    lowest = BD.lowest_eigenpair

    def frozen_or_lowest(h):
        """The band unit's orbital: the lowest eigenvector, or (first-order state) B's own, frozen."""
        lam, phi = lowest(h)
        f = FROZEN["phi"]
        if f is None:
            return lam, phi
        f = f.to(h)
        assert f.shape == phi.shape, "B's band sites must be AB's"
        return f @ h @ f, f

    BD.lowest_eigenpair = frozen_or_lowest


def evaluate(calc, atoms, charge):
    """One forward pass; a snapshot of everything the split needs."""
    for k in ("O", "own", "ind", "aug", "excl", "elim"):
        PT.STORE.pop(k, None)
    a = atoms.copy()
    a.info = {"charge": float(charge)}
    a.calc = calc
    calc.reset()
    cap = []
    mod = calc.model.electrostatics
    mod._ensemble_last = None
    ARGS.clear()
    hook = mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    E_tot = float(a.get_potential_energy())
    hook.remove()
    last = mod._ensemble_last
    assert last is not None and ARGS and "elim" in PT.STORE, "this structure did not go through the diabatic solve"
    rc = calc.model.electrostatics.redox            # the frame terms' floats, read now (the module overwrites them per forward)
    frame_now = sum(float(getattr(mod, attr)) for flag, attr in (("frame_wall", "_frame_wall_last"),
                    ("bond_pauli", "_bond_pauli_last"), ("bond_pauli_wall", "_bond_pauli_wall_last")) if getattr(rc, flag, False))
    snap = {"store": dict(PT.STORE), "args": dict(ARGS), "r": cap[-1], "E_tot": E_tot, "z": torch.as_tensor(a.numbers),
            "r_cfg": rc, "frame": frame_now,
            "configs": last["configs"].clone(), "H": last["H"].detach().clone(), "kT": float(last["kT"]), "mod": mod}
    snap["omega"] = None if snap["store"].get("aug") is None else snap["store"]["aug"]["omega"].detach()
    return snap


def term_table(s, configs):
    """{term: [C]} for arbitrary configurations in the snapshot's units (probe_t5_terms.terms per configuration)."""
    st, A, mod, r, z = s["store"], s["args"], s["mod"], s["r"], s["z"]
    states, e_hat, b, W = A["states"], A["e_hat"], A["b"], A["W"]
    U, n = int(e_hat.shape[0]), int(z.numel())
    first = int(round(float(states[0])))
    _, E_Z, _, _ = formal_ladder(z, int(mod.redox.max_charge))
    pad = int(states.numel()) - int(E_Z.shape[1])
    E_Z = torch.cat([torch.full((n, pad), float("inf")), E_Z], 1)
    shift = r.state_energies.detach()[:n] - E_Z
    sigma = mod.free_widths(z) if mod.width is not None else st["elim"].assembly.params.sigma
    S0 = gram_self_energy(sigma.detach().to(torch.float64))
    k = (configs - first).clamp(0, int(states.numel()) - 1)
    rows = torch.arange(n)
    C = int(configs.shape[0])
    pick = lambda T_: torch.stack([torch.nan_to_num(T_[rows, k[c, :n]], posinf=0.0).sum() for c in range(C)])
    out = {"ladder": pick(E_Z), "shifts": pick(shift)}
    nq = configs[:, :n].to(torch.float64)
    out["self"] = -0.5 * (S0.unsqueeze(0) * nq * nq).sum(1)
    known = torch.zeros(C)
    if "own" in st:
        out["overlap"] = pick(st["own"].detach())
        known = known + out["overlap"]
    if "ind" in st:
        out["ind26"] = pick(st["ind"][0].detach())
        known = known + out["ind26"]
    if "excl" in st:
        out["exclusion18"] = pick(st["excl"].detach())
        known = known + out["exclusion18"]
    residue = pick(e_hat[:n]) - (out["ladder"] + out["shifts"] + out["self"] + known)
    assert float(residue.abs().max()) <= 1e-9 * (1.0 + float(pick(e_hat[:n]).abs().max())), "(k1) unnamed one-body"
    out["band"] = torch.stack([e_hat[n:, k[c, n:]].sum() if U > n else torch.tensor(0.0) for c in range(C)])
    cf = configs.to(torch.float64)
    elim = st["elim"]
    Om = s["omega"] if s["omega"] is not None else torch.eye(n, dtype=torch.float64)
    # (k3): the solver's b, W are omega u0 and omega W_atoms omega^T
    assert torch.allclose(b, Om @ elim.u0.detach(), rtol=0, atol=1e-9 * (1 + float(b.abs().max()))), "(k3) b"
    assert torch.allclose(W, Om @ elim.W.detach() @ Om.T, rtol=0, atol=1e-9 * (1 + float(W.abs().max()))), "(k3) W"
    m = cf @ Om                                                          # [C, n]: the atoms' charges
    J = elim.J.detach()
    out["bare"] = cf @ b + 0.5 * torch.einsum("ci,ij,cj->c", m, J, m)
    out["response"] = 0.5 * torch.einsum("ci,ij,cj->c", m, elim.W.detach() - J, m)
    if "O" in st:
        out["overlap"] = out.get("overlap", 0.0) + PT.pair_energy(configs, states, U, st["O"])
    if "ind" in st:
        out["ind26"] = out["ind26"] + PT.pair_energy(configs, states, U, st["ind"][1])
    if st.get("aug") is not None and st["aug"].get("pairs") is not None:
        out["band"] = out["band"] + PT.pair_energy(configs, states, U, st["aug"]["pairs"])
    if A.get("diag_correction") is not None:
        with torch.no_grad():
            out["alpha28"] = A["diag_correction"](configs)
    E = energies(s, configs)
    total = sum(out.values())
    assert float((total - E).abs().max()) <= 1e-9 * (1.0 + float(E.abs().max())), "(k1) terms != E"
    return out


def energies(s, configs):
    A = s["args"]
    E = configuration_energies(configs, A["e_hat"], A["states"], A["b"], A["W"],
                               pairs=A.get("pairs"))
    if A.get("diag_correction") is not None:
        with torch.no_grad():
            E = E + A["diag_correction"](configs)
    return E.detach()


def hamiltonian(s, configs):
    """The solver's H on an arbitrary configuration set (same couplings, switch and spin factors)."""
    A, rx = s["args"], s["mod"].redox
    E = energies(s, configs)
    if A["T"] is None:
        return torch.diag(E)
    if rx.diabatic_selection == "amplitude":
        return switched_hamiltonian(E, configs, A["K"], A["T"], float(rx.selection_amplitude_off),
                                    float(rx.selection_amplitude_on), s["kT"], electrons=A.get("electrons")).detach()
    return diabatic_hamiltonian(E, configs, A["K"], A["T"], electrons=A.get("electrons")).detach()


def ensemble(s, configs=None, rho=None, minus_TS=None):
    """{term: <term>, 'coherent', 'entropy'} and F for the snapshot's own canonical ensemble, or for a given rho."""
    if configs is None:
        configs, H = s["configs"], s["H"]
        rho = density_matrix(H, s["kT"]).detach()
        F = float(canonical_free_energy(H, s["kT"]))
        H_check = hamiltonian(s, configs)
        assert float((H_check - H).abs().max()) <= 1e-9 * (1 + float(H.abs().max())), "(k2) H not reproduced"
        minus_TS = F - float((rho * H).sum())
    else:
        H = hamiltonian(s, configs)
    t = term_table(s, configs)
    p = torch.diagonal(rho)
    out = {k: float(p @ v) for k, v in t.items()}
    out["coherent"] = float((rho * H).sum()) - float(p @ torch.diagonal(H))
    out["entropy"] = float(minus_TS)
    return out, sum(out.values())


def product_state(sAB, sA, sB, a_idx, b_idx):
    """rho_A (x) rho_B in AB's units: configurations [P, U_AB], rho [P, P], -kT(S_A + S_B), dropped weight."""
    nAB = int(sAB["z"].numel())
    U_AB = int(sAB["args"]["e_hat"].shape[0])
    rhoA = density_matrix(sA["H"], sA["kT"]).detach()
    rhoB = density_matrix(sB["H"], sB["kT"]).detach()
    nA, nB = int(sA["z"].numel()), int(sB["z"].numel())
    cA, cB = sA["configs"], sB["configs"]
    assert int(cA.shape[1]) == nA, "the isolated dimer has no band unit"
    band_B = int(cB.shape[1]) - nB
    band_AB = U_AB - nAB
    assert band_AB == band_B, "AB and B must carry the same band unit"
    pB = torch.diagonal(rhoB)
    keepB = pB > 1e-10
    dropped = float(pB[~keepB].sum())
    iB = keepB.nonzero().flatten()
    pairs = [(i, j) for i in range(int(cA.shape[0])) for j in iB.tolist()]
    P = torch.zeros(len(pairs), U_AB, dtype=cA.dtype)
    for q, (i, j) in enumerate(pairs):
        P[q, a_idx] = cA[i]
        P[q, b_idx] = cB[j, :nB]
        if band_B:
            P[q, nAB:] = cB[j, nB:]                       # B's band electrons, in B's own (frozen) orbital
    ia = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    jb = torch.tensor([j for _, j in pairs], dtype=torch.long)
    rho = rhoA[ia][:, ia] * rhoB[jb][:, jb]
    rho = rho / torch.trace(rho)
    FA = float(canonical_free_energy(sA["H"], sA["kT"]))
    FB = float(canonical_free_energy(sB["H"], sB["kT"]))
    minus_TS = (FA - float((rhoA * sA["H"]).sum())) + (FB - float((rhoB * sB["H"]).sum()))
    return P, rho, minus_TS, dropped


def bound_energy(elim, y, m):
    """U(y; m) = U_b(y) + m^T A y + 1/2 m^T J m: the bound sector at moments y with free charges m (eliminate_bound_sector's
    functional; quadratic)."""
    a = elim.assembly
    n, nb = a.n_atoms, elim.n_bonds
    theta = y[nb + 3 * n:].reshape(n, 5) if a.quadrupoles else None
    zero = torch.zeros(n, dtype=y.dtype)
    with torch.no_grad():
        Ub = a.energy(y[:nb], zero, y[nb:nb + 3 * n].reshape(n, 3), theta)
    return float(Ub + m @ (elim.A.detach() @ y) + 0.5 * m @ (elim.J.detach() @ m))


def frozen_bound(sAB1, sA, sB, P, rho_p):
    """<U_AB(y_A* (+) y_B*; m) - U_AB(y_AB*; m)>_p >= 0: the bound moments frozen at each fragment's own optimum
    (the analogue of the frozen densities) against their joint optimum, per product configuration.  Bond charges
    must vanish in both fragments (atom units, SPEC 22), so the frozen bonds are zero.  Checks U(y*) against the
    module's closed form e_b0 + u0.m + m W m / 2 (1e-9)."""
    eAB, eA, eB = (s["store"]["elim"] for s in (sAB1, sA, sB))
    nAB, nA, nB = (int(s["z"].numel()) for s in (sAB1, sA, sB))
    OmAB = sAB1["omega"] if sAB1["omega"] is not None else torch.eye(nAB, dtype=torch.float64)
    OmB = sB["omega"] if sB["omega"] is not None else torch.eye(nB, dtype=torch.float64)
    p = torch.diagonal(rho_p)
    quad = eAB.assembly.quadrupoles
    out, worst = 0.0, 0.0
    for c in range(int(P.shape[0])):
        if float(p[c]) < 1e-12:
            continue
        mAB = P[c].to(torch.float64) @ OmAB
        mA = P[c, AU[0]:AU[1] + 1].to(torch.float64)
        nB_units = int(OmB.shape[0])
        cB = torch.cat([P[c, :AU[0]], P[c, nAB:nAB + nB_units - nB]]).to(torch.float64)
        mB = cB @ OmB
        yA, yB = eA.bound_moments(mA).detach(), eB.bound_moments(mB).detach()
        tA, muA, thA = eA.split_moments(yA)
        tB, muB, thB = eB.split_moments(yB)
        tmax = lambda t: float(t.abs().max()) if t.numel() else 0.0
        assert tmax(tA) < 1e-10 and tmax(tB) < 1e-10, \
            "fragment bond charges are not zero: the frozen bonds need a bond map"
        mu = torch.cat([muB, muA]).reshape(-1)
        parts = [torch.zeros(eAB.n_bonds, dtype=torch.float64), mu]
        if quad:
            parts.append(torch.cat([thB, thA]).reshape(-1))
        y_fr = torch.cat(parts)
        y_star = eAB.bound_moments(mAB).detach()
        U_star = bound_energy(eAB, y_star, mAB)
        closed = float(eAB.e_b0 + eAB.u0 @ mAB + 0.5 * mAB @ (eAB.W @ mAB))
        worst = max(worst, abs(U_star - closed) / (1.0 + abs(closed)))
        out += float(p[c]) * (bound_energy(eAB, y_fr, mAB) - U_star)
    assert worst < 1e-9, f"(k6) the bound functional does not reproduce the closed form: {worst:.1e}"
    return out


def split(calc, AB, charge_A, charge_B):
    """The interaction energy of one geometry, split (eV): returns dict and the dimer's total energy."""
    sAB = evaluate(calc, AB, 0.0)
    # A in AB's own cell: its interaction with its periodic images (+18.97 meV for neutral Au2 against the isolated
    # dimer, measured, step 17 s42) is in AB too, and cancels in AB - A; the isolated dimer gives the deformation
    au = Atoms(AB.get_chemical_symbols()[AU[0]:AU[1] + 1], positions=AB.positions[AU[0]:AU[1] + 1], cell=AB.cell,
               pbc=AB.pbc)
    sA = evaluate(calc, au, charge_A)
    s_iso = evaluate(calc, Atoms(au.get_chemical_symbols(), positions=au.positions, pbc=False), charge_A)
    slab = AB[:AU[0]]
    sB = evaluate(calc, slab, charge_B)
    # the first-order Hamiltonian: AB with the band unit's orbital frozen at B's own (the doped slab's two band
    # electrons stay in the orbital they have without Au2; their response to Au2 is relaxation, B's polarisation)
    augB = sB["store"].get("aug")
    FROZEN["phi"] = None if augB is None else augB["phi"].detach()
    try:
        sAB1 = evaluate(calc, AB, 0.0)
    finally:
        FROZEN["phi"] = None
    if augB is not None:
        assert torch.equal(sAB1["store"]["aug"]["sites"], augB["sites"]), "B's band sites must be AB's"
    a_idx, b_idx = torch.tensor(AU), torch.arange(AU[0])
    P, rho_p, mTS, dropped = product_state(sAB1, sA, sB, a_idx, b_idx)
    full, F_AB = ensemble(sAB)
    prod, F1 = ensemble(sAB1, P, rho_p, mTS)
    assert abs(float(sAB1["store"]["elim"].e_b0 - sAB["store"]["elim"].e_b0)) <= 1e-12 and \
        abs((sAB1["E_tot"] - float(sAB1["r"].energy)) - (sAB["E_tot"] - float(sAB["r"].energy))) <= 1e-9, \
        "the frozen orbital must touch the band unit only"
    a_terms, F_A = ensemble(sA)
    b_terms, F_B = ensemble(sB)
    # the frame terms added outside the ensemble (the frame pair's Pauli wall, SPEC 33's law and its wall) are part of
    # the module's energy but not of e_b0 + F: named apart (geometric physics, configuration independent)
    frame = lambda s: s["frame"]
    eq = lambda s, F: float(s["store"]["elim"].e_b0) + F + frame(s)
    for s, F in ((sAB, F_AB), (sA, F_A), (sB, F_B)):
        assert abs(eq(s, F) - float(s["r"].energy)) <= 1e-8 * (1 + abs(float(s["r"].energy))), "(k2) E_q"
    bb = lambda s: s["E_tot"] - float(s["r"].energy)
    d = {"backbone": bb(sAB) - bb(sA) - bb(sB), "frame": frame(sAB) - frame(sA) - frame(sB),
         "e_b0": float(sAB["store"]["elim"].e_b0 - sA["store"]["elim"].e_b0 - sB["store"]["elim"].e_b0)}
    for k in set(prod) | set(a_terms) | set(b_terms):
        d["1:" + k] = prod.get(k, 0.0) - a_terms.get(k, 0.0) - b_terms.get(k, 0.0)
    for k in set(full) | set(prod):
        d["R:" + k] = full.get(k, 0.0) - prod.get(k, 0.0)
    dE = sAB["E_tot"] - sA["E_tot"] - sB["E_tot"]
    assert abs(sum(d.values()) - dE) <= 1e-8 * (1 + abs(dE)), ("sum", sum(d.values()), dE)
    d["dE"] = dE
    d["E_AB"] = sAB["E_tot"]
    d["frozen_bound"] = frozen_bound(sAB1, sA, sB, P, rho_p)       # moves from POL to E1 (pieces)
    d["gibbs"] = F_AB - F1
    d["dropped_B"] = dropped
    d["E_A"] = sA["E_tot"]
    d["E_A_iso"] = s_iso["E_tot"]
    d["q_A"] = (torch.diagonal(density_matrix(sA["H"], sA["kT"])) @ sA["configs"].to(torch.float64)).tolist()
    return d


def pieces(d):
    """E1: the first-order state with the bound moments frozen at each fragment's own optimum (the screened terms
    and e_b0 at the joint optimum, plus <dU_frozen>); POL: -<dU_frozen> (the bound moments' mutual relaxation) +
    SPEC 26 + SPEC 28; CT: the configurations' relaxation.  (Before 2026-10-27 evening the screened term's whole
    W - J part and e_b0 were counted as POL; that counted the fragments' own induced moments, e.g. the slab's surface
    dipoles acting on Au2-, as polarisation, which the frozen densities of the first-principles side contain.)"""
    E1 = sum(v for k, v in d.items() if k.startswith("1:") and k[2:] not in ("ind26", "alpha28")) + d["e_b0"] \
        + d["frozen_bound"]
    POL = sum(v for k, v in d.items() if k.startswith("1:") and k[2:] in ("ind26", "alpha28")) - d["frozen_bound"]
    CT = sum(v for k, v in d.items() if k.startswith("R:"))
    return {"E1": E1, "POL": POL, "CT": CT, "backbone": d["backbone"], "frame": d.get("frame", 0.0)}


def separated_check(calc, refs):
    a = refs[("undoped", "wetting")].copy()
    a.positions[AU, 2] += 29.15 - float(a.positions[AU, 2].mean())
    d = split(calc, a, 0.0, 0.0)
    ok = abs(d["gibbs"]) < 1e-4
    print(f"(k4) separated limit (undoped, Au2 at mid-vacuum): F_AB - F1_AB = {d['gibbs'] * 1e3:+.4f} meV, dE "
          f"{d['dE'] * 1e3:+.2f} meV, frozen-moment penalty {d['frozen_bound'] * 1e3:+.4f} meV -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    _install()
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        if os.environ.get("T5_BOP"):                 # frozen check: SPEC 33's law (+ wall with T5_BOP=wall) switched on
            mod.redox = dataclasses.replace(mod.redox, bond_pauli=True, bond_pauli_wall=os.environ["T5_BOP"] == "wall")
        name = os.path.basename(os.path.dirname(model)) + (f" + SPEC 33 ({os.environ['T5_BOP']}, frozen)" if os.environ.get("T5_BOP") else "")
        print(f"=== {name}", flush=True)
        separated_check(calc, refs)
        if os.environ.get("PROBE_K4_ONLY"):
            continue
        D = {}
        for doping, (qA, qB) in (("undoped", (0.0, 0.0)), ("doped", (-1.0, 1.0))):
            for geom in ("wetting", "non-wetting"):
                D[(doping, geom)] = split(calc, refs[(doping, geom)], qA, qB)
                d = D[(doping, geom)]
                ps = pieces(d)
                print(f"  {doping} {geom}: dE {d['dE']:+.4f} eV = E1 {ps['E1']:+.4f} + POL {ps['POL']:+.4f} + CT "
                      f"{ps['CT']:+.4f} + backbone {ps['backbone']:+.4f}; Gibbs F - F1 {d['gibbs'] * 1e3:+.3f} meV; "
                      f"B weight dropped (p < 1e-10) {d['dropped_B']:.1e}; A charges {np.round(d['q_A'], 3).tolist()}",
                      flush=True)
        rows = {}
        for label, dop in (("wet_u", "undoped"), ("wet_d", "doped")):
            f, u = D[(dop, "wetting")], D[(dop, "non-wetting")]
            pf, pu = pieces(f), pieces(u)
            rows[label] = {k: 1e3 * (pf[k] - pu[k]) for k in pf}
            rows[label]["deformation"] = 1e3 * (f["E_A"] - u["E_A"])
            rows[label]["model"] = 1e3 * (f["E_AB"] - u["E_AB"])
            rows[label]["deformation_iso"] = 1e3 * (f["E_A_iso"] - u["E_A_iso"])
            rows[label]["terms"] = {k: 1e3 * (f[k] - u[k]) for k in f if k[:2] in ("1:", "R:") or k in ("e_b0",)}
        for label in ("wet_u", "wet_d"):                 # the cell dimer's deformation = isolated + its images' change
            rows[label]["images"] = rows[label]["deformation"] - rows[label]["deformation_iso"]
            rows[label]["deformation"] = rows[label]["deformation_iso"]
        rows["T5"] = {k: rows["wet_d"][k] - rows["wet_u"][k]
                      for k in ("E1", "POL", "CT", "backbone", "frame", "deformation", "images")}
        rows["T5"]["terms"] = {k: rows["wet_d"]["terms"].get(k, 0) - rows["wet_u"]["terms"].get(k, 0)
                               for k in set(rows["wet_d"]["terms"]) | set(rows["wet_u"]["terms"])}
        print(f"  {'(meV)':6s} {'total':>8s} {'E1':>8s} {'POL':>8s} {'CT':>8s} {'relax':>8s} {'deform':>8s} {'images':>8s} "
              f"{'backbone':>9s} {'frame':>8s}   | PBE cluster: E1, relax, deform   DFT", flush=True)
        for label in ("wet_u", "wet_d", "T5"):
            x = rows[label]
            tot = x["E1"] + x["POL"] + x["CT"] + x["backbone"] + x["frame"] + x["deformation"] + x["images"]
            if "model" in x:
                assert abs(tot - x["model"]) <= 1e-5, (label, tot, x["model"])     # the slab cancels exactly
            c = CLUSTER[label]
            print(f"  {label:6s} {tot:+8.1f} {x['E1']:+8.1f} {x['POL']:+8.1f} {x['CT']:+8.1f} {x['POL'] + x['CT']:+8.1f} "
                  f"{x['deformation']:+8.1f} {x['images']:+8.1f} {x['backbone']:+9.1f} {x['frame']:+8.1f}   | {c[0]:+8.1f} "
                  f"{c[1]:+8.1f} {c[2]:+6.1f}   {DFT[label]:+7.1f}", flush=True)
        print("  PBE cluster: wet_u converged (28 ions, bare boundary); wet_d and T5 from the one-Al 10-ion cluster with "
              "TIPs (not the 4G three-Al cell: no band electrons; theory note erratum 2026-10-27)", flush=True)
        print(f"  deformation of the isolated dimer (meV): wet_u {rows['wet_u']['deformation_iso']:+.1f} (PBE +10.3), "
              f"wet_d {rows['wet_d']['deformation_iso']:+.1f} (PBE Au2- -22.8)", flush=True)
        print("  T5 by term (meV; 1: first order, R: the configurations' relaxation):", flush=True)
        for k, v in sorted(rows["T5"]["terms"].items(), key=lambda kv: -abs(kv[1])):
            if abs(v) >= 0.05:
                print(f"    {k:14s} {v:+9.1f}", flush=True)


if __name__ == "__main__":
    main()
