"""The Pauli deformation dipole of the Mg2+ - O2- contact in bulk MgO (2026-11-07; theory note section 4e, the Z* fix's
deciding measurement, pre-registered): an O2- with its six Mg2+ (quantum, PBE / def2-tzvp, density fitting) embedded in
the Evjen point lattice of cluster_interaction.rocksalt_lattice (bulk, n = 5; TIPs on the point cations within 3.5 A of
a quantum ion, as wetting_eda does).  Fragment A = the Mg2+ on the +z side of the O, moved along the bond to distance R;
fragment B = the O2- and the other five Mg2+.  From almo_split's machinery: the frozen Heitler-London state (x = 0 of
DirectionalEnergy: the antisymmetrised product of the fragments' own SCF orbitals) and the ALMO state (fragment-local
polarisation, no charge transfer).  Reported per R (electronic dipoles, e A, origin independent because the electron
count matches):
    mu_def = mu_HL - mu_A - mu_B     the Pauli (exchange) deformation dipole of the pair, the shell model's
                                     "deformation dipole" (Dick-Overhauser / Hardy; ESTABLISHED), field independent;
    mu_pol = mu_ALMO - mu_HL         the field-induced (polarisation) part, for the record;
and I_FRZ, I_ALMO (eV).  Then d mu_def,z / dR by central differences at the PBE distance and the Born-charge prediction
Z*_def(O) = -2 d mu_def,z / dR (one axial contact shortens and the other lengthens when the O moves), to be added to the
model's -2.43.  Usage: python tools/electrostatics/deformation_dipole.py OUT.json [R ...]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import almo_split as AS, cluster_interaction as CI, wetting_eda as W  # noqa: E402
BOHR = 0.529177210903
D_PBE = 4.255656884758435 / 2                 # Togo's PBE lattice constant: the nearest-neighbour distance


def setup(R, n=5):
    """Quantum ions (symbol, position), charges, point lattice, TIPs for the +z Mg at distance R from the O."""
    d = W.D                                   # the point lattice at the 4G spacing (2.1325 A); the quantum cluster at d
    o = np.array([0.0, 0.0, d])
    sites = [(0, 0, 1), (0, 0, 0), (0, 0, 2), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)]
    A = [("Mg", o + np.array([0.0, 0.0, R]))]
    B = [("O", o)] + [("Mg", d * np.array(s, float)) for s in sites if s not in ((0, 0, 1), (0, 0, 2))]
    coords, q = CI.rocksalt_lattice(d, n, False, exclude=sites)
    tips, lattice = W.tip_capped(A + B, (coords, q))
    return A, B, lattice, tips


def mol_of(atoms, charge, tips, with_tips):
    from pyscf import gto
    spec = [[sym, tuple(x)] for sym, x in atoms]; basis = {sym: W.BASIS for sym, _ in atoms}; ecp = {}
    if with_tips and len(tips):
        spec += [["Mg1", tuple(t)] for t in tips]; basis["Mg1"] = W.TIP_BASIS; ecp["Mg1"] = "stuttgart"; charge += 2 * len(tips)
    return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=0, unit="Angstrom", verbose=0)


def dipole(mol, dm):
    """Electronic dipole -Tr[D r] in e A."""
    r = mol.intor("int1e_r")                  # [3, nao, nao], a.u.
    return -np.einsum("xij,ji->x", r, dm) * BOHR


def one(R):
    from pyscf import dft
    A, B, lattice, tips = setup(R)
    molA, molB = mol_of(A, 2, tips, False), mol_of(B, 8, tips, True)
    dimer = mol_of(A + B, 10, tips, True)
    nA = molA.nao_nr(); assert nA + molB.nao_nr() == dimer.nao_nr()
    grids = dft.gen_grid.Grids(dimer); grids.level = 3; grids.build()
    t0 = time.time()
    mfA, eA = W._solve(W._rks(molA, AS._grid_like(molA, grids), lattice))
    mfB, eB = W._solve(W._rks(molB, AS._grid_like(molB, grids), lattice))
    mfAB = W._rks(dimer, grids, lattice)
    se = AS.SpanEnergy(mfAB)
    de = AS.DirectionalEnergy(se, mfA, mfB, nA, {"A": ("A",), "B": ("B",)})
    C0 = de.orbitals(de.x0()); P0 = C0 @ np.linalg.inv(C0.T @ se.S @ C0) @ C0.T
    e_hl = se(C0)[0]
    e_frz, e_almo, res = AS.minimise_rot(de, label=f" R={R:.3f} ALMO"); assert res.success, "ALMO did not converge"
    C1 = de.orbitals(res.x); P1 = C1 @ np.linalg.inv(C1.T @ se.S @ C1) @ C1.T
    muA, muB = dipole(molA, mfA.make_rdm1()), dipole(molB, mfB.make_rdm1())
    muHL, muALMO = dipole(dimer, 2.0 * P0), dipole(dimer, 2.0 * P1)
    assert abs(e_hl - e_frz) < 1e-8, (e_hl, e_frz)
    mfAB, eAB = W._solve(W._rks(dimer, grids, lattice))                      # the full SCF: ALMO + charge transfer
    muSCF = dipole(dimer, mfAB.make_rdm1())
    out = {"R": R, "mu_A": muA.tolist(), "mu_B": muB.tolist(), "mu_HL": muHL.tolist(), "mu_ALMO": muALMO.tolist(),
           "mu_SCF": muSCF.tolist(), "mu_CT": (muSCF - muALMO).tolist(),
           "mu_def": (muHL - muA - muB).tolist(), "mu_pol": (muALMO - muHL).tolist(),
           "I_FRZ_eV": (e_frz - eA - eB) * W.HARTREE, "I_ALMO_eV": (e_almo - eA - eB) * W.HARTREE, "I_SCF_eV": (eAB - eA - eB) * W.HARTREE,
           "n_tips": len(tips), "nao": dimer.nao_nr(), "seconds": time.time() - t0}
    print(f"R {R:.3f}: mu_def {np.round(out['mu_def'], 5).tolist()} e A, mu_pol {np.round(out['mu_pol'], 5).tolist()}, mu_CT {np.round(out['mu_CT'], 5).tolist()}, "
          f"I_FRZ {out['I_FRZ_eV']:+.4f}, I_ALMO {out['I_ALMO_eV']:+.4f}, I_SCF {out['I_SCF_eV']:+.4f} eV ({out['seconds']:.0f} s, {out['nao']} AOs, {len(tips)} TIPs)", flush=True)
    return out


def main():
    out_json = sys.argv[1]; Rs = [float(x) for x in sys.argv[2:]] or [1.928, 2.028, D_PBE, 2.228, 2.328]
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for R in Rs:
        key = f"{R:.4f}"
        if key not in res or "mu_CT" not in res[key]:
            res[key] = one(R); json.dump(res, open(out_json, "w"), indent=1)
    keys = sorted(res, key=float); Rv = np.array([float(k) for k in keys]); mz = np.array([res[k]["mu_def"][2] for k in keys])
    if len(keys) >= 3:
        i = int(np.argmin(np.abs(Rv - D_PBE))); total = -2.0
        for name in ("mu_def", "mu_pol", "mu_CT"):
            if all(name in res[k] for k in keys):
                m = np.array([res[k][name][2] for k in keys]); slope = np.gradient(m, Rv)[i]; total += -2 * slope
                print(f"{name},z(R): {np.round(m, 5).tolist()} e A; d/dR at {Rv[i]:.3f}: {slope:+.4f} e; Z*_{name[3:]}(O) = -2 d mu/dR = {-2 * slope:+.4f} e", flush=True)
        print(f"pair budget Z*(O) = -2 + the three channels = {total:+.3f} e against PBE -1.992 (no lattice local-field sum in the pair)", flush=True)


if __name__ == "__main__":
    main()
