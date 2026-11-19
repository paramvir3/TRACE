"""The bond-oriented law for the anion Au2- (2026-11-06; theory note section 4c, pre-registered): the dimer Au2- (UKS
doublet) with its contact Au 2.26 A above the O-top O, rotated theta toward the tilted state's far-Au direction, over
almo_tilt.py's 6-ion cluster in the doped point lattice of wetting_eda (one Al3+; TIPs on the cations within 3.5 A of the
quantum ions, as the doped decomposition used).  I_FRZ = E[P_HL] - E_A - E_B and I_ALMO = E_ALMO - E_A - E_B per spin
(almo_split_open's unrestricted machinery), for the dimer and for each neutral Au atom alone (doublet);
D-(theta) = I(Au2-) - I(Au0 contact) - I(Au0 far).  Usage: almo_angular_anion.py PATH.json OUT.json [H] [theta ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import almo_split as AS, almo_split_open as AO, cluster_interaction as CI, wetting_eda as W  # noqa: E402
HARTREE = W.HARTREE


def almo_open(au, charge, surf, qB, lattice, tips):
    """(I_FRZ, I_ALMO) in eV of the Au fragment ``au`` (positions; total charge ``charge``, a doublet) over the cluster."""
    from pyscf import dft, gto
    A_atoms = [("Au", p) for p in au]

    def mol(atoms, ch, spin, ghosts=(), with_tips=False):
        spec = [[sym, tuple(x)] for sym, x in atoms] + [["ghost-" + sym, tuple(x)] for sym, x in ghosts]
        basis = {sym: W.BASIS for sym, _ in list(atoms)} | {"ghost-" + sym: W.BASIS for sym, _ in ghosts}
        ecp = {"Au": W.BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
        if with_tips and len(tips):
            spec += [["Mg1", tuple(t)] for t in tips]; basis["Mg1"] = W.TIP_BASIS; ecp["Mg1"] = "stuttgart"; ch += 2 * len(tips)
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=ch, spin=spin, unit="Angstrom", verbose=0)

    dimer = mol(A_atoms + surf, qB + charge, 1, with_tips=True)
    grids = dft.gen_grid.Grids(dimer); grids.level = 3; grids.build()
    molA = mol(A_atoms, charge, 1); molB = mol(surf, qB, 0, with_tips=True)
    nA = molA.nao_nr(); assert nA + molB.nao_nr() == dimer.nao_nr()
    mfA, eA = W._solve(W._uks(molA, AS._grid_like(molA, grids)))
    mfB, eB = W._solve(W._rks(molB, AS._grid_like(molB, grids), lattice))
    mfAB = W._uks(dimer, grids, lattice)
    de = AO.DirectionalEnergyU(AO.SpanEnergyU(mfAB), mfA, mfB, nA, {"A": ("A",), "B": ("B",)})
    e_frz, e_almo, res = AS.minimise_rot(de, label=" anion ALMO")
    assert res.success, "ALMO did not converge"
    return (e_frz - eA - eB) * HARTREE, (e_almo - eA - eB) * HARTREE


P = json.load(open(sys.argv[1])); out_json = sys.argv[2]
H = float(sys.argv[3]) if len(sys.argv) > 3 else 2.26                      # the contact height (A)
thetas = [float(x) for x in sys.argv[4:]] or [0.0, 40.0, 70.0]
tilted = [np.array(p) for p in P["cluster_au"]["1.0"]]; upright = [np.array(p) for p in P["cluster_au"]["upright"]]
sites = W.quantum_sites(tilted + upright, r_au=3.6)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
lattice, _ = W.doped_lattice(sites, "upright"); tips, lattice = W.tip_capped(surf, lattice)
contact = np.array([upright[0][0], upright[0][1], H])
low = int(np.argmin([p[2] for p in tilted])); e = tilted[1 - low][:2] - tilted[low][:2]; e = np.array([*(e / np.linalg.norm(e)), 0.0])
print(f"{len(surf)} quantum ions, {len(tips)} TIPs, one Al3+ in the point lattice", flush=True)
res = json.load(open(out_json)) if os.path.exists(out_json) else {}
if "contact" not in res:
    res["contact"] = almo_open([contact], 0, surf, qB, lattice, tips); json.dump(res, open(out_json, "w"), indent=1)
print(f"contact Au0 alone: I_FRZ {res['contact'][0]:+.4f}, I_ALMO {res['contact'][1]:+.4f}", flush=True)
for th in thetas:
    key = f"{th:.0f}"
    far = contact + 2.545 * (np.sin(np.radians(th)) * e + np.cos(np.radians(th)) * np.array([0, 0, 1.0]))
    if f"far {key}" not in res:
        res[f"far {key}"] = almo_open([far], 0, surf, qB, lattice, tips); json.dump(res, open(out_json, "w"), indent=1)
    if f"dimer {key}" not in res:
        res[f"dimer {key}"] = almo_open([contact, far], -1, surf, qB, lattice, tips); json.dump(res, open(out_json, "w"), indent=1)
    D = [res[f"dimer {key}"][k] - res["contact"][k] - res[f"far {key}"][k] for k in (0, 1)]
    res[f"D {key}"] = D; json.dump(res, open(out_json, "w"), indent=1)
    print(f"  theta {th:3.0f}: Au2- I_FRZ {res[f'dimer {key}'][0]:+.4f} I_ALMO {res[f'dimer {key}'][1]:+.4f}; far Au0 {res[f'far {key}'][0]:+.4f} / "
          f"{res[f'far {key}'][1]:+.4f} -> D-_FRZ {D[0]:+.4f}, D-_ALMO {D[1]:+.4f} eV", flush=True)
k0, k1 = f"D {thetas[0]:.0f}", f"D {thetas[-1]:.0f}"
print(f"D-_ALMO({thetas[-1]:.0f}) - D-_ALMO({thetas[0]:.0f}) = {res[k1][1] - res[k0][1]:+.4f} eV (neutral +0.700); frozen {res[k1][0] - res[k0][0]:+.4f} (neutral +0.929)", flush=True)
