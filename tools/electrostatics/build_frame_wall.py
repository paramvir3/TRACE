"""Build trace_redox/data/frame_wall.npz: the frame pair's own Pauli wall (2026-10-31).

    python tools/electrostatics/build_frame_wall.py

For each listed frame pair (here Mg2+ - Au0, the 4G pair whose training contacts stop at 2.43 A while the flat-dimer
relaxations of the frozen model fall to ~1.8 A) the wall is E_w(R) = K_TF * T_TF^nad[rho_a, rho_b](R), the
non-additive Thomas-Fermi kinetic energy of the frozen frame densities of trace_redox/data/ion_densities.npz (the SPEC
25 library; gk_parts' 'tf' functional) times one constant K_TF fitted, by least squares on E, to the frame pair's own
frozen-density PBE non-electrostatic interaction ('sph_nonelst_pbe', compare_overlap.run_pair; cached in
$TRACE_FP_CACHE/pauli_fp (default ./fp_cache/pauli_fp) at the Pauli build's window points 2.66-3.49 A and the wall's points 1.8-2.4 A).  Tabulated as a
Chebyshev series of R E_w on [R_MIN, R_CUT] (the penetration tables' range and order), with direct check values K tf(R)
at 25 points in 1.6-4.0 A and the first-principles targets stored beside it."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gk_parts as G
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BOHR, HARTREE = 0.52917721092, 27.211386245988
R_MIN, R_CUT, N_CHEB = 0.5, 8.0, 96
PAIRS = {(12, 79): ((12, 2), (79, 0), "Mg+2_Au+0")}
d = np.load(os.path.join(REPO, "trace_redox", "data", "ion_densities.npz"), allow_pickle=False)
r = d["grid"]
dens = {(int(k[0]), int(k[1])): d["densities"][i] for i, k in enumerate(d["density_keys"])}
radial = G.radial
nodes = radial.chebyshev_nodes(N_CHEB, R_MIN, R_CUT)
pairs, Ks, coeffs, tR, tE, cR, cE = [], [], [], [], [], [], []
for (za, zb), (ka, kb, label) in PAIRS.items():
    tf = lambda R: G.gk_parts(r, dens[ka], dens[kb], R / BOHR)["tf"] * HARTREE
    rows = sorted((json.load(open(f))["pair"][2], json.load(open(f))["fp"]["sph_nonelst_pbe"])
                  for f in glob.glob(os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "pauli_fp", f"{label}_*.json")))
    R = np.array([x for x, _ in rows]); E = np.array([y for _, y in rows])
    T = np.array([tf(x) for x in R])
    K = float(E @ T / (T @ T))
    resid = (K * T - E)
    print(f"{label}: {len(R)} targets at R {np.round(R, 3).tolist()}; K_TF {K:.4f}; residuals (eV) {np.round(resid, 4).tolist()}", flush=True)
    c = radial.chebyshev_fit(np.array([x * K * tf(x) for x in nodes]), N_CHEB)
    Rc = np.linspace(1.6, 4.0, 25); Ec = np.array([K * tf(x) for x in Rc])
    pairs.append([za, zb]); Ks.append(K); coeffs.append(c); tR.append(R); tE.append(E); cR.append(Rc); cE.append(Ec)
out = os.path.join(REPO, "trace_redox", "data", "frame_wall.npz")
np.savez(out, pairs=np.array(pairs), K=np.array(Ks), coefficients=np.array(coeffs), r_min=R_MIN, r_cut=R_CUT,
         targets_R=np.array(tR), targets_E=np.array(tE), checks_R=np.array(cR),
         checks_E=np.array(cE), method=np.array("K_TF x non-additive Thomas-Fermi kinetic energy of the frozen frame "
                                                "densities (SPEC 25 library), K fitted to frozen-density PBE "
                                                "sph_nonelst_pbe; build_frame_wall.py 2026-10-31"))
print("wrote", out)
