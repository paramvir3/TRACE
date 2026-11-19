"""Where does the model's Au2 ionic gap come from? (2026-11-01)  For the gas-phase dimer at bond lengths R the
configuration energies E(0,0) and E(+1,-1) the module builds, split term by term, and Delta_ion = E(+1,-1) - E(0,0)
of each term:

    Delta_ion = [ladder(inf) + on-site W(inf)]          (= IE - EA - S0 terms + S0 terms: SPEC 24's isolated ions)
              + [ladder(R) - ladder(inf)]               (the learned environment shift, gated to zero when isolated)
              + [on-site W(R) - on-site W(inf)]         (the bound sector's screening of each site's own charge)
              - K/R                                     (point Coulomb of the carriers)
              - [W_12 - J^G_12]                         (bound-sector screening of the inter-site Gaussian Coulomb)
              + D_12                                    (SPEC 23: true-density penetration, electrostatic)
              + G_12                                    (SPEC 25: Gordon-Kim Pauli, first-principles table)
              + SPEC 26 induction (one-body + pairs) + SPEC 18.5 exclusion (one-body + pairs) + SPEC 28 + band pairs

with J^G_12 = K erf(R / sqrt(2 sigma^2)) / R the Gaussian kernel that SPEC 23's excess K/R - J^G replaces.  The sum is
checked against the module's own ensemble Hamiltonian (H[ion, ion] - H[cov, cov]).  Usage (repo root):
python experiments/benchmarks_4g/probe_dimer_decompose.py MODEL.pt [...]"""
import sys, math, dataclasses, warnings
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase import Atoms  # noqa: E402
import trace_redox.module as M  # noqa: E402
import trace_redox.induction as IND  # noqa: E402
import trace_redox.penetration as PEN  # noqa: E402
import trace_redox.exclusion as EXC  # noqa: E402
import trace_redox.polarizability as POL  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

K_E = 14.399645
CAP = {}
_solve, _ind, _pp, _own = M.RedoxElectrostatics._diabatic_solution, IND.induction_terms, PEN.penetration_pairs, PEN.penetration_self_terms
_exc_fwd, _exc_pairs = EXC.Exclusion.forward, EXC.Exclusion.pair_terms
det = lambda p: None if p is None else tuple(x.detach().clone() for x in p)


def solve(self, e_hat, states, b, W, L, K, Q, T=None, **kw):
    CAP.update(e_hat=e_hat.detach().clone(), states=states.detach().clone(), b=b.detach().clone(), W=W.detach().clone(),
               T=None if T is None else T.detach().clone(), pairs=det(kw.get("pairs")), diag=kw.get("diag_correction"))
    return _solve(self, e_hat, states, b, W, L, K, Q, T=T, **kw)


def ind(*a, **k):
    out = _ind(*a, **k); CAP["ind_E1"], CAP["ind_pairs"] = out[0].detach().clone(), det(out[1]); return out


def pp(*a, **k):
    out = _pp(*a, **k)
    es = _pp(*a, **{**k, "model": "electrostatic"})
    CAP["pen_pairs"], CAP["pen_es"], CAP["sigma"] = det(out), det(es), a[3].detach().clone()
    return out


def own(*a, **k):
    out = _own(*a, **k); CAP["pen_own"] = out.detach().clone(); return out


def exc_fwd(self, *a, **k):
    out = _exc_fwd(self, *a, **k); CAP["exc_one"] = out.detach().clone(); return out


def exc_pairs(self, *a, **k):
    out = _exc_pairs(self, *a, **k); CAP["exc_pairs"] = det(out); return out


M.RedoxElectrostatics._diabatic_solution, IND.induction_terms, PEN.penetration_pairs = solve, ind, pp
PEN.penetration_self_terms, EXC.Exclusion.forward, EXC.Exclusion.pair_terms = own, exc_fwd, exc_pairs


def pair_sum(pairs, idx):
    if pairs is None or not int(pairs[0].numel()):
        return 0.0
    pi, pj, Xi = pairs
    p = torch.arange(int(pi.numel()))
    return float(Xi[p, idx[pi], idx[pj]].sum())


def terms(n2):
    """Every term of E(n) for the dimer's two units in configuration n2 (other units at 0)."""
    e, st, b, W = CAP["e_hat"], CAP["states"], CAP["b"], CAP["W"]
    U = int(e.shape[0])
    n = torch.zeros(U); n[:2] = torch.tensor(n2, dtype=torch.float64)
    idx = torch.round(n - st[0]).long()
    one = lambda x: 0.0 if x is None else float(sum(x[u, idx[u]] for u in range(U)))
    e_tot = float(sum(e[u, idx[u]] for u in range(U)))
    corr = {k: one(CAP.get(k)) for k in ("exc_one", "ind_E1", "pen_own")}
    out = {"ladder": e_tot - sum(corr.values()), **corr, "b": float(b @ n),
           "onsite": 0.5 * float((torch.diagonal(W) * n * n).sum()), "inter": float(n[0] * n[1] * W[0, 1])}
    for k in ("exc_pairs", "ind_pairs"):
        out[k] = pair_sum(CAP.get(k), idx)
    pen, es = pair_sum(CAP.get("pen_pairs"), idx), pair_sum(CAP.get("pen_es"), idx)
    out["pen_es"], out["pauli"] = es, pen - es
    total_pairs = pair_sum(CAP["pairs"], idx)
    out["band_pairs"] = total_pairs - pen - out["exc_pairs"] - out["ind_pairs"]
    cfg = torch.zeros(1, U, dtype=torch.long); cfg[0, :2] = torch.tensor(n2)
    out["spec28"] = 0.0 if CAP["diag"] is None else float(CAP["diag"](cfg)[0])
    return out


def run(calc, mod, R):
    for k in list(CAP):
        CAP.pop(k)
    at = Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]], pbc=False); at.info = {"charge": 0.0}
    at.calc = calc; calc.reset(); at.get_potential_energy()
    a, c = terms((1, -1)), terms((0, 0))
    d = {k: a[k] - c[k] for k in a}
    last = mod._ensemble_last; H = last["H"].detach().numpy(); cfg = last["configs"].tolist()
    rows = {tuple(x[:2]): i for i, x in enumerate(cfg)}
    i, c0 = rows.get((1, -1)), rows.get((0, 0))
    ok = i is not None and c0 is not None
    d["H"] = H[i, i] - H[c0, c0] if ok else float("nan")       # the ionic configuration is unselected when uncoupled
    d["coupling"] = abs(H[c0, i]) if ok else float("nan")
    s = float(CAP["sigma"][0]) if "sigma" in CAP else float("nan")
    d["JG"] = K_E * math.erf(R / math.sqrt(2 * s * s)) / R
    return d


for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu")
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False)
    r = mod.redox
    print(f"=== {model.split('/')[-2][-34:]}: exclusion {r.exclusion}/{r.exclusion_pairs}, induction {r.carrier_induction}, "
          f"penetration {r.carrier_penetration} ({r.carrier_overlap}, {r.pauli_table}), band {r.carrier_band}, "
          f"SPEC 28 {r.state_polarizability}, shift_gate {r.shift_gate}", flush=True)
    inf = run(calc, mod, 30.0)
    iso = inf["ladder"] + inf["onsite"]
    print(f"  isolated (R = 30): ladder + on-site W {iso:+.4f} eV (IE - EA of Au 9.225554 - 2.308610 = 6.916944); sum of "
          f"terms {inf['ladder'] + inf['onsite'] + inf['inter'] + sum(inf[k] for k in ('pen_es', 'pauli', 'ind_E1', 'ind_pairs', 'exc_one', 'exc_pairs', 'spec28', 'band_pairs', 'b', 'pen_own')):+.4f} vs IE - EA - K/R {6.916944 - K_E / 30:+.4f}", flush=True)
    print("   R    | Delta_ion (H) | sum   | shift  | on-site scr | -K/R   | inter scr | D_pen  | Pauli  | induction | "
          "exclusion | SPEC 28 | band  | b     | |H_cov,ion|")
    for R in (2.30, 2.47, 2.60, 2.80):
        d = run(calc, mod, R)
        shift = d["ladder"] - inf["ladder"]
        on_scr = d["onsite"] - inf["onsite"]
        inter_scr = d["inter"] + d["JG"]            # d["inter"] = -W12 (n1 n2 = -1); screening part = -(W12 - J^G)
        # SPEC 23's pair value already carries dq (K/R - J^G) (dq = -1 here): the true-density penetration alone is
        # D_12 = pen_es + (K/R - J^G); counting the excess once more double-counted it (the first run, 2026-11-01)
        D = d["pen_es"] + (K_E / R - d["JG"])
        indu = d["ind_E1"] + d["ind_pairs"]
        excl = d["exc_one"] + d["exc_pairs"]
        total = iso + shift + on_scr - K_E / R + inter_scr + D + d["pauli"] + indu + excl + d["spec28"] \
            + d["band_pairs"] + d["b"] + d["pen_own"]
        print(f"  {R:.2f}  | {d['H']:+7.3f}       | {total:+6.3f}| {shift:+6.3f} | {on_scr:+7.3f}     | {-K_E / R:+6.3f} | "
              f"{inter_scr:+7.3f}   | {D:+6.3f} | {d['pauli']:+6.3f} | {indu:+7.3f}   | {excl:+7.3f}   | "
              f"{d['spec28']:+6.3f}  | {d['band_pairs']:+5.3f} | {d['b']:+5.3f} | {d['coupling']:.3f}", flush=True)
