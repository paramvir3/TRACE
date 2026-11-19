"""A failure dump's configuration energies split into exact terms, per configuration class (2026-10-20).

Usage (repo root): python experiments/benchmarks_4g/probe_dump_terms.py RUN_DIR/redox_failure.pt [CAP]
       (REPLAY_REDOX / REPLAY_PAULI_SCALE as in replay_redox_failure.py)

The cap refusal of steps 15-16 is a window flooded by charge-transfer configurations (Au+ + a band electron at the
bottom, O- + band electrons within 0.7 eV).  Which term puts them there?  E(n) is a sum of distinct terms (SPEC 0;
probe_t5_terms.py, whose decomposition this reuses and whose checks it keeps: the terms sum to H's diagonal to 1e-9):

  ladder, shifts, self, screened (b^T n + 1/2 n^T W n: Coulomb + bound-sector induction), overlap (SPEC 23 + 25),
  ind26, alpha28, band (the band unit's own energies + its pair corrections), exclusion18.

For the lowest configuration of each class the terms are printed relative to the frame configuration.  The band
electron's energy lambda_1 = sum_v phi_v^2 eps_v - <phi|tau|phi> is split further, phi^2-weighted over the sites:

  eps_v = [e(f-1) - e(f)]_v - (b + W f)_v + 1/2 W_vv + d_eps_v
          ladder+shift+self+one-body   field (Coulomb+induction)   pair corrections (overlap, SPEC 26)

so the site energy and the hopping are separated.
"""
import collections
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import replay_redox_failure as RF  # noqa: E402
import trace_redox.band as BD  # noqa: E402
import trace_redox.exclusion as EX  # noqa: E402
import trace_redox.induction as I  # noqa: E402
import trace_redox.module as M  # noqa: E402
import trace_redox.penetration as P  # noqa: E402
from trace_electrostatics.kernels import gram_self_energy  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import configuration_energies, density_matrix, formal_ladder  # noqa: E402

STORE = {}


def _wrap(name, f, method=False, inputs=None):
    if method:
        def g(self, *a, **k):
            out = f(self, *a, **k)
            STORE[name] = out
            if inputs:
                import inspect
                STORE[inputs] = inspect.signature(f).bind(self, *a, **k).arguments
            return out
    else:
        def g(*a, **k):
            out = f(*a, **k)
            STORE[name] = out
            return out
    return g


def _install():
    sol = M.RedoxElectrostatics._diabatic_solution

    def capture(self, e_hat, states, b, W, L, K, Q, T=None, **kwargs):
        STORE.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), pairs=kwargs.get("pairs"),
                     diag=kwargs.get("diag_correction"))
        return sol(self, e_hat, states, b, W, L, K, Q, T=T, **kwargs)

    M.RedoxElectrostatics._diabatic_solution = capture
    M.eliminate_bound_sector = _wrap("elim", M.eliminate_bound_sector)
    P.penetration_pairs = _wrap("O", P.penetration_pairs)
    P.penetration_self_terms = _wrap("own", P.penetration_self_terms)
    I.induction_terms = _wrap("ind", I.induction_terms)
    BD.BandUnit.augment = _wrap("aug", BD.BandUnit.augment, method=True, inputs="aug_in")
    EX.Exclusion.forward = _wrap("excl", EX.Exclusion.forward, method=True)


def pair_energy(configs, states, U, pairs):
    if pairs is None:
        return torch.zeros(int(configs.shape[0]), dtype=torch.float64)
    zero_e = torch.zeros(U, int(states.numel()), dtype=torch.float64)
    return configuration_energies(configs, zero_e, states, torch.zeros(U, dtype=torch.float64),
                                  torch.zeros(U, U, dtype=torch.float64), pairs=tuple(x.detach() for x in pairs))


def config_terms(mod, r, z):
    """{term: [C]} per configuration (probe_t5_terms.terms without the ensemble average), checked against H."""
    last = mod._ensemble_last
    configs, H = last["configs"], last["H"].detach()
    states, e_hat = STORE["states"], STORE["e_hat"]
    U, n = int(e_hat.shape[0]), int(z.numel())
    first = int(round(float(states[0])))
    _, E_Z, _, _ = formal_ladder(z, int(mod.redox.max_charge))
    pad = int(states.numel()) - int(E_Z.shape[1])
    E_Z = torch.cat([torch.full((n, pad), float("inf")), E_Z], 1)
    shift = r.state_energies.detach()[:n] - E_Z
    sigma = mod.free_widths(z) if mod.width is not None else STORE["elim"].assembly.params.sigma
    S0 = gram_self_energy(sigma.detach().to(torch.float64))
    k = (configs - first).clamp(0, int(states.numel()) - 1)
    rows = torch.arange(n)
    C = int(configs.shape[0])
    out = {"ladder": torch.stack([E_Z[rows, k[c, :n]].sum() for c in range(C)]),
           "shifts": torch.stack([torch.nan_to_num(shift[rows, k[c, :n]], posinf=0.0).sum() for c in range(C)])}
    nq = configs[:, :n].to(torch.float64)
    out["self"] = -0.5 * (S0.unsqueeze(0) * nq * nq).sum(1)
    cf = configs.to(torch.float64)
    out["screened"] = cf @ STORE["b"] + 0.5 * torch.einsum("ci,ij,cj->c", cf, STORE["W"], cf)
    known = torch.zeros(C)
    if "own" in STORE:
        out["overlap"] = torch.stack([STORE["own"][rows, k[c, :n]].sum() for c in range(C)])
        known = known + out["overlap"]
    if "ind" in STORE:
        E1 = STORE["ind"][0].detach()
        out["ind26"] = torch.stack([E1[rows, k[c, :n]].sum() for c in range(C)])
        known = known + out["ind26"]
    if "excl" in STORE:
        X = STORE["excl"].detach()
        out["exclusion18"] = torch.stack([torch.nan_to_num(X[rows, k[c, :n]], posinf=0.0).sum() for c in range(C)])
        known = known + out["exclusion18"]
    seen = torch.stack([torch.nan_to_num(e_hat[rows, k[c, :n]], posinf=0.0).sum() for c in range(C)])
    residue = seen - (out["ladder"] + out["shifts"] + out["self"] + known)
    assert float(residue.abs().max()) <= 1e-9 * (1.0 + float(seen.abs().max())), float(residue.abs().max())
    out["band"] = torch.stack([e_hat[n:, k[c, n:]].sum() if U > n else torch.tensor(0.0) for c in range(C)])
    if "O" in STORE:
        out["overlap"] = out.get("overlap", 0.0) + pair_energy(configs, states, U, STORE["O"])
    if "ind" in STORE:
        out["ind26"] = out["ind26"] + pair_energy(configs, states, U, STORE["ind"][1])
    if STORE.get("aug") is not None and STORE["aug"].get("pairs") is not None:
        out["band"] = out["band"] + pair_energy(configs, states, U, STORE["aug"]["pairs"])
    if STORE["diag"] is not None:
        with torch.no_grad():
            out["alpha28"] = STORE["diag"](configs)
    total = sum(out.values())
    diag = torch.diagonal(H)
    assert float((total - diag).abs().max()) <= 1e-9 * (1.0 + float(diag.abs().max())), \
        float((total - diag).abs().max())
    return out, configs, H, float(last["kT"])


def band_split(z):
    """The band electron's lambda_1 split phi^2-weighted (eV), and the sites' composition."""
    aug, a = STORE["aug"], STORE["aug_in"]
    e_hat, states, b, W, frames = a["e_hat"], a["states"], a["b"], a["W"], a["frames"]
    first = int(round(float(states[0])))
    U = int(e_hat.shape[0])
    rows = torch.arange(U)
    f = frames.to(torch.long)
    col = lambda m: (m - first).clamp(0, int(e_hat.shape[1]) - 1)
    sites = aug["sites"]
    e_f, e_m1 = e_hat[rows, col(f)], e_hat[rows, col(f - 1)]
    field = (b + W @ f.to(torch.float64))
    one = (e_m1 - e_f)[sites]
    fld = -field[sites]
    half = 0.5 * torch.diagonal(W)[sites]
    eps = aug["eps"].detach()
    d_eps = eps - (one + fld + half)
    w = aug["phi"].detach() ** 2
    lam = float(aug["lambda1"])
    tau = aug["tau"].detach()
    hop = float(aug["phi"].detach() @ tau @ aug["phi"].detach())
    zs = z[sites]
    comp = {int(Z): float(w[zs == Z].sum()) for Z in torch.unique(zs).tolist()}
    return {"lambda1": lam, "<ladder+shift+self+1body>": float(w @ one), "<-(b+Wf)>": float(w @ fld),
            "<W_vv/2>": float(w @ half), "<pair corrections>": float(w @ d_eps), "-<phi|tau|phi>": -hop,
            "sum check": float(w @ eps) - hop - lam, "min eps": float(eps.min()), "n sites": int(sites.numel()),
            "phi^2 by element": comp, "participation 1/sum phi^4": float(1.0 / (w * w).sum()),
            "tau factor": float(STORE.get("band_factor", float("nan")))}


def main():
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    _install()
    module, args, kwargs, dump = RF.reproduce(path)
    RF.apply_overrides(module, cap)
    STORE.clear()
    with torch.no_grad():
        _, r = module(*args, **kwargs)
    z = kwargs["data"]["z"]
    n = int(z.numel())
    out, configs, H, kT = config_terms(module, r, z)
    if getattr(module, "band", None) is not None:
        STORE["band_factor"] = float(module.band.factor)
    fr = formal_frame(z)
    p = torch.diagonal(density_matrix(H, kT))
    classes = collections.defaultdict(list)
    for c in range(int(configs.shape[0])):
        changed = (configs[c, :n] != fr).nonzero().flatten().tolist()
        key = tuple(sorted((RF.SYMBOL.get(int(z[j]), str(int(z[j]))), int(configs[c, j]) - int(fr[j])) for j in changed))
        band = int(configs[c, n:].abs().sum()) if configs.shape[1] > n else 0
        classes[(key, band)].append(c)
    frame_c = [c for (key, band), cs in classes.items() if not key and band == 0 for c in cs]
    if not frame_c:
        print("no frame configuration in the ensemble")
        return
    f0 = frame_c[0]
    diag = torch.diagonal(H)
    names = [k for k in ("ladder", "shifts", "self", "screened", "overlap", "ind26", "alpha28", "band", "exclusion18")
             if k in out]
    print(f"{int(configs.shape[0])} configurations; kT {kT}; energies relative to the frame configuration (eV)")
    print(("{:38s} {:>6s} {:>8s} " + " ".join(["{:>9s}"] * len(names))).format("class (lowest member)", "pop", "E", *names))
    rows = sorted(classes.items(), key=lambda kv: min(float(diag[c]) for c in kv[1]))
    for (key, band), cs in rows[:12]:
        c = min(cs, key=lambda i: float(diag[i]))
        label = (" + ".join(f"{s}{q:+d}" for s, q in key) or "frame") + f" | band {band}"
        vals = [float(out[k][c] - out[k][f0]) for k in names]
        print(("{:38s} {:6.3f} {:+8.3f} " + " ".join(["{:+9.3f}"] * len(names))).format(
            label, float(sum(p[i] for i in cs)), float(diag[c] - diag[f0]), *vals))
    if STORE.get("aug") is not None:
        print("\nband electron (SPEC 20), phi^2-weighted over the sites:")
        for k_, v in band_split(z).items():
            print(f"  {k_:28s} {v if not isinstance(v, float) else round(v, 4)}")


if __name__ == "__main__":
    main()
