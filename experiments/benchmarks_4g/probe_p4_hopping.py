# Frozen test probe (2026-10-09): pre-registered test P4 (docs/ELECTROSTATICS_THEORY_2026-10-09_band.md §6).
# Usage (repo root): python experiments/benchmarks_4g/probe_p4_hopping.py RUN/model.pt [N_STRUCTURES]
"""Does coherent hopping of the free carriers lift the training-signal blockage for Au2?

The trained model's single-molecule free sector (Au2 is bonded to the slab: one molecule, weight ~1)
is the concave dual  D(z, mu) = -1/2 z^T W z + mu Q + sum_u Omega_u(nu_u),  nu = mu 1 - b - W z,
with Omega_u the per-unit Mermin ensemble over n = -1, 0, +1 (state energies ehat_x, W_x after the
certified SIC, as in trace_redox.module's exact branch).

Coherent carriers replace sum_u Omega_u by
    Omega(nu) = -kT Tr ln(1 + e^{-(H^a + nu)/kT}) - kT Tr ln(1 + e^{(H^d + nu)/kT}),
    H^a = diag(eps^a) + T,  H^d = diag(eps^d) + T,  eps^a = ehat_x[:, n=-1],  eps^d = -ehat_x[:, n=+1],
    T_uv = -t s_uv on chemical bonds (s_uv the bond switch of trace_redox.fragments.chemical_bonds).
c = -dOmega/dnu (holes - electrons), X = dc/dnu >= 0 by Daleckii-Krein; same Newton scheme as
free_solver with diag(x) -> X.  At t = 0 this is the module's ensemble to kT e^{-G/kT} per unit.

Measured: R = d q_free(Au2) / d eps^a_Au (both Au acceptor levels shifted together; self-consistent,
central finite difference), at t = 0 and t = 0.3 eV.
Mechanics check first: at t = 0 the free charges must equal free_solver.differentiable_solution's.
Pre-registered criterion (P4): |R(t = 0.3 eV)| >= 1e-3 e/eV.
"""
import sys
import warnings

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from probe_one_curve import capture  # noqa: E402
from trace_redox.bound import assemble_neutral, eliminate_bound_sector  # noqa: E402
from trace_redox.fragments import chemical_bonds  # noqa: E402
from trace_redox.free_solver import differentiable_solution  # noqa: E402
from trace_redox.module import certified_removal  # noqa: E402

AU, AL = 79, 13
KT = 0.1


def fermi(x):
    return 0.5 * (1.0 - np.tanh(0.5 * x / KT))                       # 1 / (1 + e^{x/kT}), overflow-free


def dk_kernel(lam, V):
    """K_uv = d diag(f(H + nu))_u / d nu_v  (Daleckii-Krein; finite at degeneracy)."""
    f = fermi(lam)
    d = lam[:, None] - lam[None, :]
    fp = -f * (1.0 - f) / KT
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(np.abs(d) > 1e-10, (f[:, None] - f[None, :]) / d, 0.5 * (fp[:, None] + fp[None, :]))
    VV = V * V                                                         # not used directly; see below
    # K_uv = sum_ij V_ui V_uj F_ij V_vi V_vj
    return np.einsum("ui,uj,ij,vi,vj->uv", V, V, F, V, V, optimize=True)


def carriers(nu, ea, ed, T):
    """(c, X, Omega): free charge per unit, susceptibility dc/dnu, grand potential."""
    la, Va = np.linalg.eigh(np.diag(ea + nu) + T)
    ld, Vd = np.linalg.eigh(np.diag(ed + nu) + T)
    ne = (Va * Va) @ fermi(la)                                         # electrons in acceptor orbitals
    nh = 1.0 - (Vd * Vd) @ fermi(ld)                                   # holes in donor orbitals
    X = -(dk_kernel(la, Va) + dk_kernel(ld, Vd))
    omega = -KT * np.sum(np.logaddexp(0.0, -la / KT)) - KT * np.sum(np.logaddexp(0.0, ld / KT))
    return nh - ne, 0.5 * (X + X.T), omega


def solve(b, W, ea, ed, T, Q, tol=1e-11, max_iter=200):
    U = len(b)
    z, mu = np.zeros(U), 0.0
    one = np.ones(U)

    def dual(z, mu):
        nu = mu * one - b - W @ z
        c, X, om = carriers(nu, ea, ed, T)
        return -0.5 * z @ W @ z + mu * Q + om, c, X

    D, c, X = dual(z, mu)
    for _ in range(max_iter):
        gz, gm = W @ (c - z), Q - c.sum()
        if max(np.abs(gz).max(), abs(gm)) < tol:
            return z, mu, c
        A = W + W @ X @ W
        wx = W @ (X @ one)
        Ainv = np.linalg.pinv(0.5 * (A + A.T), rcond=1e-13)
        a, v = Ainv @ gz, Ainv @ wx
        s = one @ X @ one - wx @ v
        dmu = (gm + wx @ a) / s if s > 1e-14 else 0.0
        dz = a + v * dmu
        step = 1.0
        for _ in range(60):
            Dn, cn, Xn = dual(z + step * dz, mu + step * dmu)
            if Dn >= D - 1e-13 * (1 + abs(D)):
                break
            step *= 0.5
        z, mu, D, c, X = z + step * dz, mu + step * dmu, Dn, cn, Xn
    raise RuntimeError("P4 solver did not converge")


def main():
    ck = sys.argv[1]
    n_struct = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    doped = [a for a in test if a.info["system"].startswith("Al")][:n_struct]
    calc = TransformersACECalculator(model_path=ck, device="cpu")
    results = []
    for s, atoms in enumerate(doped):
        z = torch.tensor(atoms.numbers)
        mod, (args, kw), res = capture(calc, atoms)
        r = mod.redox
        h, pos, ei, vec, L = [x.detach() for x in args[:5]]
        keys = {k: kw[k] for k in ("pbc", "cell", "edge_shift") if k in kw}
        asm = assemble_neutral(h, pos, ei, vec, L, mod.heads, mod.config, species=z, **keys)
        sigma_f = (mod.width(z) if mod.width is not None else asm.params.sigma).detach()
        elim = eliminate_bound_sector(asm, sigma_f, method=r.bound_elimination)
        W = 0.5 * (elim.W + elim.W.T).detach()
        J = elim.J.detach()
        states = res.states.detach()
        ehat = res.state_energies.detach() - 0.5 * torch.diagonal(J).unsqueeze(1) * (states * states).unsqueeze(0)
        removal = certified_removal(W, J, theta=r.sic_theta, smoothing=r.sic_smoothing).detach()
        W_x = 0.5 * ((W - torch.diag(removal)) + (W - torch.diag(removal)).T)
        e_x = ehat + 0.5 * removal.unsqueeze(1) * (states * states).unsqueeze(0)
        b = elim.u0.detach()
        idx = {int(v): k for k, v in enumerate(states.tolist())}
        assert set(idx) == {-1, 0, 1}, "probe written for states -1, 0, +1"
        ea, ed = e_x[:, idx[-1]].numpy(), -e_x[:, idx[1]].numpy()
        bn, Wn = b.numpy(), W_x.numpy()
        bi, bj, bs = chemical_bonds(z, ei, L, coverage=float(r.attention_cutoff))
        S = np.zeros((len(z), len(z)))
        for i, j, w in zip(bi.tolist(), bj.tolist(), bs.detach().tolist()):
            S[i, j] = S[j, i] = max(S[i, j], w)
        au = atoms.numbers == AU

        # mechanics: t = 0 equals the package's single-molecule solve
        ref = differentiable_solution(b, W_x, e_x, states, 0.0, KT).charge.detach().numpy()
        _, _, c0 = solve(bn, Wn, ea, ed, np.zeros_like(S), 0.0)
        mech = float(np.abs(c0 - ref).max())

        row = {"mechanics_max_dc": mech, "free_Au2_t0": float(c0[au].sum())}
        for t in (0.0, 0.3):
            T = -t * S
            delta = 0.05
            q = []
            for sgn in (+1, -1):
                ea2 = ea.copy()
                ea2[au] += sgn * delta
                _, _, c = solve(bn, Wn, ea2, ed, T, 0.0)
                q.append(float(c[au].sum()))
            _, _, cc = solve(bn, Wn, ea, ed, T, 0.0)
            row[f"R_t{t}"] = (q[0] - q[1]) / (2 * delta)
            row[f"free_Au2_t{t}"] = float(cc[au].sum())
            row[f"free_Al_t{t}"] = float(cc[atoms.numbers == AL].sum())
        results.append(row)
        print(f"structure {s}: mechanics max|c(t=0) - package| = {mech:.2e} e;  "
              f"R(t=0) = {row['R_t0.0']:+.3e}  R(t=0.3) = {row['R_t0.3']:+.3e} e/eV;  "
              f"free Au2: t=0 {row['free_Au2_t0.0']:+.5f}, t=0.3 {row['free_Au2_t0.3']:+.5f};  "
              f"free Al: t=0 {row['free_Al_t0.0']:+.3f}, t=0.3 {row['free_Al_t0.3']:+.3f}")
    ok = all(abs(r_["R_t0.3"]) >= 1e-3 for r_ in results)
    mech_ok = all(r_["mechanics_max_dc"] < 1e-6 for r_ in results)
    print(f"mechanics (t = 0 reproduces the package to < 1e-6 e): {'OK' if mech_ok else 'FAILED -- result not valid'}")
    print(f"P4 (|R(t = 0.3 eV)| >= 1e-3 e/eV in every structure): {'PASS' if ok and mech_ok else 'FAIL' if mech_ok else 'INVALID'}")


if __name__ == "__main__":
    main()
