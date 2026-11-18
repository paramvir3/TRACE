# Frozen test probe (2026-10-08): pre-registered test P1 of docs/ELECTROSTATICS_THEORY_2026-10-08.md.
# Usage (repo root): python experiments/benchmarks_4g/probe_one_curve.py RUN/model.pt [--validate-only] [N_STRUCTURES]
"""One atom, one curve (Theorem 2): does the consistent on-site coupling move the donor electron's
acceptor from O toward Au2?  Integer transfers on doped Au2/MgO:Al test structures, trained redox model.

The model's own energy of an integer free-charge configuration n (units = atoms, lam = 0):

    E(n) = E_b0 + sum_u Ehat_u(n_u) + b^T n + 1/2 n^T W n,

with b = A y0, W = J - A K^-1 A^T from the exact bound-sector elimination (trace_redox.bound), and
Ehat_u(n) = E_un - J_uu n^2 / 2 (the module's self-Coulomb-removed ladder; the certified SIC cancels
exactly on integer states).  dE(X) = E(Al: +1, X: -1) - E(0) for X in {each Au, the Al's nearest
O, the Al's nearest Mg}.

One atom, one curve: the CHEMICAL (non-Coulomb) on-site hardness of atom u acts on its total
charge q_b + n, while the Coulomb energy stays that of the actual Gaussians (bound width sigma_b,
free width sigma_f).  With D_u = eta_u - J_bb(u, u) (eta_u the bound sector's on-site curvature,
J_bb its bound Gaussian's local self-Coulomb):

    A'  = A + diag(D) [B | 0]                                  (free-bound, on-site)
    J'  = J + diag(D)                                          (free-free, on-site)
    Ehat'_u(n) = Ehat_u(n) - D_u n^2 / 2                        (isolated-atom IE / EA unchanged)
    b' = A' y0,   W' = J' - A' K^-1 A'^T.

The joint form is then Coulomb(actual densities) + 1/2 sum_u D_u (q_b + n)^2 + bond and dipole
stiffnesses, so W' is positive definite whenever D >= 0 (checked and printed).

AMENDMENT (2026-10-08, after the first run, before reading a valid result): the first version set
both on-site couplings to eta, i.e. D_fb = eta - J_fb and D_ff = eta - J_ff.  That deletes the
on-site Coulomb self-energy of the bound and free Gaussians, which differ in width here
(width='hardness'); W' was then indefinite in all 6 structures and the test invalid.  The
criterion below is unchanged.

Pre-registered criterion (P1): averaged over structures and Al sites,
    [dE(Al->O) - dE(Al->Au)]_one-curve - [dE(Al->O) - dE(Al->Au)]_model  >=  +0.5 eV.
--validate-only prints only the mechanics checks (D = 0 reproduces the model; E_b0 equals the
module's energy_bound), not the physics, for development on an unfinished run.
"""
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.bound import assemble_neutral, charge_coulomb_block, eliminate_bound_sector  # noqa: E402

AU, AL, MG, O = 79, 13, 12, 8


def capture(calc, atoms):
    mod = calc.model.electrostatics
    cap_in, cap_out = [], []
    h1 = mod.register_forward_pre_hook(lambda m, a, k: cap_in.append((a, k)), with_kwargs=True)
    h2 = mod.register_forward_hook(lambda m, i, o: cap_out.append(o[1]))
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    a.get_potential_energy()
    h1.remove()
    h2.remove()
    return mod, cap_in[0], cap_out[0]


def operators(mod, inputs, z):
    """(E_b0, Ehat [U, S], states, b, W, b', W', Ehat', D_fb, D_ff, q_b0, PD(W'))."""
    args, kwargs = inputs
    h, pos, ei, vec, L = [x.detach() for x in args[:5]]
    keys = {k: kwargs[k] for k in ("pbc", "cell", "edge_shift") if k in kwargs}
    asm = assemble_neutral(h, pos, ei, vec, L, mod.heads, mod.config, species=z, **keys)
    sigma_f = (mod.width(z) if mod.width is not None else asm.params.sigma).detach()
    elim = eliminate_bound_sector(asm, sigma_f, method=mod.redox.bound_elimination)
    A, J, W, y0 = elim.A.detach(), elim.J.detach(), elim.W.detach(), elim.y0.detach()
    n, nb, m = asm.n_atoms, asm.n_bonds, int(A.shape[1])
    C = torch.zeros(n, m)
    C[:, :nb] = asm._dense_incidence().detach()
    eta = asm.params.eta.detach()
    sigma_b = asm.params.sigma.detach()
    # the chemical on-site hardness: eta minus the bound Gaussian's LOCAL self-Coulomb (open-boundary
    # integral; the periodic image / background parts are interactions and stay in J and A)
    J_bb = torch.diagonal(charge_coulomb_block(pos, sigma_b, sigma_b))
    return elim, A, J, W, y0, C, eta, J_bb


def integer_energy(n_vec, e0, ehat, index, b, W):
    return e0 + sum(float(ehat[u, index[int(v)]]) for u, v in enumerate(n_vec.tolist()) if v != 0) \
        + float(b @ n_vec) + 0.5 * float(n_vec @ (W @ n_vec))


def main():
    ck = sys.argv[1]
    validate = "--validate-only" in sys.argv
    rest = [x for x in sys.argv[2:] if not x.startswith("--")]
    n_struct = int(rest[0]) if rest else 6
    test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    doped = [a for a in test if a.info["system"].startswith("Al")][:n_struct]
    calc = TransformersACECalculator(model_path=ck, device="cpu")
    rows, checks, pds = [], [], []
    for atoms in doped:
        z = torch.tensor(atoms.numbers)
        mod, inputs, res = capture(calc, atoms)
        if mod.redox.self_interaction != 0.0 or res.omega is not None:
            raise SystemExit("probe written for lam = 0 and units = atoms")
        elim, A, J, W, y0, C, eta, J_bb = operators(mod, inputs, z)
        states = res.states.detach()
        index = {int(s): k for k, s in enumerate(states.tolist())}
        # the module's own split: E_un = Ehat_un + J_uu n^2 / 2 with J_uu = diag(J) (periodic)
        ehat = res.state_energies.detach() - 0.5 * torch.diagonal(J).unsqueeze(1) * (states * states).unsqueeze(0)
        e0 = float(elim.e_b0.detach())
        b = elim.u0.detach()
        checks.append((abs(e0 - float(res.energy_bound.detach())),))

        def corrected(scale):
            d = scale * (eta - J_bb)
            A2 = A + torch.diag(d) @ C
            KinvAT = elim.operator.solve(A2.transpose(0, 1)).detach()
            W2 = J + torch.diag(d) - A2 @ KinvAT
            W2 = 0.5 * (W2 + W2.T)
            b2 = A2 @ y0
            e2 = ehat - 0.5 * d.unsqueeze(1) * (states * states).unsqueeze(0)
            return b2, W2, e2, d

        b0c, W0c, e0c, _ = corrected(0.0)
        checks[-1] += (float((W0c - W).abs().max()), float((b0c - b).abs().max()))
        b1, W1, e1, d = corrected(1.0)
        pd = int(torch.linalg.cholesky_ex(W1).info) == 0
        pd_model = int(torch.linalg.cholesky_ex(W).info) == 0
        pds.append(pd)
        q_b0 = C @ y0
        pos = atoms.get_positions()
        au = np.where(atoms.numbers == AU)[0]
        for al in np.where(atoms.numbers == AL)[0]:
            dist = atoms.get_distances(al, np.arange(len(atoms)), mic=True)     # not `d`: that is D
            near_o = int(np.where(atoms.numbers == O)[0][np.argmin(dist[atoms.numbers == O])])
            near_mg = int(np.where(atoms.numbers == MG)[0][np.argmin(dist[atoms.numbers == MG])])
            out = {}
            for tag, (bb, WW, ee) in (("model", (b, W, ehat)), ("one-curve", (b1, W1, e1))):
                E_ref = integer_energy(torch.zeros(len(atoms)), e0, ee, index, bb, WW)
                for name, acc in (("Au", int(au[0])), ("Au'", int(au[1])), ("O", near_o), ("Mg", near_mg)):
                    nv = torch.zeros(len(atoms))
                    nv[al], nv[acc] = 1.0, -1.0
                    out[(tag, name)] = integer_energy(nv, e0, ee, index, bb, WW) - E_ref
            rows.append(out)
        if not validate:
            print(f"  structure: W positive definite {pd_model}, W' positive definite {pd};  D (eV/e^2) and q_b0 (e) by species: " + "  ".join(
                f"{s} {float(d[z == Zs].mean()):+.2f} q {float(q_b0[z == Zs].mean()):+.3f}"
                for s, Zs in (("Al", AL), ("O", O), ("Mg", MG), ("Au", AU))))
    c = np.array(checks)
    print(f"mechanics: max |E_b0 - energy_bound| {c[:, 0].max():.2e} eV;  D = 0: max |W' - W| {c[:, 1].max():.2e}, "
          f"max |b' - b| {c[:, 2].max():.2e}   ({len(doped)} structures)")
    if validate:
        return
    for tag in ("model", "one-curve"):
        mean = {name: np.mean([r[(tag, name)] for r in rows]) for name in ("Au", "Au'", "O", "Mg")}
        print(f"{tag:10s} dE(Al->X), eV, mean over {len(rows)} Al sites: " + "  ".join(f"{k} {v:+.3f}" for k, v in mean.items()))
    diff = {tag: np.array([r[(tag, "O")] - min(r[(tag, "Au")], r[(tag, "Au'")]) for r in rows]) for tag in ("model", "one-curve")}
    shift = diff["one-curve"] - diff["model"]
    print(f"[dE(Al->O) - dE(Al->Au)]: model {diff['model'].mean():+.3f}, one-curve {diff['one-curve'].mean():+.3f};  "
          f"shift {shift.mean():+.3f} eV (min {shift.min():+.3f}, max {shift.max():+.3f})")
    if not all(pds):
        print(f"P1: INVALID -- W' is not positive definite in {pds.count(False)} of {len(pds)} structures")
    else:
        print(f"P1 (shift >= +0.5 eV): {'PASS' if shift.mean() >= 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
