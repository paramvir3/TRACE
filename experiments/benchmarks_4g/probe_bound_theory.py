# Frozen test probe (2026-10-08): theory checks on the bound (SPEC 1) sector of bulk MgO.
# Usage (repo root): python experiments/benchmarks_4g/probe_bound_theory.py RUN/model.pt [...]
"""Four checks on a trained checkpoint, bulk MgO (rocksalt, a = 4.212 A, 2x2x2 cubic, 64 atoms).

C1  Loewner certificate.  K = K_loc + C on the solver's variables (t, mu_hat), with
    K_loc = diag(kappa) (+) blockdiag(A_mu) the local stiffnesses and C the rest (Coulomb).
    If N^T C N >= 0 on the null space N of the constraints, then
        alpha = D^T N (N^T K N)^-1 N^T D  <=  D^T N (N^T K_loc N)^-1 N^T D  <=  D^T K_loc^-1 D
              = sum_b d_b d_b^T / kappa_b + sum_i D_i^T A_mu,i^-1 D_i      (Loewner order),
    so eps_inf <= 1 + 4 pi K_E / Omega * lambda_max(that sum).  Printed: lambda_min(N^T C N),
    eps (exact minimiser), and the two bounds.
C2  The bound, split by bond shell and species pair: which bonds make eps_inf.
C3  What-if hardness settings, exact minimiser eps_inf, and the screened self-term W~_uu of a
    free charge (trace_redox elimination) against the bare J_uu:
      (a) as trained;  (b) Mg-Mg bonds stiff (kappa = 1e6);  (c) nearest-neighbour Mg-O only,
      with kappa_MgO from the two-level bond law kappa = G^3 / (4 h^2) (G = 7.8 eV, the MgO gap;
      h = 1.42 hbar^2/(m d^2) = 2.44 eV, Harrison's V_sp_sigma at d = 2.106 A): kappa = 19.9.
"""
import dataclasses
import math
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from ase.build import bulk  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics.constants import K_E  # noqa: E402
from trace_electrostatics.energy import electrostatic_state  # noqa: E402
from trace_electrostatics.observables import null_space_basis  # noqa: E402
from trace_redox.bound import eliminate_bound_sector  # noqa: E402

TWO_LEVEL_KAPPA = 7.8 ** 3 / (4.0 * (1.42 * 7.62 / 2.106 ** 2) ** 2)


def blocks(assembly):
    """(H, N, D, K_loc) on the solver's free variables (t, mu_hat); rho is a fixed carrier."""
    H = assembly.dense_hessian().detach()
    A = assembly.constraint_operator().detach()
    D = assembly.dipole_derivative().detach()
    nb, n = assembly.n_bonds, assembly.n_atoms
    free = torch.cat((torch.arange(nb), torch.arange(nb + n, H.shape[0])))
    H = H.index_select(0, free).index_select(1, free)
    A = A.index_select(1, free)
    D = D.index_select(0, free)
    K_loc = torch.block_diag(torch.diag(assembly.params.kappa.detach()),
                             *assembly.params.A_mu.detach().unbind(0))
    N = null_space_basis(A, H.shape[0], dtype=H.dtype, device=H.device)
    return 0.5 * (H + H.T), N, D, K_loc


def eps_of(alpha, volume):
    return 1.0 + 4.0 * math.pi * K_E / volume * alpha


def main():
    atoms = bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat((2, 2, 2))
    atoms.info["charge"] = 0.0
    volume = float(atoms.get_volume())
    z = torch.tensor(atoms.numbers)
    for ck in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=ck, device="cpu")
        mod = calc.model.electrostatics
        cap = []
        hook = mod.register_forward_pre_hook(lambda m, a, k: cap.append((a, k)), with_kwargs=True)
        a = atoms.copy()
        a.info = dict(atoms.info)
        a.calc = calc
        a.get_potential_energy()
        hook.remove()
        args, kwargs = cap[0]
        h, pos, ei, vec, L = [x.detach() for x in args[:5]]
        keys = {k: kwargs[k] for k in ("pbc", "cell", "edge_shift") if k in kwargs}
        dense = dataclasses.replace(mod.config, solver="dense")   # the periodic dense lattice blocks
        st = electrostatic_state(h, pos, ei, vec, L, mod.heads, dense, species=z, **keys)
        asm = st.assembly
        print(f"\n=== {ck}   ({asm.n_atoms} atoms, {asm.n_bonds} bonds, Omega {volume:.2f} A^3)")

        H, N, D, K_loc = blocks(asm)
        C = H - K_loc
        NCN = N.T @ C @ N
        lam_c = float(torch.linalg.eigvalsh(0.5 * (NCN + NCN.T)).min())
        alpha = D.T @ N @ torch.linalg.solve(N.T @ H @ N, N.T @ D)
        alpha_c = D.T @ N @ torch.linalg.solve(N.T @ K_loc @ N, N.T @ D)
        alpha_l = D.T @ torch.linalg.solve(K_loc, D)
        e = [float(torch.linalg.eigvalsh(0.5 * (x + x.T)).max()) for x in (alpha, alpha_c, alpha_l)]
        print(f"C1  lambda_min(N^T C N) = {lam_c:+.3e} eV/e^2  (>= 0 needed)")
        nb_ = asm.n_bonds
        C_tt = C[:nb_, :nb_]
        print(f"    charge sector only: lambda_min(C_tt) = {float(torch.linalg.eigvalsh(0.5 * (C_tt + C_tt.T)).min()):+.3e} eV/e^2"
              f"   dipole sector only: lambda_min(C_mumu) = {float(torch.linalg.eigvalsh(0.5 * (C[nb_:, nb_:] + C[nb_:, nb_:].T)).min()):+.3e}")
        print(f"    eps_inf: exact {eps_of(e[0], volume):9.2f}   bound N^T K_loc N {eps_of(e[1], volume):9.2f}"
              f"   local bound {eps_of(e[2], volume):9.2f}   (max eigenvalue)")

        # C2: shell / pair decomposition of the local bound (trace over xyz / 3)
        nb = asm.n_bonds
        bi, bj = asm.bond_index[0].cpu(), asm.bond_index[1].cpu()
        kappa = asm.params.kappa.detach()
        d = D[:nb]
        per_bond = (d * d).sum(1) / kappa / 3.0
        length = torch.linalg.norm(d, dim=1)
        dip = D[nb:]
        per_dip = float(torch.einsum("ia,ij,ja->", dip, torch.linalg.inv(K_loc[nb:, nb:]), dip)) / 3.0
        print(f"C2  local bound split (isotropic part, eps - 1 contribution):  dipoles {4*math.pi*K_E/volume*per_dip:8.3f}")
        rows = {}
        for b in range(nb):
            pair = "-".join(sorted(("Mg" if int(z[bi[b]]) == 12 else "O", "Mg" if int(z[bj[b]]) == 12 else "O")))
            key = (pair, round(float(length[b]), 2))
            r = rows.setdefault(key, [0, 0.0, []])
            r[0] += 1
            r[1] += float(per_bond[b])
            r[2].append(float(kappa[b]))
        for (pair, dist), (count, s, ks) in sorted(rows.items(), key=lambda x: x[0][1]):
            print(f"    {pair:6s} |d| {dist:5.2f} A  n {count:5d}  kappa median {np.median(ks):10.4g}"
                  f"  eps-1 contribution {4*math.pi*K_E/volume*s:10.3f}")

        # C3: what-if hardness settings
        mgmg = torch.tensor([(int(z[bi[b]]) == 12 and int(z[bj[b]]) == 12) for b in range(nb)])
        mgo = torch.tensor([(int(z[bi[b]]) == 12) != (int(z[bj[b]]) == 12) for b in range(nb)])
        nn = mgo & (length > 2.0) & (length < 2.2)        # f_b = 1 nearest-neighbour shell, |d| = 2.106 A
        print(f"    nearest-neighbour Mg-O bonds selected: {int(nn.sum())} (expected 64 * 6 / 2 = 192)")
        settings = {
            "(a) as trained": kappa,
            "(b) Mg-Mg stiff": torch.where(mgmg, torch.full_like(kappa, 1e6), kappa),
            f"(c) NN Mg-O only, kappa {TWO_LEVEL_KAPPA:.1f}": torch.where(nn, torch.full_like(kappa, TWO_LEVEL_KAPPA),
                                                                       torch.full_like(kappa, 1e6)),
        }
        o_site = int(torch.nonzero(z == 8)[0])
        mg_site = int(torch.nonzero(z == 12)[0])
        for name, kap in settings.items():
            asm2 = dataclasses.replace(asm, params=dataclasses.replace(asm.params, kappa=kap))
            H2, N2, D2, _ = blocks(asm2)
            al = D2.T @ N2 @ torch.linalg.solve(N2.T @ H2 @ N2, N2.T @ D2)
            eps2 = eps_of(torch.linalg.eigvalsh(0.5 * (al + al.T)), volume)
            elim = eliminate_bound_sector(asm2, asm2.params.sigma.detach(), method="dense")
            W, J = elim.W.detach(), elim.J.detach()
            print(f"C3  {name:32s} eps_inf {[round(float(x), 3) for x in eps2]}   "
                  f"W~_OO {float(W[o_site, o_site]):.3f} (J {float(J[o_site, o_site]):.3f})   "
                  f"W~_MgMg {float(W[mg_site, mg_site]):.3f} (J {float(J[mg_site, mg_site]):.3f})")


if __name__ == "__main__":
    main()
