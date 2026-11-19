"""Induction by the carriers' true fields (trace_redox SPEC 26; theory note 2026-10-18 section 3, Proposition 4).

The bound sector's moments feel a free charge on atom i through A[i, n_b + 3 j + c] = K lambda_3 (r_i - r_j)_c / r^3
ell_j, with lambda_3 = P(3/2, r^2 / (sigma_f,i^2 + w_j^2)) the enclosed fraction of the Gaussian free charge smeared by
moment j's Gaussian (width w_j).  A carrier (the charge Q = n - f a state adds to its frame) has its true density; by
Gauss its field factor is lambda_c(R, w), the fraction of that density, smeared by w, inside R
(trace_redox.radial.field_factor; 2-D Chebyshev tables in the ion-density library).  The coupling of atom i in state s
is therefore A_i n_i + Q_is v_is with the correction column

    v_is[n_b + 3 j + c] = K (lambda_c,s(R_ij, w_j) - lambda_3(R_ij)) (r_i - r_j)_c / R^3 ell_j  s(R)

(frame charge on the Gaussian, as before; s the C^2 switch of the penetration tables).  The bound energy of a
configuration is quadratic, so the correction is exact:

    dE(n) = dF^T (y0 - M n) - 1/2 dF^T K^-1 dF,   dF = sum_i Q_i v_i(s_i),  M = K^-1 A^T,  y0 = -K^-1 g,

and it splits exactly into one-body terms E1 (added to the ladder) and two-body terms E2 (the pair slot):
:func:`overlap_induction_terms`.  ESTABLISHED: Gauss's law; the charge-penetration damping of induction in force fields.
NEW-TO-TRACE: state-resolved carrier fields from first-principles densities, exact in the integer-configuration
energies.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from .penetration import DATA, K_E, R_CUT, _switch, pairs_within


class FieldFactorTable:
    """lambda_c(R, w) of every carrier state (Z, n), 2-D Chebyshev in (R, w)."""

    def __init__(self, path: str = DATA) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it with tools/electrostatics/build_ion_densities.py")
        d = np.load(path)
        if "field_keys" not in d.files:
            raise ValueError("this ion-density library has no field-factor tables (SPEC 26): rebuild it")
        self.coefficients = torch.as_tensor(d["field_coefficients"])                 # [T, N_R, N_W]
        self.index = {tuple(int(v) for v in k): t for t, k in enumerate(d["field_keys"])}
        self.w_range = tuple(float(x) for x in d["field_w_range"])
        self.r_min, self.r_cut = float(d["r_min"]), float(d["r_cut"])

    @staticmethod
    def _basis(x: Tensor, n: int) -> Tensor:
        T = [torch.ones_like(x), x]
        for _ in range(2, n):
            T.append(2.0 * x * T[-1] - T[-2])
        return torch.stack(T[:n], dim=-1)

    def evaluate(self, types: Tensor, R: Tensor, w: Tensor) -> Tensor:
        coeff = self.coefficients.to(R.dtype).to(R.device)
        x = (2.0 * R.clamp(self.r_min, self.r_cut) - (self.r_cut + self.r_min)) / (self.r_cut - self.r_min)
        lo, hi = self.w_range
        y = (2.0 * w.clamp(lo, hi) - (hi + lo)) / (hi - lo)
        Tx, Ty = self._basis(x, coeff.shape[1]), self._basis(y, coeff.shape[2])            # [P, N_R], [P, N_W]
        out = torch.zeros_like(R)
        for t in torch.unique(types).tolist():                                            # one matmul per type
            sel = (types == t).nonzero().flatten()
            out = out.index_put((sel,), ((Tx[sel] @ coeff[t]) * Ty[sel]).sum(-1))
        return out


_TABLE: FieldFactorTable | None = None


def table() -> FieldFactorTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = FieldFactorTable()
    return _TABLE


def _carrier_factor(tab: FieldFactorTable, types: Tensor, R: Tensor, w: Tensor, gauss: Tensor) -> Tensor:
    """The carriers' field factor (a seam for the tests: replacing it by ``gauss`` must give no correction)."""
    return tab.evaluate(types, R, w)


def dipole_correction_columns(species: Tensor, states: Tensor, frames: Tensor, sigma_free: Tensor, w: Tensor,
                              ell: Tensor, pos: Tensor, cell: Tensor | None = None, pbc=None) -> Tensor:
    """``V_d [3 n, n * S]``: the correction column of every (atom, state) on the moments' dipole rows (3 j + c), the
    only rows it has; zero at frames and for states without a tabulated density.  O(n S) non-zeros, stored on
    3 n rows instead of the n_b + 3 n of the variational basis (n_b ~ 40 n in a dense oxide)."""
    from torch.special import gammainc
    tab = table()
    dtype, device = pos.dtype, pos.device
    n, S = int(species.shape[0]), int(states.numel())
    st = [int(round(float(s))) for s in states.tolist()]
    V = torch.zeros(3 * n, n * S, dtype=dtype, device=device)
    # an atom's own images are skipped: their fields at its moment cancel exactly in +-T pairs (inversion of T)
    i, j, R, vec = pairs_within(pos, cell, pbc, R_CUT, return_vectors=True, allow_self=True)[:4]   # r_j + T - r_i
    if int(i.numel()) == 0:
        return V
    sw = _switch(R)
    for carrier, moment, sep in ((i, j, -vec), (j, i, vec)):                           # sep = r_carrier - r_moment
        s01 = torch.sqrt(sigma_free[carrier] ** 2 + w[moment] ** 2)
        gauss = gammainc(torch.tensor(1.5, dtype=dtype, device=device), (R / s01) ** 2)
        geometry = K_E * sep / (R ** 3).unsqueeze(1) * (ell[moment] * sw).unsqueeze(1)     # [P, 3]
        zc = species[carrier]
        for k, s in enumerate(st):
            for Z in sorted({int(z) for z in zc.tolist()}):
                key = (Z, s)
                if key not in tab.index:
                    continue
                sel = ((zc == Z) & (frames[carrier] != s)).nonzero().flatten()
                if not int(sel.numel()):
                    continue
                lam = _carrier_factor(tab, torch.full_like(sel, tab.index[key]), R[sel], w[moment[sel]], gauss[sel])
                value = (lam - gauss[sel]).unsqueeze(1) * geometry[sel]                    # [p, 3]
                rows = 3 * moment[sel].unsqueeze(1) + torch.arange(3, device=device)
                cols = (carrier[sel] * S + k).unsqueeze(1).expand(-1, 3)
                V = V.index_put((rows.reshape(-1), cols.reshape(-1)), value.reshape(-1), accumulate=True)
    return V


def correction_columns(species: Tensor, states: Tensor, frames: Tensor, sigma_free: Tensor, w: Tensor, ell: Tensor,
                       pos: Tensor, n_columns: int, n_bonds: int, cell: Tensor | None = None, pbc=None) -> Tensor:
    """``V [n_columns, n * S]`` in the variational basis (dipole rows n_bonds + 3 j + c; every other row zero): the
    dense embedding of :func:`dipole_correction_columns`, kept as the reference route (``USE_DENSE_REFERENCE``)."""
    Vd = dipole_correction_columns(species, states, frames, sigma_free, w, ell, pos, cell=cell, pbc=pbc)
    V = torch.zeros(n_columns, int(Vd.shape[1]), dtype=Vd.dtype, device=Vd.device)
    return torch.cat([V[:n_bonds], Vd, V[n_bonds + int(Vd.shape[0]):]], dim=0)


def dipole_block_inverse(operator, n_bonds: int, n_dipole: int) -> Tensor:
    """``(K^-1)_dd [3 n, 3 n]``, the dipole block of the bound operator's inverse.

    For the condensed operator (:class:`trace_redox.bound._CondensedK`), block elimination with a zero bond source
    gives x_p = S^-1 r_p, so (K^-1)_pp = S^-1 with S = P - T^T Lam T the multipole Schur complement it has already
    factorised: no bond-block work at all.  Otherwise a solve with the 3 n dipole unit columns."""
    factor = getattr(operator, "S_factor", None)
    if factor is not None:
        return torch.cholesky_inverse(factor)[:n_dipole, :n_dipole]
    K = getattr(operator, "K", None)
    K = operator.matrix() if K is None else K
    unit = torch.zeros(int(K.shape[0]), n_dipole, dtype=K.dtype, device=K.device)
    unit[n_bonds + torch.arange(n_dipole, device=K.device), torch.arange(n_dipole, device=K.device)] = 1.0
    return operator.solve(unit)[n_bonds:n_bonds + n_dipole]


class InductionColumns:
    """N = K^-1 V of the correction columns, never formed (``V_d`` on the dipole rows, ``C_dd = (K^-1)_dd``).

    What its consumers take from N: its rows at given dipole indices (SPEC 28's moments of the softened atoms),
    N[cols] = (K^-1)[cols, d] V_d = C_dd[cols - n_b] V_d; and full columns contracted with a configuration's carrier
    charges (the SPEC 30 record), K^-1 (V Q) by one solve per configuration."""

    def __init__(self, Vd: Tensor, Cdd: Tensor, n_bonds: int, operator, n_rows: int) -> None:
        self.Vd, self.Cdd, self.n_bonds, self.operator, self.n_rows = Vd, Cdd, int(n_bonds), operator, int(n_rows)

    def rows(self, cols: Tensor) -> Tensor:
        local = cols - self.n_bonds
        if bool((local < 0).any()) or bool((local >= int(self.Cdd.shape[0])).any()):
            raise IndexError("InductionColumns.rows takes dipole rows only (n_bonds <= index < n_bonds + 3 n)")
        return self.Cdd[local] @ self.Vd

    def solve_contracted(self, columns: list, weights: list) -> Tensor:
        """``[m, M]``: K^-1 (V[:, columns[c]] @ weights[c]) for each configuration c."""
        rhs = torch.zeros(self.n_rows, len(columns), dtype=self.Vd.dtype, device=self.Vd.device)
        d = self.Vd.shape[0]
        block = torch.stack([self.Vd[:, cols] @ wts for cols, wts in zip(columns, weights)], dim=1)
        rhs = torch.cat([rhs[:self.n_bonds], block, rhs[self.n_bonds + d:]], dim=0)
        return self.operator.solve(rhs)


#: tests only: the original dense route (V on every variational row, N = K^-1 V formed)
USE_DENSE_REFERENCE = False


def induction_terms(elim, species: Tensor, states: Tensor, frames: Tensor, sigma_free: Tensor, w: Tensor, ell: Tensor,
                    pos: Tensor, Q: Tensor, cell: Tensor | None = None, pbc=None):
    """SPEC 26 for the module: ``(E1 [n, S], pairs, N)`` with N the correction columns' K^-1 V, as a dense tensor on
    the reference route and as :class:`InductionColumns` otherwise.  Same terms to round-off; the scalable route keeps
    O(n S) non-zeros of V on 3 n rows, takes (K^-1)_dd from the condensed factor, and forms no [m, n S] matrix."""
    nb, n = int(elim.n_bonds), int(elim.A.shape[0])
    M = elim.operator.solve(elim.A.transpose(0, 1))
    if USE_DENSE_REFERENCE:
        V = correction_columns(species, states, frames, sigma_free, w, ell, pos, int(elim.A.shape[1]), nb, cell=cell,
                               pbc=pbc)
        N = elim.operator.solve(V)
        E1, pairs = overlap_induction_terms(V, M, N, elim.y0, frames.to(Q.dtype), Q)
        return E1, pairs, N
    Vd = dipole_correction_columns(species, states, frames, sigma_free, w, ell, pos, cell=cell, pbc=pbc)
    Cdd = dipole_block_inverse(elim.operator, nb, 3 * n)
    E1, pairs = overlap_induction_terms_dipole(Vd, M, Cdd, elim.y0, frames.to(Q.dtype), Q, nb)
    return E1, pairs, InductionColumns(Vd, Cdd, nb, elim.operator, int(elim.A.shape[1]))


def _terms(c0: Tensor, C1: Tensor, C2: Tensor, Q: Tensor):
    n, S = int(Q.shape[0]), int(Q.shape[1])
    q = Q.reshape(-1)
    idx = torch.arange(n * S, device=C1.device)
    unit = idx // S
    E1 = (q * c0 - q * q * C1[idx, unit] - 0.5 * q * q * C2[idx, idx]).reshape(n, S)
    iu, iw = torch.triu_indices(n, n, 1, device=C1.device)
    C1r, C2r = C1.reshape(n, S, n), C2.reshape(n, S, n, S)
    a = C1r[iu, :, iw]                                                                 # [P, S]: v_u(s) . M_w
    b = C1r[iw, :, iu]                                                                 # [P, S]: v_w(t) . M_u
    c = C2r[iu, :, iw, :]                                                              # [P, S, S]: v_u(s) . N_w(t)
    Xi = -(Q[iu].unsqueeze(2) * Q[iw].unsqueeze(1)) * (a.unsqueeze(2) + b.unsqueeze(1) + c)
    return E1, (iu, iw, Xi)


def overlap_induction_terms(V: Tensor, M: Tensor, N: Tensor, y0: Tensor, f: Tensor, Q: Tensor):
    """``(E1 [n, S], (pi, pj, Xi [P, S, S]))`` of the module docstring from the correction columns ``V [m, n S]``,
    M = K^-1 A^T [m, n], N = K^-1 V, y0 [m], frames f [n] and carrier charges Q [n, S] (zero at frames)."""
    c0 = V.transpose(0, 1) @ (y0 - M @ f)                                              # [n S]
    C1 = V.transpose(0, 1) @ M                                                         # [n S, n]
    C2 = V.transpose(0, 1) @ N                                                         # [n S, n S]
    return _terms(c0, C1, C2, Q)


def overlap_induction_terms_dipole(Vd: Tensor, M: Tensor, Cdd: Tensor, y0: Tensor, f: Tensor, Q: Tensor,
                                   n_bonds: int):
    """The same terms from ``V_d [3 n, n S]`` (the dipole rows, the only non-zero ones), ``C_dd = (K^-1)_dd``: c0 =
    V_d^T (y0 - M f)_d, C1 = V_d^T M_d, C2 = V_d^T C_dd V_d.  O((3n)^2 n S) for C2 instead of O(m (n S)^2)."""
    d = slice(int(n_bonds), int(n_bonds) + int(Vd.shape[0]))
    c0 = Vd.transpose(0, 1) @ (y0 - M @ f)[d]
    C1 = Vd.transpose(0, 1) @ M[d]
    C2 = Vd.transpose(0, 1) @ (Cdd @ Vd)
    return _terms(c0, C1, C2, Q)
