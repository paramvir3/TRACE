"""A band unit for delocalised carriers (trace_redox SPEC 20; theory note 2026-10-16 §10).

The extra-electron states of the cation-frame atoms (the host's cation sublattice and its dopants) are carried by
one unit whose charge lives in the lowest orbital phi_1 of the one-carrier Hamiltonian over those sites,

    h = diag(eps) - tau,     eps_v = E(f - e_v) - E(f)   (the exact configuration energy of one electron at v)
    tau_vv' = V_sss(R) exp(ln 20 tanh s) c(R),    V_sss = 1.40 hbar^2 / (m R^2)  (Harrison's universal s-s sigma
                                                  coupling; ESTABLISHED), s learned, c a C^2 switch

holding m = 0, 1, 2 electrons in phi_1:

    E_b(1) = lambda_1                       exact one-carrier tight binding (theory note, Theorem B)
    E_b(2) = 2 lambda_1 + J_11,   J_11 = sum_uv phi_u^2 phi_v^2 V_uv,
             V_vv = e(f-2) - 2 e(f-1) + e(f) + W_vv,  V_uv = W_uv      (a spin pair in a frozen phi_1)

The band's charge -m sits on the sites with weights phi_1^2 (exact for m <= 2: both electrons share phi_1), which
fixes its rows of b and W; its ladder e_b(m) is set so that the band alone reproduces E_b(m) exactly.  Hopping between
a unit u and the band is sum_v T_uv phi_v.

phi_1 and lambda_1 come from a detached eigh refined by two differentiable Newton steps on the bordered
eigen-equation: Newton converges quadratically, so the derivatives are exact to third order wherever lambda_1 is
non-degenerate (Perron-Frobenius on a connected hopping graph), whatever the degeneracy of the higher states.

SPEC 38 (``RedoxConfig(band_form='fermi')``, :meth:`augment_fermi`, :class:`BandCharges`) keeps h and V and lets the
unit hold any number N of electrons, 0 <= N <= 2 B, in thermal Fermi-Dirac occupations of h at the model's kT, with
the mean-field free energy F_b(N) of ``trace_redox.fermi.band_free_energy`` and the site charges -rho_N(v).  The
phi_1 band above is its gapped N <= 2 limit and stays the default (``band_form='phi1'``).  In the Fermi form the
unit's couplings to the other units are zero (a stated limit, SPEC 38.2), and a supercell's band is the band of
the whole supercell: its filling is not fixed by weights trained on the single cell (SPEC 38.4).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

#: hbar^2 / m_e in eV A^2 (as in trace_electrostatics.bond_law).
HBAR2_OVER_ME = 7.619964
#: Harrison's universal s-s sigma coefficient (Harrison, Electronic Structure and the Properties of Solids, 1980).
V_SS_SIGMA = 1.40


def lowest_eigenpair(h: Tensor):
    """``(lambda_1, phi_1)`` of a real symmetric matrix, phi_1 >= 0 summed sign, differentiable to third order at a
    non-degenerate lowest eigenvalue (two Newton steps on [[h - l, -phi], [-phi^T, 0]] from a detached eigh)."""
    n = int(h.shape[0])
    with torch.no_grad():
        ev, V = torch.linalg.eigh(h.detach())
        if n > 1 and float(ev[1] - ev[0]) <= 1e-9 * (1.0 + abs(float(ev[0]))):
            raise ValueError("the band's lowest one-carrier level is degenerate (a disconnected hopping graph?)")
        phi = V[:, 0]
        if float(phi.sum()) < 0:
            phi = -phi
        lam = ev[0]
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    for _ in range(2):
        top = torch.cat([h - lam * eye, -phi.unsqueeze(1)], dim=1)
        bottom = torch.cat([-phi.unsqueeze(0), torch.zeros(1, 1, dtype=h.dtype, device=h.device)], dim=1)
        M = torch.cat([top, bottom], dim=0)
        rhs = -torch.cat([(h - lam * eye) @ phi, (0.5 * (1.0 - phi @ phi)).reshape(1)])
        step = torch.linalg.solve(M, rhs)
        phi, lam = phi + step[:n], lam + step[n]
    return lam, phi


def _switch(r: Tensor, r_cut: float, onset: float = 0.8) -> Tensor:
    x = ((r - onset * r_cut) / ((1.0 - onset) * r_cut)).clamp(0.0, 1.0)
    return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


class BandUnit(nn.Module):
    """The band unit (SPEC 20).  One learned scalar: the hopping factor exp(ln 20 tanh s), initial ``initial_factor``
    (1, Harrison's universal value, unless set; RedoxConfig.band_initial_factor)."""

    def __init__(self, r_cut: float = 3.5, initial_factor: float = 1.0) -> None:
        super().__init__()
        self.r_cut = float(r_cut)
        s0 = math.atanh(math.log(float(initial_factor)) / math.log(20.0))      # 0 exactly for the default 1
        self.log_factor = nn.Parameter(torch.full((), s0))

    @property
    def factor(self) -> Tensor:
        return torch.exp(math.log(20.0) * torch.tanh(self.log_factor))

    def sites(self, frames: Tensor, L: Tensor) -> Tensor:
        """Cation-frame atoms that can take an extra electron (state f - 1 allowed)."""
        return ((frames > 0) & (L <= frames - 1)).nonzero(as_tuple=True)[0]

    def hopping(self, sites: Tensor, edge_index: Tensor, edge_len: Tensor, n_atoms: int) -> Tensor:
        """``tau [B, B]`` between band sites (symmetric, zero diagonal)."""
        dtype, device = edge_len.dtype, edge_len.device
        local = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        local[sites] = torch.arange(int(sites.numel()), device=device)
        src, dst = edge_index[0], edge_index[1]
        keep = (local[src] >= 0) & (local[dst] >= 0) & (src != dst) & (edge_len < self.r_cut)
        a, b, r = local[src[keep]], local[dst[keep]], edge_len[keep]
        value = V_SS_SIGMA * HBAR2_OVER_ME / (r * r) * self.factor.to(dtype) * _switch(r, self.r_cut)
        B = int(sites.numel())
        tau = torch.zeros(B, B, dtype=dtype, device=device).index_put((a, b), value, accumulate=True)
        return 0.5 * (tau + tau.transpose(0, 1))

    def _carriers(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, L: Tensor, f: Tensor,
                  edge_index: Tensor, edge_len: Tensor, pairs):
        """SPEC 20's one-carrier arrays, shared by both forms of the unit: ``(sites [B], eps [B], V [B, B],
        onsite [B], tau [B, B], band_pairs)``, or None when no atom is a band site.  ``onsite`` is V's diagonal before
        the pair tables' part (infinite where a site has no second-electron state)."""
        dtype, device = e_hat.dtype, e_hat.device
        U = int(e_hat.shape[0])
        first = int(round(float(states[0])))
        sites = self.sites(f, L)
        B = int(sites.numel())
        if B == 0:
            return None
        rows = torch.arange(U, device=device)
        col = lambda n: (n - first).clamp(0, int(e_hat.shape[1]) - 1)
        e_f, e_m1 = e_hat[rows, col(f)], e_hat[rows, col(f - 1)]
        e_m2 = torch.where(f - 2 >= L, e_hat[rows, col(f - 2)], torch.full_like(e_f, float("inf")))
        fd = f.to(dtype)
        field = b + W @ fd
        diag = torch.diagonal(W)
        eps = (e_m1 - e_f - field + 0.5 * diag)[sites]                               # one electron at v
        V = W[sites][:, sites].clone()
        onsite = (e_m2 - 2.0 * e_m1 + e_f + diag)[sites]
        V = V - torch.diag(torch.diagonal(V)) + torch.diag(onsite)
        band_pairs = None
        if pairs is not None and int(pairs[0].numel()):
            d_eps, d_V, band_pairs = _band_pair_corrections(pairs, f, first, sites, U, int(e_hat.shape[1]))
            eps = eps + d_eps
            V = V + d_V
        tau = self.hopping(sites, edge_index, edge_len, U)
        return sites, eps, V, onsite, tau, band_pairs

    def augment(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, L: Tensor, K: Tensor, frames: Tensor,
                T: Tensor | None, edge_index: Tensor, edge_len: Tensor, pairs=None):
        """Atom-level arrays -> the same arrays with the band unit appended (last), the band sites' electron states
        removed, and ``omega [U + 1, U]`` mapping unit charges to atoms.  Returns a dict, or None when no atom is a
        band site (an isolated atom, a neutral-frame cluster): then there is no band unit."""
        dtype, device = e_hat.dtype, e_hat.device
        U = int(e_hat.shape[0])
        first = int(round(float(states[0])))
        f = frames.to(torch.long)
        carriers = self._carriers(e_hat, states, b, W, L, f, edge_index, edge_len, pairs)
        if carriers is None:
            return None                                   # no cation-frame site: no band unit (the arrays are unchanged)
        sites, eps, V, onsite, tau, band_pairs = carriers
        fd = f.to(dtype)
        lam, phi = lowest_eigenpair(torch.diag(eps) - tau)
        w = phi * phi                                                                 # charge weights, sum 1
        finite_v = torch.isfinite(onsite)
        J11 = (w[finite_v].unsqueeze(1) * w[finite_v].unsqueeze(0) * V[finite_v][:, finite_v]).sum()
        if not bool(finite_v.all()):
            J11 = J11 + torch.where(w[~finite_v].sum() > 0, torch.tensor(float("inf"), dtype=dtype, device=device),
                                    torch.zeros((), dtype=dtype, device=device))
        # omega: atoms are their own units; the band unit's charge sits on the sites with weights phi^2
        omega = torch.zeros(U + 1, U, dtype=dtype, device=device)
        omega[torch.arange(U), torch.arange(U)] = 1.0
        omega = omega.index_put((torch.full_like(sites, U), sites), w)
        b2 = torch.cat([b, (w * b[sites]).sum().reshape(1)])
        W_col = W[:, sites] @ w                                                       # [U]
        W_bb = w @ W[sites][:, sites] @ w
        W2 = torch.cat([torch.cat([W, W_col.unsqueeze(1)], dim=1),
                        torch.cat([W_col, W_bb.reshape(1)]).unsqueeze(0)], dim=0)
        # the band's ladder at states 0, -1, -2 so that the band alone reproduces E_b(m) exactly:
        # E(f with band at -m) - E(f) = e_b(-m) - m b_b - m (W f)_b + m^2 W_bb / 2
        Wf_b = (W_col * fd).sum()
        S = int(e_hat.shape[1])
        row = torch.full((S,), float("inf"), dtype=dtype, device=device)
        target = {0: torch.zeros((), dtype=dtype, device=device), 1: lam, 2: 2.0 * lam + J11}
        for m, E_m in target.items():
            c = -m - first
            if 0 <= c < S:
                row = row.index_put((torch.tensor([c], device=device),),
                                    (E_m + m * b2[U] + m * Wf_b - 0.5 * m * m * W_bb).reshape(1))
        e2 = torch.cat([e_hat, row.unsqueeze(0)], dim=0)
        # the band sites lose their extra-electron states (the band carries them)
        L2 = L.clone().to(torch.long)
        L2[sites] = f[sites]
        L_b = -2 if bool(torch.isfinite(J11)) and (-2 - first) >= 0 else -1
        L2 = torch.cat([L2, torch.tensor([L_b], dtype=torch.long, device=L2.device)])
        K2 = torch.cat([K.to(torch.long), torch.zeros(1, dtype=torch.long, device=K.device)])
        T2 = torch.zeros(U + 1, U + 1, dtype=dtype, device=device)
        if T is not None:
            T2[:U, :U] = T
            T_band = (T[:, sites] * phi.unsqueeze(0)).sum(1)                         # [U]
            T_band = T_band.index_put((sites,), torch.zeros_like(sites, dtype=dtype))  # sites couple only via phi
            T2[:U, U] = T_band
            T2[U, :U] = T_band
        if band_pairs is not None:
            # the band's charge -m in phi_1 against every other unit's carrier, bilinear (SPEC 23)
            u_idx, delta = band_pairs                                                 # [Q], [Q, S, B]
            agg = (delta * w.unsqueeze(0).unsqueeze(0)).sum(-1)                        # [Q, S]
            Xi_b = torch.zeros(int(u_idx.numel()), S, S, dtype=dtype, device=device)
            for m in (1, 2):
                c = -m - first
                if 0 <= c < S:
                    Xi_b[:, :, c] = m * agg
            band_pairs = (u_idx, torch.full_like(u_idx, U), Xi_b)
        return dict(e_hat=e2, b=b2, W=0.5 * (W2 + W2.transpose(0, 1)), L=L2, K=K2, T=T2, omega=omega, sites=sites,
                    phi=phi, lambda1=lam, J11=J11, eps=eps, tau=tau, pairs=band_pairs)


    def augment_fermi(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, L: Tensor, K: Tensor, frames: Tensor,
                      T: Tensor | None, edge_index: Tensor, edge_len: Tensor, pairs=None, *, kT: float,
                      reference: int):
        """SPEC 38: atom-level arrays -> the same arrays with the Fermi band unit appended (last).

        The unit's state s stands for the filling N = N0 - s of the band, N0 = ``reference`` (its charge is -N, so the
        solver is given Q + N0).  It holds the fillings 0 .. N_max that fit on the state axis, N_max = 2 B, or 1 when
        a site has no second-electron state.  With the atoms at n,

            E(n, N) = E_at(n) + F_b(N) + sum_v rho_N(v) c_v(n),
            c_v(n)  = - sum_u W_vu (n_u - f_u) + sum_{u not a site} delta_uv(n_u),

        F_b and rho_N from ``fermi.band_free_energy`` at ``kT``, eps, V, tau and delta as in SPEC 20.  The coupling
        depends on the filling through rho_N, so it is carried by one pair table per atom with the band (the band is
        the second unit) and the band's entries of b and W are zero; the ladder row is F_b, +inf for the states the
        band does not have, whose table columns are zero.  Hopping between a unit and the band is not built: the
        band's row of T is zero.  Returns a dict (``omega`` None; ``charges`` a :class:`BandCharges`), or None when no
        atom is a band site.  A reference outside 0 .. N_max is refused."""
        from .fermi import band_free_energy
        dtype, device = e_hat.dtype, e_hat.device
        U, S = int(e_hat.shape[0]), int(e_hat.shape[1])
        first = int(round(float(states[0])))
        last = first + S - 1
        f = frames.to(torch.long)
        carriers = self._carriers(e_hat, states, b, W, L, f, edge_index, edge_len, pairs)
        if carriers is None:
            return None
        sites, eps, V, onsite, tau, band_pairs = carriers
        B = int(sites.numel())
        capacity = 2 * B if bool(torch.isfinite(onsite).all()) else 1
        if int(reference) != reference or not 0 <= int(reference) <= capacity:
            raise ValueError(f"the band's reference filling {reference} is outside what it holds, 0 .. {capacity}")
        N0 = int(reference)
        K_b, L_b = min(last, N0), max(first, N0 - capacity)
        fillings = [N0 - s for s in range(L_b, K_b + 1)]
        F, rho = band_free_energy(torch.diag(eps) - tau, V, fillings, float(kT))      # [n_f], [n_f, B]
        column = torch.arange(L_b, K_b + 1, device=device) - first                    # the held states' columns
        row = torch.full((S,), float("inf"), dtype=dtype, device=device).index_put((column,), F)
        e2 = torch.cat([e_hat, row.unsqueeze(0)], dim=0)
        b2 = torch.cat([b, torch.zeros(1, dtype=dtype, device=device)])
        W2 = torch.nn.functional.pad(W, (0, 1, 0, 1))
        L2 = L.clone().to(torch.long)
        L2[sites] = f[sites]                                                          # the band carries their electrons
        L2 = torch.cat([L2, torch.tensor([L_b], dtype=torch.long, device=L2.device)])
        K2 = torch.cat([K.to(torch.long), torch.tensor([K_b], dtype=torch.long, device=K.device)])
        T2 = torch.zeros(U + 1, U + 1, dtype=dtype, device=device)
        if T is not None:
            T2[:U, :U] = T
        # one table per atom u with the band: -(n_u - f_u) sum_v W_uv rho_N(v) + sum_v delta_uv(n_u) rho_N(v)
        off_frame = (torch.arange(S, device=device) + first).to(dtype).unsqueeze(0) - f.to(dtype).unsqueeze(1)
        W_sites = 0.5 * (W[:, sites] + W[sites].transpose(0, 1))                      # [U, B]
        table = -off_frame.unsqueeze(2) * (W_sites @ rho.transpose(0, 1)).unsqueeze(1)   # [U, S, n_f]
        if band_pairs is not None:
            u_idx, delta = band_pairs                                                 # [Q], [Q, S, B]
            table = table.index_add(0, u_idx, delta @ rho.transpose(0, 1))
        Xi_b = torch.zeros(U, S, S, dtype=dtype, device=device)
        Xi_b[:, :, column] = table
        atoms = torch.arange(U, device=device)
        shape = torch.zeros(S, U, dtype=dtype, device=device)
        shape = shape.index_put((column.unsqueeze(1), sites.unsqueeze(0)), -rho)      # the band's charge on the atoms
        return dict(e_hat=e2, b=b2, W=W2, L=L2, K=K2, T=T2, omega=None, charges=BandCharges(shape, first),
                    sites=sites, eps=eps, tau=tau, V=V, pairs=(atoms, torch.full_like(atoms, U), Xi_b),
                    reference=N0, capacity=capacity, fillings=fillings, F=F, rho=rho)


class BandCharges:
    """The atom charges of configurations whose last unit is a Fermi band unit (SPEC 38): the band's charge at state
    s is ``shape[s - first]`` [n] (minus the filling's charge shape rho_N on the sites, zero rows for the states the
    band does not have), where SPEC 20's unit has one fixed row of ``omega``."""

    def __init__(self, shape: Tensor, first: int) -> None:
        self.shape = shape                                # [S, n]
        self.first = int(first)

    def _index(self, configs: Tensor) -> Tensor:
        return configs[:, -1].to(torch.long) - self.first

    def onehot(self, configs: Tensor) -> Tensor:
        """``[M, S]``: 1 at the band unit's state in each configuration; ``onehot @ shape`` is :meth:`band`."""
        out = torch.zeros(int(configs.shape[0]), int(self.shape.shape[0]), dtype=self.shape.dtype,
                          device=self.shape.device)
        return out.scatter(1, self._index(configs).to(self.shape.device).unsqueeze(1), 1.0)

    def band(self, configs: Tensor) -> Tensor:
        """``[M, n]``: the band's charge on the atoms."""
        return self.shape[self._index(configs).to(self.shape.device)]

    def atoms(self, configs: Tensor) -> Tensor:
        """``[M, n]``: the atoms' own charges plus the band's."""
        n = int(self.shape.shape[1])
        return configs[:, :n].to(self.shape.dtype) + self.band(configs)


def atom_charges(configs: Tensor, omega, dtype) -> Tensor:
    """``[M, n]`` atom charges of ``configs [M, U]`` for every kind of unit map: None (the units are the atoms), a
    tensor ``omega [U, n]`` (SPEC 20's band unit, orbital units) or :class:`BandCharges` (SPEC 38)."""
    if omega is None:
        return configs.to(dtype)
    if isinstance(omega, BandCharges):
        return omega.atoms(configs).to(dtype)
    return configs.to(dtype) @ omega.to(dtype)


def _band_pair_corrections(pairs, frames: Tensor, first: int, sites: Tensor, U: int, S: int):
    """The two-body pair terms (SPEC 18.5, 23) seen by the band's carriers, exactly for one electron and bilinearly for two:

    * d_eps_v = sum over pairs of v with w at its frame of Xi(f_v - 1, f_w): the configuration energy of one electron
      at v includes them (the band's diagonal is that exact energy);
    * d_V_vv = sum Xi(f_v - 2, f_w) - 2 Xi(f_v - 1, f_w), d_V_vv' = Xi(f-1, f-1) - Xi(f-1, f) - Xi(f, f-1) + Xi(f, f)
      for two band sites (the second differences that J_11 needs);
    * for a non-site unit u and site v: delta_uv(n_u) = Xi(n_u, f_v - 1) - Xi(n_u, f_v) - Xi(f_u, f_v - 1) + Xi(f_u, f_v),
      the change when u leaves its frame next to one band electron at v, returned per (u, v) as ``(u, [Q, S, B])``
      for the caller to weight by phi_1^2."""
    pi, pj, Xi = pairs
    dtype, device = Xi.dtype, Xi.device
    B = int(sites.numel())
    local = torch.full((U,), -1, dtype=torch.long, device=device)
    local[sites] = torch.arange(B, device=device)
    col = lambda n: (n - first).clamp(0, S - 1)
    valid = lambda n: (n - first >= 0) & (n - first < S)
    fi, fj = frames[pi], frames[pj]
    p = torch.arange(int(pi.numel()), device=device)
    si, sj = local[pi] >= 0, local[pj] >= 0
    d_eps = torch.zeros(B, dtype=dtype, device=device)
    d_two = torch.zeros(B, dtype=dtype, device=device)
    x1_i = Xi[p, col(fi - 1), col(fj)]
    x1_j = Xi[p, col(fi), col(fj - 1)]
    x2_i = torch.where(valid(fi - 2), Xi[p, col(fi - 2), col(fj)], torch.zeros_like(x1_i))
    x2_j = torch.where(valid(fj - 2), Xi[p, col(fi), col(fj - 2)], torch.zeros_like(x1_j))
    d_eps = d_eps.index_add(0, local[pi[si]], x1_i[si]).index_add(0, local[pj[sj]], x1_j[sj])
    d_two = d_two.index_add(0, local[pi[si]], x2_i[si] - 2.0 * x1_i[si]).index_add(0, local[pj[sj]], x2_j[sj] - 2.0 * x1_j[sj])
    d_V = torch.diag(d_two)
    both = si & sj
    if bool(both.any()):
        q = p[both]
        cross = (Xi[q, col(fi[both] - 1), col(fj[both] - 1)] - Xi[q, col(fi[both] - 1), col(fj[both])]
                 - Xi[q, col(fi[both]), col(fj[both] - 1)] + Xi[q, col(fi[both]), col(fj[both])])
        a, c = local[pi[both]], local[pj[both]]
        d_V = d_V.index_put((a, c), cross, accumulate=True).index_put((c, a), cross, accumulate=True)
    # one site and one non-site unit
    rows, deltas = [], []
    for site_first in (True, False):
        mask = (si & ~sj) if site_first else (sj & ~si)
        if not bool(mask.any()):
            continue
        q = p[mask]
        if site_first:
            v, u, fv, fu = pi[q], pj[q], fi[q], fj[q]
            table = Xi[q].transpose(1, 2)                                             # [Q, state of u, state of v]
        else:
            v, u, fv, fu = pj[q], pi[q], fj[q], fi[q]
            table = Xi[q]
        k = torch.arange(int(q.numel()), device=device)
        lo = table[k, :, col(fv - 1)] - table[k, :, col(fv)]                          # [Q, S] over n_u
        ref = (table[k, col(fu), col(fv - 1)] - table[k, col(fu), col(fv)]).unsqueeze(1)
        Qn = int(q.numel())
        delta = torch.zeros(Qn, S, B, dtype=dtype, device=device).index_put(
            (k.repeat_interleave(S), torch.arange(S, device=device).repeat(Qn), local[v].repeat_interleave(S)),
            (lo - ref).reshape(-1))
        rows.append(u)
        deltas.append(delta)
    band_pairs = None
    if rows:
        u_all, d_all = torch.cat(rows), torch.cat(deltas)
        uniq, inverse = torch.unique(u_all, return_inverse=True)
        agg = torch.zeros(int(uniq.numel()), S, B, dtype=dtype, device=device).index_add(0, inverse, d_all)
        band_pairs = (uniq, agg)
    return d_eps, d_V, band_pairs
