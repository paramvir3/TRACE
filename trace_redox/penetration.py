"""Charge penetration of charge-state densities (trace_redox SPEC 23).

The configuration energies of the formal model carry every unit's free charge on a compact Gaussian.  At contact
the real charge distributions of two ions overlap: a diffuse carrier (the electron of Au-, the hole of Au+) enters
the neighbour's electron cloud and sees part of its bare nucleus.  The exact electrostatic energy of two spherical
ions a, b (nucleus Z plus spherically averaged electron density rho, N = Z - q electrons) at distance R is

    E_es(R) = Z_a Z_b / R - Z_a phi_b(R) - Z_b phi_a(R) + (rho_a | rho_b)(R)                       (atomic units)

phi_x(d) = 4 pi [ (1/d) int_0^d rho_x r^2 dr + int_d^inf rho_x r dr ] the electron potential of a spherical density,
(rho_a | rho_b)(R) = int 4 pi r^2 rho_a(r) <phi_b>_shell(r; R) dr, with the angular average of phi_b over the sphere
of radius r about a point at distance R,  <phi_b> = [W(R + r) - W(|R - r|)] / (2 r R),  W(x) = int_0^x phi_b(d) d dd.

The pair term added to the configuration energies (SPEC 18.5's two-body slot) replaces the model's Gaussian Coulomb
of the carriers by this, relative to the frames (zero at frames; the frame-frame part is state independent):

    Xi_ab(n_a, n_b; R) = s(R) { D_ab(n_a, n_b; R) + (n_a n_b - f_a f_b) [K / R - J^G_ab(R)] },
    D_ab = [E_es(n_a, n_b) - E_es(f_a, f_b)] - (n_a n_b - f_a f_b) K / R       (the penetration, -> 0 at large R)

with J^G_ab = K erf(R / sqrt(sigma_a^2 + sigma_b^2)) / R the model's free-free Gaussian kernel (rho ~ exp(-r^2/sigma^2))
and s a C^2 switch to zero at r_cut.  R D_ab is tabulated per (element, state) pair as a Chebyshev series in R
(tools/electrostatics/build_ion_densities.py; densities: PBE0, X2C for Z > 36, ANO-RCC + diffuse s/p; exact
Katriel-Davidson tails spliced on; O2- in a Watson sphere matched to the rock-salt MgO Madelung potential).

ESTABLISHED: the electrostatics of overlapping spherical ions (Gordon & Kim, J. Chem. Phys. 56, 3122 (1972)); charge
penetration as a correction to point multipoles in force fields; Watson spheres (Phys. Rev. 111, 1108 (1958)).
NEW-TO-TRACE: state-resolved penetration of first-principles charge-state densities as a two-body term of the
integer-configuration energies.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch import Tensor

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
K_E = 14.399645
#: The tables' range (A): below R_MIN the series is clamped (no two atoms come that close), beyond R_CUT the term is 0.
R_MIN = 0.5
R_CUT = 8.0
#: The C^2 switch runs over [R_CUT - SWITCH_WIDTH, R_CUT].
SWITCH_WIDTH = 1.5
N_CHEBYSHEV = 96
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ion_densities.npz")
#: The first-principles Pauli overlay (RedoxConfig.pauli_table='first_principles'; tools/electrostatics/build_pauli_fp.py)
DATA_PAULI_FP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pauli_fp.npz")


from .radial import (RadialDensity, chebyshev_fit, chebyshev_nodes, electron_count,  # noqa: E402,F401
                     pair_electrostatic_energy, potential)


# ---- the runtime library (torch, differentiable in R and in the Gaussian widths) -----------------------------

def _switch(R: Tensor) -> Tensor:
    x = ((R - (R_CUT - SWITCH_WIDTH)) / SWITCH_WIDTH).clamp(0.0, 1.0)
    return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


class PenetrationLibrary:
    """The tabulated R D_ab(n_a, n_b; R) of SPEC 23, keyed by (Z_a, n_a, Z_b, n_b) with Z_a <= Z_b."""

    def __init__(self, path: str = DATA, pauli_path: str | None = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it with tools/electrostatics/build_ion_densities.py")
        data = np.load(path, allow_pickle=False)
        keys = data["pair_keys"]                       # [T, 4] (Za, na, Zb, nb)
        self.coefficients = torch.as_tensor(data["pair_coefficients"])   # [T, N_CHEBYSHEV]
        # SPEC 25: R K dS, the Pauli half (K from the Gordon-Kim functional at contact), same types (absent in libraries
        # built before it)
        self.gk_coefficients = (torch.as_tensor(data["pair_pauli_coefficients"]) if "pair_pauli_coefficients" in data.files
                                else None)
        if pauli_path is not None:
            # 2026-10-25 (step 20): an overlay replacing only the Pauli series R K dS (K fitted to first principles);
            # it must carry this library's pair keys in the same order
            if not os.path.exists(pauli_path):
                raise FileNotFoundError(f"{pauli_path} is missing: build it with tools/electrostatics/build_pauli_fp.py")
            over = np.load(pauli_path, allow_pickle=False)
            if over["pair_keys"].shape != keys.shape or not np.array_equal(over["pair_keys"], keys):
                raise ValueError(f"{pauli_path}: its pair keys differ from the ion-density library's (rebuild it)")
            self.gk_coefficients = torch.as_tensor(np.array(over["pair_pauli_coefficients"]))
        self.index = {tuple(int(v) for v in k): t for t, k in enumerate(keys)}
        # exchange symmetry: (Z, n, Z, m) and (Z, m, Z, n) describe one pair; the builder fits them separately
        # (measured asymmetry <= 7.8e-6 eV A electrostatic, <= 2.4e-2 eV A Pauli), so hold their mean (exact: the
        # series is linear in its coefficients)
        for (za, na, zb, nb), t in self.index.items():
            if za == zb and na < nb and (za, nb, zb, na) in self.index:
                u = self.index[(za, nb, zb, na)]
                for table in (self.coefficients, self.gk_coefficients):
                    if table is not None:
                        mean = 0.5 * (table[t] + table[u])
                        table[t], table[u] = mean, mean.clone()
        self.elements = {int(z) for z in data["elements"]}
        self.r_min, self.r_cut = float(data["r_min"]), float(data["r_cut"])
        if abs(self.r_cut - R_CUT) > 1e-12 or abs(self.r_min - R_MIN) > 1e-12:
            raise ValueError("the penetration tables were built for another range")

    def has(self, Za: int, na: int, Zb: int, nb: int) -> bool:
        key = (Za, na, Zb, nb) if Za <= Zb else (Zb, nb, Za, na)
        return key in self.index

    def r_times_D(self, types: Tensor, R: Tensor) -> Tensor:
        """R D (eV A) for type indices ``types`` at distances ``R`` (A), both [P]."""
        c = self.coefficients.to(R.dtype).to(R.device)[types]                          # [P, M]
        x = (2.0 * R.clamp(self.r_min, self.r_cut) - (self.r_cut + self.r_min)) / (self.r_cut - self.r_min)
        T_prev, T = torch.ones_like(x), x
        out = c[:, 0] * T_prev + c[:, 1] * T
        for m in range(2, c.shape[1]):
            T_prev, T = T, 2.0 * x * T - T_prev
            out = out + c[:, m] * T
        return out


_LIBRARIES: dict = {}


def library(pauli: str = "gordon_kim") -> PenetrationLibrary:
    """The tables, one instance per Pauli table choice ('gordon_kim': the library's own; 'first_principles': the
    overlay at DATA_PAULI_FP)."""
    if pauli not in ("gordon_kim", "first_principles"):
        raise ValueError(f"pauli table must be 'gordon_kim' or 'first_principles', got {pauli!r}")
    if pauli not in _LIBRARIES:
        _LIBRARIES[pauli] = PenetrationLibrary(pauli_path=DATA_PAULI_FP if pauli == "first_principles" else None)
    return _LIBRARIES[pauli]


def pairs_within(pos: Tensor, cell: Tensor | None, pbc, r_cut: float, return_vectors: bool = False,
                 allow_self: bool = False):
    """Every pair of distinct atoms (i < j) and its periodic images within ``r_cut``: (i, j, R [P]), and with
    ``return_vectors`` also vec [P, 3] = r_j + T - r_i.  An atom meeting its own image within ``r_cut`` is refused
    unless ``allow_self``: then the self-image pairs are left to :func:`self_images_within` (one-body terms)."""
    n = int(pos.shape[0])
    if cell is None or not any(bool(p) for p in (pbc or (False, False, False))):
        i, j = torch.triu_indices(n, n, 1, device=pos.device)
        vec = pos[j] - pos[i]
        R = torch.linalg.norm(vec, dim=-1)
        keep = R < r_cut
        if return_vectors:
            return i[keep], j[keep], R[keep], vec[keep]
        return i[keep], j[keep], R[keep]
    cell = cell.to(pos.dtype)
    flags = [bool(p) for p in pbc]
    inv = torch.linalg.inv(cell.detach())
    heights = 1.0 / torch.linalg.norm(inv, dim=0)                                     # plane spacings
    reach = [int(math.ceil(r_cut / float(h))) if f else 0 for h, f in zip(heights, flags)]
    shifts = torch.tensor([[a, b, c] for a in range(-reach[0], reach[0] + 1) for b in range(-reach[1], reach[1] + 1)
                           for c in range(-reach[2], reach[2] + 1)], dtype=pos.dtype, device=pos.device)
    upper = torch.ones(n, n, dtype=torch.bool, device=pos.device).triu(1)
    # 2026-11-13: the shifts are scanned around the MINIMUM-IMAGE difference.  Scanned around the raw difference
    # (as before) a pair was lost once two atoms were more than about one cell apart in unwrapped coordinates (an MD
    # trajectory, a long relaxation); wrapped inputs give the same pairs as before.  The integer lattice vector is
    # detached: it is piecewise constant, and the distances stay differentiable in the positions and the cell.
    delta = pos.unsqueeze(0) - pos.unsqueeze(1)                                        # [i, j]: r_j - r_i
    nearest = torch.round(delta.detach() @ inv) * torch.tensor([1.0 if f else 0.0 for f in flags], dtype=pos.dtype,
                                                                device=pos.device)
    base = delta - nearest @ cell
    I, J, Rs, Vs = [], [], [], []
    for s in shifts:
        vec = base + (s @ cell)                                                        # [i, j]: r_j + T - r_i
        R2 = (vec * vec).sum(-1)
        if not allow_self and not bool((s == 0).all()) and bool((torch.diagonal(R2) < r_cut * r_cut).any()):
            raise ValueError(f"an atom meets its own periodic image within the penetration cutoff {r_cut} A")
        # i < j over every shift T (both signs): {i, j + T} and its mirror {j, i - T} are one pair, listed once.
        # The norm is taken on the selected pairs only: the table's zero-length self-pairs must not enter it (its
        # backward is masked at first order but 0/0 at second, which detect_anomaly flags in every force-loss step)
        ii, jj = ((R2 < r_cut * r_cut) & upper).nonzero(as_tuple=True)
        I.append(ii)
        J.append(jj)
        Rs.append(torch.sqrt(R2[ii, jj]))
        Vs.append(vec[ii, jj])
    i, j, R = torch.cat(I), torch.cat(J), torch.cat(Rs)
    if return_vectors:
        return i, j, R, torch.cat(Vs)
    return i, j, R


def self_images_within(pos: Tensor, cell: Tensor | None, pbc, r_cut: float):
    """``(i, R)``: every atom and each of its periodic images within ``r_cut``, one of each pair +-T (the half
    space whose first non-zero index is positive), R = |T|.  Empty without periodic boundaries."""
    n = int(pos.shape[0])
    empty = (torch.zeros(0, dtype=torch.long, device=pos.device), pos.new_zeros(0))
    if cell is None or not any(bool(p) for p in (pbc or (False, False, False))):
        return empty
    cell = cell.to(pos.dtype)
    flags = [bool(p) for p in pbc]
    inv = torch.linalg.inv(cell.detach())
    heights = 1.0 / torch.linalg.norm(inv, dim=0)
    reach = [int(math.ceil(r_cut / float(h))) if f else 0 for h, f in zip(heights, flags)]
    I, Rs = [], []
    for a in range(-reach[0], reach[0] + 1):
        for b in range(-reach[1], reach[1] + 1):
            for c in range(-reach[2], reach[2] + 1):
                first = a if a != 0 else (b if b != 0 else c)
                if first <= 0:
                    continue
                T = torch.tensor([a, b, c], dtype=pos.dtype, device=pos.device) @ cell
                R = torch.linalg.norm(T)
                if float(R) < r_cut:
                    I.append(torch.arange(n, device=pos.device))
                    Rs.append(R.expand(n))
    if not I:
        return empty
    return torch.cat(I), torch.cat(Rs)


def _check_library(species: Tensor, lib, model: str) -> None:
    if model not in ("electrostatic", "gordon_kim"):
        raise ValueError(f"model must be 'electrostatic' or 'gordon_kim', got {model!r}")
    if model == "gordon_kim" and lib.gk_coefficients is None:
        raise ValueError("this ion-density library has no Gordon-Kim tables: rebuild it (build_ion_densities.py tables)")
    missing = sorted({int(z) for z in species.tolist()} - lib.elements)
    if missing:
        raise ValueError(f"carrier penetration has no ion densities for Z = {missing}: extend "
                         "tools/electrostatics/build_ion_densities.py")


def _pair_values(zi: Tensor, zj: Tensor, fi: Tensor, fj: Tensor, R: Tensor, sig2_i: Tensor, sig2_j: Tensor,
                 states: Tensor, lib, model: str) -> Tensor:
    """``Xi [P, S, S]`` (state of i, state of j) for pairs of atoms (Z_i, f_i) and (Z_j, f_j) at distance R."""
    dtype, device = R.dtype, R.device
    S = int(states.numel())
    P = int(R.numel())
    Xi = torch.zeros(P, S, S, dtype=dtype, device=device)
    if P == 0:
        return Xi
    st = [int(round(float(x))) for x in states.tolist()]
    st_t = torch.tensor(st, dtype=dtype, device=device)
    fi, fj = fi.to(dtype), fj.to(dtype)
    gauss = K_E * torch.erf(R / torch.sqrt(sig2_i + sig2_j)) / R
    excess = K_E / R - gauss                                                       # K/R - J^G  [P]
    sw = _switch(R)
    coeff = lib.coefficients.to(dtype).to(device)
    if model == "gordon_kim":
        coeff = coeff + lib.gk_coefficients.to(dtype).to(device)       # R (D + G): one series, linear in the coefficients
    M = int(coeff.shape[1])
    lo_z, hi_z = torch.minimum(zi, zj), torch.maximum(zi, zj)
    for za, zb in sorted({(int(a), int(b)) for a, b in zip(lo_z.tolist(), hi_z.tolist())}):
        rows = ((lo_z == za) & (hi_z == zb)).nonzero().flatten()
        # the coefficient rows of every (state of za, state of zb), zero where a density is missing
        C = torch.zeros(S * S, M, dtype=dtype, device=device)
        for a, na in enumerate(st):
            for b, nb in enumerate(st):
                key = (za, na, zb, nb)
                if key in lib.index:
                    C[a * S + b] = coeff[lib.index[key]]
        Rg = R[rows]
        x = (2.0 * Rg.clamp(lib.r_min, lib.r_cut) - (lib.r_cut + lib.r_min)) / (lib.r_cut - lib.r_min)
        T = [torch.ones_like(x), x]
        for _ in range(2, M):
            T.append(2.0 * x * T[-1] - T[-2])
        RD = (torch.stack(T, dim=1) @ C.transpose(0, 1)).reshape(-1, S, S)           # [P_g, S(za), S(zb)]
        # orient to (state of i, state of j): transpose where atom i is the heavier element
        flip = (zi[rows] > zj[rows]).view(-1, 1, 1)
        RD = torch.where(flip, RD.transpose(1, 2), RD)
        fa, fb = fi[rows], fj[rows]
        dq = st_t.view(1, S, 1) * st_t.view(1, 1, S) - (fa * fb).view(-1, 1, 1)
        value = RD / Rg.view(-1, 1, 1) + dq * excess[rows].view(-1, 1, 1)
        at_frame = (st_t.view(1, S, 1) == fa.view(-1, 1, 1)) & (st_t.view(1, 1, S) == fb.view(-1, 1, 1))
        value = torch.where(at_frame, torch.zeros_like(value), value) * sw[rows].view(-1, 1, 1)
        # zero where either state has no density (the Gaussian part alone would be no correction at all)
        has = (C.abs().sum(1) > 0).view(1, S, S)
        has = torch.where(flip, has.transpose(1, 2), has)
        value = torch.where(has, value, torch.zeros_like(value))
        Xi = Xi.index_put((rows,), value)
    return Xi


def penetration_self_terms(species: Tensor, states: Tensor, frames: Tensor, sigma: Tensor, pos: Tensor,
                           cell: Tensor | None = None, pbc=None, lib: PenetrationLibrary | None = None,
                           model: str = "electrostatic") -> Tensor:
    """``[n, S]`` (eV): the pair terms between an atom and its own periodic images, which are always in the same
    state, as one-body ladder terms: sum over half the lattice vectors T (|T| < R_CUT) of Xi_ii(|T|; s, s), i.e.
    1/2 of the sum over all T != 0 per cell.  Zero without periodic images inside the cutoff."""
    lib = library() if lib is None else lib
    _check_library(species, lib, model)
    n, S = int(species.shape[0]), int(states.numel())
    out = pos.new_zeros(n, S)
    i, R = self_images_within(pos, cell, pbc, R_CUT)
    if int(i.numel()) == 0:
        return out
    sig2 = sigma.to(pos.dtype) ** 2
    Xi = _pair_values(species[i], species[i], frames[i], frames[i], R, sig2[i], sig2[i], states, lib, model)
    return out.index_add(0, i, torch.diagonal(Xi, dim1=1, dim2=2))


def penetration_pairs(species: Tensor, states: Tensor, frames: Tensor, sigma: Tensor, pos: Tensor,
                      cell: Tensor | None = None, pbc=None, lib: PenetrationLibrary | None = None,
                      model: str = "electrostatic"):
    """SPEC 23 pair terms ``(pi, pj, Xi [P, S, S])`` (eV) for the configuration energies.

    Every pair within R_CUT (periodic images included, each once; an atom's own images are the one-body
    :func:`penetration_self_terms`).  Per element pair, one Chebyshev basis matrix
    T [P_g, M] and one matmul with the constant coefficients of every state pair: Xi = T C^T / R + Gaussian part.
    State pairs without a tabulated density (an anion unbound in vacuum, Mg-) are zero; frame-frame entries are zero.
    An element missing from the library is refused (the physics would be silently incomplete)."""
    lib = library() if lib is None else lib
    _check_library(species, lib, model)
    i, j, R = pairs_within(pos, cell, pbc, R_CUT, allow_self=True)
    sig2 = sigma.to(pos.dtype) ** 2
    Xi = _pair_values(species[i], species[j], frames[i], frames[j], R, sig2[i], sig2[j], states, lib, model)
    return i, j, Xi
    st = [int(round(float(x))) for x in states.tolist()]
    st_t = torch.tensor(st, dtype=dtype, device=device)
    frames_d = frames.to(dtype).to(device)
    sig2 = sigma.to(dtype) ** 2
    gauss = K_E * torch.erf(R / torch.sqrt(sig2[i] + sig2[j])) / R
    excess = K_E / R - gauss                                                       # K/R - J^G  [P]
    sw = _switch(R)
    coeff = lib.coefficients.to(dtype).to(device)
    if model == "gordon_kim":
        coeff = coeff + lib.gk_coefficients.to(dtype).to(device)       # R (D + G): one series, linear in the coefficients
    M = int(coeff.shape[1])
    zi, zj = species[i], species[j]
    lo_z, hi_z = torch.minimum(zi, zj), torch.maximum(zi, zj)
    for za, zb in sorted({(int(a), int(b)) for a, b in zip(lo_z.tolist(), hi_z.tolist())}):
        rows = ((lo_z == za) & (hi_z == zb)).nonzero().flatten()
        # the coefficient rows of every (state of za, state of zb), zero where a density is missing
        C = torch.zeros(S * S, M, dtype=dtype, device=device)
        for a, na in enumerate(st):
            for b, nb in enumerate(st):
                key = (za, na, zb, nb)
                if key in lib.index:
                    C[a * S + b] = coeff[lib.index[key]]
        Rg = R[rows]
        x = (2.0 * Rg.clamp(lib.r_min, lib.r_cut) - (lib.r_cut + lib.r_min)) / (lib.r_cut - lib.r_min)
        T = [torch.ones_like(x), x]
        for _ in range(2, M):
            T.append(2.0 * x * T[-1] - T[-2])
        RD = (torch.stack(T, dim=1) @ C.transpose(0, 1)).reshape(-1, S, S)           # [P_g, S(za), S(zb)]
        # orient to (state of i, state of j): transpose where atom i is the heavier element
        flip = (zi[rows] > zj[rows]).view(-1, 1, 1)
        RD = torch.where(flip, RD.transpose(1, 2), RD)
        fa, fb = frames_d[i[rows]], frames_d[j[rows]]
        dq = st_t.view(1, S, 1) * st_t.view(1, 1, S) - (fa * fb).view(-1, 1, 1)
        value = RD / Rg.view(-1, 1, 1) + dq * excess[rows].view(-1, 1, 1)
        at_frame = (st_t.view(1, S, 1) == fa.view(-1, 1, 1)) & (st_t.view(1, 1, S) == fb.view(-1, 1, 1))
        value = torch.where(at_frame, torch.zeros_like(value), value) * sw[rows].view(-1, 1, 1)
        # zero where either state has no density (the Gaussian part alone would be no correction at all)
        has = (C.abs().sum(1) > 0).view(1, S, S)
        has = torch.where(flip, has.transpose(1, 2), has)
        value = torch.where(has, value, torch.zeros_like(value))
        Xi = Xi.index_put((rows,), value)
    return i, j, Xi
