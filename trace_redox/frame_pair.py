"""The frame pair's own interaction, for every element pair of the density library (trace_redox SPEC 39; 2026-11-13).

NEW-TO-TRACE: the frame pair's whole frozen-density interaction as the configuration-independent baseline of the
charge-configuration ensemble, by one procedure for every pair, so that a bond is (this repulsion) + (the ensemble's
hopping energy) and the learned couplings are no longer asked to stand in for a wall.  ESTABLISHED ingredients: the
electrostatics and the electron-gas Pauli repulsion of overlapping frozen densities (Gordon and Kim, J. Chem. Phys. 56,
3122 (1972)); the tight-binding total energy as band energy plus pair repulsion (Harrison, *Electronic Structure and the
Properties of Solids*, as cited in SPEC 27 / 31; cited from memory).

Why (measured 2026-11-12 / 13, experiments/benchmarks_4g/README.md).  SPEC 23 (penetration) and SPEC 25 (Pauli) hold
each carrier state relative to the frame pair and vanish at it, so the frame pair's own interaction was no term of the
model.  The backbone carried it where there are data; below them the charge module's only Au-Au repulsion was the
learned coupling shrinking under compression, which crosses zero at 2.06-2.13 A (DFT bond force +15.2 eV/A at 2.089 A,
the models -12 and +6).  The one-pair wall of 2026-10-31 (trace_redox/wall.py, Mg-Au, K_TF T_TF) is not a law: its
constant spans 0.40-1.02 over six pairs and it holds no electrostatics, which for two neutral atoms is as large as the
Pauli term (Au0-Au0 at 2.331 A: +8.77 - 6.30 = +2.47 eV).

The term.  For every pair of atoms within R_CUT (periodic images included; an atom's own images once per +-T),

    Phi_ab(R) = s(R) { D_ff(R) + f_a f_b [K / R - J^G_ab(R)] + c_ab P_ab(R) },

    D_ff = E_es(f_a, f_b) - f_a f_b K / R      the exact electrostatics of the two frozen frame densities beyond point
                                               charges (penetration; tabulated)
    J^G  = K erf(R / sqrt(sigma_a^2 + sigma_b^2)) / R   the model's own Gaussian kernel of the frame charges (SPEC 23's)
    P    = T_TF^nad + E_x^nad                  the non-additive Thomas-Fermi kinetic and Dirac exchange energy of the
                                               frame densities (tabulated)
    c_ab(R) = 1, or the first-principles ratio where the frame pair's frozen-density PBE interaction has been
           computed: E_fp(R_k) / P(R_k) at the computed distances, a monotone cubic between them; beyond the first
           and the last it keeps its end slope and saturates within 0.3 A (six pairs on 2026-11-13).  One constant
           per pair is not enough: the ratio drifts
           with distance (Au0-Au0 0.843 at 2.68 A to 0.880 at 2.10 A; Au0-O2- 0.842 at 2.0 A to 0.690 at 4.0 A; the
           other four within 1.00-1.07x), and the term is a difference of two large parts.

s the penetration tables' C^2 switch.  With SPEC 23, whose state-dependent entry is Xi = s { D(n) + (n_a n_b - f_a f_b)
[K / R - J^G] }, the pair's electrostatics becomes E_es(n_a, n_b) - n_a n_b J^G for every state pair: SPEC 23 made
absolute; and SPEC 25's carrier-minus-frame Pauli gets its frame.  The term is the same in every configuration (it does
not move the ensemble's populations) and is trained with: the backbone then learns the residual.

Scope, stated as rules and not as lists.  (i) Every pair of the library's elements, at every distance to the cutoff;
an element outside the library is refused.  (ii) Where another term holds a frame interaction measured in situ, that
measurement stands and this term supplies the rest: SPEC 33's wall is the lone s atom's interaction with its WHOLE
site (the anion, its four in-plane cations and the cation below; not a pair function), so while it is on the pair sum
of that site at the measurement's geometry is taken out again, with the wall's own switch (:meth:`site_reference`).
On the measured site the frame terms then sum to the measurement exactly, and elsewhere, and beyond the wall's 4.3 A,
the pair sum acts.  (iii) The first-principles points of two open shells are their high-spin product, the state in
which no electron hops between the two; the bond is the ensemble's.  P itself is the spin-compensated electron gas of
the spin-summed densities, so two open-shell frames WITHOUT first-principles points (c = 1) carry the spin-averaged
value, below the high-spin one (H0-H0 at 0.74 A: 2.60 against 4.67 eV in the same functional): H, C and Ag pairs need
their points before the term is used on them.  Limits: rigid free-ion densities (O2- in the Watson sphere of
rock-salt MgO); c = 1 is 21 % high to 28 % low on the six pairs measured; the pair sum is not the whole frozen
interaction of a bonded atom (SPEC 33's law is the correction where it has been measured); the energy is C^1 at the
knots of c(R).
tools/electrostatics/build_frame_pair.py -> trace_redox/data/frame_pair.npz.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from .penetration import K_E, SWITCH_WIDTH, pairs_within, self_images_within

DATA_FRAME_PAIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frame_pair.npz")

#: The site on which SPEC 33's wall tables were measured (tools/electrostatics/otop_lone_e1.py and otop_lone_almo.py:
#: W.quantum_sites(..., r_au=3.6)), keyed like bond_pauli.WALL_TABLES by (s atom, anion): the lone atom above the anion
#: of a rock-salt (100) surface, the quantum cluster being the anion, its four in-plane cations and the cation below,
#: at the nearest-neighbour spacing d (A; the 4G MgO slab, a / 2 = 2.1325).  The pair sum of that cluster at height R
#: is what the in-situ table already contains.
IN_SITU_SITES = {(79, 8): {"cation": 12, "d": 2.1325}}


class FramePair:
    def __init__(self, path: str = DATA_FRAME_PAIR) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it with tools/electrostatics/build_frame_pair.py")
        d = np.load(path, allow_pickle=False)
        self.pairs = [tuple(int(x) for x in p) for p in d["pairs"]]
        self.index = {p: i for i, p in enumerate(self.pairs)}
        self.frames = [[int(x) for x in f] for f in d["frames"]]
        self.elements = sorted({z for p in self.pairs for z in p})
        self.frame_of = {}
        for (a, b), (fa, fb) in zip(self.pairs, self.frames):
            self.frame_of[a], self.frame_of[b] = fa, fb
        self.c = {p: float(d["c"][i]) for p, i in self.index.items()}     # the least-squares mean ratio, for the record
        self.measured = {p: bool(d["c_measured"][i]) for p, i in self.index.items()}
        n_t = [int(x) for x in d["targets_n"]]
        self._knots = {}                                   # pair row -> (R_k, c_k, slopes): the first-principles ratio
        for p, i in self.index.items():
            if n_t[i]:
                self._knots[i] = tuple(torch.tensor(d[key][i][:n_t[i]], dtype=torch.float64)
                                       for key in ("targets_R", "ratio_c", "ratio_slope"))
        self.ratio = {p: self._knots[i][:2] for p, i in self.index.items() if i in self._knots}
        self.ratio_tail = float(d["ratio_tail"])           # beyond its end points c keeps its slope and saturates in this
        self._es = torch.tensor(d["es_coefficients"], dtype=torch.float64)
        self._pauli = torch.tensor(d["pauli_coefficients"], dtype=torch.float64)
        self.r_min, self.r_cut = float(d["r_min"]), float(d["r_cut"])
        self.check_R = torch.tensor(d["check_R"], dtype=torch.float64)
        self.check_es = {p: torch.tensor(d["check_es"][i], dtype=torch.float64) for p, i in self.index.items()}
        self.check_pauli = {p: torch.tensor(d["check_pauli"][i], dtype=torch.float64) for p, i in self.index.items()}
        self.check_pauli_bare = {p: torch.tensor(d["check_pauli_bare"][i], dtype=torch.float64) for p, i in self.index.items()}
        self.targets = {}
        for p, i in self.index.items():
            n = int(d["targets_n"][i])
            self.targets[p] = (torch.tensor(d["targets_R"][i][:n], dtype=torch.float64),
                               torch.tensor(d["targets_E"][i][:n], dtype=torch.float64))
        z_max = max(self.elements)
        lookup = torch.full((z_max + 1, z_max + 1), -1, dtype=torch.long)
        for (a, b), i in self.index.items():
            lookup[a, b] = lookup[b, a] = i
        self._lookup = lookup
        self._frame = torch.zeros(z_max + 1, dtype=torch.float64)
        for z, f in self.frame_of.items():
            self._frame[z] = float(f)

    def _series(self, coefficients: Tensor, R: Tensor) -> Tensor:
        """The Chebyshev series (R times the function) of each row of ``coefficients`` at its ``R``."""
        x = (2.0 * R.clamp(self.r_min, self.r_cut) - (self.r_cut + self.r_min)) / (self.r_cut - self.r_min)
        T_prev, T = torch.ones_like(x), x
        out = coefficients[:, 0] * T_prev + coefficients[:, 1] * T
        for m in range(2, int(coefficients.shape[1])):
            T_prev, T = T, 2.0 * x * T - T_prev
            out = out + coefficients[:, m] * T
        return out

    def _switch(self, R: Tensor) -> Tensor:
        x = ((R - (self.r_cut - SWITCH_WIDTH)) / SWITCH_WIDTH).clamp(0.0, 1.0)
        return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)

    def _ratio(self, types: Tensor, R: Tensor) -> Tensor:
        """c(R) of each pair-table row: 1 without first-principles points; else the monotone cubic Hermite through the
        ratios at the computed distances, and beyond the first and the last point c_end + m_end w tanh((R - R_end) / w)
        with w = ``ratio_tail``: C^1 at the end, bounded beyond it."""
        c = torch.ones_like(R)
        w = self.ratio_tail
        for t in torch.unique(types).tolist():
            if t not in self._knots:
                continue
            x, y, m = (v.to(R.dtype).to(R.device) for v in self._knots[t])
            sel = types == t
            r = R[sel]
            if int(x.numel()) == 1:
                value = y[0].expand_as(r)
            else:
                k = (torch.searchsorted(x, r.detach().contiguous(), right=True) - 1).clamp(0, int(x.numel()) - 2)
                h = x[k + 1] - x[k]
                u = ((r - x[k]) / h).clamp(0.0, 1.0)
                value = ((2 * u ** 3 - 3 * u ** 2 + 1) * y[k] + (u ** 3 - 2 * u ** 2 + u) * h * m[k]
                         + (-2 * u ** 3 + 3 * u ** 2) * y[k + 1] + (u ** 3 - u ** 2) * h * m[k + 1])
                value = torch.where(r < x[0], y[0] + m[0] * w * torch.tanh((r - x[0]) / w), value)
                value = torch.where(r > x[-1], y[-1] + m[-1] * w * torch.tanh((r - x[-1]) / w), value)
            c = c.index_put((sel.nonzero().flatten(),), value)
        return c

    def _typed(self, types: Tensor, R: Tensor, scaled: bool = True):
        """(D_ff, c P) at ``R`` for pair-table rows ``types`` (eV; no switch, no Gaussian part); ``scaled=False``: P."""
        dtype, device = R.dtype, R.device
        es = self._series(self._es.to(dtype).to(device)[types], R) / R
        pauli = self._series(self._pauli.to(dtype).to(device)[types], R) / R
        if scaled:
            pauli = pauli * self._ratio(types, R)
        return es, pauli

    def _key(self, Za: int, Zb: int) -> tuple[int, int]:
        key = (min(int(Za), int(Zb)), max(int(Za), int(Zb)))
        if key not in self.index:
            raise ValueError(f"no frame-pair table for the element pair {key}: the density library holds {self.elements}")
        return key

    def pair_parts(self, Za: int, Zb: int, R: Tensor, scaled: bool = True):
        """``(D_ff, c P)`` of the element pair at distances ``R`` (eV); ``scaled=False`` gives P itself."""
        types = torch.full((int(R.numel()),), self.index[self._key(Za, Zb)], dtype=torch.long, device=R.device)
        return self._typed(types, R, scaled)

    def pair_energy(self, Za: int, Zb: int, R: Tensor, sig2_a: Tensor | None = None, sig2_b: Tensor | None = None) -> Tensor:
        """Phi_ab (eV) at distances ``R``; ``sig2`` the squared free-charge widths (needed only when both frames carry a
        charge)."""
        key = self._key(Za, Zb)
        es, pauli = self.pair_parts(Za, Zb, R)
        value = es + pauli
        ff = float(self.frame_of[key[0]] * self.frame_of[key[1]])
        if ff != 0.0:
            if sig2_a is None or sig2_b is None:
                raise ValueError(f"the frame pair {key} carries charges: its Gaussian part needs the free-charge widths")
            value = value + ff * (K_E / R - K_E * torch.erf(R / torch.sqrt(sig2_a + sig2_b)) / R)
        return value * self._switch(R) * (R < self.r_cut).to(R.dtype)

    def energy(self, species: Tensor, sigma: Tensor, pos: Tensor, cell: Tensor | None, pbc) -> Tensor:
        """The total frame interaction (eV): every pair within the cutoff, periodic images included, an atom's own
        images once.  ``sigma`` [n]: the free-charge widths (rho ~ exp(-r^2 / sigma^2), the model's kernel)."""
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        dtype = pos.dtype
        zero = pos.sum() * 0.0
        if int(z.numel()) == 0:
            return zero
        self._check(z)
        sig2 = (sigma.to(dtype).to(pos.device).reshape(-1)) ** 2
        lookup = self._lookup.to(pos.device)
        frame = self._frame.to(dtype).to(pos.device)
        i, j, R = pairs_within(pos, cell, pbc, self.r_cut, allow_self=True)[:3]
        si, sR = self_images_within(pos, cell, pbc, self.r_cut)
        i, j, R = torch.cat([i, si]), torch.cat([j, si]), torch.cat([R, sR.to(dtype)])
        if int(i.numel()) == 0:
            return zero
        es, pauli = self._typed(lookup[z[i], z[j]], R)
        ff = frame[z[i]] * frame[z[j]]
        gauss = K_E * torch.erf(R / torch.sqrt(sig2[i] + sig2[j])) / R
        value = es + pauli + ff * (K_E / R - gauss)
        return (value * self._switch(R)).sum()

    def _check(self, z: Tensor) -> None:
        unknown = sorted({int(v) for v in z.tolist()} - set(self.elements))
        if unknown:
            raise ValueError(f"no frame-pair table for Z = {unknown}: the density library holds {self.elements} "
                             "(tools/electrostatics/build_ion_densities.py, then build_frame_pair.py)")

    def site_reference(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc, level: str = "almo") -> Tensor:
        """What SPEC 33's wall already holds of this term (eV), to be subtracted while that wall is on.

        The wall is ``s_c(R) I(R)`` over every (s atom, anion) pair, with I the lone atom's in-situ interaction with
        its whole site (:data:`IN_SITU_SITES`) at height R.  The pair sum of that site at the same height,

            ref(R) = Phi(s, anion; R) + 4 Phi(s, cation; sqrt(R^2 + d^2)) + Phi(s, cation; R + d),

        is contained in I(R), so the same pairs, the same cutoff and the same switch s_c take it out of the frame-pair
        sum: on the measured site the frame terms then add up to I(R) exactly, and beyond the wall's cutoff nothing
        is removed."""
        from .bond_pauli import R_CUT as R_WALL
        from .bond_pauli import R_ON, WALL_TABLES
        from .bond_pauli import _switch as wall_switch
        from .constants import formal_frame
        from .downfold import acceptor_mask
        if level not in WALL_TABLES:
            raise ValueError(f"bond_pauli_level must be one of {sorted(WALL_TABLES)}, got {level!r}")
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        zero = pos.sum() * 0.0
        s_class = acceptor_mask(z).to(pos.device)
        anion = formal_frame(z).to(pos.device) < 0
        if not bool(s_class.any()) or not bool(anion.any()):
            return zero
        self._check(z)
        i, j, R = pairs_within(pos, cell, pbc, R_WALL, allow_self=True)[:3]       # the wall's own pair list
        if int(i.numel()) == 0:
            return zero
        sa, as_ = s_class[i] & anion[j], anion[i] & s_class[j]
        a = torch.cat([i[sa], j[as_]])
        if int(a.numel()) == 0:
            return zero
        Rp = torch.cat([R[sa], R[as_]])
        z_an = torch.cat([z[j][sa], z[i][as_]])
        E = zero
        for key in sorted({(int(x), int(y)) for x, y in zip(z[a].tolist(), z_an.tolist())}):
            if key not in WALL_TABLES[level]:
                continue                                    # the wall itself refuses an untabulated pair
            if key not in IN_SITU_SITES:
                raise ValueError(f"SPEC 33's wall holds the pair {key} but frame_pair has no record of the site it was "
                                 "measured on (trace_redox.frame_pair.IN_SITU_SITES): its pair sum would be counted twice")
            site = IN_SITU_SITES[key]
            if self.frame_of[key[0]] != 0:
                raise NotImplementedError(f"the in-situ reference of a charged s atom (Z = {key[0]}) needs its Gaussian part")
            Rk = Rp[(z[a] == key[0]) & (z_an == key[1])]
            d = float(site["d"])
            ref = (self.pair_energy(key[0], key[1], Rk) + 4.0 * self.pair_energy(key[0], site["cation"], torch.sqrt(Rk * Rk + d * d))
                   + self.pair_energy(key[0], site["cation"], Rk + d))
            E = E + (wall_switch(Rk, R_ON, R_WALL) * ref).sum()
        return E


_FRAME_PAIR: dict = {}


def frame_pair(path: str = DATA_FRAME_PAIR) -> FramePair:
    if path not in _FRAME_PAIR:
        _FRAME_PAIR[path] = FramePair(path)
    return _FRAME_PAIR[path]
