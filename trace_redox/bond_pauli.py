"""The bond-oriented Pauli repulsion of a bonded s-class atom against a closed-shell anion (trace_redox SPEC 33;
2026-11-05).  NEW-TO-TRACE: a frame term whose anisotropy is set by the direction of the atom's own covalent bond.
ESTABLISHED ingredients: exchange repulsion as the square of the orbital overlap (Murrell, Randic and Williams, Proc. R.
Soc. A 284, 566 (1965)); anisotropic atom-atom repulsion from the molecular density (Stone and Price, J. Phys. Chem. 92,
3325 (1988)); the ALMO energy decomposition that measured it (Khaliullin et al., J. Phys. Chem. A 111, 8753 (2007)).

Why (measured, theory note 2026-10-18 section 4c, 2026-11-03/04): the sigma bond of Au2 draws the contact atom's 6s
density along the bond axis.  Standing on an O-top site the bond points away from the O and shields the contact atom:
its frozen (Pauli + electrostatic) repulsion against the O2- is 0.76 eV below a lone Au atom's at the same 2.26 A.
Tilted by 70 deg the shielding is gone (+0.19 eV above the lone atom).  PBE therefore puts the tilted dimer 0.8 eV above
the upright one; the model's overlap terms (SPEC 23 / 25) are isotropic atom pairs and its backbone saw only upright
contacts in the data, so every model of steps 20-24 made the tilted state 0.5-1.1 eV too stable and the flat dimer of
the T5 protocol rolled over into it.  Charge transfer (the O -> Au donation keeps 95 % at 70 deg) and field polarisation
(which favours the tilt) were measured and are not the cause.

The law, measured on Au2 (2.545 A) over the O-top O of a MgO(100) cluster, D = E_int(dimer) - E_int(atom) - E_int(atom)
at theta = 0, 20, 40, 60, 70, 90 deg (theta the bond's angle from the surface normal; the cos / P2 fits hold to <= 8 meV)
and R = 2.26, 2.4, 2.6, 2.9 A (frozen; relaxed at 2.26, 2.6 and 2.9 A), fitted jointly to this form:

    E_bp(a, j) = lambda_a s_c(R_aj) [ D1(R_aj) cos t + D2(R_aj) P2(cos t) ],    cos t = r_aj . (-e_a),
    D1(R) = C1 f(R)^p1,   D2(R) = C2 f(R)^p2,   f(R) = E1(R) / E1(2.26 A),

e_a the unit vector from a toward its s-class partner(s) (lambda_a its bond weight, 1 for one partner within 2.9 A,
switched off to 3.4 A; several partners add their weighted unit vectors and lambda_a is the resultant's length, capped at
1 -- a symmetric site has no lobe and no term; measured for one partner), s_c a C2 switch from 1 at 3.8 A to 0 at
4.3 A, E1(R) the lone Au0's frozen repulsion on the O-top site (1.4254 / 0.8454 / 0.3891 / 0.1045 eV at 2.26 / 2.4 / 2.6 /
2.9 A; linear in ln E1 between the points, the end slopes beyond: an extrapolation, stated).  The isotropic part D0(R) of
the measured non-additivity (+0.18 eV at 2.26 A relaxed, <= 0.04 frozen) is left out: a pair function of R alone that
the local backbone represents.  Levels: 'almo' (the default), the orbitals relaxed with charge transfer forbidden,
C1, C2 = -0.069, -0.497 eV, p1, p2 = 1.877, 0.382 (three R) -- the physical target, since TRACE's bound sector answers
electric fields only and cannot supply the Pauli-driven relaxation; 'frz', the frozen densities, -0.072, -0.665 eV,
p1, p2 = 2.29, 0.500 (four R).  Tabulated for (Au, O) only; a bonded
s-class atom within the cutoff of an anion of an untabulated pair is refused (the energy would be silently incomplete).
Configuration independent (a frame term, as the frame wall), trained with: the backbone learns the residual.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .constants import formal_frame
from .downfold import acceptor_mask
from .penetration import pairs_within

#: the lone Au0's frozen interaction (penetration electrostatics + Pauli) on the O-top site of MgO(100), eV
#: (tools/electrostatics/frz_angular.py at 2.26-2.9 A; otop_lone_e1.py at 1.8-2.1 and 3.2-3.8 A, 2026-11-05).  It turns
#: slightly attractive beyond 3.3 A (penetration electrostatics over Pauli), so the law's f(R) uses the positive points
#: in ln E1 and the wall interpolates E1 itself across the sign change.
E1_TABLE = {(79, 8): ((1.8, 2.0, 2.1, 2.26, 2.4, 2.6, 2.9, 3.2, 3.5, 3.8),
                      (7.7324, 3.6975, 2.5648, 1.4254, 0.8454, 0.3891, 0.1045, 0.0132, -0.0121, -0.0165))}
E1_REF_R = 2.26
#: the lone Au0's relaxed, charge-transfer-free interaction I_ALMO(R) on the same site (tools/electrostatics/
#: almo_angular.py at 2.26-2.9 A; otop_lone_almo.py at 1.8-2.1 and 3.2-3.8 A): the reference of the relaxed law, and the
#: wall at the relaxed level (the orbitals relax in each other's frozen potential, Pauli included; the lone atom's
#: relaxation recovers 0.90 of 1.37 eV at 2.26 A, 0.51 of 1.04 at 1.8); a shallow well of -0.035 eV near 3.2 A.
I_ALMO_TABLE = {(79, 8): ((1.8, 2.0, 2.1, 2.26, 2.6, 2.9, 3.2, 3.5, 3.8),
                          (3.8140, 1.6372, 1.0388, 0.4718, 0.0466, -0.0278, -0.0349, -0.0273, -0.0194))}
WALL_TABLES = {"frz": E1_TABLE, "almo": I_ALMO_TABLE}
#: per level: (C1, C2) in eV at R = 2.26 A and the radial exponents (p1, p2) of D1 = C1 f^p1, D2 = C2 f^p2, fitted
#: jointly over the measured R and theta = 0-70 deg with this module's own two-atom form and this table's f(R)
#: (tools/electrostatics/fit_bond_pauli.py; the far atom's own term is in the fit; 90 deg excluded: the far Au then sits
#: 2.26 A over a Mg, whose non-additivity has no law here).  Frozen: four R (2.26-2.9 A), largest residual 3.7 meV,
#: p2 = 0.500 (exchange ~ S^2, the anisotropy ~ S).  Relaxed (ALMO): three R (2.26, 2.6, 2.9 A), p1 held at 1.877 (three
#: R do not determine the small cos term's decay), largest residual 6.2 meV (theory note 2026-10-18 s. 4c, 2026-11-05).
LEVELS = {"almo": (-0.0694, -0.4973, 1.877, 0.3818), "frz": (-0.0720, -0.6649, 2.2925, 0.5001)}
#: SPEC 33 per charge state (step 27, 2026-11-08): the anion's law, measured on Au2- over the same O-top O
#: (tools/electrostatics/almo_angular_anion.py at 2.26 A, theta = 0-70 deg, and 2.6 A, 0 / 40 / 70 deg) and fitted with the
#: neutral's form (fit_bond_pauli.py --anion: free D0 per R, p1 held at the neutral's).  Relaxed: C1, C2, p2 jointly over
#: both R, largest residual 6.3 meV (adopted under the README rule, written before the fit: p2 inside its bounds, residual
#: <= 10 meV).  Frozen: the joint fit runs to the p2 bound (18.1 meV), so the 2.26 A five-angle coefficients with the
#: neutral's exponents (the pre-registered default).  The sigma* electron restores the axial density the sigma bond
#: removed: the anion's tilt costs +0.20 eV at 2.26 A against the neutral's +0.70 (relaxed).  The cation and the higher
#: states take the neutral's law (unmeasured; stated).
ANION_LEVELS = {"almo": (-0.1094, -0.0957, 1.877, 0.4561), "frz": (0.1011, -0.2905, 2.2925, 0.5001)}
R_BOND, R_BOND_OFF = 2.9, 3.4
R_ON, R_CUT = 3.8, 4.3


def _fc_slopes(x: Tensor, y: Tensor) -> Tensor:
    """Fritsch-Carlson monotone knot slopes (the shape-preserving cubic Hermite interpolant; ESTABLISHED: Fritsch and
    Carlson, SIAM J. Numer. Anal. 17, 238 (1980)): the harmonic mean of adjacent secants where they agree in sign, zero
    where they do not, one-sided secants at the ends."""
    h = x[1:] - x[:-1]
    d = (y[1:] - y[:-1]) / h
    m = torch.zeros_like(y)
    same = (d[:-1] * d[1:]) > 0
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    inner = torch.where(same, (w1 + w2) / (w1 / d[:-1] + w2 / d[1:]), torch.zeros_like(d[:-1]))
    m = m.index_put((torch.arange(1, int(y.numel()) - 1, device=y.device),), inner)
    m = m.index_put((torch.tensor([0], device=y.device),), d[:1])
    m = m.index_put((torch.tensor([int(y.numel()) - 1], device=y.device),), d[-1:])
    return m


def _hermite(R: Tensor, x: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """Cubic Hermite interpolation of (x, y, slopes m) at R, C^1; linear with the end slopes outside [x_0, x_n]."""
    out = torch.where(R < x[0], y[0] + m[0] * (R - x[0]), y[-1] + m[-1] * (R - x[-1]))
    for k in range(int(x.numel()) - 1):
        h = x[k + 1] - x[k]
        t = ((R - x[k]) / h).clamp(0.0, 1.0)
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        seg = h00 * y[k] + h10 * h * m[k] + h01 * y[k + 1] + h11 * h * m[k + 1]
        out = torch.where((R >= x[k]) & (R < x[k + 1]), seg, out)
    return out


def _switch(R: Tensor, a: float, b: float) -> Tensor:
    """C2 switch: 1 for R <= a, 0 for R >= b."""
    u = ((R - a) / (b - a)).clamp(0.0, 1.0)
    return 1.0 - u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


class BondPauli:
    def __init__(self, level: str = "almo") -> None:
        if level not in LEVELS:
            raise ValueError(f"bond_pauli_level must be one of {sorted(LEVELS)}, got {level!r}")
        self.level = level
        self.C1, self.C2, self.p1, self.p2 = LEVELS[level]
        self.anion = ANION_LEVELS[level]
        self.tables = {}
        for k, (r, e) in E1_TABLE.items():
            pos = [(x, y) for x, y in zip(r, e) if y is not None and y > 0.0]
            rr = torch.tensor([x for x, _ in pos], dtype=torch.float64)
            ln_e = torch.log(torch.tensor([y for _, y in pos], dtype=torch.float64))
            ref = torch.log(torch.tensor(dict(pos)[E1_REF_R], dtype=torch.float64))
            self.tables[k] = (rr, ln_e, _fc_slopes(rr, ln_e), ref)

    def _f(self, key: tuple[int, int], R: Tensor) -> Tensor:
        """E1(R) / E1(2.26 A) over the positive points: a monotone cubic Hermite interpolant in ln E1 (C^1, so the
        forces are continuous at the knots), linear in ln E1 with the end slopes beyond."""
        r, ln_e, m, ref = (x.to(R.dtype).to(R.device) for x in self.tables[key])
        return torch.exp(_hermite(R, r, ln_e, m) - ref)

    def pair_energy(self, key: tuple[int, int], R: Tensor, cos: Tensor) -> Tensor:
        """D1(R) cos t + D2(R) P2(cos t) (eV) for the tabulated pair ``key``; no switch, no bond weight."""
        f = self._f(key, R)
        return self.C1 * f ** self.p1 * cos + self.C2 * f ** self.p2 * 0.5 * (3.0 * cos * cos - 1.0)

    def state_coefficients(self, n) -> tuple[float, float]:
        """(C1, C2) of the law for an s atom in charge state ``n``: the anion's for n = -1, the neutral's otherwise."""
        return (self.anion[0], self.anion[1]) if int(round(float(n))) == -1 else (self.C1, self.C2)

    def state_exponents(self, n) -> tuple[float, float]:
        """(p1, p2) of the law for an s atom in charge state ``n`` (as ``state_coefficients``)."""
        return (self.anion[2], self.anion[3]) if int(round(float(n))) == -1 else (self.p1, self.p2)

    def _geometry(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc):
        """The s-anion pairs of the law: (a [P] s-atom index, R [P], cos t [P], lambda_a [P], anion species [P], keys),
        or None when there are none.  Refuses a bonded s atom near an anion of an untabulated pair."""
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        n = int(z.numel())
        s_class = acceptor_mask(z).to(pos.device)
        anion = formal_frame(z).to(pos.device) < 0
        if int(s_class.sum()) < 2 or not bool(anion.any()):
            return None
        i, j, R, vec = pairs_within(pos, cell, pbc, R_CUT, return_vectors=True, allow_self=True)
        if int(i.numel()) == 0:
            return None
        unit = vec / R.unsqueeze(1)
        # the bond lobe of every s-class atom: the switch-weighted sum of the unit vectors toward its s-class partners
        ss = (s_class[i] & s_class[j]).to(pos.dtype) * _switch(R, R_BOND, R_BOND_OFF)
        v = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        v = v.index_add(0, i, ss.unsqueeze(1) * unit).index_add(0, j, -ss.unsqueeze(1) * unit)
        lam2 = (v * v).sum(1)
        has = s_class & (lam2 > 1e-12)
        norm = torch.sqrt(torch.where(has, lam2, torch.ones_like(lam2)))     # the norm on the lobed atoms only
        e_hat = v / norm.unsqueeze(1)
        lam = torch.where(has, norm, torch.zeros_like(norm)).clamp(max=1.0)
        # the s-anion pairs, oriented from the s atom to the anion
        sa, as_ = s_class[i] & anion[j], anion[i] & s_class[j]
        a = torch.cat([i[sa], j[as_]])
        if int(a.numel()) == 0:
            return None
        d = torch.cat([unit[sa], -unit[as_]])
        Rp = torch.cat([R[sa], R[as_]])
        z_an = torch.cat([z[j][sa], z[i][as_]])
        keys = {(int(za), int(zb)) for za, zb in zip(z[a].tolist(), z_an.tolist())}
        bonded = lam[a] > 0
        missing = {k for k, b in zip(zip(z[a].tolist(), z_an.tolist()), bonded.tolist()) if b and k not in E1_TABLE}
        if missing:
            raise ValueError(f"bond_pauli has no table for the (s atom, anion) pairs {sorted(missing)}: extend "
                             "tools/electrostatics/frz_angular.py / almo_angular.py")
        cos = (d * (-e_hat[a])).sum(1)
        return a, Rp, cos, lam[a], z[a], z_an, keys

    def energy(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc) -> Tensor:
        """E_bp (eV) summed over every bonded s-class atom and the anions within R_CUT, periodic images included."""
        zero = pos.sum() * 0.0
        g = self._geometry(species, pos, cell, pbc)
        if g is None:
            return zero
        a, Rp, cos, lam_a, z_a, z_an, keys = g
        E = zero
        for key in sorted(keys & set(E1_TABLE)):
            sel = (z_a == key[0]) & (z_an == key[1])
            E = E + (lam_a[sel] * _switch(Rp[sel], R_ON, R_CUT) * self.pair_energy(key, Rp[sel], cos[sel])).sum()
        return E

    def geometric_sums(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc, state=0) -> tuple[Tensor, Tensor]:
        """Per atom ``[n]``: A1 = sum_j lambda_a s_c f^p1 cos t and A2 = sum_j lambda_a s_c f^p2 P2(cos t), with the
        exponents of charge state ``state``; zero for every atom that is not a bonded s atom near an anion."""
        n = int(torch.as_tensor(species).numel())
        A1 = pos.new_zeros(n) + pos.sum() * 0.0
        A2 = pos.new_zeros(n) + pos.sum() * 0.0
        g = self._geometry(species, pos, cell, pbc)
        if g is None:
            return A1, A2
        a, Rp, cos, lam_a, z_a, z_an, keys = g
        p1, p2 = self.state_exponents(state)
        for key in sorted(keys & set(E1_TABLE)):
            sel = (z_a == key[0]) & (z_an == key[1])
            w = lam_a[sel] * _switch(Rp[sel], R_ON, R_CUT)
            f = self._f(key, Rp[sel])
            c = cos[sel]
            A1 = A1.index_add(0, a[sel], w * f ** p1 * c)
            A2 = A2.index_add(0, a[sel], w * f ** p2 * 0.5 * (3.0 * c * c - 1.0))
        return A1, A2

    def state_terms(self, species: Tensor, states: Tensor, pos: Tensor, cell: Tensor | None, pbc) -> Tensor:
        """``[n, S]``: the law of each atom in each charge state of ``states``, C1(n) A1(n) + C2(n) A2(n) (SPEC 33 per
        charge state).  The state-0 column summed over atoms is ``energy``; rows of atoms without the law are zero."""
        cache: dict = {}
        cols = []
        for s in [float(x) for x in torch.as_tensor(states).tolist()]:
            ex = self.state_exponents(s)
            if ex not in cache:
                cache[ex] = self.geometric_sums(species, pos, cell, pbc, state=s)
            A1, A2 = cache[ex]
            C1, C2 = self.state_coefficients(s)
            cols.append(C1 * A1 + C2 * A2)
        return torch.stack(cols, dim=1)


class BondPauliWall:
    """The isotropic frame wall of an s-class atom against a closed-shell anion at the law's own level: E_wall(a, j) =
    s_c(R) I(R), the lone Au0's interaction with the O-top site -- frozen (E1: penetration electrostatics + Pauli) for
    level 'frz', relaxed with charge transfer forbidden (I_ALMO) for level 'almo' -- the quantity the bond-oriented law
    of that level is measured relative to, so that wall + law is the dimer's interaction at that level (2026-11-05: a
    frozen wall under the relaxed law was launched as step 26 and withdrawn before any result).  Why (2026-11-05): SPEC 23 / 25 are frame-relative and the frame wall covers Mg2+-Au0
    only, so the Au0-O2- frozen interaction was the backbone's alone, learned at the data's 2.0-2.5 A and a cliff below
    (the frozen step-20 energy falls by hundreds of eV at 1.6 A); the law's theta = 0 shielding, attractive relative to
    the lone atom, tips that backbone over.  With the wall explicit, E1 + D is repulsive at every angle and rises below
    the data by physics.  Bonded or not, every s-class atom gets it (the law then adds the bond's modulation)."""

    def __init__(self, level: str = "almo") -> None:
        if level not in WALL_TABLES:
            raise ValueError(f"bond_pauli_level must be one of {sorted(WALL_TABLES)}, got {level!r}")
        self.level = level
        self.tables = {}
        for k, (r, e) in WALL_TABLES[level].items():
            pts = [(x, y) for x, y in zip(r, e) if y is not None]
            rr = torch.tensor([x for x, _ in pts], dtype=torch.float64)
            ee = torch.tensor([y for _, y in pts], dtype=torch.float64)
            m = _fc_slopes(rr, ee)                                            # E1-space slopes (the tail, sign change)
            pos = ee > 0.0
            if int(pos.sum()) >= 2:                                           # the exponential wall: ln-space slopes
                m_ln = _fc_slopes(rr[pos], torch.log(ee[pos]))
                m = m.index_put((pos.nonzero().flatten(),), ee[pos] * m_ln)
            self.tables[k] = (rr, ee, m)

    def _e1(self, key: tuple[int, int], R: Tensor) -> Tensor:
        """E1(R): a C^1 cubic Hermite interpolant in E1 with the positive knots' slopes from the ln E1 interpolant (the
        exponential wall) and the tail's from E1 itself (the sign change beyond 3.3 A); below the first point the
        exponential continuation with the first ln slope, beyond the last point linear (the switch takes it to zero)."""
        r, e, m = (x.to(R.dtype).to(R.device) for x in self.tables[key])
        inside = _hermite(R, r, e, m)
        below = e[0] * torch.exp((m[0] / e[0]) * (R - r[0]))
        return torch.where(R < r[0], below, inside)

    def energy(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc) -> Tensor:
        """E_wall (eV) over every s-class atom and the anions within R_CUT, periodic images included."""
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        zero = pos.sum() * 0.0
        s_class = acceptor_mask(z).to(pos.device)
        anion = formal_frame(z).to(pos.device) < 0
        if not bool(s_class.any()) or not bool(anion.any()):
            return zero
        i, j, R = pairs_within(pos, cell, pbc, R_CUT, allow_self=True)[:3]
        if int(i.numel()) == 0:
            return zero
        sa, as_ = s_class[i] & anion[j], anion[i] & s_class[j]
        a = torch.cat([i[sa], j[as_]])
        if int(a.numel()) == 0:
            return zero
        Rp = torch.cat([R[sa], R[as_]])
        z_an = torch.cat([z[j][sa], z[i][as_]])
        keys = {(int(za), int(zb)) for za, zb in zip(z[a].tolist(), z_an.tolist())}
        missing = sorted(keys - set(E1_TABLE))
        if missing:
            raise ValueError(f"bond_pauli_wall ({self.level}) has no table for the (s atom, anion) pairs {missing}: "
                             "extend tools/electrostatics/otop_lone_e1.py / otop_lone_almo.py")
        E = zero
        for key in sorted(keys):
            sel = (z[a] == key[0]) & (z_an == key[1])
            E = E + (_switch(Rp[sel], R_ON, R_CUT) * self._e1(key, Rp[sel])).sum()
        return E


_LAW: dict = {}
_WALL: dict = {}


def bond_pauli_wall(level: str = "almo") -> BondPauliWall:
    if level not in _WALL:
        _WALL[level] = BondPauliWall(level)
    return _WALL[level]


def bond_pauli(level: str = "almo") -> BondPauli:
    if level not in _LAW:
        _LAW[level] = BondPauli(level)
    return _LAW[level]
