"""The frame pair's own Pauli wall (2026-10-31; NEW-TO-TRACE, ESTABLISHED ingredient: Gordon & Kim, J. Chem. Phys. 56,
3122 (1972)).

SPEC 25's overlap term holds each carrier state's Pauli repulsion relative to the frame pair (the first-principles K are
fitted to "carrier minus frame"), so the frame pair's own repulsion -- Au0 against Mg2+, for instance -- is no physics
term of the model: the learned backbone carries it, and the data constrain it only above the pair's smallest sampled
contact (Au-Mg 2.43 A in the 4G AuMgO set).  Below that the backbone extrapolates: 0.11 of PBE's wall at 2.0 A for a
dimer standing over a surface Mg (theory note 2026-10-18 section 4c), where the frozen model's flat-dimer relaxations
end.  This term supplies the frame pair's frozen-density repulsion as a physics baseline:

    E_w = sum over frame pairs (i, j) of the listed element pairs within R_CUT of  K_TF T_TF^nad[rho_i^f, rho_j^f](R_ij),

T_TF^nad the non-additive Thomas-Fermi kinetic energy of the frozen frame densities (the SPEC 25 library), K_TF one
constant per pair fitted to the frame pair's frozen-density PBE non-electrostatic interaction (0.918 for Mg2+-Au0: within
4.5 % from 1.8 to 2.4 A, 0.04 eV absolute to 3.5 A), tabulated as a Chebyshev series of R E_w with the penetration
tables' range and switch (tools/electrostatics/build_frame_wall.py -> trace_redox/data/frame_wall.npz).  It is
configuration-independent (the frame states only: the carriers' differences stay SPEC 25's) and trained with (the
backbone then learns the residual at the data distances).

LEGACY since 2026-11-13 (kept for the step 24 checkpoints): this is one pair's fit, not a law.  Measured on six frame
pairs K_TF spans 0.399-1.017 and falls with R inside each pair, and the wall holds no electrostatics, which for two
neutral atoms is as large as the Pauli term.  The general term is SPEC 39, trace_redox/frame_pair.py
(RedoxConfig(frame_pair=True)), which holds this pair too and is refused together with frame_wall."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from .penetration import SWITCH_WIDTH, pairs_within

DATA_FRAME_WALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frame_wall.npz")


class FrameWall:
    def __init__(self, path: str = DATA_FRAME_WALL):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it with tools/electrostatics/build_frame_wall.py")
        d = np.load(path, allow_pickle=False)
        self.pairs = [tuple(int(x) for x in p) for p in d["pairs"]]
        self.index = {p: i for i, p in enumerate(self.pairs)}
        self.K = torch.tensor(d["K"], dtype=torch.float64)
        self.coefficients = torch.tensor(d["coefficients"], dtype=torch.float64)
        self.r_min, self.r_cut = float(d["r_min"]), float(d["r_cut"])
        self.targets = {p: (torch.tensor(d["targets_R"][i]), torch.tensor(d["targets_E"][i])) for p, i in self.index.items()}
        self.checks = {p: (torch.tensor(d["checks_R"][i]), torch.tensor(d["checks_E"][i])) for p, i in self.index.items()}

    def _series(self, types: Tensor, R: Tensor) -> Tensor:
        c = self.coefficients.to(R.dtype).to(R.device)[types]
        x = (2.0 * R.clamp(self.r_min, self.r_cut) - (self.r_cut + self.r_min)) / (self.r_cut - self.r_min)
        T_prev, T = torch.ones_like(x), x
        out = c[:, 0] * T_prev + c[:, 1] * T
        for m in range(2, c.shape[1]):
            T_prev, T = T, 2.0 * x * T - T_prev
            out = out + c[:, m] * T
        return out

    def _switch(self, R: Tensor) -> Tensor:
        x = ((R - (self.r_cut - SWITCH_WIDTH)) / SWITCH_WIDTH).clamp(0.0, 1.0)
        return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)

    def pair_energy(self, Za: int, Zb: int, R: Tensor) -> Tensor:
        """E_w (eV) of the frame pair (Za, Zb) at distances ``R`` (A)."""
        key = (min(Za, Zb), max(Za, Zb))
        types = torch.full((int(R.numel()),), self.index[key], dtype=torch.long, device=R.device)
        return self._series(types, R) / R * self._switch(R) * (R < self.r_cut).to(R.dtype)

    def energy(self, species: Tensor, pos: Tensor, cell: Tensor | None, pbc) -> Tensor:
        """The total wall (eV) over every frame pair of the tabulated element pairs, periodic images included."""
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        zero = pos.sum() * 0.0
        listed = torch.zeros(int(z.max()) + 1 if int(z.numel()) else 1, dtype=torch.bool, device=pos.device)
        for a, b in self.pairs:
            for e in (a, b):
                if e < int(listed.numel()):
                    listed[e] = True
        if int((listed[z] if int(z.max()) < int(listed.numel()) else torch.zeros_like(z, dtype=torch.bool)).sum()) < 2:
            return zero
        i, j, R = pairs_within(pos, cell, pbc, self.r_cut, allow_self=True)[:3]
        if int(i.numel()) == 0:
            return zero
        a, b = torch.minimum(z[i], z[j]), torch.maximum(z[i], z[j])
        types = torch.full_like(i, -1)
        for p, t in self.index.items():
            types = torch.where((a == p[0]) & (b == p[1]), torch.full_like(i, t), types)
        keep = types >= 0
        if not bool(keep.any()):
            return zero
        Rk = R[keep]
        return (self._series(types[keep], Rk) / Rk * self._switch(Rk)).sum()


_WALL: dict = {}


def frame_wall(path: str = DATA_FRAME_WALL) -> FrameWall:
    if path not in _WALL:
        _WALL[path] = FrameWall(path)
    return _WALL[path]
