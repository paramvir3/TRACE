"""Shared helpers of the trace_redox suite (deliberately NOT a conftest.py).

A ``conftest.py`` here would be imported under the module name ``conftest``
and replace ``tests/electrostatics/conftest.py`` in ``sys.modules``, which
broke ``from conftest import require`` in a combined run (measured). So the
helpers live in this uniquely named module, and every test module imports the
autouse fixture below.

float64 is set at IMPORT time, before any e3nn module is built
(trace_electrostatics SPEC 3), and again per test AFTER the session-wide dtype
fixture of ``tests/conftest.py`` (which gives float64 only to node ids that
contain "electrostatics"); depending on ``_default_dtype_isolation`` orders it.
"""

import torch

torch.set_default_dtype(torch.float64)

import warnings  # noqa: E402

import pytest  # noqa: E402

R_HEADS = 3.5


@pytest.fixture(autouse=True)
def redox_float64(_default_dtype_isolation):
    torch.set_default_dtype(torch.float64)
    yield


def make_heads(seed=0, quadrupoles=False):
    from trace_electrostatics.heads import ElectrostaticHeads

    extra = dict(quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True) if quadrupoles else {}
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElectrostaticHeads(r_max=R_HEADS, hardness_margin=1.0, **extra)


def cluster(n, dim, seed, scale=1.1):
    """n atoms of a random cluster, features, and the full directed neighbour list."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64) * scale
    h = torch.randn(n, dim, generator=g, dtype=torch.float64)
    return pos, h, neighbour_list(pos)


def neighbour_list(pos, r_max=R_HEADS):
    distance = torch.cdist(pos, pos)
    src, dst = torch.nonzero((distance < r_max) & (distance > 0), as_tuple=True)
    return torch.stack([src, dst])


def edges(pos, edge_index):
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    return vec, vec.norm(dim=-1)


def lattice_graph(pos, cell, pbc, r_max=R_HEADS, shells=1):
    """Directed neighbour list with integer image shifts (both directions), and edge geometry.

    ``edge_vec = pos[src] - pos[dst] + shift @ cell``, the flashace convention.
    """
    n = int(pos.shape[0])
    ranges = [range(-shells, shells + 1) if p else range(0, 1) for p in pbc]
    src, dst, shifts = [], [], []
    for i in range(n):
        for j in range(n):
            for a in ranges[0]:
                for b in ranges[1]:
                    for c in ranges[2]:
                        shift = torch.tensor([float(a), float(b), float(c)])
                        d = float((pos[i] - pos[j] + shift @ cell).norm())
                        if 0.0 < d < r_max:
                            src.append(i)
                            dst.append(j)
                            shifts.append(shift)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_shift = torch.stack(shifts) if shifts else torch.zeros(0, 3)
    return edge_index, edge_shift


def periodic_edges(pos, cell, edge_index, edge_shift):
    vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
    return vec, vec.norm(dim=-1)
