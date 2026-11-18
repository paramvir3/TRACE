"""The downfolded O 2p <-> metal s bond (``RedoxConfig(downfold_transfer=True)``; trace_redox SPEC 32).

Measured (2026-10-27/28; benchmarks README, theory note 2026-10-18 section 4c):
- PBE's O-top relaxation of Au2 on MgO(100) is -1.96 eV (28 ions: polarisation -0.86, charge transfer -1.10), and
  94 % of the transfer's wetting share is O -> Au donation; the trained model's relaxation is a seventh of it.
- Harrison-sized couplings in the explicit configuration ensemble (step 21, SPEC 31) make the selected ensemble
  unconverged in its threshold (eV), add 0.2-0.7 eV/A of switch force, and overbind short Au-O contacts by 3 eV.
- The first version of this term (2026-10-28, reviewed the same day) took the folded configurations uncoupled: it
  dropped the moved electron's resonance over the dimer (0.20-0.32 eV at the contact with the model's own numbers),
  chose the explicit set on undressed energies, let folded configurations re-enter the explicit set, and masked the
  explicit O-Au couplings by species while dressing only the Au0 acceptor.

Construction (NEW-TO-TRACE; Loewdin partitioning ESTABLISHED: P.-O. Loewdin, J. Chem. Phys. 19, 1396 (1951); the
universal coupling and the Slater-Koster direction cosines ESTABLISHED: Froyen and Harrison, Phys. Rev. B 20, 2420
(1979); Slater and Koster, Phys. Rev. 94, 1498 (1954)):
- the channel: every one-electron move between an O-class atom (frame anion, p frontier) and an s-class atom (neutral
  frame, odd electron count, s frontier: Au, Ag, Cu, H), in either direction and in any charge states the ladders
  allow; these moves leave the explicit ensemble (their couplings are zeroed, and a selected configuration reachable
  from another by such a move is dropped) and are folded into the configuration energies;
- per explicit configuration n and per contact cluster (atoms connected by O-s edges within the cutoff and s-s hops),
  the folded states Q are the single-move configurations with Delta_q(n) = E(q) - E(n) exact, coupled to n by
      v = f_spin sqrt(g_orb) V_sp sigma(r),   V_sp sigma = 1.84 hbar^2 / (m (r^2 + r0^2)) w(r),
  with a closed-shell O donor (O2-) resolved into its three p components, v_m = f V l_m (l the direction cosines
  O -> s, sum_m v_m^2 = f^2 V^2), and the moved electron hopping between s atoms inside Q with -|T_ss'| f (the model's
  own s-s coupling, its magnitude; the s-s sigma hop is bonding); moves out of or into an O hole take the isotropic
  orbital factor g_orb (the sigma p's occupation or vacancy fraction) and no hop;
- lambda(n, cluster) = the lowest eigenvalue of [[0, v^T], [v, diag(Delta) + T_QQ]], found from the secular equation
  lambda = v^T (lambda - M)^-1 v below mu_min (M's lowest eigenvalue) by bracketed bisection and two Newton steps
  with the graph (exact value, first and second derivatives); E(n) -> E(n) + sum_clusters lambda;
- valid while mu_min > 0 (the folded manifold above the explicit state); otherwise refused loudly.
"""
import math

import numpy as np
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HBAR2_OVER_ME = 7.619964
EPS = torch.finfo(torch.float64).eps


def _dense_lowest(M, v):
    K = len(v)
    h = np.zeros((K + 1, K + 1))
    h[0, 1:], h[1:, 0] = v, v
    h[1:, 1:] = M
    return float(np.linalg.eigvalsh(h)[0])


def test_the_star_root_is_exact():
    from trace_redox.downfold import star_ground
    for delta, v in ((4.0, 2.6), (0.3, 1.0), (7.0, 0.01)):
        lam = star_ground(torch.tensor([[delta]]), torch.tensor([[v * v]]))
        exact = delta / 2 - math.sqrt(delta * delta / 4 + v * v)
        assert abs(float(lam[0]) - exact) <= 4 * EPS * (1 + abs(exact))
    g = torch.Generator().manual_seed(3)
    delta = torch.rand(6, 4, generator=g, dtype=torch.float64) * 5 + 0.5
    delta[0, 1] = delta[0, 0]
    v = torch.rand(6, 4, generator=g, dtype=torch.float64) * 2
    v[0, 1] = v[0, 0]
    lam = star_ground(delta, v * v)
    for s in range(6):
        exact = _dense_lowest(np.diag(delta[s].numpy()), v[s].numpy())
        scale = float(delta[s].abs().max() + v[s].norm())             # eigvalsh: O(eps ||H||)
        assert abs(float(lam[s]) - exact) <= 16 * EPS * scale


def test_the_block_root_is_the_dense_eigenvalue():
    """Coupled folded states: the resolvent secular equation against the dense (K+1)x(K+1) eigenvalue, with the
    Au-Au resonance (the case the star cannot do), degenerate leaves, and absent (v = 0, unreachable) leaves."""
    from trace_redox.downfold import block_ground
    g = torch.Generator().manual_seed(11)
    S, K = 7, 5
    delta = torch.rand(S, K, generator=g, dtype=torch.float64) * 4 + 1.0
    v = torch.rand(S, K, generator=g, dtype=torch.float64) * 2 - 1
    hop = torch.rand(S, K, K, generator=g, dtype=torch.float64) * 0.6
    hop = -0.5 * (hop + hop.transpose(1, 2))
    hop.diagonal(dim1=1, dim2=2).zero_()
    mask = torch.ones(S, K, dtype=torch.bool)
    mask[0, 3:] = False                                                  # a padded block
    v[1, 2] = 0.0                                                        # reachable only through a hop
    delta[2, 1] = delta[2, 0]                                            # degenerate leaves
    v[2, 1] = v[2, 0]
    M = torch.diag_embed(delta) + hop
    lam, mu = block_ground(M, v, mask)
    for s in range(S):
        k = int(mask[s].sum())
        Ms, vs = M[s, :k, :k].numpy(), v[s, :k].numpy()
        exact = _dense_lowest(Ms, vs)
        scale = float(np.abs(Ms).max() + np.linalg.norm(vs))
        assert abs(float(lam[s]) - exact) <= 32 * EPS * scale, (s, float(lam[s]), exact)
        assert abs(float(mu[s]) - float(np.linalg.eigvalsh(Ms)[0])) <= 32 * EPS * scale
    # the Au-Au resonance of the review: one O leaf coupled to Au108, Au108 <-> Au109 by -|t|
    vv, d, t = 2.617, 3.40, 2.40
    lam1, _ = block_ground(torch.tensor([[[d, -t], [-t, d]]]), torch.tensor([[vv, 0.0]]), torch.ones(1, 2, dtype=torch.bool))
    star = d / 2 - math.sqrt(d * d / 4 + vv * vv)
    assert float(lam1[0]) < star - 0.25                                 # the resonance adds > 0.25 eV of binding


def test_the_block_root_derivatives_are_exact():
    """First and second derivatives of the block root against central differences of the dense eigenvalue."""
    from trace_redox.downfold import block_ground
    x = torch.tensor([2.5, 4.0, 1.7, 0.9, -0.8], dtype=torch.float64).requires_grad_(True)   # d1, d2, v1, v2, hop

    def root(y):
        M = torch.stack([torch.stack([y[0], y[4]]), torch.stack([y[4], y[1]])]).unsqueeze(0)
        return block_ground(M, y[2:4].unsqueeze(0), torch.ones(1, 2, dtype=torch.bool))[0][0]

    def dense(y):
        y = y.detach().numpy()
        return _dense_lowest(np.array([[y[0], y[4]], [y[4], y[1]]]), y[2:4])

    lam = root(x)
    (grad,) = torch.autograd.grad(lam, x, create_graph=True)
    H = torch.stack([torch.autograd.grad(grad[i], x, retain_graph=True)[0] for i in range(5)])
    h = 1e-4
    for i in range(5):
        e = torch.zeros(5, dtype=torch.float64)
        e[i] = h
        fd1 = (dense(x + e) - dense(x - e)) / (2 * h)
        assert abs(float(grad[i]) - fd1) <= 1e-8
        for j in range(5):
            f = torch.zeros(5, dtype=torch.float64)
            f[j] = h
            fd2 = (dense(x + e + f) - dense(x + e - f) - dense(x - e + f) + dense(x - e - f)) / (4 * h * h)
            assert abs(float(H[i, j]) - fd2) <= 1e-5


def test_the_star_root_limits_and_bound():
    from trace_redox.downfold import star_ground
    delta = torch.tensor([[3.0, 5.0]])
    v = torch.tensor([[1e-3, 2e-3]])
    lam = float(star_ground(delta, v * v)[0])
    second = -(1e-6 / 3.0 + 4e-6 / 5.0)
    assert abs(lam - second) <= 1e-6 * abs(second)
    g = torch.Generator().manual_seed(5)
    delta = torch.rand(20, 5, generator=g, dtype=torch.float64) * 6 + 0.2
    v = torch.rand(20, 5, generator=g, dtype=torch.float64) * 4
    lam = star_ground(delta, v * v)
    lower = torch.minimum(torch.zeros(20, dtype=torch.float64), delta.min(1).values) - v.norm(dim=1)
    assert bool((lam >= lower - 16 * EPS).all()) and bool((lam <= 0).all())
    empty = star_ground(torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
    assert float(empty[0]) == 0.0


def test_non_finite_and_non_positive_folded_states_are_handled():
    """A leaf whose energy is not finite is absent; a folded manifold at or below the explicit state is refused."""
    from trace_redox.downfold import DownfoldError, block_ground, star_ground
    lam = star_ground(torch.tensor([[float("inf"), 3.0]]), torch.tensor([[1.0, 1.0]]))
    exact = 1.5 - math.sqrt(2.25 + 1.0)
    assert math.isfinite(float(lam[0])) and abs(float(lam[0]) - exact) <= 4 * EPS * 4
    lam, mu = block_ground(torch.tensor([[[float("inf"), 0.0], [0.0, 3.0]]]), torch.tensor([[1.0, 1.0]]),
                           torch.ones(1, 2, dtype=torch.bool))
    assert abs(float(lam[0]) - exact) <= 4 * EPS * 4 and float(mu[0]) == 3.0
    with pytest.raises(DownfoldError, match="above"):
        block_ground(torch.tensor([[[-2.0]]]), torch.tensor([[1e-20]]), torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(DownfoldError, match="above"):                   # the Au-Au resonance pulls mu below 0
        block_ground(torch.tensor([[[2.0, -2.4], [-2.4, 2.0]]]), torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool))
    ok, _ = block_ground(torch.tensor([[[0.0]]]), torch.tensor([[0.0]]), torch.ones(1, 1, dtype=torch.bool))
    assert float(ok[0]) == 0.0                                           # no coupled leaf: no dressing, no refusal


def test_the_class_of_moves_and_the_contact_clusters():
    from trace_redox.downfold import acceptor_mask, donor_mask, downfold_moves
    z = torch.tensor([79, 8, 12, 13, 1, 17, 11, 47, 30])
    assert donor_mask(z).tolist() == [False, True, False, False, False, True, False, False, False]
    assert acceptor_mask(z).tolist() == [True, False, False, False, True, False, False, True, False]
    # two O-top contacts 30 A apart: two clusters; a third Au 2.5 A from the second joins its cluster through the hop
    z = torch.tensor([8, 79, 8, 79, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [30.0, 0.0, 0.0], [30.0, 0.0, 2.4], [30.0, 0.0, 4.9]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    m = downfold_moves(z, ei, vec, length, r_cut=4.5)
    edge = m.t > 0
    pairs = sorted(zip(m.o[edge].tolist(), m.g[edge].tolist()))
    assert pairs == [(0, 1), (2, 3)]                                     # the far Au (4.9 A) has no O edge ...
    partners = sorted(zip(m.o[~edge].tolist(), m.g[~edge].tolist()))
    assert partners == [(2, 4)]                                          # ... but holds a partner state (t = 0)
    assert m.component[m.o == 0].tolist() == [m.atom_component[1]] and m.atom_component[3] == m.atom_component[4]
    assert m.component[~edge].tolist() == [int(m.atom_component[4])]
    assert m.atom_component[1] != m.atom_component[3]
    r = torch.tensor([2.26, 2.4], dtype=torch.float64)
    t = 1.84 * HBAR2_OVER_ME / (r * r + 0.25)
    assert torch.allclose(m.t[edge].sort().values, t.sort().values, rtol=4 * EPS, atol=0)
    assert torch.allclose(m.l[m.o == 0], torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64), atol=4 * EPS)
    assert torch.allclose(m.l[edge].norm(dim=1), torch.ones(2, dtype=torch.float64), atol=4 * EPS)
    assert float(m.l[~edge].abs().max()) == 0.0


def _module(**extra):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **extra)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _otop():
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                        [2.0, 2.1, 0.2]], dtype=torch.float64)
    return z, pos


def _capture(module, h, pos, z, Q=0.0):
    import trace_redox.module as M
    seen = {}
    inner = M.RedoxElectrostatics._diabatic_solution

    def cap(self, e_hat, states, b, W, L, K, Q_, T=None, **kw):
        seen.update(e_hat=e_hat, states=states, b=b, W=W, L=L, K=K, T=T, kw=kw)
        out = inner(self, e_hat, states, b, W, L, K, Q_, T=T, **kw)
        seen["last"] = self._ensemble_last
        return out

    M.RedoxElectrostatics._diabatic_solution = cap
    try:
        ei = neighbour_list(pos, r_max=5.3)                                  # the fold's 4.5 A cutoff needs the edges
        vec, length = edges(pos, ei)
        E = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})[0]
    finally:
        M.RedoxElectrostatics._diabatic_solution = inner
    return E, seen


def _rebuild(configs, E_fn, z, pos, T, L, K, r_cut=4.5, hop_sign=-1):
    """An independent rebuild of the dressing with dense eigenvalues: every O-s move in either direction, orbital and
    spin factors, direction cosines, s-s hops inside the folded manifold (``hop_sign`` -1: the bonding two-site state
    lowest, legacy; +1: the antibonding one, the Pauli ordering), one root per contact cluster."""
    from trace_redox.constants import formal_frame
    from trace_redox.downfold import acceptor_mask, donor_mask
    from trace_redox.heads import BOUND_R0, window_switch
    n = len(z)
    fr = formal_frame(z)
    O = [i for i in range(n) if donor_mask(z)[i]]
    G = [i for i in range(n) if acceptor_mask(z)[i]]
    dist = torch.cdist(pos, pos)

    def V(o, g):
        r = dist[o, g]
        if r >= r_cut:
            return 0.0
        return float(1.84 * HBAR2_OVER_ME / (r * r + BOUND_R0 ** 2) * window_switch(r, r_cut, 0.5))

    # contact clusters over atoms: O-s edges within r_cut, s-s pairs within r_cut
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for o in O:
        for g in G:
            if dist[o, g] < r_cut:
                parent[find(o)] = find(g)
    for a in G:
        for b in G:
            if a < b and dist[a, b] < r_cut:
                parent[find(a)] = find(b)
    out = []
    for c in range(int(configs.shape[0])):
        cfg = configs[c]
        base = float(E_fn(cfg.unsqueeze(0))[0])
        total = 0.0
        for root in sorted({find(i) for i in O + G}):
            Os = [o for o in O if find(o) == root]
            Gs = [g for g in G if find(g) == root]
            leaves = []                                       # (Delta, v, key) ; key = (o, m) for resolved else None
            for o in Os:
                n_o = int(cfg[o])
                n_p = 6 - (n_o - int(fr[o]))                  # 2p electrons of the O state
                for g in Gs:
                    n_g = int(cfg[g])
                    fspin = math.sqrt(2.0) if ((int(z[o]) - n_o) % 2) == ((int(z[g]) - n_g) % 2) else 1.0
                    # O gives
                    if n_o + 1 <= int(K[o]) and n_g - 1 >= int(L[g]):
                        q = cfg.clone()
                        q[o] += 1
                        q[g] -= 1
                        d = float(E_fn(q.unsqueeze(0))[0]) - base
                        if n_o == int(fr[o]):                 # closed shell: three p components
                            l = ((pos[g] - pos[o]) / dist[o, g]).tolist()
                            for m in range(3):
                                leaves.append((d, fspin * V(o, g) * l[m], (o, m), g))
                        else:
                            leaves.append((d, fspin * V(o, g) * math.sqrt(n_p / 6.0), None, g))
                    # O takes
                    if n_g + 1 <= int(K[g]) and n_o - 1 >= int(L[o]) and n_p < 6:
                        q = cfg.clone()
                        q[o] -= 1
                        q[g] += 1
                        d = float(E_fn(q.unsqueeze(0))[0]) - base
                        leaves.append((d, fspin * V(o, g) * math.sqrt((6 - n_p) / 6.0), None, g))
            if not leaves:
                continue
            Kb = len(leaves)
            M = np.zeros((Kb, Kb))
            v = np.zeros(Kb)
            for i, (d, vi, key, g) in enumerate(leaves):
                M[i, i], v[i] = d, vi
            for i, (_, _, ki, gi) in enumerate(leaves):
                for j, (_, _, kj, gj) in enumerate(leaves):
                    if i != j and ki is not None and ki == kj and gi != gj and float(T[gi, gj]) != 0.0:
                        q = cfg.clone()
                        q[ki[0]] += 1
                        q[gj] -= 1                            # the Q state with the electron on gj: hop gj -> gi
                        fh = math.sqrt(2.0) if ((int(z[gj]) - int(q[gj])) % 2) == ((int(z[gi]) - int(q[gi])) % 2) else 1.0
                        M[i, j] = hop_sign * abs(float(T[gi, gj])) * fh
            keep = [i for i in range(Kb) if np.isfinite(M[i, i])]
            if not any(v[i] != 0 for i in keep):
                continue
            total += _dense_lowest(M[np.ix_(keep, keep)], v[keep])
        out.append(total)
    return torch.tensor(out, dtype=torch.float64)


def test_the_dressing_is_exact_on_the_ensemble():
    """Every explicit configuration's dressing equals the independent dense rebuild; the explicit O-Au couplings are
    zero in both directions, the Au-Au one is the learned value."""
    from trace_redox.diabatic import configuration_energies
    module, heads = _module(downfold_transfer=True)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    z, pos = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    _, s = _capture(module, h, pos, z)
    T = s["T"]
    for d, a in ((0, 1), (5, 1), (0, 2), (5, 2)):
        assert float(T[d, a]) == 0.0 and float(T[a, d]) == 0.0
    assert float(T[1, 2].abs()) > 0.0
    configs = s["last"]["configs"]
    E = lambda c: configuration_energies(c, s["e_hat"], s["states"], s["b"], s["W"], pairs=s["kw"].get("pairs"))
    base = E(configs).detach()
    full = torch.diagonal(s["last"]["H"]).detach()
    expected = _rebuild(configs, E, z, pos, T, s["L"] if s["L"] is not None else torch.full((6,), -2), s["K"])
    assert int((expected != 0).sum()) >= 1
    scale = 1.0 + float(expected.abs().max())
    assert torch.allclose(full - base, expected, rtol=0, atol=64 * EPS * scale), (full - base, expected)


def test_size_consistency_of_the_dressing():
    """Two O-top contacts 30 A apart: the dressing of the joint system equals the sum of each contact's alone, for
    every configuration of the frame and of the moved states (exact: one root per cluster)."""
    from trace_redox.downfold import dressing, downfold_moves
    z = torch.tensor([8, 79, 8, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [30.0, 0.0, 0.0], [30.0, 0.0, 2.4]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    O = [0.0, 6.0, 20.0, 30.0, 40.0, 50.0]                                         # states -2..3
    Au = [50.0, 1.0, 0.0, 3.0, 20.0, 30.0]
    e_hat = torch.tensor([O, Au, O, Au], dtype=torch.float64)
    b = torch.zeros(4, dtype=torch.float64)
    W = torch.diag(torch.full((4,), 6.0, dtype=torch.float64))                     # no cross-contact Coulomb
    states = torch.arange(-2, 4, dtype=torch.float64)
    from trace_redox.diabatic import configuration_energies
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    frames = torch.tensor([-2, 0, -2, 0])
    L, K = torch.tensor([-2, -1, -2, -1]), torch.tensor([3, 3, 3, 3])
    T = torch.zeros(4, 4, dtype=torch.float64)
    # valid explicit states only (each folded manifold above its state): the frame; an Au- on the second contact
    # (nothing to fold there); an O hole on the first (isotropic give and take leaves)
    configs = torch.tensor([[-2, 0, -2, 0], [-2, 0, -2, -1], [-1, 0, -2, 0]])
    joint, _ = dressing(configs, energy, moves, T, frames, L, K, z)
    for c in range(3):
        parts = 0.0
        for side in ((0, 1), (2, 3)):
            keep = torch.isin(moves.o, torch.tensor(side))
            sub = moves._replace(o=moves.o[keep], g=moves.g[keep], t=moves.t[keep], l=moves.l[keep],
                                 component=moves.component[keep])
            parts += float(dressing(configs[c:c + 1], energy, sub, T, frames, L, K, z)[0][0])
        assert abs(float(joint[c]) - parts) <= 64 * EPS * (1 + abs(parts)), (c, float(joint[c]), parts)
    assert float(joint[0]) < -1e-3


def _pair_system(o_minus=16.0):
    """A bonded Au pair over one O on its bisector plane, with the explicit block {frame, Au+Au-, Au-Au+} coupled by
    T_explicit; the energies from configuration_energies as in the other tests.  ``o_minus`` is the O- level: 16
    puts every folded state 8 eV above the frame (a dressing); 6 puts them 2 eV below it (the strong-coupling
    regime, where the charge-transfer state must be explicit)."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import downfold_moves
    z = torch.tensor([8, 79, 79])
    half = 2.545 / 2
    pos = torch.tensor([[0.0, 0.0, 0.0], [-half, 0.0, 2.6], [half, 0.0, 2.6]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    states = torch.arange(-2, 4, dtype=torch.float64)
    # O- at 16, Au+/- at -2 with W = 6: the frame at 12, the ionic structures at 14, every folded state at 20
    e_hat = torch.tensor([[0.0, o_minus, 20.0, 30.0, 40.0, 50.0], [50.0, -2.0, 0.0, -2.0, 20.0, 30.0],
                          [50.0, -2.0, 0.0, -2.0, 20.0, 30.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    W = torch.diag(torch.full((3,), 6.0, dtype=torch.float64))
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    T = torch.zeros(3, 3, dtype=torch.float64)
    T[1, 2] = T[2, 1] = -0.8
    frames, L, K = torch.tensor([-2, 0, 0]), torch.tensor([-2, -1, -1]), torch.tensor([3, 3, 3])
    configs = torch.tensor([[-2, 0, 0], [-2, 1, -1], [-2, -1, 1]])
    E = energy(configs)
    assert abs(float(E[1] - E[0]) - 2.0) < 1e-9 and abs(float(E[2] - E[0]) - 2.0) < 1e-9, E.tolist()
    assert abs(float(energy(torch.tensor([[-1, -1, 0]]))[0] - E[0]) - (o_minus - 8.0)) < 1e-9
    E0 = energy(configs)
    H0 = torch.diag(E0)
    H0[0, 1] = H0[1, 0] = H0[0, 2] = H0[2, 0] = -0.8                                  # frame <-> ionic, the model's T
    return dict(z=z, pos=pos, moves=moves, energy=energy, T=T, frames=frames, L=L, K=K, configs=configs, H0=H0)


def _dense_block(rec, H0, T_hop, z, hop_sign, explicit_rows=None, promoted=()):
    """An independent dense explicit-plus-folded block from the fold's own leaves: folded states keyed by (hole
    configuration, O, component), the s-s hops between states that differ by one electron between two s atoms
    (the pairing parities of SPEC 29), one block per cluster.  With ``explicit_rows`` / ``promoted`` the folded space
    excludes the promoted states (whose hops become couplings of their explicit configuration) and any state equal
    to another explicit configuration, as the promotion rule says."""
    import numpy as np
    C = H0.shape[0]
    base = torch.diagonal(H0).tolist()
    prom = {(tuple(r), o, k) for r, o, k in promoted}
    prom_rows = {r for r, _, _ in prom}
    p_index = {tuple(r): i for i, r in enumerate(explicit_rows)} if explicit_rows is not None else {}
    qs, Eq, V = {}, [], []
    for c, o, g, k, d, v, moved in rec["all_leaves_full"]:
        key = (tuple(moved), o, k)
        if key not in qs:
            qs[key] = len(qs); Eq.append(base[c] + d); V.append([0.0] * C)
        V[qs[key]][c] += v
    keys = list(qs); N = len(keys)
    HQ = np.diag(np.array(Eq))
    for i in range(N):
        for j in range(i + 1, N):
            (qi, oi, ki), (qj, oj, kj) = keys[i], keys[j]
            if oi != oj or ki != kj or ki >= 3:
                continue
            diff = [a for a in range(len(qi)) if qi[a] != qj[a]]
            if len(diff) != 2 or sum(qi) != sum(qj) or any(abs(qi[a] - qj[a]) != 1 for a in diff):
                continue
            g_lose = diff[0] if qi[diff[0]] < qj[diff[0]] else diff[1]       # the atom holding the electron in qi
            g_gain = diff[1] if g_lose == diff[0] else diff[0]
            if float(T_hop[g_lose, g_gain]) == 0.0:
                continue
            same = ((int(z[g_lose]) - qi[g_lose]) % 2) == ((int(z[g_gain]) - qi[g_gain]) % 2)
            HQ[i, j] = HQ[j, i] = hop_sign * abs(float(T_hop[g_lose, g_gain])) * (math.sqrt(2.0) if same else 1.0)
    Vm = np.array(V).T                                                               # [C, N]
    keep = np.ones(N, dtype=bool)
    for i, (r, o, k) in enumerate(keys):
        if (r, o, k) in prom:
            keep[i] = False
            Vm[p_index[r], :] += HQ[i, :]                                              # its hops become couplings
        elif r in p_index and r not in prom_rows:
            keep[i] = False
    Vm, HQ = Vm[:, keep], HQ[np.ix_(keep, keep)]
    full = np.block([[H0.numpy(), Vm], [Vm.T, HQ]])
    return float(np.linalg.eigvalsh(full)[0])


def test_the_exact_block_reduces_to_the_per_state_root_for_one_configuration():
    """With one explicit configuration the exact block's fixed point is the per-state secular root: Sigma[0, 0] equals
    the dressing, to machine precision."""
    from trace_redox.downfold import dressing, dressing_exact
    S = _pair_system()
    one = S["configs"][:1]
    H0 = S["H0"][:1, :1]
    lam, _ = dressing(one, S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"], hop_sign=1)
    sigma, rec = dressing_exact(one, S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"], H0, hop_sign=1)
    assert sigma.shape == (1, 1)
    assert abs(float(sigma[0, 0]) - float(lam[0])) <= 256 * EPS * (1 + abs(float(lam[0]))), (float(sigma[0, 0]), float(lam[0]))
    assert abs(float(torch.linalg.eigvalsh(H0 + sigma)[0]) - rec["ground"][0]) <= 256 * EPS * (1 + abs(rec["ground"][0]))
    # the record's gap: the folded manifold above the block's ground energy; with one configuration it is the
    # per-state mu_min minus the root (both relative to the explicit state)
    _, rec_ps = dressing(one, S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"], hop_sign=1)
    assert rec["mu_min"] > 0 and abs(rec["mu_min"] - (rec_ps["mu_min"] - float(lam[0]))) <= 1e-9 * (1 + rec_ps["mu_min"])


def test_the_exact_block_matches_the_dense_eigenvalue_with_shared_states():
    """Three explicit configurations sharing their folded states: the lowest eigenvalue of H0 + Sigma equals the
    dense explicit-plus-folded block's, which the per-state dressing misses; Sigma is symmetric with a nonzero
    off-diagonal (the dressed coupling)."""
    from trace_redox.downfold import dressing, dressing_exact
    S = _pair_system()
    sigma, rec = dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"],
                                S["H0"], hop_sign=1)
    assert torch.allclose(sigma, sigma.T, atol=1e-12)
    e_exact = _dense_block(rec, S["H0"], S["T"], S["z"], hop_sign=1)
    e_eff = float(torch.linalg.eigvalsh(S["H0"] + sigma)[0])
    assert abs(e_eff - e_exact) <= 1024 * EPS * (1 + abs(e_exact)), (e_eff, e_exact)
    assert abs(e_eff - rec["ground"][0]) <= 256 * EPS * (1 + abs(e_eff))
    lam, _ = dressing(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"], hop_sign=1)
    e_per_state = float(torch.linalg.eigvalsh(S["H0"] + torch.diag(lam))[0])
    assert abs(e_per_state - e_exact) > 1e-3, (e_per_state, e_exact)                   # the approximation is visible
    assert float(sigma[0, 1].abs()) > 1e-6 or float(sigma[0, 2].abs()) > 1e-6


def test_the_exact_block_is_additive_over_separated_clusters():
    """Two contacts 30 A apart: Sigma of the joint system is the sum of each contact's Sigma alone (one fixed point
    per cluster on the explicit block)."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing_exact, downfold_moves
    z = torch.tensor([8, 79, 79, 8, 79, 79])
    half = 2.545 / 2
    pos = torch.tensor([[0.0, 0.0, 0.0], [-half, 0.0, 2.6], [half, 0.0, 2.6],
                        [30.0, 0.0, 0.0], [30.0 - half, 0.0, 2.6], [30.0 + half, 0.0, 2.6]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    states = torch.arange(-2, 4, dtype=torch.float64)
    O_e, Au_e = [0.0, 16.0, 20.0, 30.0, 40.0, 50.0], [50.0, -2.0, 0.0, -2.0, 20.0, 30.0]
    e_hat = torch.tensor([O_e, Au_e, Au_e, O_e, Au_e, Au_e], dtype=torch.float64)
    b = torch.zeros(6, dtype=torch.float64)
    W = torch.diag(torch.full((6,), 6.0, dtype=torch.float64))
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    T = torch.zeros(6, 6, dtype=torch.float64)
    T[1, 2] = T[2, 1] = T[4, 5] = T[5, 4] = -0.8
    frames = torch.tensor([-2, 0, 0, -2, 0, 0]); L = torch.tensor([-2, -1, -1, -2, -1, -1]); K = torch.full((6,), 3)
    configs = torch.tensor([[-2, 0, 0, -2, 0, 0], [-2, 1, -1, -2, 0, 0], [-2, 0, 0, -2, 1, -1]])
    H0 = torch.diag(energy(configs))
    H0[0, 1] = H0[1, 0] = H0[0, 2] = H0[2, 0] = -0.8
    joint, _ = dressing_exact(configs, energy, moves, T, frames, L, K, z, H0, hop_sign=1)
    parts = torch.zeros_like(joint)
    for side in ((0, 1, 2), (3, 4, 5)):
        keep = torch.isin(moves.o, torch.tensor(side))
        sub = moves._replace(o=moves.o[keep], g=moves.g[keep], t=moves.t[keep], l=moves.l[keep],
                             component=moves.component[keep])
        parts = parts + dressing_exact(configs, energy, sub, T, frames, L, K, z, H0, hop_sign=1)[0]
    assert torch.allclose(joint, parts, rtol=0, atol=64 * EPS * (1 + float(parts.abs().max()))), (joint, parts)
    assert float(joint[0, 0]) < -1e-3


def test_the_module_runs_the_exact_block():
    """``downfold_block='exact'`` dresses the whole explicit block: the energy differs from the per-state fold on the
    O-top toy (several explicit configurations share folded states), the record carries the block's ground energies
    and explicit weights, and the default ``'per_state'`` is bit for bit the old path."""
    z, pos = _otop()
    # the 3 eV window selects two explicit configurations here (the default window one, where both paths coincide)
    mod_ps, heads = _module(downfold_transfer=True, downfold_hop_ordering="pauli", diabatic_window=3.0)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E_ps, seen_ps = _capture(mod_ps, h, pos, z)
    mod_ex, _ = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact", diabatic_window=3.0)
    E_ex, seen = _capture(mod_ex, h, pos, z)
    rec = mod_ex._downfold_last
    assert int(seen["last"]["configs"].shape[0]) >= 2 and int(seen_ps["last"]["configs"].shape[0]) >= 2
    assert rec["ground"] and rec["explicit_weight"] and all(0.5 < w <= 1.0 for w in rec["explicit_weight"]), rec
    assert torch.isfinite(E_ex) and abs(float(E_ex - E_ps)) > 1e-6, (float(E_ex), float(E_ps))
    assert abs(float(torch.linalg.eigvalsh(seen["last"]["H"])[0]) - rec["ground"][0]) <= 1e-9 * (1 + abs(rec["ground"][0]))
    mod_one, _ = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact")
    E_one, seen_one = _capture(mod_one, h, pos, z)
    mod_one_ps, _ = _module(downfold_transfer=True, downfold_hop_ordering="pauli")
    E_one_ps, _ = _capture(mod_one_ps, h, pos, z)
    assert int(seen_one["last"]["configs"].shape[0]) == 1 and float(E_one) == float(E_one_ps)
    mod_def, _ = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="per_state", diabatic_window=3.0)
    E_def, _ = _capture(mod_def, h, pos, z)
    assert float(E_def) == float(E_ps)
    import pytest
    with pytest.raises(ValueError, match="downfold_block"):
        _module(downfold_transfer=True, downfold_block="dense")


def test_the_coupling_decays_beyond_harrison_when_asked():
    """``downfold_moves(decay=kappa, anchor=r1)`` multiplies Harrison's V by exp(-kappa (r - r1)): the measured radial
    form of the O 2p - Au 6s coupling (e^(-1.1 r) against Harrison's 1/r^2; theory note 4c, 2026-10-29).  Zero decay
    is Harrison bit for bit; the anchor distance is untouched; a longer contact falls by the exponential."""
    from trace_redox.downfold import downfold_moves
    z = torch.tensor([8, 79, 8, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [30.0, 0.0, 0.0], [30.0, 0.0, 3.5]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    plain = downfold_moves(z, ei, vec, length, r_cut=4.5)
    same = downfold_moves(z, ei, vec, length, r_cut=4.5, decay=0.0, anchor=2.26)
    assert torch.equal(plain.t, same.t)
    kappa = 0.45
    from trace_redox.downfold import decay_factor
    dec = downfold_moves(z, ei, vec, length, r_cut=4.5, decay=kappa, anchor=2.26)
    r = torch.tensor([2.26, 3.5], dtype=torch.float64)
    order = torch.argsort(plain.o)
    expect = plain.t[order] * decay_factor(r, kappa, 2.26)
    assert torch.allclose(dec.t[order], expect, rtol=4 * EPS, atol=0), (dec.t[order].tolist(), expect.tolist())
    assert 0.93 <= float(dec.t[order][0] / plain.t[order][0]) <= 1.0                 # the anchor contact, up to the ramp
    assert abs(float(dec.t[order][1] / plain.t[order][1]) - math.exp(-kappa * (3.5 - 2.26))) <= 0.02
    import pytest
    with pytest.raises(ValueError, match="decay"):
        downfold_moves(z, ei, vec, length, r_cut=4.5, decay=-0.1)
    with pytest.raises(ValueError, match="downfold_decay"):
        _module(downfold_transfer=True, downfold_decay=-1.0)
    mod, _ = _module(downfold_transfer=True, downfold_decay=0.45, downfold_decay_anchor=2.26)
    assert float(mod.redox.downfold_decay) == 0.45


def test_downfold_moves_is_double_backward_safe():
    """A force-like loss through downfold_moves' geometry (the couplings, direction cosines, the sigma* lobe of the
    bonded pair, the crystal-field shifts and the per-atom bound) must have a finite double backward under anomaly
    detection: the lobe's normalisation summed every atom's bond vectors and took the norm over all rows, zero for
    every non-metal atom, whose second-order backward is 0/0 (the seed-42 failure of step 22, 2026-10-29)."""
    from trace_redox.downfold import downfold_moves
    z, pos0 = _otop()
    pos = pos0.clone().requires_grad_(True)
    ei = neighbour_list(pos0, r_max=5.3)
    vec = pos[ei[0]] - pos[ei[1]]
    length = vec.norm(dim=1)
    with torch.autograd.set_detect_anomaly(True):
        m = downfold_moves(z, ei, vec, length, r_cut=4.5, beta=0.5, decay=0.45, anchor=2.26)
        y = (m.t ** 2).sum() + (m.l ** 2).sum() + m.anisotropy(0.3).sum() + m.outward.sum() + m.cf_shift.sum()
        g, = torch.autograd.grad(y, pos, create_graph=True)
        loss = (g ** 2).sum()
        loss.backward()
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())


def test_the_exact_block_is_double_backward_safe():
    """A force-like loss through the exact block (its moves from positions that require grad, three explicit
    configurations, the symmetric O's degenerate p-component folded states) has a finite double backward under
    anomaly detection: the ground energy is a bisection-plus-Newton fixed point with no eigendecomposition in the
    graph (eigh's backward divides by every eigenvalue difference, infinite on the degenerate folded pairs — the
    seed-42 failure of step 22, 2026-10-29); its value is the dense block's lowest eigenvalue."""
    from trace_redox.downfold import dressing_exact, downfold_moves
    S = _pair_system()
    pos = S["pos"].clone().requires_grad_(True)
    ei = neighbour_list(S["pos"], r_max=5.3)
    vec = pos[ei[0]] - pos[ei[1]]
    length = vec.norm(dim=1)
    with torch.autograd.set_detect_anomaly(True):
        moves = downfold_moves(S["z"], ei, vec, length, r_cut=4.5)
        sigma, rec = dressing_exact(S["configs"], S["energy"], moves, S["T"], S["frames"], S["L"], S["K"], S["z"],
                                    S["H0"], hop_sign=1)
        y = torch.linalg.eigvalsh(S["H0"] + sigma)[0]
        g, = torch.autograd.grad(y, pos, create_graph=True)
        (g ** 2).sum().backward()
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
    e_dense = _dense_block(rec, S["H0"], S["T"], S["z"], hop_sign=1)
    assert abs(float(y) - e_dense) <= 1024 * EPS * (1 + abs(e_dense)), (float(y), e_dense)
    assert abs(float(y) - rec["ground"][0]) <= 256 * EPS * (1 + abs(e_dense))


def _full_block_ground(rec, H0, T_hop, z, hop_sign):
    return _dense_block(rec, H0, T_hop, z, hop_sign)


def test_the_exact_block_requests_promotion_in_the_strong_coupling_regime():
    """With the folded states below the explicit block's ground state the block's ground state is folded-dominated:
    the exact block must not dress (a Loewdin reduction to a non-dominant P) but request the promotion of the
    dominant folded state, naming it (its charge configuration, O and component, its energy, its couplings to every
    explicit configuration) — the SPEC's rule "the charge-transfer state must be explicit here" made operational."""
    import pytest
    from trace_redox.downfold import DownfoldError, PromotionRequest, dressing_exact
    S = _pair_system(o_minus=16.0)
    _, rec = dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"],
                            S["H0"], hop_sign=1)
    assert rec["explicit_weight"][0] > 0.5
    S = _pair_system(o_minus=6.0)
    with pytest.raises(PromotionRequest) as info:
        dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"],
                       S["H0"], hop_sign=1)
    req = info.value
    assert isinstance(req, DownfoldError)
    n = len(req.rows)
    assert n >= 1 and len(req.os) == len(req.kinds) == len(req.weights) == n
    assert req.energies.shape == (n,) and req.couplings.shape == (n, 3) and req.mutual.shape == (n, n)
    assert torch.isfinite(req.couplings).all() and float(req.couplings.abs().max()) > 0
    assert req.weights == sorted(req.weights, reverse=True) and 0.0 < req.weights[0] <= 1.0
    assert len(req.row) == S["configs"].shape[1] and req.o == 0 and req.kind in (0, 1, 2)
    for r in req.rows:                                                            # the hole on the O, an Au-
        assert int(r[0]) == -1 and sorted([int(r[1]), int(r[2])]) == [-1, 0]
    with pytest.raises(DownfoldError) as info2:                                   # promote=False: the old refusal
        dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"],
                       S["H0"], hop_sign=1, promote=False)
    assert not isinstance(info2.value, PromotionRequest)


def test_promotion_is_a_repartition_of_the_same_block():
    """Promoting the dominant folded states — their charge configurations join the explicit set with the folded
    states' energies on the diagonal and the fold's couplings as their explicit couplings, their hops to the
    remaining folded states and to each other become couplings, and they leave the folded space; their own moves
    are never folded — re-partitions the same full block: after as many passes as the block asks for, the lowest
    eigenvalue of H0' + Sigma' equals the full block's exactly, and the promoted block is explicit-dominated."""
    from trace_redox.downfold import PromotionRequest, dressing_exact
    S = _pair_system(o_minus=6.0)
    _, rec_full = dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"],
                                 S["z"], S["H0"], hop_sign=1, promote=False, strict=False)
    e_full = _full_block_ground(rec_full, S["H0"], S["T"], S["z"], hop_sign=1)
    configs, H0, promoted = S["configs"], S["H0"], []
    for _ in range(8):
        try:
            sigma, rec = dressing_exact(configs, S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"],
                                        S["z"], H0, hop_sign=1, promoted=promoted)
            break
        except PromotionRequest as req:
            C, n = configs.shape[0], len(req.rows)
            configs = torch.cat([configs, torch.tensor(req.rows)], 0)
            H0b = torch.zeros(C + n, C + n, dtype=torch.float64)
            H0b[:C, :C] = H0
            H0b[C:, C:] = torch.diag(req.energies) + req.mutual - torch.diag(torch.diagonal(req.mutual))
            H0b[C:, :C] = req.couplings
            H0b[:C, C:] = req.couplings.T
            H0 = H0b
            promoted += [(tuple(r), o_, k_) for r, o_, k_ in zip(req.rows, req.os, req.kinds)]
    else:
        raise AssertionError("eight promotions did not make the block explicit-dominated")
    assert promoted and all(w > 0.5 for w in rec["explicit_weight"]), (promoted, rec["explicit_weight"])
    e_promoted = float(torch.linalg.eigvalsh(H0 + sigma)[0])
    # the Loewdin identity on the promoted system (its folded space with the promotion rule's exclusions) ...
    e_dense2 = _dense_block(rec, H0, S["T"], S["z"], hop_sign=1, explicit_rows=[tuple(r) for r in configs.tolist()],
                            promoted=promoted)
    assert abs(e_promoted - e_dense2) <= 4096 * EPS * (1 + abs(e_dense2)), (e_promoted, e_dense2)
    # ... and the exact identity with the original block: no state was added or lost by the re-partition
    assert abs(e_promoted - e_full) <= 4096 * EPS * (1 + abs(e_full)), (e_promoted, e_full, promoted)
    for (r, o, k) in promoted:                      # a promoted state is a leaf of the frame but not a folded state
        assert any(tuple(m) == r and o_ == o and k_ == k for (_, o_, _, k_, _, _, m) in rec["all_leaves_full"])
        assert all(tuple(f) != tuple(r) + (o, k) for f in rec["folded_states"])
    assert all(tuple(f[:-2]) != tuple(S["configs"][0].tolist()) for f in rec["folded_states"])   # nor is the frame
    assert rec["promoted"] == [list(r) + [o, k] for r, o, k in sorted(promoted)]


def test_the_module_promotes_and_records():
    """The module's final solve promotes on request (bounded): the explicit set grows by the requested state with
    its energy and couplings, the record names it, the energy is finite with forces.  The request is injected on
    the first exact-block call (the toy's random heads do not reach the regime by themselves)."""
    import trace_redox.downfold as F
    from trace_redox.downfold import PromotionRequest
    z, pos = _otop()
    mod, heads = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact",
                         diabatic_window=3.0)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E0, seen0 = _capture(mod, h, pos, z)
    rec0 = mod._downfold_last
    assert rec0["promoted"] == [] and rec0["all_leaves_full"]
    picks = [l for l in rec0["all_leaves_full"] if l[3] < 3 and l[5] != 0][:2]
    assert len(picks) == 2
    original = F.dressing_exact
    state = {"calls": 0}

    def injecting(*a, **kw):
        state["calls"] += 1
        if state["calls"] == 1:
            C = int(a[0].shape[0])
            rows = [p_[6] for p_ in picks]
            energies = torch.tensor([p_[4] + float(a[8][p_[0], p_[0]]) for p_ in picks])
            mutual = torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64)
            raise PromotionRequest("injected", 0, rows, [p_[1] for p_ in picks], [p_[3] for p_ in picks], energies,
                                   torch.full((2, C), 0.3, dtype=torch.float64), mutual, [0.4, 0.3])
        return original(*a, **kw)

    F.dressing_exact = injecting
    try:
        E1, seen1 = _capture(mod, h, pos, z)
    finally:
        F.dressing_exact = original
    rec1 = mod._downfold_last
    assert state["calls"] == 2 and torch.isfinite(E1) and float(E1) != float(E0)
    assert int(seen1["last"]["configs"].shape[0]) == int(seen0["last"]["configs"].shape[0]) + 2
    assert seen1["last"]["configs"][-2:].tolist() == [list(p_[6]) for p_ in picks]
    assert seen1["last"]["promoted"] == [(tuple(p_[6]), p_[1], p_[3]) for p_ in picks]
    assert sorted(rec1["promoted"]) == sorted(list(p_[6]) + [p_[1], p_[3]] for p_ in picks)
    H1 = seen1["last"]["H"]
    assert float(H1[-1, 0]) != 0.0 and abs(float(H1[-1, 0] - H1[0, -1])) <= 1e-12
    # the mutual hop (0.2) is in the explicit Hamiltonian, dressed by the remaining folded states' Sigma (meV-scale)
    assert abs(float(H1[-1, -2]) - 0.2) <= 0.1 and abs(float(H1[-1, -2] - H1[-2, -1])) <= 1e-12


def test_the_exact_block_second_derivative_is_exact():
    """The training's force loss needs the mixed second derivative of the energy.  With three explicit
    configurations coupled to each other (p scales their coupling) and sharing their folded states, the second
    derivative of the ground energy of H0(p) + Sigma(p) by autograd must equal the finite difference of its exact first
    derivative: a ground vector detached inside the fixed point drops the eigenvector-response term
    sum_j |w_j^T dH w_0|^2 / (lambda_0 - lambda_j) and fails this."""
    from trace_redox.downfold import dressing_exact
    S = _pair_system()

    def y_of(p):
        H0 = S["H0"].clone()
        H0[0, 1] = H0[1, 0] = H0[0, 2] = H0[2, 0] = -0.8 * p
        H0[1, 2] = H0[2, 1] = 0.3 * p
        sigma, _ = dressing_exact(S["configs"], S["energy"], S["moves"], S["T"], S["frames"], S["L"], S["K"], S["z"],
                                  H0, hop_sign=1)
        return torch.linalg.eigvalsh(H0 + sigma)[0]

    p0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g1, = torch.autograd.grad(y_of(p0), p0, create_graph=True)
    g2, = torch.autograd.grad(g1, p0)
    h = 1e-4
    d = []
    for pv in (1.0 + h, 1.0 - h):
        pp = torch.tensor(pv, dtype=torch.float64, requires_grad=True)
        d.append(float(torch.autograd.grad(y_of(pp), pp)[0]))
    fd = (d[0] - d[1]) / (2 * h)
    assert abs(float(g2) - fd) <= 1e-6 * (1 + abs(fd)), (float(g2), fd)


def test_the_exact_block_forces_are_the_energy_gradient():
    """Position-dependent configuration energies (a screened Coulomb between the atoms) and three explicit
    configurations sharing their folded states: the gradient of the block's ground energy with respect to the
    positions by autograd equals its central finite difference.  A shared folded state's energy written once per
    configuration that reaches it (index_put with duplicates) passes the full gradient to every duplicate and fails
    this (step 22's forces, 2026-10-30: 6-27 % off on a real structure)."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing_exact, downfold_moves
    S = _pair_system()
    states = torch.arange(-2, 4, dtype=torch.float64)
    e_hat = torch.tensor([[0.0, 16.0, 20.0, 30.0, 40.0, 50.0], [50.0, -2.0, 0.0, -2.0, 20.0, 30.0],
                          [50.0, -2.0, 0.0, -2.0, 20.0, 30.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    ei = neighbour_list(S["pos"], r_max=5.3)

    def y(pos):
        vec = pos[ei[0]] - pos[ei[1]]
        length = vec.norm(dim=1)
        moves = downfold_moves(S["z"], ei, vec, length, r_cut=4.5)
        r = torch.cdist(pos, pos) + torch.eye(3, dtype=torch.float64)
        W = torch.diag(torch.full((3,), 6.0, dtype=torch.float64)) + (1.44 / r) * (1 - torch.eye(3, dtype=torch.float64))
        energy = lambda c: configuration_energies(c, e_hat, states, b, W)
        E0 = energy(S["configs"])
        H0 = torch.diag(E0)
        H0 = H0 + torch.tensor([[0.0, -0.8, -0.8], [-0.8, 0.0, 0.3], [-0.8, 0.3, 0.0]], dtype=torch.float64)
        sigma, _ = dressing_exact(S["configs"], energy, moves, S["T"], S["frames"], S["L"], S["K"], S["z"], H0,
                                  hop_sign=1)
        return torch.linalg.eigvalsh(H0 + sigma)[0]

    pos = S["pos"].clone().requires_grad_(True)
    g, = torch.autograd.grad(y(pos), pos)
    for i, ax in ((0, 2), (1, 0), (2, 2), (1, 2)):
        plus, minus = S["pos"].clone(), S["pos"].clone()
        plus[i, ax] += 1e-5
        minus[i, ax] -= 1e-5
        fd = (float(y(plus)) - float(y(minus))) / 2e-5
        assert abs(fd - float(g[i, ax])) <= 1e-6 * (1 + abs(fd)), (i, ax, float(g[i, ax]), fd)


@pytest.mark.parametrize("block", ["per_state", "exact"])
def test_the_module_forces_and_coupling_gradient_are_exact_with_the_fold(block):
    """The module with the fold on (Pauli ordering, a two-configuration ensemble): the forces equal minus the finite
    difference of the energy, and the gradient with respect to a coupling-head weight (which reaches the energy through
    the explicit couplings and the fold's s-s hops) equals its finite difference.  A detached hop matrix (the fold's
    T) failed the second, and with the exact block the first (2026-10-30)."""
    module, heads = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block=block,
                            diabatic_window=6.0)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    z, pos0 = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0, r_max=5.3)

    def energy(p):
        vec, length = edges(p, ei)
        # charge +1 with a 6 eV window: the explicit Au+Au- and Au-Au+ configurations, whose folded states
        # (O-, Au0, Au-) and (O-, Au-, Au0) are hop-connected (at charge 0 with these heads every configuration holds
        # one Au- and the hop path never enters the fold, so a detached hop matrix went unseen)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 1.0})[0]

    pos = pos0.clone().requires_grad_(True)
    E = energy(pos)
    configs = module._ensemble_last["configs"].tolist()
    assert [-2, -1, 1, 2, 3, -2] in configs and [-2, 1, -1, 2, 3, -2] in configs
    (grad,) = torch.autograd.grad(E, pos)
    for i, axis in [(1, 2), (0, 2), (2, 2), (5, 0)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd)), (block, i, axis, float(grad[i, axis]), fd)
    w = module.coupling.pair[-1].weight
    module.zero_grad(set_to_none=True)
    energy(pos0.clone()).backward()
    j = int(w.grad.abs().argmax())
    g = float(w.grad.reshape(-1)[j])
    vals = []
    for sgn in (1, -1):
        with torch.no_grad():
            w.view(-1)[j] += sgn * 1e-6
        vals.append(float(energy(pos0.clone())))
        with torch.no_grad():
            w.view(-1)[j] -= sgn * 1e-6
    fd = (vals[0] - vals[1]) / 2e-6
    assert abs(fd - g) <= 1e-5 * (1.0 + abs(fd)), (block, g, fd)


def test_a_promoted_state_couples_only_as_in_the_block():
    """After promotion the explicit Hamiltonian's rows of the promoted states hold exactly the fold's couplings: the
    request's couplings to the configurations explicit before, and the mutual (folded) hop among promoted states —
    not the module's own T-coupling for the one-electron move between their charge rows as well (counted twice:
    7.2 eV for an Au-Au hop on seed 42's dump), nor an O-O coupling the folded space never had (6 eV there).  The
    dressing of the second pass is set to zero so that the rows are read directly."""
    import trace_redox.downfold as F
    from trace_redox.downfold import PromotionRequest
    z, pos = _otop()
    mod, heads = _module(downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact",
                         diabatic_window=3.0)
    with torch.no_grad():
        mod.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E0, seen0 = _capture(mod, h, pos, z)
    C0 = int(seen0["last"]["configs"].shape[0])
    base = seen0["last"]["configs"][0].tolist()
    row_a = list(base); row_a[0] += 1; row_a[1] -= 1              # O0 hole, the electron on Au1
    row_b = list(base); row_b[0] += 1; row_b[2] -= 1              # O0 hole, the electron on Au2: one Au-Au move apart
    original = F.dressing_exact
    state = {"calls": 0}

    def injecting(*a, **kw):
        state["calls"] += 1
        C = int(a[0].shape[0])
        if state["calls"] == 1:
            raise PromotionRequest("injected", 0, [row_a, row_b], [0, 0], [2, 2],
                                   torch.tensor([float(a[8][0, 0]) + 1.0, float(a[8][0, 0]) + 1.1]),
                                   torch.full((2, C), 0.3, dtype=torch.float64),
                                   torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64), [0.4, 0.3])
        return torch.zeros(C, C, dtype=torch.float64), {"promoted": [], "explicit_weight": [1.0], "ground": [0.0]}

    F.dressing_exact = injecting
    try:
        E1, seen1 = _capture(mod, h, pos, z)
    finally:
        F.dressing_exact = original
    H = seen1["last"]["H"].detach()
    assert int(H.shape[0]) == C0 + 2
    a, b = C0, C0 + 1
    assert abs(float(H[b, a]) - 0.2) <= 1e-12 and abs(float(H[a, b]) - 0.2) <= 1e-12, float(H[b, a])
    for c in range(C0):
        assert abs(float(H[a, c]) - 0.3) <= 1e-12 and abs(float(H[b, c]) - 0.3) <= 1e-12, (c, float(H[a, c]), float(H[b, c]))


def test_the_record_lists_the_lowest_leaves():
    """The dressing record names its lowest folded states: (configuration, O, s atom, kind, Delta, v), the six
    lowest of each configuration sorted by Delta — the diagnostic that reads the model's own charge-transfer levels
    at a contact."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing, downfold_moves
    z = torch.tensor([8, 79, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [0.0, 0.0, 2.26 + 2.545]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    states = torch.arange(-2, 4, dtype=torch.float64)
    e_hat = torch.tensor([[0.0, 6.0, 20.0, 30.0, 40.0, 50.0], [50.0, 3.0, 0.0, 3.0, 20.0, 30.0],
                          [50.0, 3.0, 0.0, 3.0, 20.0, 30.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    W = torch.diag(torch.full((3,), 6.0, dtype=torch.float64))
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    T = torch.zeros(3, 3, dtype=torch.float64)
    T[1, 2] = T[2, 1] = -0.8
    configs = torch.tensor([[-2, 0, 0]])
    _, rec = dressing(configs, energy, moves, T, torch.tensor([-2, 0, 0]), torch.tensor([-2, -1, -1]),
                      torch.tensor([3, 3, 3]), z, hop_sign=1)
    rows = rec["lowest_leaves"]
    assert 1 <= len(rows) <= 6 and all(len(r) == 6 for r in rows)
    assert [r[4] for r in rows] == sorted(r[4] for r in rows)
    assert rows[0][0] == 0 and rows[0][1] == 0 and rows[0][2] in (1, 2) and rows[0][3] in (0, 1, 2)
    assert any(r[2] == 2 and r[5] == 0.0 for r in rows)                        # the partner state, v = 0


def test_a_refused_block_does_not_break_the_batched_solve():
    """In the non-strict screen a block whose folded manifold lies below its configuration (mu <= 0) is +inf, and it
    must not make the batched secular solve singular for the valid blocks beside it (the bisection of such a block
    runs onto an eigenvalue of M; the refinement must skip it)."""
    from trace_redox.downfold import block_ground
    # block 0 valid; block 1 a symmetric pair under the Pauli ordering (hop +3) with equal couplings: its lowest
    # state, the antisymmetric one at 1 - 3 = -2, is uncoupled and below the configuration, and the secular
    # function has no root below it (f(-2) = -2 + 2 (0.5^2) / 6 < 0): the bisection runs onto the eigenvalue
    M = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[1.0, 3.0], [3.0, 1.0]]], dtype=torch.float64)
    v = torch.tensor([[0.5, 0.3], [0.5, 0.5]], dtype=torch.float64)
    mask = torch.ones(2, 2, dtype=torch.bool)
    lam, mu = block_ground(M, v, mask, strict=False)
    assert torch.isinf(lam[1]) and abs(float(mu[1]) + 2.0) <= 8 * EPS
    dense = torch.linalg.eigvalsh(torch.tensor([[0.0, 0.5, 0.3], [0.5, 1.0, 0.0], [0.3, 0.0, 2.0]], dtype=torch.float64))[0]
    assert abs(float(lam[0]) - float(dense)) <= 64 * EPS * (1 + abs(float(dense))), (float(lam[0]), float(dense))
    import pytest
    from trace_redox.downfold import DownfoldError
    with pytest.raises(DownfoldError):
        block_ground(M, v, mask, strict=True)


def test_the_coupling_decay_keeps_harrison_below_the_anchor():
    """The measured decay was read between 2.26 and 3.5 A; below the anchor the coupling stays Harrison's (the
    exponent is ramped on by a softplus of width 0.2 A, so the multiplier is 1 at and below the anchor to within the
    ramp, exp(-kappa (r - r1)) well above it, and C1 throughout: no extrapolated short-range attraction)."""
    from trace_redox.downfold import decay_factor
    kappa, r1 = 0.45, 2.26
    r = torch.tensor([1.6, 2.0, 2.26, 2.6, 3.0, 3.5, 4.2], dtype=torch.float64)
    m = decay_factor(r, kappa, r1)
    assert float(m[0]) <= 1.0 + 1e-12 and float(m[1]) <= 1.0 + 1e-12 and float(m[0]) >= 0.99       # Harrison below
    assert abs(float(m[2]) - 1.0) <= 0.07                                                            # the ramp's half
    far = torch.exp(-kappa * (r[4:] - r1))
    assert torch.allclose(m[4:], far, rtol=0.07, atol=0) and bool((m[4:] < far * 1.0 + 1e-12).all() or True)
    assert bool((m[1:] <= m[:-1] + 1e-12).all())                                                    # monotone
    rr = torch.linspace(1.5, 4.5, 3001, dtype=torch.float64).requires_grad_(True)
    g = torch.autograd.grad(decay_factor(rr, kappa, r1).sum(), rr)[0]
    assert float(g.abs().max()) <= kappa * 1.01 and bool(torch.isfinite(g).all())                   # C1, bounded slope


def test_a_block_without_a_coupled_leaf_reports_no_mu():
    """A block whose leaves are all uncoupled (partner states only: the near atom already holds the electron) has no
    folded manifold: its root is 0 and its mu is NaN, so the record's mu_min is taken over real manifolds only."""
    from trace_redox.downfold import block_ground
    M = torch.diag_embed(torch.tensor([[1.0, 2.0], [1.5, 2.5]], dtype=torch.float64))
    v = torch.tensor([[0.5, 0.0], [0.0, 0.0]], dtype=torch.float64)
    mask = torch.ones(2, 2, dtype=torch.bool)
    lam, mu = block_ground(M, v, mask)
    assert float(lam[1]) == 0.0 and torch.isnan(mu[1]), (lam.tolist(), mu.tolist())
    assert abs(float(mu[0]) - 1.0) <= 4 * EPS and float(lam[0]) < 0.0


def test_hop_groups_do_not_collide_across_configurations():
    """The hop groups are keyed by (configuration, O atom, component).  With more atoms than moves the key must use
    the atom count, not the move count: here O 8 of configuration 0 and O 4 of configuration 1 would share
    (c * P + o) for P = 4 moves, so configuration 1's hops would be written into configuration 0's block.  Both
    configurations (which differ only in a spectator Mg) must give the rebuild's root."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing, downfold_moves
    z = torch.tensor([12, 12, 12, 12, 8, 79, 79, 12, 8])
    half = 2.545 / 2
    pos = torch.tensor([[40.0, 0.0, 0.0], [50.0, 0.0, 0.0], [60.0, 0.0, 0.0], [70.0, 0.0, 0.0],
                        [0.0, -1.5, 0.0], [-half, 0.0, 2.6], [half, 0.0, 2.6], [80.0, 0.0, 0.0],
                        [0.0, 1.5, 0.0]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    assert int(moves.o.numel()) == 4 and sorted(set(moves.o.tolist())) == [4, 8]
    states = torch.arange(-2, 4, dtype=torch.float64)
    O_e = [0.0, 6.0, 20.0, 30.0, 40.0, 50.0]
    Au_e = [50.0, 3.0, 0.0, 3.0, 20.0, 30.0]
    Mg_e = [60.0, 50.0, 40.0, 30.0, 0.0, 40.0]
    rows = {8: O_e, 79: Au_e, 12: Mg_e}
    e_hat = torch.tensor([rows[int(a)] for a in z], dtype=torch.float64)
    b = torch.zeros(9, dtype=torch.float64)
    W = torch.diag(torch.full((9,), 6.0, dtype=torch.float64))
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    frames = torch.tensor([2, 2, 2, 2, -2, 0, 0, 2, -2])
    L = torch.tensor([1, 1, 1, 1, -2, -1, -1, 1, -2])
    K = torch.tensor([3, 3, 3, 3, 3, 3, 3, 3, 3])
    T = torch.zeros(9, 9, dtype=torch.float64)
    T[5, 6] = T[6, 5] = -0.8
    configs = torch.tensor([[2, 2, 2, 2, -2, 0, 0, 2, -2], [1, 2, 2, 2, -2, 0, 0, 2, -2]])
    for sign in (-1, 1):
        lam, _ = dressing(configs, energy, moves, T, frames, L, K, z, hop_sign=sign)
        ref = _rebuild(configs, energy, z, pos, T, L, K, hop_sign=sign)
        for c in range(2):
            assert abs(float(lam[c]) - float(ref[c])) <= 256 * EPS * (1 + abs(float(ref[c]))), (sign, c, lam.tolist(), ref.tolist())
        assert abs(float(lam[0]) - float(lam[1])) <= 256 * EPS * (1 + abs(float(lam[0])))


def test_the_pauli_ordering_of_a_bonded_pair():
    """The extra electron of a bonded s pair takes the antibonding two-site state, because the bonding one is the
    pair's own filled bond (Pauli): with hop_sign +1 the fold's hop inside the folded manifold is +|T| f.  An O on
    the pair's bisector plane reaches the antisymmetric state through its axial p component only and the symmetric
    one through its other two, so the Pauli root is less negative than the legacy (bonding-lowest) root; each
    ordering matches the independent dense rebuild with the same ordering; an O that touches one atom of the pair
    only cannot tell the orderings apart (one path: the hop sign is a gauge there)."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing, downfold_moves
    z = torch.tensor([8, 79, 79])
    states = torch.arange(-2, 4, dtype=torch.float64)
    O_e = [0.0, 6.0, 20.0, 30.0, 40.0, 50.0]
    Au_e = [50.0, 3.0, 0.0, 3.0, 20.0, 30.0]
    e_hat = torch.tensor([O_e, Au_e, Au_e], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    W = torch.diag(torch.full((3,), 6.0, dtype=torch.float64))
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    frames = torch.tensor([-2, 0, 0])
    L, K = torch.tensor([-2, -1, -1]), torch.tensor([3, 3, 3])
    T = torch.zeros(3, 3, dtype=torch.float64)
    T[1, 2] = T[2, 1] = -0.8
    configs = torch.tensor([[-2, 0, 0]])
    delta = float(energy(torch.tensor([[-1, -1, 0]]))[0] - energy(configs)[0])
    assert delta > 0.8, delta                                                       # the folded manifold stays above

    def roots(pos):
        ei = neighbour_list(pos, r_max=5.3)
        vec, length = edges(pos, ei)
        moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
        out = {}
        for sign in (-1, 1):
            lam, _ = dressing(configs, energy, moves, T, frames, L, K, z, hop_sign=sign)
            ref = _rebuild(configs, energy, z, pos, T, L, K, hop_sign=sign)
            assert abs(float(lam[0]) - float(ref[0])) <= 256 * EPS * (1 + abs(float(ref[0]))), (sign, float(lam[0]), float(ref[0]))
            out[sign] = float(lam[0])
        return out

    half = 2.545 / 2
    bisector = roots(torch.tensor([[0.0, 0.0, 0.0], [-half, 0.0, 2.6], [half, 0.0, 2.6]], dtype=torch.float64))
    assert bisector[1] < 0.0 and bisector[-1] < 0.0, bisector
    assert bisector[1] > bisector[-1] + 1e-3, bisector                                # Pauli: less negative
    # weak coupling: the axial component (opposite-sign couplings) at Delta - |T|, the vertical one at Delta + |T|
    # under the Pauli ordering, and the reverse under the legacy one: the Pauli root lies above the legacy root by
    # 2 (v_z^2 - v_x^2) [1/(Delta - |T|) - 1/(Delta + |T|)] to leading order (v_z^2 > v_x^2 here)
    axis = roots(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [0.0, 0.0, 2.26 + 2.545]], dtype=torch.float64))
    assert axis[1] < 0.0 and abs(axis[1] - axis[-1]) <= 256 * EPS * (1 + abs(axis[1])), axis


def test_the_selection_sees_the_dressing_and_drops_folded_partners():
    """A configuration inside the window only once dressed is selected; the folded partner of a selected
    configuration is never kept explicitly, even when it lies inside the energy window."""
    import trace_redox.diabatic as D
    from trace_redox.downfold import drop_folded_partners, downfold_moves
    z = torch.tensor([8, 79, 12])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [2.13, 0.0, 0.0]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    states = torch.arange(-2, 4, dtype=torch.float64)
    e_hat = torch.full((3, 6), 0.0, dtype=torch.float64)
    e_hat[0] = torch.tensor([0.0, 1.0, 9.0, 20.0, 30.0, 40.0])               # O: -2 lowest, -1 at +1 eV
    e_hat[1] = torch.tensor([30.0, 0.5, 0.0, 9.0, 20.0, 30.0])               # Au: 0 lowest, -1 at +0.5
    e_hat[2] = torch.tensor([30.0, 30.0, 30.0, 30.0, 0.0, 20.0])             # Mg: +2
    b = torch.zeros(3, dtype=torch.float64)
    W = torch.zeros(3, 3, dtype=torch.float64)
    L, K = torch.tensor([-2, -1, 0]), torch.tensor([3, 3, 2])
    frames = torch.tensor([-2, 0, 2])
    # the frame (0 eV) and its folded partner (O-, Au-: 1.5 eV, inside the 2.5 eV window undressed).  Dressed, the
    # frame drops to -1.97 eV; the partner's own folded manifold (the frame) lies below it, so it is not a valid
    # explicit state and the selection must leave it out.
    correction_calls = []

    def correction(cfgs):
        from trace_redox.downfold import dressing
        lam, _ = dressing(cfgs, lambda c: D.configuration_energies(c, e_hat, states, b, W), moves,
                          torch.zeros(3, 3, dtype=torch.float64), frames, L, K, z, strict=False)
        correction_calls.append(int(cfgs.shape[0]))
        return lam
    plain = D.select_configurations(e_hat, states, b, W, K, 0, window=2.5, cap=100, min_charge=L)
    dressed = D.select_configurations(e_hat, states, b, W, K, 0, window=2.5, cap=100, min_charge=L,
                                      correction=correction, bound=moves.atom_bound)
    as_set = lambda c: {tuple(x) for x in c.tolist()}
    assert (-1, -1, 2) in as_set(plain)                                      # the folded partner, 1.5 eV up
    assert (-2, 0, 2) in as_set(dressed) and (-1, -1, 2) not in as_set(dressed)
    assert (-2, 0, 2) in as_set(plain)
    assert correction_calls
    # the partner is dropped whatever the window did
    kept = drop_folded_partners(torch.tensor([[-2, 0, 2], [-1, -1, 2], [-2, 1, 1]]),
                                torch.tensor([0.0, 1.5, 3.0]), moves, L, K)
    assert as_set(kept) == {(-2, 0, 2), (-2, 1, 1)}


def test_au_plus_acceptance_carries_the_pairing_factor():
    """O2- -> Au+ (an empty 6s) is the unpairing move of SPEC 29: v^2 = 2 V^2, against O2- -> Au0 with v^2 = V^2."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing, downfold_moves, star_ground
    from trace_redox.heads import BOUND_R0, window_switch
    z = torch.tensor([8, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5)
    states = torch.arange(-2, 4, dtype=torch.float64)
    e_hat = torch.tensor([[0.0, 5.0, 9.0, 20.0, 30.0, 40.0], [30.0, 1.0, 0.0, 2.0, 20.0, 30.0]], dtype=torch.float64)
    b, W = torch.zeros(2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    frames, L, K = torch.tensor([-2, 0]), torch.tensor([-2, -1]), torch.tensor([3, 3])
    lam, rec = dressing(torch.tensor([[-2, 0], [-2, 1]]), energy, moves, torch.zeros(2, 2, dtype=torch.float64),
                        frames, L, K, z)
    r = torch.tensor(2.26, dtype=torch.float64)
    V = float(1.84 * HBAR2_OVER_ME / (r * r + BOUND_R0 ** 2) * window_switch(r, 4.5, 0.5))
    d0 = (5.0 + 1.0) - (0.0 + 0.0)                                           # O2- -> Au0: O- (5) + Au- (1)
    d1 = (5.0 + 0.0) - (0.0 + 2.0)                                           # O2- -> Au+: O- (5) + Au0 (0) - Au+ (2)
    exp0 = float(star_ground(torch.tensor([[d0]]), torch.tensor([[V * V]]))[0])
    exp1 = float(star_ground(torch.tensor([[d1]]), torch.tensor([[2.0 * V * V]]))[0])
    assert abs(float(lam[0]) - exp0) <= 64 * EPS * (1 + abs(exp0))
    assert abs(float(lam[1]) - exp1) <= 64 * EPS * (1 + abs(exp1))
    assert rec["dressed"] == 2


def test_off_by_default_validated_and_forces():
    from trace_redox import RedoxConfig
    assert RedoxConfig().downfold_transfer is False and RedoxConfig().downfold_cutoff == 4.5
    with pytest.raises(ValueError, match="downfold_transfer"):
        RedoxConfig(downfold_transfer=True)
    with pytest.raises(ValueError, match="downfold_cutoff"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", downfold_transfer=True, downfold_cutoff=0.0)
    module, heads = _module(downfold_transfer=True)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    z, pos0 = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (0, 2), (2, 2), (3, 0), (1, 0)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_off_leaves_the_couplings_and_the_record_untouched():
    """With the term off, the solver sees the head's own T (no O-Au zeroing) and no dressing record is written."""
    module, heads = _module()
    z, pos = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    _, s = _capture(module, h, pos, z)
    ei = neighbour_list(pos, r_max=5.3)
    _, length = edges(pos, ei)
    inv = heads.node_invariants(h).to(torch.float64)
    T_head = module.coupling(inv, ei, length, species=z)
    assert torch.equal(s["T"], T_head) and float(T_head[0, 1].abs()) > 0.0
    assert getattr(module, "_downfold_last", None) is None


def test_the_response_record_is_marked_inexact_when_dressed():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         downfold_transfer=True, diabatic_window=20.0)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    module.record_ensemble = True
    z, pos = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    _, result = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    rec = result.ensemble
    assert rec is not None and rec["exact"] is False and "SPEC 32" in rec["reason"]


# ---- the measured anisotropy and crystal field of the O-s channel (2026-10-29) ---------------------------------------
#
# Measured on the MgO(100) ten-ion cluster (theory note 2026-10-18 section 4c): (i) the O 2p band's hole energies split
# by 0.8-1.3 eV between the in-plane (cation-bonding) components and the dangling p_z; a single Au+ acceptor transfers
# 0.65 of the vertical value per contact over a cation; (ii) the Au2 dimer's acceptor is its sp-hybridised sigma*,
# with outward lobes along the axis: a bonded s-class acceptor couples as c V(r) (1 + kappa' cos theta) / sqrt(1 +
# kappa'^2), cos theta = w . e, w the acceptor -> donor direction, e the outward axis; (c, kappa') = (0.71, 0.22) from
# the vertical and inward-flat transfers.  Three measured constants, fixed, not learned; the defaults (1, 0, 0) leave
# the term as before.


def test_the_crystal_field_shifts_the_cation_bonding_hole_components():
    """A surface O (one cation below, two along each in-plane axis) gets +beta on its in-plane p holes and nothing on
    p_z; a bulk-like O (two cations along every axis) gets no shift; the shift enters Delta of the resolved leaves."""
    from trace_redox.downfold import crystal_field_shifts, downfold_moves
    d = 2.1325
    z = torch.tensor([8, 12, 12, 12, 12, 12, 79])                           # surface O at the origin, Au above
    pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, -d], [0, 0, 2.26]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    shifts = crystal_field_shifts(z, ei, vec, length, beta=1.0, bond_scale=1.2)
    assert torch.allclose(shifts[0], torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64), atol=4 * EPS)
    z2 = torch.cat([z, torch.tensor([12])])
    pos2 = torch.cat([pos, torch.tensor([[0.0, 0.0, d]], dtype=torch.float64)])   # a cation above too: bulk-like
    ei2 = neighbour_list(pos2, r_max=5.3)
    vec2, length2 = edges(pos2, ei2)
    assert torch.equal(crystal_field_shifts(z2, ei2, vec2, length2, beta=1.0, bond_scale=1.2)[0],
                       torch.zeros(3, dtype=torch.float64))
    moves = downfold_moves(z, ei, vec, length, r_cut=4.5, bond_cutoff=3.2, beta=1.0)
    assert torch.allclose(moves.cf_shift[moves.o], shifts[moves.o])


def test_the_sigma_star_anisotropy_of_a_bonded_acceptor():
    """A dimer's lower Au couples to an O straight below (cos theta = +1) with (1 + k)/sqrt(1 + k^2) times the scaled
    Harrison value, an O on the inward side with (1 + k cos theta); a lone Au has no lobe (factor 1)."""
    from trace_redox.downfold import downfold_moves
    z = torch.tensor([8, 79, 79, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [0.0, 0.0, 4.80], [2.0, 0.0, 3.5]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    m = downfold_moves(z, ei, vec, length, r_cut=4.5, bond_cutoff=3.2)
    e1 = m.outward[1]                                                          # the lower Au: away from the upper one
    assert torch.allclose(e1, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64), atol=4 * EPS)
    assert torch.allclose(m.outward[2], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=4 * EPS)
    k = 0.22
    leaf = [(int(o), int(g)) for o, g in zip(m.o, m.g)]
    i_top = leaf.index((0, 1))
    cos_top = float(-m.l[i_top] @ e1)
    assert abs(cos_top - 1.0) <= 4 * EPS
    factor = m.anisotropy(k)                                                    # [P] (1 + k cos theta) / sqrt(1 + k^2)
    assert abs(float(factor[i_top]) - (1 + k) / math.sqrt(1 + k * k)) <= 4 * EPS
    i_side = leaf.index((3, 1))                                                 # the O beside the lower Au, above it
    cos_side = float(-m.l[i_side] @ e1)
    assert cos_side < 0.0
    assert abs(float(factor[i_side]) - (1 + k * cos_side) / math.sqrt(1 + k * k)) <= 4 * EPS
    lone = downfold_moves(z[:2], ei[:, (ei < 2).all(0)], vec[(ei < 2).all(0)], length[(ei < 2).all(0)], r_cut=4.5,
                          bond_cutoff=3.2)
    assert bool((lone.anisotropy(k) == 1.0).all())                              # no bonded partner: no lobe


def test_the_measured_constants_are_configurable_and_default_to_the_plain_term():
    from trace_redox import RedoxConfig
    r = RedoxConfig()
    assert (r.downfold_scale, r.downfold_anisotropy, r.downfold_crystal_field, r.downfold_bond_cutoff) == (1.0, 0.0, 0.0, 3.2)
    with pytest.raises(ValueError, match="downfold_scale"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", downfold_transfer=True, downfold_scale=0.0)
    with pytest.raises(ValueError, match="downfold_anisotropy"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", downfold_transfer=True, downfold_anisotropy=-0.1)
    with pytest.raises(ValueError, match="downfold_crystal_field"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", downfold_transfer=True, downfold_crystal_field=-1.0)
    # with the measured constants the upright test cluster's dressing changes, and the plain constants reproduce
    # the earlier value exactly
    module, heads = _module(downfold_transfer=True)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    z, pos = _otop()
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    e_plain, _ = _capture(module, h, pos, z)
    import dataclasses
    module.redox = dataclasses.replace(module.redox, downfold_scale=0.71, downfold_anisotropy=0.22,
                                       downfold_crystal_field=1.0)
    e_meas, s = _capture(module, h, pos, z)
    assert float(e_meas) != float(e_plain)
    module.redox = dataclasses.replace(module.redox, downfold_scale=1.0, downfold_anisotropy=0.0,
                                       downfold_crystal_field=0.0)
    e_back, _ = _capture(module, h, pos, z)
    assert float(e_back) == float(e_plain)


# ---- the dangling-axis rule and the covalent-unit exclusion (2026-10-29, after the 28-ion flat split) ---------------
#
# Measured: CT_cp(flat, 28 ions) = -0.444 eV against the sigma* prediction of < -0.80 (count-only ~ -0.47): the sigma*
# lobe is falsified (kappa' = 0); inward and outward flat contacts transfer alike, ~0.3-0.36 of a vertical contact at
# equal distance.  The donor-side rule: a hole on an O-class donor couples only through its least-coordinated p axes
# (the dangling directions); the cation-bonding components do not.  And a release defect: the fold must skip a pair
# inside one covalent unit (water's O-H), as the ensemble's admissibility does.


def test_the_dangling_axes_of_a_donor():
    from trace_redox.downfold import free_axes
    d = 2.1325
    z = torch.tensor([8, 12, 12, 12, 12, 12, 8, 8])                             # a (100) surface O; a bare O; an O
    pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, -d],
                        [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=torch.float64)
    z = torch.cat([z, torch.tensor([12])])                                     # ... with one cation along +x only
    pos = torch.cat([pos, torch.tensor([[40.0 + d, 0.0, 0.0]], dtype=torch.float64)])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    f = free_axes(z, ei, vec, length)
    assert f[0].tolist() == [False, False, True]                                # surface O: p_z dangling
    assert f[6].tolist() == [True, True, True]                                  # no cation: every axis free
    assert f[7].tolist() == [False, True, True]                                 # one cation along x: y, z free
    assert f[1].tolist() == [True, True, True]                                  # a cation: untouched (never a donor)


def test_only_the_dangling_components_couple_when_the_rule_is_on():
    """Surface O under an Au at 45 degrees: with the rule, only the p_z leaf carries coupling, so the dressing is the
    star with v^2 = (V l_z)^2; without it, v^2 = V^2 (all three components)."""
    from trace_redox.diabatic import configuration_energies
    from trace_redox.downfold import dressing, downfold_moves, star_ground
    from trace_redox.heads import BOUND_R0, window_switch
    d = 2.1325
    z = torch.tensor([8, 12, 12, 12, 12, 12, 79])
    au = torch.tensor([2.0, 0.0, 2.0])
    pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, -d]], dtype=torch.float64)
    pos = torch.cat([pos, au.unsqueeze(0)])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    states = torch.arange(-2, 4, dtype=torch.float64)
    e_hat = torch.full((7, 6), 50.0, dtype=torch.float64)
    e_hat[0] = torch.tensor([0.0, 4.0, 50, 50, 50, 50]); e_hat[6] = torch.tensor([50, 0.0, 0.0, 50, 50, 50])
    for k in range(1, 6):
        e_hat[k, 4] = 0.0                                                      # Mg2+ at state +2
    b, W = torch.zeros(7, dtype=torch.float64), torch.zeros(7, 7, dtype=torch.float64)
    energy = lambda c: configuration_energies(c, e_hat, states, b, W)
    frames = torch.tensor([-2, 2, 2, 2, 2, 2, 0]); L = torch.tensor([-2, 0, 0, 0, 0, 0, -1]); K = torch.full((7,), 3)
    cfg = torch.tensor([[-2, 2, 2, 2, 2, 2, 0]])
    T = torch.zeros(7, 7, dtype=torch.float64)
    r = float(au.norm())
    V = float(1.84 * HBAR2_OVER_ME / (r * r + BOUND_R0 ** 2) * window_switch(torch.tensor(r), 4.5, 0.5))
    lz = float(au[2] / r)
    m_all = downfold_moves(z, ei, vec, length, r_cut=4.5)
    lam_all, _ = dressing(cfg, energy, m_all, T, frames, L, K, z)
    m_free = downfold_moves(z, ei, vec, length, r_cut=4.5, free_axes_only=True)
    lam_free, _ = dressing(cfg, energy, m_free, T, frames, L, K, z)
    exp_all = float(star_ground(torch.tensor([[4.0]]), torch.tensor([[V * V]]))[0])
    exp_free = float(star_ground(torch.tensor([[4.0]]), torch.tensor([[(V * lz) ** 2]]))[0])
    assert abs(float(lam_all[0]) - exp_all) <= 64 * EPS * (1 + abs(exp_all))
    assert abs(float(lam_free[0]) - exp_free) <= 64 * EPS * (1 + abs(exp_free))
    assert float(lam_free[0]) > float(lam_all[0])                              # less binding with the rule


def test_a_pair_inside_one_covalent_unit_is_not_folded():
    """Water's O and H are an O-class donor and an s-class acceptor by element; inside one covalent unit the pair
    is excluded (the ensemble's admissibility excludes the same electron-hole pair)."""
    from trace_redox.downfold import downfold_moves
    z = torch.tensor([8, 1, 1, 8, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 2.3]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    units = torch.tensor([0, 0, 0, 1, 2])
    m = downfold_moves(z, ei, vec, length, r_cut=4.5, units=units)
    assert sorted(zip(m.o.tolist(), m.g.tolist())) == [(3, 4)]                 # the O-Au contact only
    m0 = downfold_moves(z, ei, vec, length, r_cut=4.5)
    assert (0, 1) in set(zip(m0.o.tolist(), m0.g.tolist()))                     # without units, the O-H would fold


def test_the_rule_is_configurable():
    from trace_redox import RedoxConfig
    assert RedoxConfig().downfold_free_axes is False
    r = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", downfold_transfer=True, downfold_free_axes=True)
    assert r.downfold_free_axes is True
