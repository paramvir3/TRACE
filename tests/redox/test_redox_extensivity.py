"""Extensivity of TRACE-Redox (the extensivity line, branch extensivity-2026-11-10; benchmarks_4g README R12, X1).

X1 (written red before the code, 2026-11-10): SPEC 28 on the atom's own dipole stiffness.  SPEC 28 changes atom j's
stiffness by Delta_j = S_j (1 / rho_j - 1).  As first written S_j was the operator's dipole block P_jj; in a periodic
cell P_jj = A_mu,j + T2_jj, T2_jj the dipole-dipole lattice sum of the atom with its OWN periodic images, which a charge
state does not change and which depends on the cell.  Measured on Au2/MgO(001): the in-phase transfer's SPEC 28 term at
2x1 missed twice the cell's by +76.6, +19.0, -83.9 and -84.4 meV with S_j = P_jj and by <= 1e-6 meV with S_j = A_mu,j.
"""
import dataclasses
import warnings

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import cluster, edges, lattice_graph, make_heads, periodic_edges

from trace_electrostatics.config import ElectrostaticConfig
from trace_redox.bound import assemble_neutral, eliminate_bound_sector

EPS = torch.finfo(torch.float64).eps
ACCURACY = 1e-12                                    # the Ewald sums' requested accuracy: the two cells' sums differ by it
CONFIG = ElectrostaticConfig(ewald_accuracy=ACCURACY)
BULK = torch.tensor([[7.6, 0.0, 0.0], [0.3, 7.2, 0.0], [0.0, 0.4, 7.9]])
SLAB = torch.tensor([[7.4, 0.3, 0.0], [0.2, 7.1, 0.0], [0.0, 0.0, 22.0]])
STATES = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
FRAMES = torch.zeros(6, dtype=torch.long)
# neutral configurations of the cell's six atoms, and each atom's ratio alpha(state) / alpha(frame)
CONFIGS = torch.tensor([[1, -1, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0], [1, -1, -1, 1, 0, 0], [-1, 0, 0, 0, 1, 0]])
RATIO = {(0, 1): 0.37, (0, -1): 2.1, (1, -1): 1.9, (2, -1): 0.44, (3, 1): 0.6, (4, 1): 0.5}
TABLE = torch.tensor([[RATIO.get((j, int(s)), 1.0) for s in STATES.tolist()] for j in range(6)])


def _periodic(pbc, cell, reps=(1, 1, 1), seed=4, n=6):
    """The cell's atoms replicated ``reps`` times (copy-major order), every copy with the cell's own features."""
    heads = make_heads()
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g) * torch.tensor([6.0, 6.0, 4.0])
    h = torch.randn(n, heads.irreps_in.dim, generator=g)
    shifts = [i * cell[0] + j * cell[1] + k * cell[2]
              for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])]
    pos_r = torch.cat([pos + s for s in shifts])
    h_r = h.repeat(len(shifts), 1)
    cell_r = cell * torch.tensor(reps, dtype=cell.dtype).unsqueeze(1)
    ei, es = lattice_graph(pos_r, cell_r, pbc)
    vec, length = periodic_edges(pos_r, cell_r, ei, es)
    a = assemble_neutral(h_r, pos_r, ei, vec, length, heads, CONFIG, pbc=pbc, cell=cell_r, edge_shift=es)
    return a, eliminate_bound_sector(a, a.params.sigma), len(shifts)


def _condition(elim):
    ev = torch.linalg.eigvalsh(elim.operator.matrix().detach())
    return float(ev[-1] / ev[0])


def _in_phase(copies):
    return CONFIGS.repeat(1, copies), FRAMES.repeat(copies), TABLE.repeat(copies, 1)


CELLS = [((True, True, True), BULK, (2, 1, 1)), ((True, True, False), SLAB, (2, 1, 1)),
         ((True, True, False), SLAB, (2, 2, 1))]
IDS = ["bulk 2x1x1", "slab 2x1", "slab 2x2"]


@pytest.mark.parametrize("pbc,cell,reps", CELLS, ids=IDS)
def test_the_dipole_block_of_a_periodic_cell_holds_the_atoms_own_image_sum(pbc, cell, reps):
    """P_jj - A_mu,j is the lattice sum of atom j's dipole with its own images: nonzero, and not the same in the
    replicated cell (there some of the cell's images are other atoms).  A_mu,j is the same in both."""
    a1, e1, _ = _periodic(pbc, cell)
    a2, e2, copies = _periodic(pbc, cell, reps)
    for j in range(6):
        own1 = e1.operator.P[3 * j:3 * j + 3, 3 * j:3 * j + 3] - a1.params.A_mu[j]
        own2 = e2.operator.P[3 * j:3 * j + 3, 3 * j:3 * j + 3] - a2.params.A_mu[j]
        sym1 = 0.5 * (a1.t2_pair[j, j] + a1.t2_pair[j, j].transpose(0, 1))
        scale = float(a1.params.A_mu[j].abs().max())
        assert float((own1 - sym1).abs().max()) <= 8 * EPS * scale
        assert float((a2.params.A_mu[j] - a1.params.A_mu[j]).abs().max()) <= 64 * EPS * scale
        assert float(own1.abs().max()) > 1e6 * ACCURACY * scale                    # the own-image sum is there
        assert float((own2 - own1).abs().max()) > 1e6 * ACCURACY * scale           # and it depends on the cell


@pytest.mark.parametrize("pbc,cell,reps", CELLS, ids=IDS)
def test_the_in_phase_state_polarizability_term_is_extensive(pbc, cell, reps):
    """Every copy of the cell in the same configuration is the cell's own periodic charge state, so its SPEC 28 term is
    (number of copies) x the cell's.  True with Delta_j = A_mu,j (1 / rho_j - 1); not with the legacy block P_jj."""
    from trace_redox.polarizability import state_polarizability_corrections
    a1, e1, _ = _periodic(pbc, cell)
    a2, e2, copies = _periodic(pbc, cell, reps)
    kappa = max(_condition(e1), _condition(e2))
    # the frame level must be extensive first (otherwise the comparison below says nothing about SPEC 28)
    assert abs(float(e2.e_b0) - copies * float(e1.e_b0)) <= 64 * kappa * (ACCURACY + EPS) * (1.0 + abs(float(e2.e_b0)))
    cfg2, frames2, table2 = _in_phase(copies)
    one = state_polarizability_corrections(CONFIGS, e1, FRAMES, TABLE, STATES, stiffness=a1.params.A_mu)
    many = state_polarizability_corrections(cfg2, e2, frames2, table2, STATES, stiffness=a2.params.A_mu)
    assert float(one.abs().min()) > 1e-6                                           # the term is there (eV)
    tol = 64 * kappa * (ACCURACY + EPS) * (1.0 + float(many.abs().max()))
    assert float((many - copies * one).abs().max()) <= tol, (many - copies * one).tolist()
    one_b = state_polarizability_corrections(CONFIGS, e1, FRAMES, TABLE, STATES)
    many_b = state_polarizability_corrections(cfg2, e2, frames2, table2, STATES)
    assert float((many_b - copies * one_b).abs().min()) > 1e3 * tol, (many_b - copies * one_b).tolist()


@pytest.mark.parametrize("pbc,cell", [((True, True, True), BULK), ((True, True, False), SLAB)], ids=["bulk", "slab"])
def test_the_onsite_correction_is_the_exact_energy_of_the_state_s_own_stiffness(pbc, cell):
    """Brute force in a periodic cell: K(n) is K with A_mu,j -> A_mu,j / rho_j on the changed atoms and every
    interaction block (the atoms' own image sums included) untouched; dE = -1/2 s^T K(n)^-1 s + 1/2 s^T K^-1 s.  The
    response function's pieces obey the same K(n)."""
    from trace_redox.polarizability import (state_polarizability_corrections, state_polarizability_response,
                                            stiffness_defects)
    a, elim, _ = _periodic(pbc, cell)
    K, A, y0, nb = elim.operator.matrix(), elim.A, elim.y0, elim.n_bonds
    A_mu = a.params.A_mu
    dE = state_polarizability_corrections(CONFIGS, elim, FRAMES, TABLE, STATES, stiffness=A_mu)
    legacy = state_polarizability_corrections(CONFIGS, elim, FRAMES, TABLE, STATES)
    D = torch.randn(int(y0.shape[0]), 3, generator=torch.Generator().manual_seed(9))
    dM, dalpha = state_polarizability_response(CONFIGS, elim, FRAMES, TABLE, STATES, D, stiffness=A_mu)
    assert not bool(stiffness_defects(CONFIGS, elim, FRAMES, TABLE, STATES, stiffness=A_mu).any())
    g0 = -K @ y0
    kappa = _condition(elim)
    for c in range(int(CONFIGS.shape[0])):
        s = g0 + A.transpose(0, 1) @ CONFIGS[c].to(torch.float64)
        Kn = K.clone()
        for j in range(6):
            rho = float(TABLE[j, int(CONFIGS[c, j]) - int(STATES[0])])
            if int(CONFIGS[c, j]) != int(FRAMES[j]) and rho != 1.0:
                sl = slice(nb + 3 * j, nb + 3 * j + 3)
                Kn[sl, sl] = K[sl, sl] + A_mu[j] * (1.0 / rho - 1.0)
        assert float(torch.linalg.eigvalsh(Kn).min()) > 0.0
        exact = -0.5 * s @ torch.linalg.solve(Kn, s) + 0.5 * s @ torch.linalg.solve(K, s)
        tol = 64 * kappa * EPS * (1.0 + abs(float(exact)))
        assert abs(float(dE[c]) - float(exact)) <= tol, (c, float(dE[c]), float(exact))
        assert abs(float(legacy[c]) - float(exact)) > 1e3 * tol                    # the legacy block is another K(n)
        y_ref, y_n = -torch.linalg.solve(K, s), -torch.linalg.solve(Kn, s)
        want_m = D.transpose(0, 1) @ (y_n - y_ref)
        want_a = D.transpose(0, 1) @ (torch.linalg.solve(Kn, D) - torch.linalg.solve(K, D))
        assert float((dM[c] - want_m).abs().max()) <= 64 * kappa * EPS * (1.0 + float(want_m.abs().max()))
        assert float((dalpha[c] - want_a).abs().max()) <= 64 * kappa * EPS * (1.0 + float(want_a.abs().max()))


def test_in_an_open_system_the_two_forms_are_one():
    """No images: P_jj = A_mu,j, so the on-site form changes no open-boundary result (molecules, clusters, the three
    non-periodic 4G benchmarks)."""
    from trace_redox.polarizability import state_polarizability_corrections
    heads = make_heads()
    pos, h, ei = cluster(6, heads.irreps_in.dim, seed=3)
    vec, length = edges(pos, ei)
    a = assemble_neutral(h, pos, ei, vec, length, heads, CONFIG)
    elim = eliminate_bound_sector(a, a.params.sigma)
    for j in range(6):
        block = elim.operator.P[3 * j:3 * j + 3, 3 * j:3 * j + 3]
        assert float((block - a.params.A_mu[j]).abs().max()) <= 8 * EPS * float(a.params.A_mu[j].abs().max())
    on = state_polarizability_corrections(CONFIGS, elim, FRAMES, TABLE, STATES, stiffness=a.params.A_mu)
    block = state_polarizability_corrections(CONFIGS, elim, FRAMES, TABLE, STATES)
    assert float(on.abs().min()) > 1e-6
    assert float((on - block).abs().max()) <= 64 * _condition(elim) * EPS * (1.0 + float(on.abs().max()))


Z6 = torch.tensor([8, 79, 79, 12, 13, 8])
POS6 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                     [2.0, 2.1, 0.2]], dtype=torch.float64)
CELL6 = torch.tensor([[7.4, 0.0, 0.0], [0.0, 7.4, 0.0], [0.0, 0.0, 22.0]])


def _module(**redox):
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         state_polarizability=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _slab_energy(module, heads):
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    pbc = (True, True, False)
    ei, es = lattice_graph(POS6, CELL6, pbc)
    vec, length = periodic_edges(POS6, CELL6, ei, es)
    return module(h, POS6, ei, vec, length, pbc=pbc, cell=CELL6, edge_shift=es, data={"z": Z6, "total_charge": 0.0})[0]


def test_the_module_uses_the_onsite_stiffness_unless_the_legacy_form_is_asked(monkeypatch):
    """RedoxConfig.state_polarizability_stiffness: 'onsite' (default) hands SPEC 28 the assembly's A_mu, 'block' (legacy)
    hands it nothing, so the operator's block is used as before; the two energies differ in a periodic cell."""
    import trace_redox.polarizability as P
    from trace_redox import RedoxConfig
    assert RedoxConfig().state_polarizability_stiffness == "onsite"
    with pytest.raises(ValueError, match="state_polarizability_stiffness"):
        RedoxConfig(state_polarizability_stiffness="P_jj")
    seen = {}
    original = P.state_polarizability_corrections

    def spy(configs, elim, *args, **kwargs):
        seen["stiffness"], seen["A_mu"] = kwargs.get("stiffness"), elim.assembly.params.A_mu
        return original(configs, elim, *args, **kwargs)

    monkeypatch.setattr(P, "state_polarizability_corrections", spy)
    energy = {}
    for form in ("onsite", "block"):
        module, heads = _module(state_polarizability_stiffness=form)
        seen.clear()
        energy[form] = float(_slab_energy(module, heads))
        assert "stiffness" in seen                                                # SPEC 28 ran
        if form == "onsite":
            assert seen["stiffness"] is not None and torch.equal(seen["stiffness"], seen["A_mu"])
        else:
            assert seen["stiffness"] is None
    assert abs(energy["onsite"] - energy["block"]) > 1e-9


def test_a_checkpoint_older_than_the_key_keeps_the_form_it_trained():
    """train.py stores every RedoxConfig field, so a checkpoint without ``state_polarizability_stiffness`` was trained
    before the key existed, i.e. with the block form: it reloads with 'block' (and a warning when SPEC 28 is on),
    whatever the default is now.  A checkpoint that has the key keeps its value."""
    from trace_redox import RedoxConfig
    new = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", state_polarizability=True)
    saved = dataclasses.asdict(new)
    assert RedoxConfig.from_checkpoint(saved) == new
    old = {k: v for k, v in saved.items() if k != "state_polarizability_stiffness"}
    with pytest.warns(RuntimeWarning, match="state_polarizability_stiffness"):
        restored = RedoxConfig.from_checkpoint(old)
    assert restored.state_polarizability_stiffness == "block"
    assert dataclasses.replace(restored, state_polarizability_stiffness="onsite") == new
    plain = {k: v for k, v in dataclasses.asdict(RedoxConfig()).items() if k != "state_polarizability_stiffness"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                            # SPEC 28 off: nothing to warn about
        assert RedoxConfig.from_checkpoint(plain).state_polarizability_stiffness == "block"
    with pytest.raises(ValueError, match="does not know"):
        RedoxConfig.from_checkpoint({**saved, "not_a_field": 1})


def test_a_resumed_run_is_told_when_it_changes_the_statement_the_checkpoint_trained():
    """train.py's resume builds the model from the YAML and loads the checkpoint's weights into it.  A YAML written
    before the key existed then trains 'onsite' on weights fitted under 'block': ``statement_changes`` names that
    (field -> (trained, now)); nothing is named when SPEC 28 is off, when the YAML asks for the legacy form, or when
    the checkpoint already has the key with the same value."""
    from trace_redox import RedoxConfig
    from trace_redox.module import statement_changes
    on = dict(fragment_ensemble="diabatic", anchor="formal", state_polarizability=True)
    now = RedoxConfig(**on)
    old = {k: v for k, v in dataclasses.asdict(now).items() if k != "state_polarizability_stiffness"}
    assert statement_changes(old, now) == {"state_polarizability_stiffness": ("block", "onsite")}
    assert statement_changes(old, RedoxConfig(state_polarizability_stiffness="block", **on)) == {}
    assert statement_changes(dataclasses.asdict(now), now) == {}
    off = RedoxConfig()
    assert statement_changes({k: v for k, v in dataclasses.asdict(off).items()
                              if k != "state_polarizability_stiffness"}, off) == {}
    block = dataclasses.asdict(RedoxConfig(state_polarizability_stiffness="block", **on))
    assert statement_changes(block, now) == {"state_polarizability_stiffness": ("block", "onsite")}
