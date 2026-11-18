"""The model in three equations (trace_redox SPEC §0): the configuration energy the module puts on H's diagonal is

    E(n) = sum_u eps_u(n_u) + 1/2 q(n)^T J q(n) - 1/2 s(n)^T K(n)^-1 s(n) + sum_{u<v} O_uv(n_u, n_v) + const,

    eps_u(n) = E_Z(n) + g_u Delta_u(n) - 1/2 S0_u n^2 (S0 the isolated Gaussian self energy),
    s(n) = g + A^T q(n) + sum_u Q_u v_u(n_u),

with K(n) the bound stiffness with each state's own dipole blocks (SPEC 28), v the carriers' true-field columns
(SPEC 26), O the overlap tables (SPEC 23 penetration + SPEC 25 Gordon-Kim, an atom's own images one-body).  The
module never forms this closed form: it splits it into the ladder e_hat (with SPEC 26's one-body terms and the
self-image terms), b and W from the exact elimination, the pair slot (SPEC 26 two-body, SPEC 23/25 overlap) and
SPEC 28's Woodbury correction.  This test evaluates the closed form directly (dense K(n)^-1, dense s(n)) and checks
that the module's assembly equals it for every selected configuration, up to the configuration-independent constant.
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

Z6 = torch.tensor([8, 79, 79, 12, 13, 8])
POS = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                    [2.0, 2.1, 0.2]], dtype=torch.float64)


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


STACKS = {
    "ions + Coulomb + induction": {},
    "+ overlap (SPEC 23/25), gate (24)": dict(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                                              shift_gate=True),
    "+ true-field induction (26), state alpha (28)": dict(carrier_penetration=True, carrier_overlap="gordon_kim",
                                                          exclusion=False, shift_gate=True, carrier_induction=True,
                                                          state_polarizability=True,
                                                          state_polarizability_elements=(79, 8)),
}


def _capture(monkeypatch):
    import trace_redox.induction as I
    import trace_redox.module as M
    import trace_redox.penetration as P
    store = {}
    solution, eliminate = M.RedoxElectrostatics._diabatic_solution, M.eliminate_bound_sector
    pairs_f, self_f, cols_f = P.penetration_pairs, P.penetration_self_terms, I.dipole_correction_columns

    def wrapped_solution(self, e_hat, states, b, W, L, K, Q, T=None, **kwargs):
        store.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), pairs=kwargs.get("pairs"),
                     diag=kwargs.get("diag_correction"))
        return solution(self, e_hat, states, b, W, L, K, Q, T=T, **kwargs)

    def wrapped(name, f):
        def g(*args, **kwargs):
            store[name] = f(*args, **kwargs)
            return store[name]
        return g

    monkeypatch.setattr(M.RedoxElectrostatics, "_diabatic_solution", wrapped_solution)
    monkeypatch.setattr(M, "eliminate_bound_sector", wrapped("elim", eliminate))
    monkeypatch.setattr(P, "penetration_pairs", wrapped("O", pairs_f))
    monkeypatch.setattr(P, "penetration_self_terms", wrapped("own", self_f))
    monkeypatch.setattr(I, "dipole_correction_columns", wrapped("Vd", cols_f))
    return store


def _both(module, store, result, z, configs):
    """(E_module, E_closed) for configurations ``configs [M, U]`` (atoms first)."""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import configuration_energies
    states, elim = store["states"], store["elim"]
    n_atoms, nb = int(z.numel()), int(elim.n_bonds)
    first = int(round(float(states[0])))
    with torch.no_grad():
        E_module = configuration_energies(configs, store["e_hat"], states, store["b"], store["W"], pairs=store["pairs"])
        if store["diag"] is not None:
            E_module = E_module + store["diag"](configs)
    from trace_electrostatics.kernels import gram_self_energy
    J, A, K = elim.J, elim.A, elim.operator.matrix()
    g = -K @ elim.y0
    # the ISOLATED Gaussian self energy: under periodic boundaries diag(J) also holds an ion's interaction with its
    # own lattice images, which is lattice energy and stays in the Coulomb term (a reference taking S0 = diag(J)
    # was off by 27.4 eV per Mg+ O- hop in rock-salt MgO)
    sigma = module.free_widths(z) if module.width is not None else elim.assembly.params.sigma
    S0 = gram_self_energy(sigma.to(J.dtype))
    eps = result.state_energies.detach() - 0.5 * S0.unsqueeze(1) * (states * states).unsqueeze(0)
    frames = formal_frame(z)
    rho = module._state_ratio_table(z, states) if module.redox.state_polarizability else None
    S = int(states.numel())
    Q = states - frames.to(torch.float64).unsqueeze(1)                         # [n, S] carrier charges
    E_closed = []
    for c in range(int(configs.shape[0])):
        n = configs[c, :n_atoms]
        q = n.to(torch.float64)
        k = (n - first).tolist()
        total = sum(float(eps[u, k[u]]) for u in range(n_atoms)) + 0.5 * float(q @ J @ q)
        s = g + A.transpose(0, 1) @ q
        if "Vd" in store:
            for u in range(n_atoms):
                if int(n[u]) != int(frames[u]):
                    s[nb:nb + 3 * n_atoms] = s[nb:nb + 3 * n_atoms] + Q[u, k[u]] * store["Vd"][:, u * S + k[u]]
        Kn = K.clone()
        if rho is not None:
            # SPEC 28.6 (2026-11-10): the state scales the atom's OWN stiffness A_mu,j.  The legacy statement ('block')
            # scaled the operator's block K[sl, sl] = A_mu,j + T2_jj, which under periodic boundaries holds the atom's
            # dipole-dipole sum with its own images; the two are one in an open system (T2_jj = 0).
            own = None if module.redox.state_polarizability_stiffness == "block" else elim.assembly.params.A_mu.detach()
            for j in range(n_atoms):
                r = float(rho[j, k[j]])
                if int(n[j]) != int(frames[j]) and r != 1.0:
                    sl = slice(nb + 3 * j, nb + 3 * j + 3)
                    Kn[sl, sl] = K[sl, sl] / r if own is None else K[sl, sl] + own[j].to(K.dtype) * (1.0 / r - 1.0)
            assert float(torch.linalg.eigvalsh(Kn).min()) > 0.0
        total -= 0.5 * float(s @ torch.linalg.solve(Kn, s))
        if "O" in store:
            pi, pj, Xi = store["O"]
            total += float(Xi[torch.arange(int(pi.numel())), (n[pi] - first), (n[pj] - first)].sum())
        if "own" in store:
            total += sum(float(store["own"][u, k[u]]) for u in range(n_atoms))
        E_closed.append(total)
    return E_module, torch.tensor(E_closed, dtype=torch.float64)


def _assert_equal(E_module, E_closed):
    d_module, d_closed = E_module - E_module[0], E_closed - E_closed[0]
    scale = 1.0 + float(d_closed.abs().max())
    assert float((d_module - d_closed).abs().max()) <= 1e-10 * scale, (d_module.tolist(), d_closed.tolist())
    assert float(d_closed.abs().max()) > 1e-3                                  # the configurations differ


@pytest.mark.parametrize("stack", list(STACKS), ids=list(STACKS))
def test_the_module_assembles_the_closed_form_configuration_energy(stack, monkeypatch):
    store = _capture(monkeypatch)
    module, heads = _module(**STACKS[stack])
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(POS)
    vec, length = edges(POS, ei)
    with torch.no_grad():
        _, result = module(h, POS, ei, vec, length, data={"z": Z6, "total_charge": 0.0})
    configs = module._ensemble_last["configs"]
    assert int(configs.shape[0]) >= 3                                           # several configurations exercised
    _assert_equal(*_both(module, store, result, Z6, configs))


@pytest.mark.parametrize("stiffness", ["onsite", "block"])
def test_the_closed_form_holds_in_a_small_periodic_cell(stiffness, monkeypatch):
    """Rock-salt MgO, 8 atoms, a = 4.212 A below the overlap cutoff: the Ewald J and A, and every atom's overlap
    with its own images (one-body).  Constructed configurations (frame, Mg+ O- pairs), since random heads may select
    only the frame.

    2026-11-10 (X1, SPEC 28.6): this is the one test whose closed form held the legacy statement of SPEC 28 in a
    periodic cell (K(n)'s block = the operator's block / rho, own-image sum included).  It is kept verbatim under
    ``state_polarizability_stiffness='block'`` and the default 'onsite' asserts the new statement (the atom's own
    A_mu,j / rho, the own-image sum unscaled); with the old closed form the default missed by 1.9e-4 eV on the first
    Mg+ O- pair, which is the size of the statement's change here, not round-off."""
    from redox_helpers import lattice_graph, periodic_edges
    from test_redox_penetration import _library, _rocksalt
    _library()
    store = _capture(monkeypatch)
    z, pos, cell = _rocksalt()
    pos = pos + torch.tensor([[0.05, -0.03, 0.02]] + [[0.0, 0.0, 0.0]] * 7, dtype=torch.float64)
    module, heads = _module(**STACKS["+ true-field induction (26), state alpha (28)"],
                            state_polarizability_stiffness=stiffness)
    pbc = (True, True, True)
    ei, es = lattice_graph(pos, cell, pbc, r_max=4.0, shells=1)
    vec, length = periodic_edges(pos, cell, ei, es)
    h = torch.randn(8, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.no_grad():
        _, result = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es,
                           data={"z": z, "total_charge": 0.0})
    assert "own" in store and float(store["own"].abs().max()) > 1e-6         # self-image terms are present
    from trace_redox.constants import formal_frame
    U = int(store["e_hat"].shape[0])
    frame = torch.zeros(U, dtype=torch.long)
    frame[:8] = formal_frame(z)
    mg = [i for i in range(8) if int(z[i]) == 12]
    o = [i for i in range(8) if int(z[i]) == 8]
    configs = [frame.clone()]
    for a, b in ((mg[0], o[0]), (mg[1], o[2]), (mg[2], o[3])):
        c = frame.clone()
        c[a] -= 1                                                              # Mg+ (one electron from O to Mg)
        c[b] += 1                                                              # O-
        configs.append(c)
    _assert_equal(*_both(module, store, result, z, torch.stack(configs)))


def test_a_configuration_switched_off_after_construction_is_off():
    """Counterfactuals replace ``module.redox`` on a trained module.  Every term must then follow the configuration,
    not whether its submodule was built: SPEC 18's exclusion read ``self.exclusion is not None`` and stayed on when
    the configuration said ``exclusion=False`` (measured 2026-10-19 in frozen-weight probes: Pauli counted twice,
    fitted exclusion plus Gordon-Kim).  A module built with exclusion and switched off equals one built without it
    (same weights), and the switched-on one differs."""
    import dataclasses
    built_on, heads = _module(exclusion=True)
    built_off, _ = _module(exclusion=False)
    built_off.load_state_dict({k: v for k, v in built_on.state_dict().items() if not k.startswith("exclusion.")})
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(POS)
    vec, length = edges(POS, ei)

    def energy(module):
        with torch.no_grad():
            return float(module(h, POS, ei, vec, length, data={"z": Z6, "total_charge": 0.0})[0])

    e_on, e_off = energy(built_on), energy(built_off)
    assert abs(e_on - e_off) > 1e-4                                          # the exclusion term is not zero here
    built_on.redox = dataclasses.replace(built_on.redox, exclusion=False)
    assert abs(energy(built_on) - e_off) <= 1e-12 * (1.0 + abs(e_off))
