"""The Pauli table choice (``RedoxConfig(pauli_table=...)``, trace_redox SPEC 25; 2026-10-25, step 20).

``'gordon_kim'`` (default, unchanged) keeps K fitted to the Gordon-Kim electron-gas term; ``'first_principles'`` reads
an overlay (trace_redox/data/pauli_fp.npz, tools/electrostatics/build_pauli_fp.py) whose Pauli series R K dS have K
fitted to the frozen-density PBE interaction in the Watson-sphere convention (validated against the lattice-embedded
values for the six AuMgO one-carrier types, theory note 2026-10-18 section 4c).  The overlay replaces only the Pauli
series, must carry the library's pair keys in the same order, and leaves every other table untouched.
"""
import numpy as np
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _overlay(tmp_path, scale_key=None, factor=1.0, keys=None):
    from trace_redox import penetration as P
    data = np.load(P.DATA, allow_pickle=False)
    coeff = np.array(data["pair_pauli_coefficients"], copy=True)
    pk = np.array(data["pair_keys"], copy=True) if keys is None else keys
    if scale_key is not None:
        t = [tuple(int(v) for v in k) for k in data["pair_keys"]].index(scale_key)
        coeff[t] *= factor
    path = tmp_path / "pauli_overlay.npz"
    np.savez(path, pair_keys=pk, pair_pauli_coefficients=coeff)
    return str(path)


def test_the_default_is_gordon_kim_and_unchanged():
    from trace_redox import RedoxConfig
    from trace_redox import penetration as P
    assert RedoxConfig().pauli_table == "gordon_kim"
    assert P.library("gordon_kim") is P.library()


def test_the_choice_is_validated():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="pauli_table"):
        RedoxConfig(pauli_table="lda")
    with pytest.raises(ValueError, match="pauli_table"):
        RedoxConfig(pauli_table="first_principles", carrier_overlap="electrostatic")


def test_the_overlay_replaces_only_the_pauli_series(tmp_path):
    from trace_redox import penetration as P
    key = (8, -2, 12, 1)
    base = P.PenetrationLibrary()
    lib = P.PenetrationLibrary(pauli_path=_overlay(tmp_path, key, 1.5))
    t = lib.index[key]
    assert torch.equal(lib.coefficients, base.coefficients)                   # the electrostatic penetration untouched
    assert torch.allclose(lib.gk_coefficients[t], 1.5 * base.gk_coefficients[t], rtol=0, atol=0)
    others = torch.ones(len(lib.index), dtype=torch.bool)
    others[t] = False
    assert torch.equal(lib.gk_coefficients[others], base.gk_coefficients[others])


def test_an_overlay_with_other_pair_keys_is_refused(tmp_path):
    from trace_redox import penetration as P
    data = np.load(P.DATA, allow_pickle=False)
    keys = np.array(data["pair_keys"], copy=True)[::-1]
    with pytest.raises(ValueError, match="pair keys"):
        P.PenetrationLibrary(pauli_path=_overlay(tmp_path, keys=keys))


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, carrier_penetration=True,
                         carrier_overlap="gordon_kim", **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _mgo(Q_extra):
    a = 4.212
    pos, z = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pos.append([i * a / 2, j * a / 2, k * a / 2])
                z.append(12 if (i + j + k) % 2 == 0 else 8)
    z, pos = torch.tensor(z), torch.tensor(pos, dtype=torch.float64)
    from trace_redox.constants import formal_frame
    return z, pos, float(int(formal_frame(z).sum()) + Q_extra)


def _energy(module, heads, z, pos, Q):
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})[0])


def test_the_module_routes_the_choice(tmp_path, monkeypatch):
    """An identical overlay gives the default energy bit for bit; scaling one type changes it."""
    from trace_redox import penetration as P
    z, pos, Q = _mgo(-1)                                                       # one excess electron: Mg+ states in play
    ref_module, heads = _module()
    e_ref = _energy(ref_module, heads, z, pos, Q)
    monkeypatch.setattr(P, "DATA_PAULI_FP", _overlay(tmp_path))
    P._LIBRARIES.pop("first_principles", None)
    same_module, _ = _module(pauli_table="first_principles")
    same_module.load_state_dict(ref_module.state_dict())
    assert _energy(same_module, heads, z, pos, Q) == e_ref
    sub = tmp_path / "scaled"
    sub.mkdir()
    monkeypatch.setattr(P, "DATA_PAULI_FP", _overlay(sub, (8, -2, 12, 1), 3.0))
    P._LIBRARIES.pop("first_principles", None)
    scaled_module, _ = _module(pauli_table="first_principles")
    scaled_module.load_state_dict(ref_module.state_dict())
    assert _energy(scaled_module, heads, z, pos, Q) != e_ref
    P._LIBRARIES.pop("first_principles", None)
