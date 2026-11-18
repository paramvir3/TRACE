"""The NaN-localising gradient hooks (trace_redox/nanprobe.py; 2026-10-18).

Five trainings stopped on a NaN gradient that an exact replay did not reproduce; the parameter gradients localise
the entry to the free-charge width path (sigma_f -> the plane-wave Gram blocks), not the operation.  The probe
records, in backward order, every watched intermediate's gradient and whether it was finite, so the faulty operation
is bracketed by the last finite and the first non-finite watched tensor of the failing pass.  Off unless enabled.
"""
import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def test_it_brackets_the_faulty_operation(monkeypatch):
    import trace_redox.nanprobe as N
    monkeypatch.setattr(N, "ENABLED", True)
    N.drain()
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    a = N.watch("a", x * 2.0)

    class Faulty(torch.autograd.Function):                     # a backward that returns NaN from finite input
        @staticmethod
        def forward(ctx, t):
            return t * 3.0

        @staticmethod
        def backward(ctx, g):
            return g * float("nan")

    b = N.watch("b", Faulty.apply(a))
    c = N.watch("c", b.sum() * 5.0)
    c.backward()
    events = N.drain()
    assert [e[0] for e in events] == ["c", "b", "a"]
    assert [e[1] for e in events] == [True, True, False]          # finite at c and b, non-finite at a: fault in Faulty
    assert N.bracket(events) == ("b", "a")


def test_it_is_inert_when_disabled(monkeypatch):
    import trace_redox.nanprobe as N
    monkeypatch.setattr(N, "ENABLED", False)
    N.drain()
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    y = N.watch("y", x * 2.0)
    y.sum().backward()
    assert N.drain() == []


def test_the_lattice_instrumentation_is_bitwise_inert_and_reversible():
    """instrument_lattice wraps trace_electrostatics.lattice's coulomb_kernel, real_space_damping and _combined to watch
    their inputs and outputs (the bracket of the 2026-10-18 NaN: inside free_charge_lattice_blocks' sigma backward):
    the same values bitwise, the events recorded, and uninstrument restores the originals."""
    import trace_electrostatics.lattice as L
    import trace_redox.nanprobe as N
    originals = (L.coulomb_kernel, L.real_space_damping, L._combined)
    r = torch.tensor([1.5, 2.5, 4.0], dtype=torch.float64)
    s = torch.tensor([0.7, 0.9, 1.1], dtype=torch.float64, requires_grad=True)
    before = (L.coulomb_kernel(r, s), L.real_space_damping(r, s, 0.3, orders=(1,))[0], L._combined(s, s))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(N, "ENABLED", True)
    try:
        N.instrument_lattice()
        N.drain()
        after = (L.coulomb_kernel(r, s), L.real_space_damping(r, s, 0.3, orders=(1,))[0], L._combined(s, s))
        for x, y in zip(before, after):
            assert torch.equal(x, y)
        (after[0].sum() + after[1].sum() + after[2].sum()).backward()
        names = {e[0] for e in N.drain()}
        assert {"lattice.coulomb_kernel.out", "lattice.coulomb_kernel.width", "lattice.damping.out",
                "lattice.damping.width", "lattice.combined.out"} <= names, names
    finally:
        N.uninstrument_lattice()
        monkey.undo()
    assert (L.coulomb_kernel, L.real_space_damping, L._combined) == originals
