"""Localising a non-reproducible NaN gradient (2026-10-18).

Five AuMgO trainings stopped on a sub-step whose loss was finite and whose gradients were NaN; the exact replay was
finite.  The saved parameter gradients put the entry on the free-charge width path (sigma_f and the plane-wave Gram
blocks of trace_redox.bound), not on an operation.  With ``TRACE_NANPROBE=1`` (set by run_train_nanguard.py), every
watched intermediate records, in backward order, that its gradient arrived and whether it was finite; the faulty
operation lies between the last finite and the first non-finite watched tensor of the failing pass (``bracket``).
Off otherwise: ``watch`` returns its tensor untouched.
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

ENABLED = os.environ.get("TRACE_NANPROBE") == "1"
_EVENTS: list = []


def watch(name: str, tensor: Tensor) -> Tensor:
    """Record ``(name, finite, numel)`` whenever a gradient reaches ``tensor``."""
    if ENABLED and isinstance(tensor, Tensor) and tensor.requires_grad:
        def hook(grad, _name=name):
            _EVENTS.append((_name, bool(torch.isfinite(grad).all()), int(grad.numel())))
            return None
        tensor.register_hook(hook)
    return tensor


def drain() -> list:
    """The events since the last drain, in the order the gradients arrived; clears them."""
    out = list(_EVENTS)
    _EVENTS.clear()
    return out


def bracket(events: list):
    """``(last finite, first non-finite)`` watched tensors before the first non-finite one, or None."""
    last = None
    for name, finite, _ in events:
        if not finite:
            return last, name
        last = name
    return None


_ORIGINALS: dict = {}


def instrument_lattice() -> None:
    """Watch the inputs and outputs of trace_electrostatics.lattice's coulomb_kernel, real_space_damping and _combined
    (the 2026-10-18 bracket: a NaN made inside free_charge_lattice_blocks' sigma backward from finite gradients).
    Wraps the module attributes from outside (trace_electrostatics' source is unchanged); values are untouched."""
    import trace_electrostatics.lattice as L
    if not ENABLED or _ORIGINALS:
        return
    _ORIGINALS.update(coulomb_kernel=L.coulomb_kernel, real_space_damping=L.real_space_damping, _combined=L._combined)
    kernel, damping, combined = L.coulomb_kernel, L.real_space_damping, L._combined

    def coulomb_kernel(r, sigma_ij):
        watch("lattice.coulomb_kernel.width", sigma_ij)
        return watch("lattice.coulomb_kernel.out", kernel(r, sigma_ij))

    def real_space_damping(r, width, alpha, *args, **kwargs):
        watch("lattice.damping.width", width)
        out = damping(r, width, alpha, *args, **kwargs)
        return tuple(watch("lattice.damping.out", t) for t in out)

    def _combined(row, column):
        return watch("lattice.combined.out", combined(row, column))

    L.coulomb_kernel, L.real_space_damping, L._combined = coulomb_kernel, real_space_damping, _combined


def uninstrument_lattice() -> None:
    import trace_electrostatics.lattice as L
    for name, fn in _ORIGINALS.items():
        setattr(L, name, fn)
    _ORIGINALS.clear()


if ENABLED:
    instrument_lattice()
