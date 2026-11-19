import dataclasses
import importlib
import sys
import warnings

import numpy as np
import torch


def _numpy_checkpoint_globals():
    """Return the NumPy scalar types used by Transformers-ACE checkpoints."""
    try:
        core = importlib.import_module("numpy._core")
        multiarray = importlib.import_module("numpy._core.multiarray")
        numeric = importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core")
        multiarray = importlib.import_module("numpy.core.multiarray")
        numeric = importlib.import_module("numpy.core.numeric")

    # NumPy 2 renamed its private implementation package. These aliases let a
    # checkpoint written by NumPy 2 load under NumPy 1 without disabling
    # PyTorch's restricted weights-only unpickler.
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)

    safe_globals = [
        (multiarray.scalar, "numpy._core.multiarray.scalar"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
    ]
    for dtype_name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        dtype_class = type(np.dtype(dtype_name))
        if dtype_class not in safe_globals:
            safe_globals.append(dtype_class)
    return safe_globals


def load_checkpoint(path, map_location):
    """Load a tensor/config checkpoint with PyTorch's restricted unpickler."""
    safe_globals = _numpy_checkpoint_globals()
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals(safe_globals):
            return torch.load(path, map_location=map_location, weights_only=True)

    # PyTorch versions before safe_globals predate the default weights-only
    # behavior. Keep an explicit fallback for the supported torch>=2.2 range.
    return torch.load(path, map_location=map_location, weights_only=False)


def attach_checkpoint_electrostatics(model, metadata):
    """Rebuild the electrostatics a training run saved and attach it to ``model``.

    ``metadata`` is ``checkpoint['config']['electrostatics']`` as written by
    ``train.electrostatics_checkpoint_metadata``: the ``ElectrostaticConfig``
    fields, the ``ElectrostaticHeads`` keyword arguments and the feature irreps
    the heads were built on.  The heads are rebuilt on ``model.attention_irreps``
    -- the full feature block every forward hands them -- which must equal the
    saved irreps.  Call before ``load_state_dict(..., strict=True)``, which then
    loads the heads' parameters and checks their physical bounds.

    The caller must have set ``torch.set_default_dtype(torch.float64)`` before
    building ``model``: e3nn bakes the dtype into generated code at
    construction (trace_electrostatics SPEC 3).
    """
    from trace_electrostatics import ElectrostaticConfig, ElectrostaticHeads, attach_electrostatics

    if torch.get_default_dtype() is not torch.float64:
        raise RuntimeError(
            "an electrostatics checkpoint needs torch.set_default_dtype(torch.float64) "
            "before the model and the heads are constructed"
        )
    saved_irreps = metadata.get('irreps_in')
    if saved_irreps is not None and str(saved_irreps) != str(model.attention_irreps):
        raise ValueError(
            f"the checkpoint's electrostatic heads were built on {saved_irreps}, but this "
            f"model's feature irreps are {model.attention_irreps}"
        )
    fields = {field.name for field in dataclasses.fields(ElectrostaticConfig)}
    saved = dict(metadata['config'])
    unknown = sorted(set(saved) - fields)
    if unknown:
        raise ValueError(
            f"the checkpoint's ElectrostaticConfig has field(s) {unknown} that this "
            "trace_electrostatics does not know, so the trained energy functional "
            "cannot be reproduced"
        )
    missing = sorted(fields - set(saved))
    if missing:
        warnings.warn(
            f"the checkpoint's ElectrostaticConfig predates field(s) {missing}; they take "
            "this trace_electrostatics' defaults, which may define a different energy "
            "from the one that was trained",
            RuntimeWarning,
        )
    config = ElectrostaticConfig(**saved)
    heads = ElectrostaticHeads(irreps_in=str(model.attention_irreps), **dict(metadata['heads']))
    if metadata.get('redox') is not None:
        # TRACE-Redox (trace_redox): the free-charge sector the run trained.
        from trace_redox import RedoxConfig, attach_redox

        redox_fields = {field.name for field in dataclasses.fields(RedoxConfig)}
        unknown = sorted(set(metadata['redox']) - redox_fields)
        if unknown:
            raise ValueError(
                f"the checkpoint's RedoxConfig has field(s) {unknown} that this trace_redox "
                "does not know, so the trained energy functional cannot be reproduced"
            )
        # from_checkpoint: a field the checkpoint predates takes the statement it trained, not today's
        # default, where the two differ (trace_redox LEGACY_FIELD_VALUES; SPEC 28.6, 2026-11-10).
        return attach_redox(model, heads, config, RedoxConfig.from_checkpoint(metadata['redox']))
    transfer = None
    if metadata.get('transfer_coupling') is not None:
        # SPEC 22: the learned through-bond coupling head the run trained.
        from trace_electrostatics import TransferCouplingHead

        transfer = TransferCouplingHead.for_heads(heads, **dict(metadata['transfer_coupling']))
    return attach_electrostatics(model, heads, config, transfer)
