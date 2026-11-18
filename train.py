import argparse
import dataclasses
import inspect
import math
import warnings
import yaml
import os
import sys
from typing import Optional
import torch
import torch.optim as optim
import numpy as np
import time
from ase.io import read
from ase.data import atomic_numbers, chemical_symbols
from e3nn import o3
from flashace.charge_state import charge_state_from_atoms, charge_state_tensors, electrostatic_environment
from flashace.checkpoint import load_checkpoint
from flashace.model import TransformersACE, TransformersACEV3, TransformersACEV4
from flashace.optim import build_optimizer, optimizer_group_summary
from flashace.plotting import plot_metric_history
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset

# --- STABILITY SETTINGS ---
# Disable TF32 to prevent potential TensorCore precision crashes in e3nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_determinism(config) -> bool:
    """Optionally force bitwise-reproducible CUDA execution.

    TRACE scatters edge contributions onto atoms with ``index_add_``, whose CUDA
    implementation uses non-deterministic atomics: run-to-run energies differ by
    ~1e-5 meV/atom and gradients by ~1e-6 relative. That is far below the model
    error, but it makes a training run impossible to reproduce bitwise, which is
    what ``reproducibility.yaml`` promises. Enabling deterministic kernels costs
    about 7% and removes the variation entirely (measured on an RTX 5060 Ti).
    """
    if not bool(config.get('deterministic', False)):
        return False
    # cuBLAS needs this set before the first CUDA context to be reproducible.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    print("Deterministic mode: ON (CUBLAS_WORKSPACE_CONFIG=:4096:8, ~7% slower)")
    return True


def _load_config(config_arg: Optional[str]):
    """Load a YAML config, falling back to training/config.yaml if needed."""
    candidates = []
    if config_arg:
        candidates.append(config_arg)
    candidates.extend([
        "config.yaml",
        os.path.join("training", "config.yaml"),
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f), candidate

    raise FileNotFoundError(
        "No configuration file found. Provide --config or place config.yaml in the repo root or training/config.yaml."
    )


# ---------------------------------------------------------------------------
# Optional variational electrostatics (trace_electrostatics)
#
# Switched on by an ``electrostatics:`` block in the YAML config.  Absent, or
# with ``enabled: false``, the dtype, the data pipeline and the model are
# exactly those of a run without it.  The schema, the ASE keys the dataset
# reads and the loss definitions are in docs/electrostatics.md, section 7.
# ---------------------------------------------------------------------------

#: Keys of the ``electrostatics:`` block.  ``config`` and ``heads`` are passed
#: through to ``ElectrostaticConfig`` and ``ElectrostaticHeads``; everything
#: else is read here.  Any other key is a typo and is refused.
ELECTROSTATICS_KEYS = (
    'enabled',
    'config',
    'heads',
    'dipole_weight',
    'polarizability_weight',
    'born_charge_weight',
    'dielectric_weight',
    'partial_charge_weight',
    'partial_charge_key',
    'spin_key',
    'polarizability_route',
    'born_charge_method',
    'transfer_coupling',
    'redox',
    'dielectric_prior',
    'polarizability_prior',
    'field_polarizability_prior',
)
#: trace_electrostatics.observables route names: 'fixed_k' is -d2U/dE2 of the
#: energy the model evaluates (the package default; identical to the minimiser
#: response under solve_mode='converged'), 'minimiser' is D^T K^-1 D.
POLARIZABILITY_ROUTES = ('fixed_k', 'minimiser')
#: 'energy': born_charges_from_energy, -d2U/dE dr of the evaluated energy;
#: 'minimiser': born_charges_minimiser, the exact minimiser's dP/dr.
BORN_CHARGE_METHODS = ('energy', 'minimiser')

#: SPEC 23 contact law for new runs (1/A); see parse_electrostatics_config.
#: Measured on water (2 seeds, energies and forces only): polarizability
#: size-extensive, test energy error -29 %, force error -14 % at r_max 6 A.
DEFAULT_CHARGE_TRANSFER_DECAY = 3.0

#: The electrostatic observables that can carry a loss, and their units.
ELECTROSTATIC_OBSERVABLES = (
    ('dipole', 'e*A'),
    ('polarizability', 'A^3'),
    ('born_charges', 'e'),
    ('dielectric_tensor', '1'),
    ('partial_charges', 'e'),
)

#: Default atoms.arrays key of reference partial charges (Hirshfeld, Mulliken,
#: ...): a partitioning CONVENTION, trained on only when partial_charge_weight > 0.
DEFAULT_PARTIAL_CHARGE_KEY = 'partial_charges'
#: Largest |sum_i q_ref_i - Q| accepted, in e.  A reference that does not sum to
#: the structure's net charge cannot be reproduced under the charge constraint.
PARTIAL_CHARGE_SUM_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class ElectrostaticsSettings:
    """The validated ``electrostatics:`` block of a run that has it enabled."""

    config: object  # trace_electrostatics.ElectrostaticConfig, validated
    heads: dict  # ElectrostaticHeads keyword arguments; irreps_in comes from the model
    dipole_weight: float = 0.0
    polarizability_weight: float = 0.0
    born_charge_weight: float = 0.0
    dielectric_weight: float = 0.0
    #: Reference partial charges (atoms.arrays[partial_charge_key]), e.  Off by
    #: default: charges are a partitioning convention, not an observable.
    partial_charge_weight: float = 0.0
    partial_charge_key: str = DEFAULT_PARTIAL_CHARGE_KEY
    polarizability_route: str = 'fixed_k'
    born_charge_method: str = 'energy'
    #: TransferCouplingHead keyword arguments (SPEC 22), or None for no head.
    transfer_coupling: object = None
    #: atoms.info key holding the spin multiplicity 2S+1 ('spin' for OMol25).
    spin_key: str = 'multiplicity'
    #: trace_redox.RedoxConfig keyword arguments (TRACE-Redox, SPEC 25), or None:
    #: the free-charge sector attached in place of the SPEC 1 module.
    redox: object = None
    #: {'file': extxyz of 3-D periodic structures with a dielectric_tensor label and no
    #: energies, 'weight': w}, or None: an eps_inf loss evaluated once per optimizer step
    #: (see _response_prior_loss).
    dielectric_prior: object = None
    #: The same for open-boundary structures with a polarizability label (A^3): measured atomic
    #: polarizabilities, for example.
    polarizability_prior: object = None
    #: The whole model's static polarisability (A^3) by finite fields of the total energy, open-boundary
    #: structures with a DIAGONAL polarizability label: the free sector's charge-transfer response included, so
    #: structures with several configurations are accepted (see _field_polarizability_loss).
    field_polarizability_prior: object = None

    @property
    def observable_losses(self) -> bool:
        """Whether any dipole, polarizability or Born-charge loss is on."""
        return max(
            self.dipole_weight, self.polarizability_weight, self.born_charge_weight,
            self.dielectric_weight, self.partial_charge_weight,
        ) > 0.0

    @property
    def spin(self) -> bool:
        """Whether the heads carry the SPEC 9 spin sector (spin labels needed)."""
        return bool(self.heads.get('spin', False))

    def weight(self, observable):
        return {
            'dipole': self.dipole_weight,
            'polarizability': self.polarizability_weight,
            'born_charges': self.born_charge_weight,
            'dielectric_tensor': self.dielectric_weight,
            'partial_charges': self.partial_charge_weight,
        }[observable]


def _yaml_float(value, name):
    """A float-valued setting as a Python float.

    PyYAML reads ``1e-6`` -- no decimal point -- as the *string* '1e-6', and
    ``ElectrostaticConfig`` rightly refuses a string, so a float field is
    converted here, loudly if the text is not a number.  Integers are promoted
    too: ``thole_a: 0`` would otherwise be refused for its type rather than for
    its value.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, str)):
        return value
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error


def parse_electrostatics_config(config):
    """Validate the ``electrostatics:`` block; ``None`` unless it is enabled.

    Every key is checked even when ``enabled: false``, so a typo cannot hide in
    a block that is switched off today and on tomorrow.  ``config`` is built
    into an ``ElectrostaticConfig`` here, which validates its values;
    ``heads`` can only be checked for its keys until the model exists, because
    the heads are built on the model's feature irreps.
    """
    block = config.get('electrostatics')
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(
            f"electrostatics must be a mapping with keys {list(ELECTROSTATICS_KEYS)}, "
            f"got {type(block).__name__}"
        )
    unknown = sorted(set(block) - set(ELECTROSTATICS_KEYS))
    if unknown:
        raise ValueError(
            f"electrostatics: unknown key(s) {unknown}; allowed keys are "
            f"{list(ELECTROSTATICS_KEYS)}"
        )
    if not isinstance(block.get('enabled'), bool):
        raise ValueError(
            "electrostatics.enabled must be given explicitly as true or false "
            f"(got {block.get('enabled')!r}); a block without it is ambiguous"
        )

    from trace_electrostatics import ElectrostaticConfig, ElectrostaticHeads

    raw_config = block.get('config') or {}
    if not isinstance(raw_config, dict):
        raise ValueError("electrostatics.config must be a mapping of ElectrostaticConfig fields")
    fields = {field.name: field for field in dataclasses.fields(ElectrostaticConfig)}
    unknown = sorted(set(raw_config) - set(fields))
    if unknown:
        raise ValueError(
            f"electrostatics.config: unknown key(s) {unknown}; ElectrostaticConfig "
            f"fields are {sorted(fields)}"
        )
    config_kwargs = {
        key: (
            _yaml_float(value, f"electrostatics.config.{key}")
            if 'float' in str(fields[key].type)
            else value
        )
        for key, value in raw_config.items()
    }
    # New runs choose the solver route per structure (trace_electrostatics
    # resolve_solver: matrix-free for 3-D periodic cells and large molecules,
    # measured 1.5-5x cheaper per training step), written into the checkpoint.
    # The exact 'minimiser' observables exist only on the dense operator, so a
    # run that asks for them defaults to 'dense' instead.
    wants_dense = (
        block.get('polarizability_route') == 'minimiser' or block.get('born_charge_method') == 'minimiser'
    )
    config_kwargs.setdefault('solver', 'dense' if wants_dense else 'auto')
    # The solver-seed loss (SPEC 2.0.1) trains only the CG starting point. Under
    # the converged solve the energy does not depend on it, and on trained water
    # models the trained seed saved 3-9 % of CG iterations while its separate
    # assembly cost 44 % of the electrostatic overhead per training step
    # (C10H2/C10H3+, 18.8 -> 10.6 ms). New converged runs turn it off; a
    # fixed-k run -- whose energy DOES depend on the seed -- keeps it.
    if config_kwargs.get('solve_mode', 'converged') == 'converged':
        config_kwargs.setdefault('solver_seed_loss_weight', 0.0)
    electrostatic_config = ElectrostaticConfig(**config_kwargs)

    raw_heads = block.get('heads') or {}
    if not isinstance(raw_heads, dict):
        raise ValueError("electrostatics.heads must be a mapping of ElectrostaticHeads options")
    if 'irreps_in' in raw_heads:
        raise ValueError(
            "electrostatics.heads.irreps_in must not be set: the heads are built on "
            "the model's own feature irreps (model.attention_irreps), which is the "
            "full block h the model hands them"
        )
    parameters = inspect.signature(ElectrostaticHeads.__init__).parameters
    allowed = sorted(name for name in parameters if name not in ('self', 'irreps_in'))
    unknown = sorted(set(raw_heads) - set(allowed))
    if unknown:
        raise ValueError(
            f"electrostatics.heads: unknown key(s) {unknown}; ElectrostaticHeads "
            f"options are {allowed}"
        )
    # Float-default options are coerced (YAML reads 1e-3 as a string), except
    # where the option also takes a documented string: hardness_margin="legacy"
    # (the Thole kernel's floor, SPEC 17.3).
    string_values = {'hardness_margin': ('legacy',)}
    heads_kwargs = {
        key: (
            _yaml_float(value, f"electrostatics.heads.{key}")
            if isinstance(parameters[key].default, float)
            and not (isinstance(value, str) and value in string_values.get(key, ()))
            else value
        )
        for key, value in raw_heads.items()
    }
    # The heads' bond cutoff defaults to the model cutoff. It may be shorter but
    # never longer: bonds must stay in the neighbour list until their
    # dissociation envelope has reached zero (attach_electrostatics checks it
    # too; this says so in terms of the YAML keys).
    model_r_max = float(config['r_max'])
    heads_kwargs['r_max'] = float(heads_kwargs.get('r_max', model_r_max))
    # SPEC 23: new runs get the charge-transfer contact law unless the YAML says
    # otherwise (``charge_transfer_decay: null`` turns it off).  Written into
    # the checkpoint explicitly, so an older checkpoint -- whose metadata lacks
    # the key -- keeps the hardness it was trained with.
    heads_kwargs.setdefault('charge_transfer_decay', DEFAULT_CHARGE_TRANSFER_DECAY)
    if heads_kwargs['r_max'] > model_r_max:
        raise ValueError(
            f"electrostatics.heads.r_max = {heads_kwargs['r_max']} A exceeds the model "
            f"cutoff r_max = {model_r_max} A. The neighbour list must cover the heads' "
            "bond cutoff, or bonds leave it while their dissociation envelope is "
            "still non-zero and the energy and forces jump there."
        )

    # SPEC 22: the learned through-bond coupling of diabatic charge states.
    # ``transfer_coupling: {}`` builds it with defaults; absent, no head.
    transfer_kwargs = None
    raw_transfer = block.get('transfer_coupling')
    if raw_transfer is not None:
        if not isinstance(raw_transfer, dict):
            raise ValueError(
                "electrostatics.transfer_coupling must be a mapping of TransferCouplingHead "
                "options ({} for the defaults)"
            )
        from trace_electrostatics import TransferCouplingHead

        transfer_parameters = inspect.signature(TransferCouplingHead.__init__).parameters
        transfer_allowed = sorted(
            name for name in transfer_parameters if name not in ('self', 'invariant_dim')
        )
        unknown = sorted(set(raw_transfer) - set(transfer_allowed))
        if unknown:
            raise ValueError(
                f"electrostatics.transfer_coupling: unknown key(s) {unknown}; "
                f"TransferCouplingHead options are {transfer_allowed}"
            )
        transfer_kwargs = {
            key: (
                _yaml_float(value, f"electrostatics.transfer_coupling.{key}")
                if isinstance(transfer_parameters[key].default, float) or key == 'r_max'
                else value
            )
            for key, value in raw_transfer.items()
        }
        transfer_kwargs['r_max'] = float(transfer_kwargs.get('r_max', heads_kwargs['r_max']))
        if transfer_kwargs['r_max'] > model_r_max:
            raise ValueError(
                f"electrostatics.transfer_coupling.r_max = {transfer_kwargs['r_max']} A "
                f"exceeds the model cutoff r_max = {model_r_max} A"
            )

    weights = {}
    for key in ('dipole_weight', 'polarizability_weight', 'born_charge_weight', 'dielectric_weight',
                'partial_charge_weight'):
        value = _yaml_float(block.get(key, 0.0), f"electrostatics.{key}")
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"electrostatics.{key} must be a finite number >= 0, got {block.get(key)!r}")
        weights[key] = value
    from trace_electrostatics.observables import DEFAULT_ROUTE

    route = block.get('polarizability_route', DEFAULT_ROUTE)
    if route not in POLARIZABILITY_ROUTES:
        raise ValueError(
            f"electrostatics.polarizability_route must be one of {list(POLARIZABILITY_ROUTES)}, "
            f"got {route!r}"
        )
    method = block.get('born_charge_method', 'energy')
    if method not in BORN_CHARGE_METHODS:
        raise ValueError(
            f"electrostatics.born_charge_method must be one of {list(BORN_CHARGE_METHODS)}, "
            f"got {method!r}"
        )

    from flashace.charge_state import SPIN_KEYS

    spin_key = block.get('spin_key', 'multiplicity')
    if spin_key not in SPIN_KEYS:
        raise ValueError(
            f"electrostatics.spin_key must be one of {list(SPIN_KEYS)} (the atoms.info key "
            f"holding 2S+1; 'spin' for OMol25-style data), got {spin_key!r}"
        )

    partial_charge_key = block.get('partial_charge_key', DEFAULT_PARTIAL_CHARGE_KEY)
    if not isinstance(partial_charge_key, str) or not partial_charge_key:
        raise ValueError(
            f"electrostatics.partial_charge_key must be a non-empty atoms.arrays key, "
            f"got {partial_charge_key!r}"
        )

    redox_kwargs = _parse_redox_block(block, weights, heads_kwargs, transfer_kwargs, model_r_max)
    dielectric_prior = _parse_response_prior(block, redox_kwargs, 'dielectric_prior')
    polarizability_prior = _parse_response_prior(block, redox_kwargs, 'polarizability_prior')
    field_polarizability_prior = _parse_response_prior(block, None, 'field_polarizability_prior')

    if not block['enabled']:
        return None

    if config.get('use_amp', False):
        raise ValueError(
            "use_amp: true cannot be combined with electrostatics. The variational "
            "electrostatics requires float64 end to end (e3nn modules built under "
            "float64, a charge constraint closed to 1e-10, a CG solve differentiated "
            "through), and autocast would run the model in float16/bfloat16. Set "
            "use_amp: false."
        )
    return ElectrostaticsSettings(
        config=electrostatic_config,
        heads=heads_kwargs,
        polarizability_route=route,
        born_charge_method=method,
        transfer_coupling=transfer_kwargs,
        spin_key=spin_key,
        partial_charge_key=partial_charge_key,
        redox=redox_kwargs,
        dielectric_prior=dielectric_prior,
        polarizability_prior=polarizability_prior,
        field_polarizability_prior=field_polarizability_prior,
        **weights,
    )


def _parse_response_prior(block, redox_kwargs, key):
    """Validate ``electrostatics.dielectric_prior`` / ``polarizability_prior`` (``{file, weight}``); ``None`` when absent.

    The structures themselves are checked when they are read (:func:`_response_prior_items`).  With redox,
    only ``anchor='formal'``: its integer free charges leave the bound sector as the model's electronic
    response when the configuration window holds one configuration (trace_redox SPEC 17.7).  With several,
    the configuration ensemble's response is added from its exact record (trace_redox SPEC 30, 36.10), and a
    structure without an exact record is refused (checked per structure at every evaluation,
    :func:`_response_prior_loss`).
    """
    raw = block.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"electrostatics.{key} must be a mapping {{file: ..., weight: ...}}")
    unknown = sorted(set(raw) - {'file', 'weight'})
    if unknown:
        raise ValueError(f"electrostatics.{key}: unknown key(s) {unknown}; allowed keys are ['file', 'weight']")
    path = raw.get('file')
    if not isinstance(path, str) or not path:
        raise ValueError(f"electrostatics.{key}.file must name an extxyz file")
    weight = _yaml_float(raw.get('weight'), f"electrostatics.{key}.weight")
    if isinstance(weight, bool) or not isinstance(weight, float) or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"electrostatics.{key}.weight must be a finite number > 0, got {raw.get('weight')!r}")
    if redox_kwargs is not None:
        from trace_redox import RedoxConfig

        if RedoxConfig(**redox_kwargs).anchor != 'formal':
            raise ValueError(
                f"electrostatics.{key} with redox needs anchor='formal': the mean-field free "
                "charges respond to a field as well, so the bound sector is not the model's response"
            )
    return {'file': path, 'weight': weight}


def _parse_redox_block(block, weights, heads_kwargs, transfer_kwargs, model_r_max):
    """Validate ``electrostatics.redox`` (TRACE-Redox); ``None`` when absent.

    ``redox: {}`` attaches trace_redox with its defaults.  Refused with it: the
    observables that need a single SPEC 1 state (polarizability, Born charges,
    dielectric tensor), spin heads, and the SPEC 22 transfer coupling.  Dipole
    and partial-charge losses read only the solved q and mu and are allowed.
    The dielectric tensor is allowed with ``anchor='formal'``: its free charges
    are integers, so the bound sector (``RedoxElectrostatics.bound_state``) is
    the model's electronic response up to Var_p(d) / kT ~ exp(-gap / kT).
    """
    raw = block.get('redox')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("electrostatics.redox must be a mapping of trace_redox.RedoxConfig fields ({} for the defaults)")
    from trace_redox import RedoxConfig

    fields = {field.name: field for field in dataclasses.fields(RedoxConfig)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(f"electrostatics.redox: unknown key(s) {unknown}; RedoxConfig fields are {sorted(fields)}")
    kwargs = {
        key: (_yaml_float(value, f"electrostatics.redox.{key}") if 'float' in str(fields[key].type) else value)
        for key, value in raw.items()
    }
    resolved = RedoxConfig(**kwargs)                        # validates the values
    for key in ('polarizability_weight', 'born_charge_weight', 'dielectric_weight'):
        if weights[key] > 0.0:
            if key == 'dielectric_weight' and resolved.anchor == 'formal':
                # Integer free charges: the model's electronic response is the bound sector's up to
                # Var_p(d) / kT ~ exp(-gap / kT) (RedoxElectrostatics.bound_state; trace_redox SPEC 17.7).
                continue
            raise ValueError(
                f"electrostatics.{key} > 0 with redox: that observable is defined from a single "
                "SPEC 1 state, and the TRACE-Redox energy is not one (dipole and partial-charge "
                "losses are allowed; the dielectric tensor is allowed with anchor='formal', whose "
                "integer free charges leave the bound sector as the electronic response)"
            )
    if heads_kwargs.get('spin', False):
        raise ValueError("electrostatics.redox does not support spin heads (heads.spin: true)")
    if transfer_kwargs is not None:
        raise ValueError("electrostatics.redox replaces the SPEC 21/22 charge states; drop transfer_coupling")
    cutoff = float(kwargs.get('attention_cutoff', RedoxConfig().attention_cutoff))
    if cutoff > model_r_max:
        raise ValueError(
            f"electrostatics.redox.attention_cutoff = {cutoff} A exceeds the model cutoff r_max = "
            f"{model_r_max} A; neighbours would leave the list while still weighted"
        )
    return kwargs


def configure_default_dtype(electrostatics):
    """Set the global default dtype a run needs, and return it.

    An electrostatics run needs float64 BEFORE the model, the heads or any
    dataset tensor exists: e3nn bakes the default dtype into its generated code
    at construction, so a later ``.double()`` is a silent no-op
    (trace_electrostatics SPEC 3).  Without electrostatics nothing is changed,
    so the float32 path is exactly what it was.
    """
    if electrostatics is not None:
        torch.set_default_dtype(torch.float64)
    return torch.get_default_dtype()


def attach_electrostatics_from_settings(model, electrostatics):
    """Build the heads on the model's feature irreps and attach them.

    ``model.attention_irreps`` is the layout of the full feature block ``h``
    that every forward hands the electrostatics (``0e`` first, ``2e``
    present at ``l_max >= 2``, as the heads require).
    """
    from trace_electrostatics import ElectrostaticHeads, attach_electrostatics

    parameter = next(model.parameters())
    if torch.get_default_dtype() is not torch.float64 or parameter.dtype is not torch.float64:
        raise RuntimeError(
            "electrostatics needs the model AND the heads built under float64 "
            f"(default dtype {torch.get_default_dtype()}, model parameters "
            f"{parameter.dtype}); call configure_default_dtype() before building the model"
        )
    heads = ElectrostaticHeads(irreps_in=str(model.attention_irreps), **electrostatics.heads)
    if electrostatics.redox is not None:
        from trace_redox import RedoxConfig, attach_redox

        return attach_redox(model, heads, electrostatics.config, RedoxConfig(**electrostatics.redox))
    transfer = None
    if electrostatics.transfer_coupling is not None:
        from trace_electrostatics import TransferCouplingHead

        transfer = TransferCouplingHead.for_heads(heads, **electrostatics.transfer_coupling)
    return attach_electrostatics(model, heads, electrostatics.config, transfer)


def electrostatics_checkpoint_metadata(electrostatics, module):
    """What a checkpoint needs to rebuild the attached electrostatics."""
    return {
        'enabled': True,
        'default_dtype': 'float64',
        'irreps_in': str(module.heads.irreps_in),
        'heads': dict(electrostatics.heads),
        'config': dataclasses.asdict(electrostatics.config),
        'dipole_weight': electrostatics.dipole_weight,
        'polarizability_weight': electrostatics.polarizability_weight,
        'born_charge_weight': electrostatics.born_charge_weight,
        'dielectric_weight': electrostatics.dielectric_weight,
        'partial_charge_weight': electrostatics.partial_charge_weight,
        'partial_charge_key': electrostatics.partial_charge_key,
        'polarizability_route': electrostatics.polarizability_route,
        'born_charge_method': electrostatics.born_charge_method,
        'transfer_coupling': (
            None if electrostatics.transfer_coupling is None else dict(electrostatics.transfer_coupling)
        ),
        'spin_key': electrostatics.spin_key,
        # Only written for a TRACE-Redox run, so every other checkpoint is unchanged.
        # The RESOLVED RedoxConfig (every field), not the YAML block: a model trained
        # under one default must reload with that physics, whatever the defaults are
        # by then (bound_elimination and fragment_ensemble defaults changed 2026-10-08).
        **({'redox': (dataclasses.asdict(module.redox) if hasattr(module, 'redox') else dict(electrostatics.redox))}
           if electrostatics.redox is not None else {}),
    }


def save_checkpoint(
    path, epoch, model, optimizer, scheduler, scaler, config, energy_shift_per_atom, atomic_energies,
    electrostatics=None,
):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'config': {
            'architecture_version': int(model.architecture_version),
            'r_max': config['r_max'],
            'l_max': config['l_max'],
            'num_radial': config['num_radial'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'radial_basis_type': config.get('radial_basis_type', 'bessel'),
            'radial_trainable': config.get('radial_trainable', False),
            'gaussian_width': config.get('gaussian_width', 0.5),
            'energy_shift_per_atom': (
                float(energy_shift_per_atom)
                if energy_shift_per_atom is not None
                else None
            ),
            'atomic_energies': atomic_energies or {},
            'amp_dtype': config.get('amp_dtype', 'float16'),
            'use_amp': config.get('use_amp', False),
            'grad_accum_steps': max(1, int(config.get('grad_accum_steps', 1))),
            'precompute_neighbors': config.get('precompute_neighbors', False),
            'radial_mlp_hidden': config.get('radial_mlp_hidden', 32),
            'radial_mlp_layers': config.get('radial_mlp_layers', 2),
            'correlation_order': config.get('correlation_order', 4),
            'correlation_channels': config.get('correlation_channels', 16),
            'attention_num_heads': config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
            'attention_key_dim': config.get('attention_key_dim', None),
            'attention_ffn_hidden': config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
            'attention_dropout': config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
            'attention_layer_scale_init': config.get('attention_layer_scale_init', 1e-2),
            'attention_distance_penalty': config.get('attention_distance_penalty', True),
            'attention_mode': config.get('attention_mode', 'full'),
            'attention_num_shells': config.get('attention_num_shells', 4),
            'correlation_rank_initial': config.get('correlation_rank_initial', None),
            'correlation_rank_warmup_epochs': config.get('correlation_rank_warmup_epochs', 0),
        }
    }
    if electrostatics is not None:
        # Only for runs that have it, so every other checkpoint is unchanged.
        checkpoint['config']['electrostatics'] = electrostatics
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def build_neighbor_tensors(atoms, r_max, dtype=None):
    """Return neighbor-to-center edges plus periodic shifts from ASE.

    ``dtype`` (default: torch's default dtype, float32 unless an electrostatics
    run switched it) is the dtype of the shift tensor.
    """
    i, j, shifts = neighbor_list('ijS', atoms, r_max)
    edge_index = torch.stack(
        [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)],
        dim=0,
    )
    edge_shift = torch.tensor(
        shifts, dtype=torch.get_default_dtype() if dtype is None else dtype
    )
    return edge_index, edge_shift


def _stress_matrix(value):
    """Convert an ASE/ExtXYZ stress-like value to a symmetric 3x3 matrix."""
    array = np.asarray(value, dtype=float)
    if array.shape == (6,):
        xx, yy, zz, yz, xz, xy = array
        array = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=float,
        )
    elif array.size == 9:
        array = array.reshape(3, 3)
    else:
        raise ValueError(
            f"Stress must contain 6 Voigt or 9 matrix components, got shape {array.shape}"
        )
    return 0.5 * (array + array.T)


def energy_target_from_atoms(atoms):
    """Return the energy label (eV; 0.0 when absent) and whether it is to be used.

    A structure trains on forces alone when it carries no energy (Togo's displaced phonon supercells: VASP forces, no
    energies) or declares ``has_energy=F`` in its extxyz header (an energy present but on a reference the set does not
    share).  Mirrors :func:`stress_target_from_atoms`.
    """
    declared = atoms.info.get('has_energy', True)
    if isinstance(declared, str):
        declared = declared.strip().lower() not in ('f', 'false', '0', 'no')
    results = atoms.calc.results if atoms.calc is not None else {}
    if 'energy' in results:
        energy = float(results['energy'])
    elif 'energy' in atoms.info:
        energy = float(atoms.info['energy'])
    else:
        return 0.0, False
    return (energy, True) if bool(declared) else (0.0, False)


def stress_target_from_atoms(atoms):
    """Return stress in eV/Angstrom^3 and whether a label was present."""
    results = atoms.calc.results if atoms.calc is not None else {}
    if 'stress' in results:
        return _stress_matrix(results['stress']), True
    if 'stress' in atoms.info:
        return _stress_matrix(atoms.info['stress']), True

    # The atomistic-data convention is virial = -V * stress. ASE ExtXYZ
    # normally performs this conversion while reading, but retain a correct
    # fallback for Atoms objects assembled by other readers.
    if 'virial' in results:
        return -_stress_matrix(results['virial']) / atoms.get_volume(), True
    if 'virial' in atoms.info:
        return -_stress_matrix(atoms.info['virial']) / atoms.get_volume(), True

    return np.zeros((3, 3), dtype=float), False


def stress_to_voigt(stress):
    """Return ASE-order Voigt components: xx, yy, zz, yz, xz, xy."""
    return torch.stack(
        [
            stress[0, 0], stress[1, 1], stress[2, 2],
            stress[1, 2], stress[0, 2], stress[0, 1],
        ]
    )


def _calc_result(atoms, key):
    results = atoms.calc.results if atoms.calc is not None else {}
    return results.get(key)


def _rank2_label(value, name):
    """A 3x3 label from 9 components (row-major) or 6 Voigt ones (xx yy zz yz xz xy)."""
    array = np.asarray(value, dtype=float)
    if array.size == 9:
        return array.reshape(3, 3)
    if array.shape == (6,):
        return _stress_matrix(array)
    raise ValueError(
        f"{name} must hold a 3x3 tensor (9 components, or 6 in Voigt order "
        f"xx yy zz yz xz xy), got shape {array.shape}. A scalar isotropic value "
        "does not determine the tensor and is not accepted."
    )


def electrostatic_labels_from_atoms(atoms, spin=False, spin_key='multiplicity',
                                    partial_charge_key=DEFAULT_PARTIAL_CHARGE_KEY):
    """Charge state and electrostatic reference labels of one ASE structure.

    The charge state -- ``pbc``, net charge (``info['charge']`` /
    ``info['total_charge']``), fragments (``arrays['fragment_id']``,
    ``info['fragment_charges']``) and, for spin-resolved heads, the spin state
    (``info['multiplicity']`` / ``info['fragment_multiplicities']``) -- is read
    by :func:`flashace.charge_state.charge_state_from_atoms`, the reader the
    ASE calculator uses too.  On top of it, all optional (units e, A):

    ``atoms.calc.results['dipole']`` or ``atoms.info['dipole']``
        Total dipole moment in e*A about the Cartesian origin of
        ``atoms.positions`` (ASE's convention; for a net-charged structure the
        dipole depends on the origin, so the label must use this one).  Open
        boundaries only.
    ``atoms.info['polarizability']``
        Static electronic polarizability VOLUME in A^3, 3x3 (9 components
        row-major, or 6 Voigt).  Open boundaries only.
    ``atoms.arrays['born_charges']`` or ``atoms.calc.results['born_effective_charges']``
        Born effective charges in e, ``[n, 3, 3]`` (or ``[n, 9]`` row-major),
        ``Z[i, a, b] = dP_a / dr_{i, b}``: polarization component first,
        displacement second, as in trace_electrostatics.observables.

    ``atoms.calc.results['dielectric_tensor']`` or ``atoms.info['dielectric_tensor']``
        Clamped-ion (electronic) dielectric tensor eps_inf, dimensionless, 3x3
        (9 row-major or 6 Voigt).  3-D periodic structures only -- it is a bulk
        property; on any other structure it is reported as ignored.
    ``atoms.arrays[partial_charge_key]``
        Reference partial charges in e, ``[n]`` (Hirshfeld, Mulliken, ...).  A
        partitioning convention, not an observable; trained on only with
        ``partial_charge_weight > 0``.  Refused unless they sum to the net
        charge within ``PARTIAL_CHARGE_SUM_TOLERANCE``: the solve enforces
        ``sum_i q_i = Q`` exactly, so such a label could never be reproduced.

    A dipole or polarizability on a structure with any periodic direction is
    not a usable reference -- the polarization of a periodic cell is defined
    only modulo a quantum -- and is reported as ignored rather than trained on.
    """
    n_atoms = len(atoms)
    labels = charge_state_from_atoms(atoms, spin=spin, spin_key=spin_key)
    periodic = bool(labels['pbc'].any())

    dipole = _calc_result(atoms, 'dipole')
    if dipole is None:
        dipole = atoms.info.get('dipole')
    polarizability = atoms.info.get('polarizability')
    ignored = []
    if periodic and dipole is not None:
        ignored.append('dipole')
        dipole = None
    if periodic and polarizability is not None:
        ignored.append('polarizability')
        polarizability = None
    if dipole is not None:
        dipole = np.asarray(dipole, dtype=float).reshape(-1)
        if dipole.shape != (3,):
            raise ValueError(f"dipole must have 3 components, got shape {dipole.shape}")
    if polarizability is not None:
        polarizability = _rank2_label(polarizability, "atoms.info['polarizability']")
    # ASE's extxyz reader moves an info key that names a calculator property
    # ('dielectric_tensor', like 'dipole') into atoms.calc.results on reading.
    dielectric = _calc_result(atoms, 'dielectric_tensor')
    if dielectric is None:
        dielectric = atoms.info.get('dielectric_tensor')
    if dielectric is not None and not bool(labels['pbc'].all()):
        ignored.append('dielectric_tensor')
        dielectric = None
    if dielectric is not None:
        dielectric = _rank2_label(dielectric, "atoms.info['dielectric_tensor']")

    partial = atoms.arrays.get(partial_charge_key) if partial_charge_key else None
    if partial is not None:
        partial = np.asarray(partial, dtype=float).reshape(-1)
        if partial.shape != (n_atoms,):
            raise ValueError(
                f"partial charges atoms.arrays['{partial_charge_key}'] must have shape "
                f"[{n_atoms}], got {partial.shape}"
            )
        if np.all(np.isfinite(partial)) and abs(partial.sum() - labels['total_charge']) > PARTIAL_CHARGE_SUM_TOLERANCE:
            raise ValueError(
                f"partial charges atoms.arrays['{partial_charge_key}'] sum to {partial.sum():.6f} e, "
                f"but the structure's net charge is {labels['total_charge']} e; the charge "
                "constraint makes such a label impossible to reproduce"
            )

    born = atoms.arrays.get('born_charges')
    if born is None:
        born = _calc_result(atoms, 'born_effective_charges')
    if born is not None:
        born = np.asarray(born, dtype=float)
        if born.size != 9 * n_atoms:
            raise ValueError(
                f"Born charges must have shape [{n_atoms}, 3, 3] (or [{n_atoms}, 9]), "
                f"got {born.shape}"
            )
        born = born.reshape(n_atoms, 3, 3)

    for name, value in (('dipole', dipole), ('polarizability', polarizability), ('born_charges', born),
                        ('dielectric_tensor', dielectric), ('partial_charges', partial)):
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} label contains non-finite values")

    labels.update(
        dipole=dipole,
        polarizability=polarizability,
        born_charges=born,
        dielectric_tensor=dielectric,
        partial_charges=partial,
        ignored=tuple(ignored),
    )
    return labels


class AtomisticDataset(Dataset):
    """Structures as model inputs and training targets.

    ``dtype`` is the floating dtype of every tensor built here; it defaults to
    torch's default dtype *at construction* (float32 unless an electrostatics
    run switched it), and is stored, so that DataLoader workers started by
    ``spawn`` -- which re-import this module with the float32 default -- still
    build the right dtype.

    ``electrostatics=True`` adds the charge state and the electrostatic
    reference labels read by :func:`electrostatic_labels_from_atoms`, all
    parsed and validated here, before training starts.  Off, the items are
    exactly what they always were.
    """

    def __init__(
        self, atoms_list, r_max, random_rotation=False, precompute_neighbors=False,
        dtype=None, electrostatics=False, spin=False, spin_key='multiplicity',
        partial_charge_key=DEFAULT_PARTIAL_CHARGE_KEY,
    ):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.random_rotation = random_rotation
        self.precompute_neighbors = precompute_neighbors
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.electrostatics = bool(electrostatics)

        self._edge_cache = None
        if precompute_neighbors:
            self._edge_cache = []
            for atoms in atoms_list:
                self._edge_cache.append(build_neighbor_tensors(atoms, self.r_max, self.dtype))

        self._electrostatic_labels = None
        if self.electrostatics:
            self._electrostatic_labels = []
            for index, atoms in enumerate(atoms_list):
                try:
                    self._electrostatic_labels.append(
                        electrostatic_labels_from_atoms(
                            atoms, spin=spin, spin_key=spin_key, partial_charge_key=partial_charge_key,
                        )
                    )
                except ValueError as error:
                    raise ValueError(f"structure {index} ({atoms.get_chemical_formula()}): {error}") from error

    def electrostatic_label_counts(self):
        """How many structures carry each electrostatic label, and how many were ignored."""
        counts = {'charged': 0, 'open_shell': 0, 'dipole': 0, 'polarizability': 0, 'born_charges': 0,
                  'dielectric_tensor': 0, 'partial_charges': 0, 'ignored_dipole': 0, 'ignored_polarizability': 0,
                  'ignored_dielectric_tensor': 0}
        for labels in self._electrostatic_labels or ():
            counts['charged'] += labels['total_charge'] != 0.0
            counts['open_shell'] += labels.get('multiplicity') is not None and labels['multiplicity'] > 1.0
            for name in ('dipole', 'polarizability', 'born_charges', 'dielectric_tensor', 'partial_charges'):
                counts[name] += labels[name] is not None
            for name in labels['ignored']:
                counts[f'ignored_{name}'] += 1
        return counts

    def __len__(self): return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        dtype = self.dtype

        # Geometry
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=dtype)
        cell = torch.tensor(atoms.cell.array, dtype=dtype)
        # Gas-phase molecular data normally have no simulation cell. Volume is
        # used only by the periodic stress path, so a neutral placeholder keeps
        # energy/force training well-defined without fabricating periodicity.
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 1.0
        vol = torch.tensor(volume, dtype=dtype)

        # Targets
        energy, has_energy = energy_target_from_atoms(atoms)
        t_E = torch.tensor(energy, dtype=dtype)
        t_F = torch.tensor(atoms.get_forces(), dtype=dtype)

        stress, has_stress = stress_target_from_atoms(atoms)
        t_S = torch.tensor(stress, dtype=dtype)

        # Neighbors
        rot = None
        if self.random_rotation:
            # Random SO(3) rotation applied via Wigner matrices to encourage rotationally
            # equivariant learning even on small datasets. Energies remain invariant
            # while forces/stresses are rotated consistently.
            rot = o3.rand_matrix().to(dtype=pos.dtype)
            pos = pos @ rot.T
            cell = cell @ rot.T
            t_F = t_F @ rot.T
            t_S = rot @ t_S @ rot.T

        if self._edge_cache is not None:
            edge_index, edge_shift = self._edge_cache[idx]
        else:
            edge_index, edge_shift = build_neighbor_tensors(atoms, self.r_max, dtype)

        item = {
            'z': z,
            'pos': pos,
            'cell': cell,
            'edge_index': edge_index,
            'edge_shift': edge_shift,
            'volume': vol,
            't_E': t_E,
            't_F': t_F,
            't_S': t_S,
            'has_stress': torch.tensor(has_stress, dtype=torch.bool),
            'has_energy': torch.tensor(has_energy, dtype=torch.bool),
            'force_weight': torch.tensor(float(atoms.info.get('force_weight', 1.0) or 1.0), dtype=dtype),
        }
        if self.electrostatics:
            item.update(self._electrostatic_item(idx, len(atoms), rot))
        return item

    def _electrostatic_item(self, idx, n_atoms, rot):
        """The charge state and electrostatic targets of one structure.

        ``total_charge`` is a Python float because trace_electrostatics refuses
        a traced net charge; everything else is a tensor.  Vector and tensor
        targets follow the random rotation, ``R p``, ``R alpha R^T`` and
        ``R Z_i R^T``, exactly as forces and stress do.
        """
        labels = self._electrostatic_labels[idx]
        dtype = self.dtype

        def target(value, shape):
            if value is None:
                return torch.zeros(shape, dtype=dtype)
            return torch.tensor(value, dtype=dtype)

        dipole = target(labels['dipole'], (3,))
        polarizability = target(labels['polarizability'], (3, 3))
        born = target(labels['born_charges'], (n_atoms, 3, 3))
        dielectric = target(labels['dielectric_tensor'], (3, 3))
        partial = target(labels['partial_charges'], (n_atoms,))   # invariant under rotation
        if rot is not None:
            dipole = dipole @ rot.T
            polarizability = rot @ polarizability @ rot.T
            born = rot @ born @ rot.T
            dielectric = rot @ dielectric @ rot.T

        item = charge_state_tensors(labels, dtype)
        item.update({
            't_dipole': dipole,
            'has_dipole': torch.tensor(labels['dipole'] is not None),
            't_polarizability': polarizability,
            'has_polarizability': torch.tensor(labels['polarizability'] is not None),
            't_born': born,
            'has_born': torch.tensor(labels['born_charges'] is not None),
            't_dielectric': dielectric,
            'has_dielectric': torch.tensor(labels['dielectric_tensor'] is not None),
            't_partial_charges': partial,
            'has_partial_charges': torch.tensor(labels['partial_charges'] is not None),
        })
        # The structure's environment, read exactly as the ASE calculator reads
        # it: applied field (V/A), metallic atoms and electrode potentials
        # (SPEC 20), diabatic charge states and their coupling (SPEC 21, 22).
        # The field is a vector and follows the rotation; the rest is invariant.
        environment = electrostatic_environment(self.atoms_list[idx], dtype, 'cpu')
        if rot is not None and 'external_field' in environment:
            environment['external_field'] = environment['external_field'] @ rot.T
        item.update(environment)
        return item

    @staticmethod
    def collate_fn(batch): return batch

    @staticmethod
    def collate_batched(items):
        """Concatenate structures into ONE disconnected graph.

        Atom indices of structure b are offset by the running atom count, so the
        edge lists of different structures can never connect: the graph is a
        disjoint union and every per-atom quantity is unchanged. ``batch`` maps
        atom -> structure so the readout can be scattered back per structure.

        The GPU is launch-bound for single small cells, so one batched graph
        costs roughly the same as one structure while doing len(items) times the
        work. See docs/CUDA_NOTES.md.
        """
        z, pos, cells, vols, batch_idx = [], [], [], [], []
        e_idx, e_shift = [], []
        t_E, t_F, t_S, has_S, has_E, w_F = [], [], [], [], [], []
        offset = 0
        for b, it in enumerate(items):
            n = int(it['z'].shape[0])
            z.append(it['z']); pos.append(it['pos'])
            cells.append(it['cell']); vols.append(it['volume'].reshape(()))
            batch_idx.append(torch.full((n,), b, dtype=torch.long))
            e_idx.append(it['edge_index'] + offset)
            e_shift.append(it['edge_shift'])
            t_E.append(it['t_E'].reshape(())); t_F.append(it['t_F'])
            t_S.append(it['t_S']); has_S.append(it['has_stress'].reshape(()))
            has_E.append(it.get('has_energy', torch.tensor(True)).reshape(()))
            w_F.append(it.get('force_weight', torch.tensor(1.0, dtype=it['t_F'].dtype)).reshape(()).to(it['t_F'].dtype))
            offset += n
        batched = {
            'z': torch.cat(z), 'pos': torch.cat(pos),
            'cell': torch.stack(cells), 'volume': torch.stack(vols),
            'batch': torch.cat(batch_idx), 'n_structures': len(items),
            'edge_index': torch.cat(e_idx, dim=1), 'edge_shift': torch.cat(e_shift),
            't_E': torch.stack(t_E), 't_F': torch.cat(t_F),
            't_S': torch.stack(t_S), 'has_stress': torch.stack(has_S), 'has_energy': torch.stack(has_E),
            'force_weight': torch.stack(w_F),
            'n_atoms': torch.tensor([int(it['z'].shape[0]) for it in items], dtype=torch.long),
        }
        if any('total_charge' in it for it in items):
            batched.update(AtomisticDataset._collate_electrostatics(items, batched['pos'].dtype))
        return batched

    @staticmethod
    def _collate_electrostatics(items, dtype):
        """The per-structure charge specification and electrostatic targets.

        Every per-structure quantity gets a structure dimension, which is what
        lets ``forward_batched`` hand each structure its own charge state
        (flashace.model._structure_charge_specs reads this layout):

        * ``pbc`` ``[B, 3]``, ``total_charge`` ``[B]`` in e;
        * ``fragment_id`` ``[N_total]`` with labels LOCAL to each structure
          (``-1`` where a structure supplies none), ``n_fragments`` ``[B]``
          (0 = none; fragments are then derived from the bond graph, as in the
          single-structure path) and ``fragment_charges`` ``[sum F_s]`` in
          structure order;
        * ``fragment_spins`` ``[sum F_s]`` with ``n_fragment_spins`` ``[B]``;
        * targets ``t_dipole`` ``[B, 3]``, ``t_polarizability`` ``[B, 3, 3]``,
          ``t_born`` ``[N_total, 3, 3]``, ``t_dielectric`` ``[B, 3, 3]``,
          ``t_partial_charges`` ``[N_total]`` and their ``has_*`` masks ``[B]``;
        * the environment, when any structure has one: ``external_field``
          ``[B, 3]`` with ``has_external_field`` ``[B]``; ``metallic``
          ``[N_total]`` (False where absent); ``electrode_potential``
          ``[N_total]`` (NaN where absent); ``charge_states`` and
          ``charge_state_coupling`` as lists of length B (``None`` / 0 where
          absent).  Each default is the physics of "no such thing here".
        """
        if not all('total_charge' in it for it in items):
            raise ValueError(
                "collate_batched got a mix of structures with and without a charge "
                "specification; build every item from one AtomisticDataset"
            )
        batched = {
            'pbc': torch.stack([it['pbc'].reshape(3) for it in items]),
            'total_charge': torch.tensor([float(it['total_charge']) for it in items], dtype=dtype),
            't_dipole': torch.stack([it['t_dipole'] for it in items]),
            'has_dipole': torch.stack([it['has_dipole'].reshape(()) for it in items]),
            't_polarizability': torch.stack([it['t_polarizability'] for it in items]),
            'has_polarizability': torch.stack([it['has_polarizability'].reshape(()) for it in items]),
            't_born': torch.cat([it['t_born'] for it in items]),
            'has_born': torch.stack([it['has_born'].reshape(()) for it in items]),
            't_dielectric': torch.stack([it['t_dielectric'] for it in items]),
            'has_dielectric': torch.stack([it['has_dielectric'].reshape(()) for it in items]),
            't_partial_charges': torch.cat([it['t_partial_charges'] for it in items]),
            'has_partial_charges': torch.stack([it['has_partial_charges'].reshape(()) for it in items]),
        }
        if any('fragment_id' in it for it in items):
            labels, counts, charges = [], [], []
            for it in items:
                if 'fragment_id' in it:
                    labels.append(it['fragment_id'])
                    charges.append(it['fragment_charges'].reshape(-1))
                    counts.append(int(charges[-1].shape[0]))
                else:
                    labels.append(torch.full((int(it['z'].shape[0]),), -1, dtype=torch.long))
                    counts.append(0)
            batched['fragment_id'] = torch.cat(labels)
            batched['fragment_charges'] = torch.cat(charges).to(dtype)
            batched['n_fragments'] = torch.tensor(counts, dtype=torch.long)
        if any('fragment_spins' in it for it in items):
            spins = [it['fragment_spins'].reshape(-1) for it in items if 'fragment_spins' in it]
            batched['fragment_spins'] = torch.cat(spins).to(dtype)
            batched['n_fragment_spins'] = torch.tensor(
                [int(it['fragment_spins'].numel()) if 'fragment_spins' in it else 0 for it in items],
                dtype=torch.long,
            )
        if any('external_field' in it for it in items):
            batched['external_field'] = torch.stack(
                [it['external_field'].reshape(3) if 'external_field' in it
                 else torch.zeros(3, dtype=dtype) for it in items]
            ).to(dtype)
            batched['has_external_field'] = torch.tensor(['external_field' in it for it in items])
        if any('metallic' in it for it in items):
            batched['metallic'] = torch.cat(
                [it['metallic'].reshape(-1) if 'metallic' in it
                 else torch.zeros(int(it['z'].shape[0]), dtype=torch.bool) for it in items]
            )
        if any('electrode_potential' in it for it in items):
            batched['electrode_potential'] = torch.cat(
                [it['electrode_potential'].reshape(-1) if 'electrode_potential' in it
                 else torch.full((int(it['z'].shape[0]),), float('nan'), dtype=dtype) for it in items]
            ).to(dtype)
        if any('charge_states' in it for it in items):
            batched['charge_states'] = [it.get('charge_states') for it in items]
            batched['charge_state_coupling'] = [it.get('charge_state_coupling', 0.0) for it in items]
        return batched

class MetricTracker:
    """Accumulates energy, force and stress errors.

    Besides RMSE_E (Eq. 42) this also tracks the *offset-free* energy error, the
    standard deviation of the per-atom energy error about its own mean.  RMSE_E
    is dominated by a global additive shift whenever the reference energy is a
    single scalar; the offset-free number isolates the part of the error that a
    constant reference cannot absorb, and is the quantity that is stable across
    epochs and library versions.
    """

    def __init__(self): self.reset()
    def reset(self):
        self.sse_e = 0.0; self.sse_s = 0.0
        # Weighted Welford accumulators for the offset-free energy error. The
        # naive <e^2> - <e>^2 form suffers catastrophic cancellation once the
        # energy zero drifts, which is exactly the regime this metric exists to
        # diagnose, so the streaming form is used instead.
        self._w_sum = 0.0
        self._mean_e = 0.0
        self._m2_e = 0.0
        self.sum_force_mse = 0.0
        self.sum_force_mae = 0.0
        self.n_atoms = 0; self.n_stress_comp = 0; self.n_struct = 0
        self.n_atoms_e = 0           # atoms of the structures that carry an energy (the rmse_e denominator)
    def update(self, p_E, p_F, p_S, t_E, t_F, t_S, include_stress, n_ats, has_energy=True):
        # Promote to float64 *before* subtracting. Total energies are O(10^3 eV)
        # for a 40-atom CsPbI3 cell while the error is O(10^-1 eV); a float32
        # difference of two such numbers loses roughly three significant digits
        # of the quantity being measured.
        # ``.detach()`` first: p_E still carries the autograd graph here, and
        # calling float() on a requires_grad tensor warns and pins the graph alive.
        if has_energy:
            err_e = (p_E.detach().double().item() - torch.as_tensor(t_E).detach().double().item()) / n_ats
            self.sse_e += err_e**2 * n_ats
            self.n_atoms_e += n_ats
            weight = float(n_ats)
            self._w_sum += weight
            delta = err_e - self._mean_e
            self._mean_e += delta * weight / self._w_sum
            self._m2_e += weight * delta * (err_e - self._mean_e)
        diff_f = p_F - t_F
        # Per-structure force MSE/MAE averaged over 3N components.
        force_mse = diff_f.pow(2).mean().item()
        force_mae = diff_f.abs().mean().item()
        self.sum_force_mse += force_mse
        self.sum_force_mae += force_mae
        self.n_struct += 1
        if include_stress:
            self.sse_s += (
                stress_to_voigt(p_S) - stress_to_voigt(t_S)
            ).pow(2).sum().item()
            self.n_stress_comp += 6
        self.n_atoms += n_ats
    def update_batched(self, p_E, p_F, p_S, t_E, t_F, t_S, stress_mask, batch, n_atoms, has_energy=None):
        """Accumulate a whole batch without a single host synchronisation.

        Every ``.item()`` forces the CPU to wait for the GPU. With one structure
        per call that was ~4 syncs x |B| per step; here the batch statistics are
        reduced on device and merged into the running totals with Chan's
        parallel variance formula, which is algebraically identical to the
        sequential Welford update but needs no per-structure round trip. The
        only synchronisation is in ``get_metrics`` once per epoch.
        """
        n = n_atoms.double()
        h = torch.ones_like(n) if has_energy is None else has_energy.to(n.device).double()   # [B] 1 for labelled energies
        n_e = n * h
        err_e = (p_E.detach().double() - t_E.detach().double()) / n          # [B] eV/atom
        self.sse_e = self.sse_e + (err_e ** 2 * n_e).sum()
        self.n_atoms_e = self.n_atoms_e + n_e.sum()

        # ---- Chan et al. parallel merge of (weight, mean, M2), over the labelled structures ----
        w_b = n_e.sum()
        if float(w_b) > 0.0:
            mean_b = (err_e * n_e).sum() / w_b
            m2_b = (n_e * (err_e - mean_b) ** 2).sum()
            w_a = self._w_sum
            delta = mean_b - self._mean_e
            w_new = w_a + w_b
            self._mean_e = self._mean_e + delta * w_b / w_new
            self._m2_e = self._m2_e + m2_b + delta ** 2 * w_a * w_b / w_new
            self._w_sum = w_new

        diff_f = (p_F.detach() - t_F.detach()).double()                       # [N_tot, 3]
        sq = torch.zeros_like(err_e).index_add(0, batch, diff_f.pow(2).sum(dim=1))
        ab = torch.zeros_like(err_e).index_add(0, batch, diff_f.abs().sum(dim=1))
        self.sum_force_mse = self.sum_force_mse + (sq / (3.0 * n)).sum()
        self.sum_force_mae = self.sum_force_mae + (ab / (3.0 * n)).sum()

        mask = stress_mask.detach().double()
        if float(mask.sum()) > 0.0:
            d2 = ((p_S.detach() - t_S.detach()).double() ** 2).flatten(1)
            voigt = d2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)                      # xx yy zz yz xz xy
            self.sse_s = self.sse_s + (mask * voigt).sum()
            self.n_stress_comp += int(6 * mask.sum().item())

        self.n_struct += int(p_E.shape[0])
        self.n_atoms = self.n_atoms + n.sum()

    def _as_float(self, x):
        return float(x) if not torch.is_tensor(x) else float(x.detach().cpu())

    def get_metrics(self):
        """Return metrics. This is the ONLY host synchronisation point."""
        sse_e   = self._as_float(self.sse_e)
        sse_s   = self._as_float(self.sse_s)
        n_atoms = self._as_float(self.n_atoms_e)   # the structures that carry an energy
        sum_mse = self._as_float(self.sum_force_mse)
        sum_mae = self._as_float(self.sum_force_mae)
        w_sum   = self._as_float(self._w_sum)
        mean_e  = self._as_float(self._mean_e)
        m2_e    = self._as_float(self._m2_e)

        rmse_e = np.sqrt(sse_e / n_atoms) if n_atoms > 0 else 0.0
        rmse_s = np.sqrt(sse_s / self.n_stress_comp) if self.n_stress_comp > 0 else 0.0
        force_mse = (sum_mse / self.n_struct) if self.n_struct > 0 else 0.0
        force_mae = (sum_mae / self.n_struct) if self.n_struct > 0 else 0.0
        force_rmse = np.sqrt(force_mse)
        # Offset-free energy error: the N_s-weighted standard deviation of the
        # per-atom error about its own mean. Exactly invariant under a constant
        # shift of the energy reference, unlike rmse_e.
        if w_sum > 0.0:
            rmse_e_free = np.sqrt(max(m2_e / w_sum, 0.0))
        else:
            mean_e = 0.0
            rmse_e_free = 0.0
        return (
            rmse_e * 1000, force_rmse, rmse_s, force_mse, force_mae,
            rmse_e_free * 1000, mean_e * 1000,
        )


def compute_mean_energy_per_atom(atoms_seq):
    total_energy = 0.0
    total_atoms = 0
    for atoms in atoms_seq:
        energy, has_energy = energy_target_from_atoms(atoms)
        if not has_energy:
            continue                                   # forces-only structures (Round 33) carry no energy
        total_energy += energy
        total_atoms += len(atoms)
    return (total_energy / total_atoms) if total_atoms > 0 else 0.0


def atomic_energies_from_model(model, dataset, device):
    """Per-species reference energies solved on the residual ``E_DFT - E_model`` of the INITIAL model.

    The formal electrostatics of TRACE-Redox (SPEC 17.5) carries the ionisation and Madelung energy of
    the ionic state (about -8 eV/atom for Au2/MgO), which references solved on the DFT energies alone
    leave to the network.  Least squares on the residual puts that offset in the reference; the
    network then learns only what composition cannot absorb (tests/redox/test_redox_baseline.py).
    The model is evaluated in eval mode, structure by structure, as validation does.
    """
    was_training = model.training
    model.eval()
    species = sorted({int(z) for i in range(len(dataset)) for z in dataset[i]['z'].tolist()})
    counts, residuals = [], []
    for i in range(len(dataset)):
        item = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dataset[i].items()}
        if not bool(item.get('has_energy', True)):
            continue                                   # forces-only structures carry no energy to fit a baseline to
        energy = model(item, training=False, compute_stress=False)[0]
        residuals.append(float(item['t_E']) - float(energy.detach()))
        zs = item['z'].tolist()
        counts.append([zs.count(z) for z in species])
    model.train(was_training)
    X = np.array(counts, dtype=float)
    y = np.array(residuals, dtype=float)
    coeffs, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if rank < len(species):
        print(
            f"WARNING: the composition matrix has rank {rank} < {len(species)} species; the "
            "minimum-norm per-species references are used (exact per structure, not transferable)."
        )
    return {z: float(e) for z, e in zip(species, coeffs)}


def compute_atomic_energies_from_dataset(atoms_seq):
    """Solve for per-species reference energies via least squares.

    Builds the standard per-species linear system where each structure's
    total energy is expressed as the sum of per-species reference energies plus
    a residual. The least-squares solution provides offsets that remove most of
    the composition-dependent baseline from the supervised loss.
    """

    species = sorted({int(z) for atoms in atoms_seq for z in atoms.numbers})
    if not species:
        return {}

    counts = []
    energies = []
    for atoms in atoms_seq:
        energy, has_energy = energy_target_from_atoms(atoms)
        if not has_energy:
            continue                                   # forces-only structures (Round 33) carry no energy
        counts.append([np.count_nonzero(atoms.numbers == z) for z in species])
        energies.append(energy)
    if not energies:
        return {z: 0.0 for z in species}

    X = np.array(counts, dtype=float)
    y = np.array(energies, dtype=float)

    coeffs, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)

    # With a fixed stoichiometry (every frame Cs8Pb8I24, say) the composition
    # matrix is rank deficient and the individual per-species energies are not
    # identifiable -- only their composition-weighted sum is. lstsq silently
    # returns the minimum-norm solution, which splits the degenerate species
    # equally. That baseline is still exactly right for every training frame, but
    # the individual numbers are arbitrary and must not be transferred to a
    # different composition.
    if rank < len(species):
        symbols = ", ".join(chemical_symbols[z] for z in species)
        print(
            f"WARNING: the composition matrix has rank {rank} < {len(species)} species "
            f"({symbols}). Per-species reference energies are not separately "
            "identifiable from this dataset; the minimum-norm solution is used. The "
            "total reference per structure is correct, but do not reuse these "
            "per-species values for a different composition."
        )
    return {z: float(e) for z, e in zip(species, coeffs)}


def parse_atomic_energy_table(raw_table):
    """Normalize an atomic energy mapping with atomic numbers as keys."""

    table = {}
    if raw_table is None:
        return table

    for key, value in raw_table.items():
        if isinstance(key, str):
            try:
                z = atomic_numbers[key]
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Unknown chemical symbol '{key}' in atomic_energies") from exc
        else:
            z = int(key)

        table[z] = float(value)

    return table


def atomic_energy_tensor(energy_table, device, dtype=None):
    if not energy_table:
        return None

    max_z = max(energy_table)
    tensor = torch.zeros(
        max_z + 1, dtype=torch.get_default_dtype() if dtype is None else dtype, device=device
    )
    for z, val in energy_table.items():
        tensor[z] = val
    return tensor


def split_trajectory_frames(
    atoms,
    val_fraction=0.1,
    seed=42,
    mode="blocked",
    block_size=25,
    gap=0,
):
    """Split ordered trajectory frames without scattering one trajectory block.

    A frame-random split gives unrealistically optimistic validation estimates
    when adjacent molecular-dynamics frames are strongly correlated. ``blocked``
    selects whole contiguous blocks and can exclude a small boundary gap from
    training. An explicit validation file remains preferable when phase or
    trajectory labels are available.
    """
    n_frames = len(atoms)
    if n_frames < 2:
        raise ValueError("At least two structures are required for a train/validation split")

    val_target = max(1, min(n_frames - 1, int(round(n_frames * val_fraction))))
    generator = torch.Generator().manual_seed(int(seed))
    mode = str(mode).lower()

    if mode == "random":
        order = torch.randperm(n_frames, generator=generator).tolist()
        val_indices = sorted(order[:val_target])
    elif mode == "blocked":
        block_size = min(max(1, int(block_size)), max(1, n_frames // 2))
        blocks = [
            list(range(start, min(start + block_size, n_frames)))
            for start in range(0, n_frames, block_size)
        ]
        order = torch.randperm(len(blocks), generator=generator).tolist()
        selected = []
        selected_count = 0
        for block_index in order:
            if selected_count >= val_target:
                break
            if len(blocks) - len(selected) <= 1:
                break
            selected.append(block_index)
            selected_count += len(blocks[block_index])
        val_indices = sorted(i for block_index in selected for i in blocks[block_index])
    else:
        raise ValueError("split_mode must be 'blocked' or 'random'")

    val_set = set(val_indices)
    excluded = set()
    gap = max(0, int(gap))
    if gap:
        for index in val_indices:
            excluded.update(range(max(0, index - gap), min(n_frames, index + gap + 1)))
        excluded.difference_update(val_set)

    train_indices = [i for i in range(n_frames) if i not in val_set and i not in excluded]
    if not train_indices or not val_indices:
        raise ValueError("Split settings left an empty training or validation set")

    return (
        [atoms[i] for i in train_indices],
        [atoms[i] for i in val_indices],
        sorted(excluded),
    )


def redox_kT(epoch, start, end, epochs):
    """Training-time electronic temperature of TRACE-Redox (audit C3; a continuation heuristic).

    ``kT(e) = end (start / end)^(1 - e / epochs)`` for ``0 < e < epochs``, exactly ``start`` at
    ``e = 0`` and exactly ``end`` (the deployed ``redox.kT``) from ``e = epochs`` on.  A configuration
    ``Delta`` above the populated one weighs ``exp(-Delta / kT)``: hotter training lets its energy
    receive a gradient (tests/redox/test_redox_kT_schedule.py).
    """
    if epoch <= 0:
        return float(start)
    if epoch >= epochs:
        return float(end)
    return float(end) * (float(start) / float(end)) ** (1.0 - float(epoch) / float(epochs))


def batch_seed(seed, epoch, batch_index):
    """The random-number seed of one training batch: a function of (seed, epoch, batch index) only.

    ``batch_seeded_rng`` reseeds the global generator with it before each batch is fetched, so the
    batch's random rotation and attention-dropout masks do not depend on which process evaluates
    it, or on what was evaluated before.  NumPy's ``SeedSequence`` hashes the three integers, so
    distinct (epoch, batch) pairs get unrelated streams.
    """
    entropy = [int(seed) % 2 ** 64, int(epoch), int(batch_index)]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])


def _seeded_batches(loader, seed, epoch):
    """``enumerate(loader)``, with the global generator seeded by :func:`batch_seed` before each fetch."""
    iterator = iter(loader)
    index = 0
    while True:
        torch.manual_seed(batch_seed(seed, epoch, index))
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield index, batch
        index += 1


def data_parallel_world(config, environ=os.environ):
    """``(rank, world size)`` of a ``torchrun`` launch, ``(0, 1)`` for a single process.

    Data parallelism deals each optimiser step's batches out across processes (see
    :class:`_DataParallel`).  It is implemented for the batched-graph path on the CPU (gloo).
    """
    world = int(environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    device = str(config.get('device', 'cpu')).split(':')[0]
    if device != 'cpu':
        raise ValueError(f"data-parallel training is implemented on device cpu (gloo), not {device!r}")
    if not config.get('batched_graphs', False):
        raise ValueError("data-parallel training needs batched_graphs: true")
    return int(environ.get("RANK", "0")), world


def _detached(value):
    """``value`` with every tensor detached and copied (lists, tuples and dicts walked)."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_detached(v) for v in value)
    if isinstance(value, dict):
        return {k: _detached(v) for k, v in value.items()}
    return value


class _CallRecorder:
    """Stands in for a metric tracker on a data-parallel rank.

    Records each update call with its inputs detached, so that every rank can replay all ranks'
    calls into the real tracker in batch order: the same arithmetic, in the same order, as a
    sequential run.
    """

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, _detached(args), _detached(kwargs)))
        return record

    @staticmethod
    def replay(calls, tracker):
        for name, args, kwargs in calls:
            getattr(tracker, name)(*args, **kwargs)


class _DataParallel:
    """Deals each optimiser step's batches out across ``torchrun`` processes, exactly.

    Batch ``b`` of a window of ``grad_accum_steps`` batches is evaluated by rank
    ``(b mod grad_accum_steps) mod world``; validation batch ``b`` by rank ``b mod world``.  Each
    rank keeps its batches' gradients apart (:meth:`keep_gradients`); at the end of the window every
    rank gathers all of them and sums them in batch order (:meth:`sum_gradients`), which is the order
    and the arithmetic in which a sequential run's ``.grad`` accumulates them (IEEE addition is
    exactly rounded and commutative).  With ``batch_seeded_rng``, equal threads per process and
    deterministic kernels, every step, parameter and metric is the sequential run's bit for bit
    (tests/test_train_data_parallel.py).  Apple Accelerate's LAPACK is not deterministic for the
    110-atom redox structures (``cholesky_solve``, ~1 ulp, intermittent; measured 2026-10-09), so on
    a Mac neither a sequential nor a parallel run of them reproduces bitwise.
    """

    def __init__(self, rank, world):
        import torch.distributed as dist
        self.dist, self.rank, self.world = dist, rank, world
        self._kept = []

    def owns(self, position):
        return position % self.world == self.rank

    def keep_gradients(self, batch_index, parameters):
        """Take this batch's gradients out of ``.grad``, so the next batch starts from None."""
        self._kept.append((batch_index, [p.grad for p in parameters]))
        for p in parameters:
            p.grad = None

    def sum_gradients(self, parameters):
        """Set ``.grad`` to the sum of the window's batch gradients over all ranks, in batch order.

        A parameter no batch reached keeps ``grad None``, so the optimiser skips it as it would in a
        sequential run.
        """
        gathered = self.gather(self._kept)
        self._kept = []
        for j, p in enumerate(parameters):
            total = None
            for _batch, grads in gathered:
                g = grads[j]
                if g is not None:
                    total = g.clone() if total is None else total.add_(g)
            p.grad = total

    def gather(self, records):
        """All ranks' ``(batch index, payload)`` records, merged in batch order, on every rank."""
        parts = [None] * self.world
        self.dist.all_gather_object(parts, records)
        merged = sorted((r for part in parts for r in part), key=lambda r: r[0])
        if len({r[0] for r in merged}) != len(merged):
            raise RuntimeError("two data-parallel ranks evaluated the same batch")
        return merged

    def broadcast_state(self, model):
        """Overwrite every rank's parameters and buffers with rank 0's, in place.

        Called before the first step and after every step.  All ranks step from the same summed
        gradients, so they agree unless a kernel is not deterministic -- and Apple Accelerate's
        LAPACK is not (``cholesky_solve``, ~1 ulp, intermittently; measured 2026-10-09).  After this
        the run is rank 0's arithmetic by construction.  The other ranks' optimiser states need no
        broadcast: they never reach the run, because their parameters are overwritten after every
        step and rank 0 alone writes checkpoints.
        """
        for tensor in [p.data for p in model.parameters()] + list(model.buffers()):
            if tensor.is_contiguous():
                self.dist.broadcast(tensor, src=0)
            else:
                buffer = tensor.contiguous()
                self.dist.broadcast(buffer, src=0)
                tensor.copy_(buffer)

    def check_identical(self, tensors):
        """Raise on every rank unless all ranks hold bitwise-identical ``tensors``."""
        local = torch.cat([t.detach().contiguous().reshape(-1).view(torch.uint8) for t in tensors])
        reference = local.clone()
        self.dist.broadcast(reference, src=0)
        differs = torch.tensor([0 if torch.equal(local, reference) else 1], dtype=torch.int64)
        self.dist.all_reduce(differs)
        if int(differs.item()):
            raise RuntimeError(
                f"data-parallel ranks diverged: {int(differs.item())} rank(s) hold parameters or buffers "
                "that differ from rank 0's although rank 0's state is broadcast after every step"
            )


def _electrostatic_seed_loss(model):
    """The electrostatic solver-seed loss accumulated during this step's forwards.

    ``trace_electrostatics.ElectrostaticModule`` evaluates a variational loss in
    training mode that trains ONLY its solver seed (SPEC 2.0.1): every physics
    parameter is detached inside it, so it never trades against the data loss.
    It is added to the BACKWARD pass only and deliberately kept out of the
    logged loss -- it is an energy, not a squared error, and would make the
    training curves meaningless. ``None`` when no electrostatics is attached.
    """
    take = getattr(getattr(model, "electrostatics", None), "take_solver_seed_loss", None)
    return None if take is None else take()


def _born_charges(assembly, method):
    """Born charges ``Z[i, a, b] = dP_a / dr_{i, b}`` in e, with the graph kept.

    ``energy`` is ``-d2U/dE dr`` of the energy the model evaluates -- the
    derivative the forces and the field agree on.  Under
    ``solve_mode='fixed_k'`` that map has no derivative at an exactly
    symmetric point (SPEC 15.2) and the package raises; the exact-minimiser
    value it tends to is used there instead, which is what ``polarizability``
    itself does at such points.  (Under the default ``solve_mode='converged'``
    the two methods are the same derivative and the fallback never fires.)
    """
    from trace_electrostatics.observables import (
        FixedKDegenerateError,
        born_charges_from_energy,
        born_charges_minimiser,
    )

    if method == 'minimiser':
        return born_charges_minimiser(assembly, create_graph=True)
    try:
        return born_charges_from_energy(assembly, create_graph=True)
    except FixedKDegenerateError as error:
        warnings.warn(
            f"Born charges taken from the exact minimiser for one structure: {error}",
            RuntimeWarning,
        )
        return born_charges_minimiser(assembly, create_graph=True)


#: The response priors: config key -> (dataset label flag, label key, boundary condition).
RESPONSE_PRIORS = {
    'dielectric': ('has_dielectric', 't_dielectric', 'periodic'),
    'polarizability': ('has_polarizability', 't_polarizability', 'open'),
    'field_polarizability': ('has_polarizability', 't_polarizability', 'open'),
}
#: the applied-field step (V/A) of the field polarisability prior: the energy change h^2 alpha / (2 K_E) ~ 5e-5 eV
#: for alpha ~ 15 A^3 against float64 round-off ~ 1e-12 eV, and the hyperpolarisability term enters at O(h^2)
FIELD_PRIOR_STEP = 0.01


def _response_prior_items(prior, kind, r_max, dtype, electrostatics, precompute_neighbors):
    """The ``electrostatics.<kind>_prior`` structures as dataset items, checked.

    dielectric: 3-D periodic structures with a dielectric_tensor label.  polarizability: open-boundary
    structures with a polarizability label (A^3).  They need no energies or forces: a structure without them
    gets zero placeholders, which nothing reads (the prior's loss is its response alone,
    :func:`_response_prior_loss`).
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    flag, _, boundary = RESPONSE_PRIORS[kind]
    name = f"electrostatics.{kind}_prior"
    atoms_list = read(prior['file'], ':')
    if not atoms_list:
        raise ValueError(f"{name}: {prior['file']} holds no structure")
    for k, atoms in enumerate(atoms_list):
        if boundary == 'periodic' and not bool(np.all(atoms.pbc)):
            raise ValueError(
                f"{name}: structure {k} of {prior['file']} is not 3-D periodic (eps_inf is a bulk property)"
            )
        if boundary == 'open' and bool(np.any(atoms.pbc)):
            raise ValueError(
                f"{name}: structure {k} of {prior['file']} needs open boundaries (a molecular or atomic "
                "polarizability); use dielectric_prior for crystals"
            )
        if atoms.calc is None:
            atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((len(atoms), 3)))
        else:
            # keep whatever the reader stored (the label among it); fill only the gaps
            atoms.calc.results.setdefault('energy', 0.0)
            atoms.calc.results.setdefault('forces', np.zeros((len(atoms), 3)))
    dataset = AtomisticDataset(
        atoms_list, r_max, random_rotation=False, precompute_neighbors=precompute_neighbors,
        dtype=dtype, electrostatics=True, spin=electrostatics.spin, spin_key=electrostatics.spin_key,
        partial_charge_key=electrostatics.partial_charge_key,
    )
    items = [dataset[k] for k in range(len(dataset))]
    label = 'dielectric_tensor' if kind == 'dielectric' else 'polarizability'
    for k, item in enumerate(items):
        if not bool(item[flag]):
            raise ValueError(f"{name}: structure {k} of {prior['file']} carries no {label} label")
        if kind == 'field_polarizability':
            t = item[RESPONSE_PRIORS[kind][1]]
            off = t - torch.diag(torch.diagonal(t))
            if float(off.abs().max()) > 1e-9 * (1.0 + float(t.abs().max())):
                raise ValueError(
                    f"{name}: structure {k} of {prior['file']} has a non-diagonal polarizability label; orient the "
                    "structure along its principal axes (the prior fits the three diagonal components)"
                )
    return items


def _field_polarizability_loss(model, items, device, temperature_scale, training):
    """``mean_s mean_i (alpha_ii - alpha_ref,ii)^2`` and the per-structure errors ``[3]`` (A^3).

    alpha_ii = -[E(+h e_i) - 2 E(0) + E(-h e_i)] / h^2 x K_E: central differences of the TOTAL energies in a uniform
    applied field, h = FIELD_PRIOR_STEP, seven batched forwards.  With TRACE-Redox this is the whole response --
    bound sector, thermal configuration term and coherent charge transfer -- so it trains the free sector; only first
    derivatives in the weights are needed.
    """
    from trace_electrostatics.constants import K_E

    _, target, _ = RESPONSE_PRIORS['field_polarizability']
    bat = AtomisticDataset.collate_batched(items)
    for key, value in bat.items():
        if isinstance(value, torch.Tensor):
            bat[key] = value.to(device, non_blocking=True)
    n = int(bat['n_structures'])
    dtype = bat['pos'].dtype
    h = FIELD_PRIOR_STEP

    def energies(field):
        data = dict(bat)
        data['external_field'] = field
        data['has_external_field'] = torch.ones(n, dtype=torch.bool, device=device)
        with torch.enable_grad():
            energy, _, _, _ = model.forward_batched(
                data, training=training, temperature_scale=temperature_scale, compute_stress=False,
            )
        return energy.reshape(n)

    zero = torch.zeros(n, 3, dtype=dtype, device=device)
    e0 = energies(zero)
    alpha = []
    for axis in range(3):
        step = zero.clone()
        step[:, axis] = h
        alpha.append(-(energies(step) - 2.0 * e0 + energies(-step)) / (h * h) * K_E)
    alpha = torch.stack(alpha, dim=1)                                                     # [n, 3]
    errors = [alpha[k] - torch.diagonal(items[k][target].to(alpha)) for k in range(n)]
    loss = torch.stack([e.pow(2).mean() for e in errors]).mean()
    return loss, errors


def _response_prior_loss(model, items, kind, device, electrostatics, temperature_scale, training):
    """``mean_s mean_ij (R_ij - R_ref,ij)^2`` over the prior structures, and the error tensors.

    R is the dielectric tensor (dimensionless) or the polarizability (A^3) of the rebuilt state, on the
    configured route: for TRACE-Redox with formal anchors, the bound sector's.  That is the model's whole
    response only when the configuration window holds ONE configuration.  With several, the thermal term
    Var_p(d) / kT and, with couplings, the coherent charge-transfer response are added from the ensemble's
    exact response record (trace_redox SPEC 30; SPEC 36.10 for several clusters); a structure whose record
    is not exact (the core + carrier elimination, the SPEC 32 fold's field dependence) is refused at every
    evaluation.  A forward of its own: the prior structures enter no energy, force or validation loss.
    """
    from trace_electrostatics import dielectric_tensor, polarizability

    if kind == 'field_polarizability':
        return _field_polarizability_loss(model, items, device, temperature_scale, training)
    _, target, _ = RESPONSE_PRIORS[kind]
    bat = AtomisticDataset.collate_batched(items)
    for key, value in bat.items():
        if isinstance(value, torch.Tensor):
            bat[key] = value.to(device, non_blocking=True)
    redox_module = getattr(model, 'electrostatics', None)
    previous = getattr(redox_module, 'record_ensemble', None)
    if previous is not None or hasattr(redox_module, '_ensemble_record'):
        redox_module.record_ensemble = True              # SPEC 30: the ensemble response record, for this forward only
    try:
        with torch.enable_grad():
            _, _, _, aux = model.forward_batched(
                bat, training=training, temperature_scale=temperature_scale, compute_stress=False,
                return_electrostatics=True,
            )
    finally:
        if hasattr(redox_module, '_ensemble_record'):
            redox_module.record_ensemble = bool(previous) if previous is not None else False
        errors = []
        for k, (output, item) in enumerate(zip(aux['electrostatics'], items)):
            solve = output.solve_result
            ensemble_alpha = None
            if getattr(solve, 'energy_bound', None) is not None and int(solve.iterations) != 1:
                # several configurations: the whole response is the bound sector's plus the configuration
                # ensemble's (thermal and coherent charge transfer), trace_redox SPEC 30
                ensemble = getattr(solve, 'ensemble', None)
                if ensemble is None or not ensemble.get('exact', False):
                    reason = '' if ensemble is None else f" ({ensemble.get('reason', '')})"
                    raise ValueError(
                        f"electrostatics.{kind}_prior: structure {k} has {int(solve.iterations)} configurations in "
                        "the window and no exact ensemble response record" + reason + "; the bound sector alone is "
                        "the model's response only with exactly one configuration"
                    )
                from trace_redox.response import record_polarizability
                ensemble_alpha = record_polarizability(ensemble)      # one Hamiltonian, or SPEC 36.10's terms
            state = output.state()
            if kind == 'dielectric':
                response = dielectric_tensor(
                    state.solve_result, state.assembly, route=electrostatics.polarizability_route,
                    create_graph=True,
                )
                if ensemble_alpha is not None:
                    from trace_electrostatics.constants import K_E as _K_E
                    volume = torch.det(state.assembly.cell).abs()
                    response = response + (4.0 * math.pi * _K_E / volume) * ensemble_alpha.to(response)
            else:
                response = polarizability(
                    state.solve_result, state.assembly, route=electrostatics.polarizability_route,
                    as_volume=True, create_graph=True,
                )
                if ensemble_alpha is not None:
                    from trace_electrostatics.constants import K_E as _K_E
                    response = response + _K_E * ensemble_alpha.to(response)
            errors.append(response - item[target].to(response))
    loss = torch.stack([error.pow(2).mean() for error in errors]).mean()
    return loss, [error.detach() for error in errors]


def _electrostatic_errors(outputs, items, electrostatics):
    """Prediction minus reference of each electrostatic observable, per structure.

    ``outputs`` are the ``aux['electrostatics']`` records of one forward and
    ``items`` the matching dataset items.  Returns ``name -> list`` with one
    entry per structure: the error tensor, or ``None`` where the structure has
    no reference or the loss weight is zero.  Every prediction is a function of
    the model parameters with its graph kept (``create_graph``), so the losses
    built from them train the parameters.  The graph is kept in validation as
    well: in a batched graph the structures share one autograd graph, and a
    Born charge taken without it would free the graph the next structure needs.

    * dipole, e*A: ``P = sum_i (q_i r_i + mu_i)`` from the solve, about the
      Cartesian origin (``observables.polarization``);
    * polarizability, A^3: ``observables.polarizability`` -- a response in
      e^2 A^2 / eV -- converted to a volume by ``K_E`` (SPEC 0) via
      ``as_volume=True``, on the configured route;
    * Born charges, e: see :func:`_born_charges`;
    * dielectric tensor, dimensionless: ``observables.dielectric_tensor``,
      ``I + 4 pi K_E alpha_cell / Omega`` on the configured route (3-D
      periodic structures);
    * partial charges, e: the solved ``q = B diag(f) t + rho`` against a
      reference partition (Hirshfeld, ...), per atom.
    """
    from trace_electrostatics import dielectric_tensor, polarizability, polarization

    errors = {name: [] for name, _unit in ELECTROSTATIC_OBSERVABLES}
    for output, item in zip(outputs, items):
        dipole = alpha = born = epsilon = charges = None
        if electrostatics.dipole_weight > 0.0 and bool(item['has_dipole']):
            solve = output.solve_result
            predicted = polarization(solve.q, output.positions, solve.mu)
            dipole = predicted - item['t_dipole'].to(predicted)
        if electrostatics.polarizability_weight > 0.0 and bool(item['has_polarizability']):
            state = output.state()
            predicted = polarizability(
                state.solve_result,
                state.assembly,
                route=electrostatics.polarizability_route,
                as_volume=True,
                create_graph=True,
            )
            alpha = predicted - item['t_polarizability'].to(predicted)
        if electrostatics.born_charge_weight > 0.0 and bool(item['has_born']):
            predicted = _born_charges(output.state().assembly, electrostatics.born_charge_method)
            born = predicted - item['t_born'].to(predicted)
        if electrostatics.dielectric_weight > 0.0 and bool(item['has_dielectric']):
            state = output.state()
            predicted = dielectric_tensor(
                state.solve_result,
                state.assembly,
                route=electrostatics.polarizability_route,
                create_graph=True,
            )
            epsilon = predicted - item['t_dielectric'].to(predicted)
        if electrostatics.partial_charge_weight > 0.0 and bool(item['has_partial_charges']):
            predicted = output.solve_result.q
            charges = predicted - item['t_partial_charges'].to(predicted)
        errors['dipole'].append(dipole)
        errors['polarizability'].append(alpha)
        errors['born_charges'].append(born)
        errors['dielectric_tensor'].append(epsilon)
        errors['partial_charges'].append(charges)
    return errors


def _electrostatic_loss(errors, items, electrostatics, reference):
    """Weighted per-structure electrostatic loss, ``[len(items)]``.

    Normalised the way Eqs. (37)-(40) normalise energy, forces and stress: an
    extensive observable is divided by N_s before squaring, like the energy --
    the dipole ``((P - P_ref) / N_s)^2`` averaged over its 3 components, the
    polarizability ``((alpha - alpha_ref) / N_s)^2`` over its 9 -- and a
    per-atom one is averaged over its components, like the forces -- the Born
    charges over their 9 N_s, the partial charges over their N_s.  The dielectric tensor is intensive: averaged
    over its 9 components, not divided by N_s.  A structure without a reference contributes
    zero but still counts in the batch denominator, as a stress-free structure
    does.
    """
    per_structure = []
    for s, item in enumerate(items):
        n_atoms = float(item['z'].shape[0])
        loss = reference.new_zeros(())
        for name, _unit in ELECTROSTATIC_OBSERVABLES:
            error = errors[name][s]
            if error is None:
                continue
            if name in ('born_charges', 'dielectric_tensor', 'partial_charges'):
                # Intensive: averaged over components, not divided by N_s.
                term = error.pow(2).mean()
            else:
                term = (error / n_atoms).pow(2).mean()
            loss = loss + electrostatics.weight(name) * term
        per_structure.append(loss)
    return torch.stack(per_structure)


class ElectrostaticMetricTracker:
    """RMSE of each electrostatic observable over the structures that have one.

    In physical units: dipole e*A, polarizability A^3, Born charges e.  ``nan``
    for an observable nothing was evaluated on.
    """

    def __init__(self): self.reset()

    def reset(self):
        self.sse = {name: 0.0 for name, _unit in ELECTROSTATIC_OBSERVABLES}
        self.count = {name: 0 for name, _unit in ELECTROSTATIC_OBSERVABLES}

    def update(self, errors):
        for name, per_structure in errors.items():
            for error in per_structure:
                if error is not None:
                    self.sse[name] += float(error.detach().pow(2).sum())
                    self.count[name] += int(error.numel())

    def rmse(self):
        return {
            name: math.sqrt(self.sse[name] / self.count[name]) if self.count[name] else float('nan')
            for name, _unit in ELECTROSTATIC_OBSERVABLES
        }


def force_loss(per_struct_sq, n_at, force_weight):
    """``sum_s w_s |dF_s|^2 / (3 N_s)``: Eq. (37)'s force term with a per-structure weight (``atoms.info['force_weight']``,
    default 1; the bulk phonon cells' force scale is a tenth of the slabs', so at weight 1 they carry 0.4 % of the loss)."""
    return (force_weight * per_struct_sq / (3.0 * n_at)).sum()


def energy_loss(E, target_E, n_at, has_energy):
    """``sum_s has_s ((E_s - t_s) / N_s)^2``: Eq. (37)'s energy term over the structures that carry an energy."""
    return (has_energy * ((E - target_E) / n_at) ** 2).sum()


def _batched_train_step(
    model, items, device, temp_scale,
    energy_weight, force_weight, stress_weight,
    sobolev_weight, sobolev_sigma,
    baseline_energy, metrics, scaler, norm,
    electrostatics=None, electrostatic_metrics=None,
):
    """One optimiser sub-step over a batch evaluated as a single graph.

    Reproduces Eqs. (37)-(41) exactly, but vectorised over structures. The
    per-structure normalisations matter: energies are divided by N_s before
    squaring, force errors averaged over 3N_s, stress over its six Voigt
    components, and stress-free structures contribute zero while still counting
    in the batch denominator.  The optional electrostatic observable losses
    (``electrostatics``, see :func:`_electrostatic_loss`) follow the same rules.
    """
    bat = AtomisticDataset.collate_batched(items)
    for k, v in bat.items():
        if isinstance(v, torch.Tensor):
            bat[k] = v.to(device, non_blocking=True)

    n_at = bat['n_atoms'].to(device).to(bat['pos'].dtype)          # [B]
    has_s = bat['has_stress'].to(device).to(bat['pos'].dtype)      # [B]
    want_stress = bool(has_s.any().item()) and stress_weight > 0.0

    sob_active = sobolev_weight > 0.0 and sobolev_sigma > 0.0
    rng_cpu = torch.get_rng_state() if sob_active else None
    rng_dev = (
        torch.cuda.get_rng_state(device)
        if sob_active and torch.device(device).type == 'cuda' else None
    )

    observables = electrostatics is not None and electrostatics.observable_losses
    E, F, S, aux = model.forward_batched(
        bat, training=True, temperature_scale=temp_scale, compute_stress=want_stress,
        return_electrostatics=observables,
    )

    base = torch.stack([baseline_energy(it['z'].to(device)) for it in items]).reshape(-1)
    target_E = bat['t_E'].reshape(-1) - base
    has_e = bat['has_energy'].to(device).to(bat['pos'].dtype)      # [B]
    loss_e = energy_loss(E, target_E, n_at, has_e)

    dF2 = (F - bat['t_F']) ** 2                                    # [N_total, 3]
    per_struct_f = torch.zeros_like(E).index_add(0, bat['batch'], dF2.sum(dim=1))
    loss_f = force_loss(per_struct_f, n_at, bat['force_weight'].to(device).to(E.dtype))

    loss = energy_weight * loss_e + force_weight * loss_f
    if want_stress:
        dS2 = ((S - bat['t_S']) ** 2).flatten(1)                   # [B, 9]
        voigt = dS2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)              # xx yy zz yz xz xy
        loss = loss + stress_weight * (has_s * voigt / 6.0).sum()

    if observables:
        errors = _electrostatic_errors(aux['electrostatics'], items, electrostatics)
        loss = loss + _electrostatic_loss(errors, items, electrostatics, E).sum()
        if electrostatic_metrics is not None:
            electrostatic_metrics.update(errors)

    if sob_active:
        delta = torch.randn_like(bat['pos']) * sobolev_sigma
        pert = dict(bat); pert['pos'] = bat['pos'] + delta
        torch.set_rng_state(rng_cpu)
        if rng_dev is not None:
            torch.cuda.set_rng_state(rng_dev, device)
        E_p, _, _, _ = model.forward_batched(
            pert, training=True, temperature_scale=temp_scale,
            detach_pos=True, compute_stress=False,
        )
        fd_lin = torch.zeros_like(E).index_add(0, bat['batch'], (F.detach() * delta).sum(dim=1))
        loss = loss + sobolev_weight * (((E_p - E) + fd_lin) ** 2).sum()

    seed_loss = _electrostatic_seed_loss(model)
    total = loss if seed_loss is None else loss + seed_loss
    scaler.scale(total / norm).backward()

    with torch.no_grad():
        S_all = S.detach() if want_stress else torch.zeros(
            len(items), 3, 3, device=device, dtype=bat['pos'].dtype
        )
        mask = has_s if want_stress else torch.zeros_like(has_s)
        metrics.update_batched(
            E, F, S_all, target_E, bat['t_F'], bat['t_S'],
            mask, bat['batch'], bat['n_atoms'].to(device), has_energy=bat['has_energy'].to(device),
        )
    return loss.detach()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Transformers-ACE")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    args = parser.parse_args(argv)

    config, config_path = _load_config(args.config)
    # Data parallelism (torchrun): see _DataParallel.  Every rank runs this function; rank 0
    # alone prints and writes checkpoints.
    rank, world = data_parallel_world(config)
    data_parallel = None
    owns_process_group = False
    if world > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
            owns_process_group = True
        data_parallel = _DataParallel(rank, world)
        if rank > 0:
            # One log (rank 0's); the other ranks' output (e.g. a failure diagnosis) goes to a
            # file of its own next to the checkpoint.
            run_dir = os.path.dirname(os.path.abspath(config['model_save_path']))
            os.makedirs(run_dir, exist_ok=True)
            sys.stdout = open(os.path.join(run_dir, f"stdout_rank{rank}.txt"), "w", buffering=1)
    batch_seeded_rng = bool(config.get('batch_seeded_rng', False)) or data_parallel is not None
    configure_determinism(config)
    print(f"--- Loading {config_path} ---")
    if data_parallel is not None:
        print(
            f"Data parallel: {world} processes (gloo); each optimiser step's batches are dealt out "
            "and their gradients summed in batch order, so the run is the sequential one bit for bit"
        )

    # Validated first, and the dtype set before ANY tensor, dataset, model or
    # e3nn module exists: e3nn bakes the default dtype into generated code at
    # construction. Without an enabled electrostatics block nothing changes.
    electrostatics = parse_electrostatics_config(config)
    default_dtype = configure_default_dtype(electrostatics)
    if electrostatics is not None:
        print(
            f"Electrostatics: ON ({'TRACE-Redox, trace_redox' if electrostatics.redox is not None else 'trace_electrostatics'}"
            ", float64); loss weights "
            f"dipole {electrostatics.dipole_weight}, polarizability "
            f"{electrostatics.polarizability_weight}, Born charges "
            f"{electrostatics.born_charge_weight}, dielectric tensor "
            f"{electrostatics.dielectric_weight}"
        )

    if config.get('atomic_energies_from_model', False) and config.get('resume_from'):
        raise ValueError(
            "atomic_energies_from_model fits the references on the INITIAL model; a resumed run keeps "
            "its checkpoint's references (drop resume_from or the key)"
        )
    redox_kT_start = config.get('redox_kT_start')
    redox_kT_epochs = config.get('redox_kT_epochs')
    if (redox_kT_start is None) != (redox_kT_epochs is None):
        raise ValueError("redox_kT_start and redox_kT_epochs go together (a training-time electronic temperature)")
    if redox_kT_start is not None:
        if electrostatics is None or electrostatics.redox is None:
            raise ValueError("redox_kT_start needs electrostatics.redox (TRACE-Redox)")
        from trace_redox import RedoxConfig
        kT_end = float(electrostatics.redox.get('kT', RedoxConfig().kT))
        redox_kT_start, redox_kT_epochs = float(redox_kT_start), int(redox_kT_epochs)
        if not redox_kT_start > kT_end:
            raise ValueError(f"redox_kT_start = {redox_kT_start} eV must lie above the deployed kT = {kT_end} eV")
        if not 0 < redox_kT_epochs < int(config['epochs']):
            raise ValueError(
                f"redox_kT_epochs = {redox_kT_epochs} must be positive and below epochs = {int(config['epochs'])}, "
                "so that kT reaches the deployed value before training ends"
            )

    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_num_threads = int(config.get('torch_num_threads', 0) or 0)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    torch_num_interop_threads = int(config.get('torch_num_interop_threads', 0) or 0)
    if torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_num_interop_threads)
        except RuntimeError:
            # PyTorch only permits changing this before inter-op work starts.
            pass
    print(
        f"PyTorch CPU threads: {torch.get_num_threads()} intra-op, "
        f"{torch.get_num_interop_threads()} inter-op"
    )

    device = config['device']
    device_type = device.split(":")[0]

    use_amp = config.get('use_amp', False) and device_type == 'cuda'
    amp_dtype = torch.float16 if config.get('amp_dtype', 'float16') == 'float16' else torch.bfloat16
    grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))

    if device_type == "cuda":
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    

    print(f"Reading data from {config['train_file']}...")
    all_atoms = read(config['train_file'], index=":")
    stress_labels = sum(stress_target_from_atoms(atoms)[1] for atoms in all_atoms)
    print(f"Stress labels: {stress_labels}/{len(all_atoms)} structures")
    configured_stress_weight = max(
        float(config.get('stress_weight', 0.0)),
        float(config.get('stress_weight_final', config.get('stress_weight', 0.0))),
    )
    if configured_stress_weight > 0.0 and stress_labels == 0:
        raise ValueError("stress_weight is nonzero, but the dataset contains no stress labels")

    if config['valid_file']:
        val_atoms = read(config['valid_file'], index=":")
        train_atoms = all_atoms
    else:
        split_mode = config.get('split_mode', 'blocked')
        train_atoms, val_atoms, excluded_frames = split_trajectory_frames(
            all_atoms,
            val_fraction=config.get('val_split', 0.1),
            seed=seed,
            mode=split_mode,
            block_size=config.get('split_block_size', 25),
            gap=config.get('split_gap', 0),
        )
        print(
            f"{str(split_mode).title()} split: {len(train_atoms)} Training | "
            f"{len(val_atoms)} Validation | {len(excluded_frames)} Gap-excluded"
        )

    atomic_energy_map = parse_atomic_energy_table(config.get('atomic_energies'))

    if not atomic_energy_map and config.get('solve_atomic_energies', False):
        atomic_energy_map = compute_atomic_energies_from_dataset(train_atoms)
        if atomic_energy_map:
            pretty = ", ".join(
                f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
            )
            print(f"Solved per-species reference energies from training set -> {pretty}")

    if atomic_energy_map:
        energy_shift_per_atom = None
        config['atomic_energies'] = atomic_energy_map
        pretty = ", ".join(
            f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
        )
        print(f"Using per-species reference energies for normalization: {pretty}")
    elif config.get('energy_shift_per_atom') is not None:
        energy_shift_per_atom = float(config['energy_shift_per_atom'])
        print(f"Using user-provided energy shift per atom: {energy_shift_per_atom:.6f} eV")
    else:
        energy_shift_per_atom = compute_mean_energy_per_atom(train_atoms)
        print(f"Computed mean energy per atom for normalization: {energy_shift_per_atom:.6f} eV")

    # DATALOADERS
    train_ds = AtomisticDataset(
        train_atoms,
        config['r_max'],
        random_rotation=config.get('random_rotation', False),
        precompute_neighbors=config.get('precompute_neighbors', False),
        dtype=default_dtype,
        electrostatics=electrostatics is not None,
        spin=electrostatics is not None and electrostatics.spin,
        spin_key='multiplicity' if electrostatics is None else electrostatics.spin_key,
        partial_charge_key=(
            DEFAULT_PARTIAL_CHARGE_KEY if electrostatics is None else electrostatics.partial_charge_key
        ),
    )
    val_ds = AtomisticDataset(
        val_atoms,
        config['r_max'],
        random_rotation=False,
        precompute_neighbors=config.get('precompute_neighbors', False),
        dtype=default_dtype,
        electrostatics=electrostatics is not None,
        spin=electrostatics is not None and electrostatics.spin,
        spin_key='multiplicity' if electrostatics is None else electrostatics.spin_key,
        partial_charge_key=(
            DEFAULT_PARTIAL_CHARGE_KEY if electrostatics is None else electrostatics.partial_charge_key
        ),
    )
    if electrostatics is not None:
        counts = train_ds.electrostatic_label_counts()
        print(
            f"Electrostatic labels (training): net-charged {counts['charged']}, dipole "
            f"{counts['dipole']}, polarizability {counts['polarizability']}, Born charges "
            f"{counts['born_charges']}, dielectric tensor {counts['dielectric_tensor']}, partial "
            f"charges {counts['partial_charges']} of {len(train_ds)} structures"
        )
        if counts['open_shell'] and not electrostatics.spin:
            print(
                f"NOTE: {counts['open_shell']} training structure(s) are open-shell "
                f"(atoms.info['{electrostatics.spin_key}'] > 1) but the heads are "
                "closed-shell: two spin states at one geometry get one energy. Use "
                "electrostatics.heads.spin: true, or train on one spin state."
            )
        ignored_eps = (counts['ignored_dielectric_tensor']
                       + val_ds.electrostatic_label_counts()['ignored_dielectric_tensor'])
        if ignored_eps:
            print(
                f"NOTE: {ignored_eps} structure(s) that are not 3-D periodic carry a "
                "dielectric_tensor label; it is ignored (eps_inf is a bulk property)."
            )
    priors = {}                     # kind -> (settings block, items)
    for kind in RESPONSE_PRIORS:
        block = None if electrostatics is None else getattr(electrostatics, f'{kind}_prior')
        if block is None:
            continue
        if data_parallel is not None:
            raise ValueError(
                f"electrostatics.{kind}_prior under data parallelism is not implemented "
                "(its gradient would be added once per rank)"
            )
        priors[kind] = (block, _response_prior_items(
            block, kind, config['r_max'], default_dtype, electrostatics, config.get('precompute_neighbors', False),
        ))
        print(f"{kind.capitalize()} prior: {len(priors[kind][1])} structure(s) from {block['file']}, weight {block['weight']}")
        for name in ('dipole', 'polarizability'):
            ignored = counts[f'ignored_{name}'] + val_ds.electrostatic_label_counts()[f'ignored_{name}']
            if ignored:
                print(
                    f"NOTE: {ignored} periodic structure(s) carry a {name} label; it is "
                    "ignored, because the polarization of a periodic cell is defined only "
                    "modulo a quantum."
                )
        for name, _unit in ELECTROSTATIC_OBSERVABLES:
            if electrostatics.weight(name) > 0.0 and counts[name] == 0:
                raise ValueError(
                    f"the electrostatics {name} loss weight is nonzero, but no training "
                    f"structure carries a usable {name} label"
                )

    num_workers = int(config.get('num_workers', 2))
    pin_memory = device_type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              generator=torch.Generator().manual_seed(seed))
    
    valid_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, num_workers=num_workers)

    print("--- Initializing Transformers-ACE ---")
    architecture_version = int(config.get('architecture_version', 2))
    if architecture_version == 2:
        model_class = TransformersACE
    elif architecture_version == 3:
        model_class = TransformersACEV3
    elif architecture_version == 4:
        model_class = TransformersACEV4
    else:
        raise ValueError("architecture_version must be 2, 3, or 4 for training")

    model_kwargs = dict(
        r_max=config['r_max'], l_max=config['l_max'], num_radial=config['num_radial'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        radial_basis_type=config.get('radial_basis_type', 'bessel'),
        radial_trainable=config.get('radial_trainable', False),
        gaussian_width=config.get('gaussian_width', 0.5),
        attention_num_heads=config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
        attention_key_dim=config.get('attention_key_dim', None),
        attention_ffn_hidden=config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
        attention_dropout=config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
        attention_layer_scale_init=config.get('attention_layer_scale_init', 1e-2),
        attention_distance_penalty=config.get('attention_distance_penalty', True),
        attention_mode=config.get('attention_mode', 'full'),
        radial_mlp_hidden=config.get('radial_mlp_hidden', 32),
        radial_mlp_layers=config.get('radial_mlp_layers', 2),
        correlation_order=config.get('correlation_order', 4),
        correlation_channels=config.get('correlation_channels', 16),
        use_aux_force_head=False,
        use_aux_stress_head=False,
    )
    if architecture_version == 4:
        model_kwargs.update(
            attention_num_shells=config.get('attention_num_shells', 4),
            correlation_rank_initial=config.get('correlation_rank_initial', None),
        )
    model = model_class(**model_kwargs)
    electrostatics_meta = None
    if electrostatics is not None:
        # Attached before .to(device) so the heads move with the model, and
        # before the optimizer is built so their parameters are optimised.
        module = attach_electrostatics_from_settings(model, electrostatics)
        electrostatics_meta = electrostatics_checkpoint_metadata(electrostatics, module)
        print(
            f"Electrostatic heads on {module.heads.irreps_in}, bond cutoff "
            f"{module.heads.r_max} A (model cutoff {model.r_max} A)"
        )
    model = model.to(device)
    
    optimizer = build_optimizer(model, config)
    print(f"Optimizer: {config.get('optimizer', 'adam')}")
    print(f"Optimizer groups: {optimizer_group_summary(optimizer)}")
    warmup_epochs = max(0, int(config.get('lr_warmup_epochs', 0)))
    warmup_start = float(config.get('lr_warmup_start_factor', 0.1))
    scheduler_interval = str(config.get('lr_scheduler_interval', 'epoch')).lower()
    if warmup_start <= 0.0:
        raise ValueError("lr_warmup_start_factor must be > 0.0")
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError("lr_scheduler_interval must be 'epoch' or 'step'")

    steps_per_epoch = max(1, len(train_loader))
    configured_t_max = int(config.get('lr_scheduler_t_max', config['epochs']))
    if scheduler_interval == 'step':
        warmup_iters = warmup_epochs * steps_per_epoch
        total_iters = max(warmup_iters + 1, configured_t_max * steps_per_epoch)
    else:
        warmup_iters = warmup_epochs
        total_iters = max(warmup_epochs + 1, configured_t_max)
    cosine_t_max = max(1, total_iters - warmup_iters)

    use_restarts = bool(config.get('lr_scheduler_use_restarts', False))
    restart_mult = float(config.get('lr_restart_mult', 1.0))
    if use_restarts:
        cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t_max,
            T_mult=max(1.0, restart_mult),
            eta_min=config.get('lr_min', 0.0),
        )
    else:
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=config.get('lr_min', 0.0),
        )
    if warmup_iters > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start,
            total_iters=warmup_iters,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
        )
    else:
        scheduler = cosine

    # Optional force-weight annealing to ease optimization toward energies.
    force_w_start = float(config.get('forces_weight', 10.0))
    force_w_final = float(config.get('forces_weight_final', force_w_start))
    force_w_decay_epochs = int(config.get('forces_weight_decay_epochs', 0))

    def _force_weight(epoch_idx: int) -> float:
        if force_w_decay_epochs <= 0 or force_w_start == force_w_final:
            return force_w_start
        frac = min(1.0, epoch_idx / max(1, force_w_decay_epochs))
        return force_w_start + frac * (force_w_final - force_w_start)

    stress_w_start = float(config.get('stress_weight', 0.0))
    stress_w_final = float(config.get('stress_weight_final', stress_w_start))
    stress_w_ramp_epochs = int(config.get('stress_weight_ramp_epochs', 0))

    def _stress_weight(epoch_idx: int) -> float:
        if stress_w_ramp_epochs <= 0 or stress_w_start == stress_w_final:
            return stress_w_start
        frac = min(1.0, epoch_idx / max(1, stress_w_ramp_epochs))
        return stress_w_start + frac * (stress_w_final - stress_w_start)

    resume_path = config.get('resume_from')
    start_epoch = 0

    scaler = _make_grad_scaler(use_amp)

    if resume_path:
        print(f"--- Loading checkpoint from {resume_path} ---")
        checkpoint = load_checkpoint(resume_path, map_location=device)
        checkpoint_version = int(checkpoint.get('config', {}).get('architecture_version', 1))
        if checkpoint_version != architecture_version:
            raise ValueError(
                f"Cannot resume architecture v{checkpoint_version} weights in the "
                f"v{architecture_version} model. Start a matching architecture run; "
                "the calculator can still evaluate older checkpoints."
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        # The resumed model is built from this run's YAML, not the checkpoint's stored RedoxConfig: say so
        # when the two state different energies (a checkpoint older than a field trained its legacy value).
        saved_redox = ((checkpoint.get('config') or {}).get('electrostatics') or {}).get('redox')
        run_redox = getattr(getattr(model, 'electrostatics', None), 'redox', None)
        if saved_redox is not None and run_redox is not None:
            from trace_redox.module import statement_changes
            for name, (trained, now) in statement_changes(saved_redox, run_redox).items():
                print(f"WARNING: resume_from trained redox.{name} = {trained!r}; this run uses {now!r}. "
                      f"The weights continue under a different energy (set redox.{name}: {trained} to keep it).")

        if config.get('resume_load_optimizer', False) and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if config.get('resume_load_scheduler', False) and checkpoint.get('scheduler_state_dict'):
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if config.get('resume_load_scaler', False) and checkpoint.get('scaler_state_dict') and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if config.get('use_checkpoint_energy_shift', True):
            ckpt_atomic = checkpoint.get('config', {}).get('atomic_energies') or {}
            if ckpt_atomic:
                atomic_energy_map = {int(k): float(v) for k, v in ckpt_atomic.items()}
                energy_shift_per_atom = None
                config['atomic_energies'] = atomic_energy_map
                pretty = ", ".join(
                    f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
                )
                print(f"Using checkpoint atomic energy references: {pretty}")
            else:
                ckpt_shift = checkpoint.get('config', {}).get('energy_shift_per_atom')
                if ckpt_shift is not None:
                    energy_shift_per_atom = ckpt_shift
                    print(f"Using checkpoint energy shift per atom: {energy_shift_per_atom:.6f} eV")

        start_epoch = int(checkpoint.get('epoch', 0))
        print(f"Resuming training from epoch {start_epoch}")

    if config.get('atomic_energies_from_model', False):
        atomic_energy_map = atomic_energies_from_model(model, train_ds, device)
        energy_shift_per_atom = None
        config['atomic_energies'] = atomic_energy_map
        pretty = ", ".join(f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items()))
        print(f"Per-species references solved on E_DFT - E_model(initial): {pretty}")

    energy_shift = None
    if energy_shift_per_atom is not None:
        energy_shift = torch.tensor(energy_shift_per_atom, dtype=default_dtype, device=device)

    atomic_energy_vec = atomic_energy_tensor(atomic_energy_map, device, dtype=default_dtype)

    def baseline_energy(z_tensor):
        if atomic_energy_vec is not None:
            if torch.max(z_tensor).item() >= atomic_energy_vec.shape[0]:
                raise ValueError("Encountered atomic number without reference energy")
            return torch.sum(atomic_energy_vec[z_tensor])
        elif energy_shift is not None:
            return energy_shift * len(z_tensor)
        else:
            return torch.tensor(0.0, device=device)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_energy_rmse': [],
        'val_energy_rmse': [],
        'train_energy_rmse_offset_free': [],
        'val_energy_rmse_offset_free': [],
        'train_energy_bias': [],
        'val_energy_bias': [],
        'train_force_rmse': [],
        'val_force_rmse': [],
        'train_stress_rmse': [],
        'val_stress_rmse': [],
    }
    electrostatic_observables = electrostatics is not None and electrostatics.observable_losses
    if electrostatic_observables:
        # RMSE per observable in physical units (e*A, A^3, e); nan where nothing
        # was evaluated. Only for runs that train on them, so every other run's
        # history -- and its CSV -- is unchanged.
        for name, _unit in ELECTROSTATIC_OBSERVABLES:
            history[f'train_{name}_rmse'] = []
            history[f'val_{name}_rmse'] = []
    for kind in priors:
        history[f'train_{kind}_prior_rmse'] = []
        history[f'val_{kind}_prior_rmse'] = []
    prior_stats = {kind: {'sse': 0.0, 'count': 0} for kind in priors}

    ckpt_interval = int(config.get('checkpoint_interval', 0) or 0)

    # Attention-temperature curriculum.
    #
    # T_att must be a deterministic function of the epoch alone and must reach
    # exactly 1 before training ends, because validation, the ASE calculator and
    # every deployed path evaluate the model at T_att = 1.  Any residual
    # T_att != 1 at the final epoch means the optimizer is shaping a *different*
    # function from the one that is later deployed.
    temp_scale_start = float(config.get('temperature_scale_start', 1.0))
    temp_scale_end = float(config.get('temperature_scale_end', 1.0))
    temp_scale_epochs = int(config.get('temperature_scale_epochs', 0) or 0)

    if abs(temp_scale_end - 1.0) > 1e-12:
        raise ValueError(
            "temperature_scale_end must be 1.0: validation and inference evaluate the "
            f"model at T_att = 1, but the schedule ends at {temp_scale_end}. Training "
            "would optimize a different function from the one deployed."
        )
    if temp_scale_epochs > 0 and temp_scale_epochs >= int(config['epochs']):
        raise ValueError(
            f"temperature_scale_epochs ({temp_scale_epochs}) must be smaller than "
            f"epochs ({int(config['epochs'])}) so that T_att reaches 1 before training ends."
        )
    # The force-loss feedback term made T_att data-dependent and left it at
    # ~4.4 at convergence while validation used 1.0.  It is no longer supported.
    for _legacy_key in ('temperature_force_ref', 'temperature_force_exponent'):
        if float(config.get(_legacy_key, 0.0) or 0.0) != 0.0:
            raise ValueError(
                f"'{_legacy_key}' is no longer supported: it made the attention "
                "temperature depend on the running force loss, so training and "
                "inference used different temperatures. Remove it from the config "
                "(the epoch schedule already anneals T_att to 1)."
            )

    def _temperature_scale(epoch_idx: int) -> float:
        """T_att as a pure function of epoch, annealing to exactly 1."""
        if temp_scale_epochs <= 0:
            return 1.0
        frac = min(1.0, epoch_idx / float(temp_scale_epochs))
        return temp_scale_start + frac * (temp_scale_end - temp_scale_start)

    # Evaluate a whole batch as one disconnected graph. The GPU is launch-bound
    # for single small cells, so this is a large throughput win; it is
    # numerically identical to the per-structure path (tests/test_batched_graph.py).
    batched_graphs = bool(config.get('batched_graphs', False))
    force_consistency_weight = float(config.get('force_consistency_weight', 0.0))
    if batched_graphs and force_consistency_weight > 0.0:
        raise ValueError(
            "batched_graphs does not support force_consistency_weight; the batched "
            "path always differentiates the batched position tensor."
        )
    displacement_prob = float(config.get('displacement_prob', 0.0))
    displacement_sigma = float(config.get('displacement_sigma', 0.0))
    if displacement_prob > 0.0:
        raise ValueError(
            "displacement augmentation cannot reuse the original DFT labels; "
            "set displacement_prob and displacement_sigma to 0"
        )
    aux_force_weight = float(config.get('aux_force_weight', 0.0))
    aux_stress_weight = float(config.get('aux_stress_weight', 0.0))
    sobolev_weight = float(config.get('sobolev_weight', 0.0))
    sobolev_sigma = float(config.get('sobolev_sigma', 0.0))

    early_stopping_patience = max(0, int(config.get('early_stopping_patience', 0)))
    early_stopping_min_epoch = max(
        stress_w_ramp_epochs,
        int(config.get('early_stopping_min_epoch', stress_w_ramp_epochs)),
    )
    early_stopping_min_delta = float(config.get('early_stopping_min_delta', 0.0))
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_saved = False
    epochs_completed = start_epoch
    
    print(
        f"{'Epoch':>5} | {'Loss':>10} | {'E (meV)':>10} | {'E_free':>9} | {'force_RMSE':>12} | {'force_MSE':>12} | {'force_MAE':>12} | {'S_RMSE':>10} || "
        f"{'Val Loss':>10} | {'Val E':>10} | {'Val E_free':>10} | {'Val force_RMSE':>16} | {'Val force_MSE':>16} | {'Val S_RMSE':>12}"
    )
    print("-" * 170)
    
    force_loss_ema = None
    trainable = [p for p in model.parameters() if p.requires_grad]
    _save_checkpoint = save_checkpoint if rank == 0 else (lambda *args, **kwargs: None)

    def _optimizer_step():
        """Clip, step and clear; a data-parallel run first sums the window's batch gradients.

        The response priors, if any, add their gradients once per step, before clipping.
        """
        for kind, (block, items_k) in priors.items():
            prior_loss, prior_errors = _response_prior_loss(
                model, items_k, kind, device, electrostatics, _temperature_scale(epoch), training=True,
            )
            scaler.scale(block['weight'] * prior_loss).backward()
            prior_stats[kind]['sse'] += float(sum(float(e.pow(2).sum()) for e in prior_errors))
            prior_stats[kind]['count'] += sum(e.numel() for e in prior_errors)
        if data_parallel is not None:
            data_parallel.sum_gradients(trainable)
        if config.get('clip_grad_norm', None):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        if data_parallel is not None:
            data_parallel.broadcast_state(model)
        optimizer.zero_grad(set_to_none=True)
        if scheduler_interval == 'step':
            scheduler.step()

    if data_parallel is not None:
        data_parallel.broadcast_state(model)
    for epoch in range(start_epoch, config['epochs']):
        if architecture_version == 4:
            max_rank = int(model.ace.irreps_correlation[0].mul)
            initial_rank = int(config.get('correlation_rank_initial') or max_rank)
            rank_warmup = max(0, int(config.get('correlation_rank_warmup_epochs', 0)))
            if rank_warmup > 0:
                fraction = min(1.0, float(epoch + 1) / float(rank_warmup))
                active_rank = round(initial_rank + fraction * (max_rank - initial_rank))
            else:
                active_rank = max_rank
            model.set_correlation_rank(active_rank)
        force_weight = _force_weight(epoch)
        stress_weight = _stress_weight(epoch)
        if redox_kT_start is not None:
            redox_module = getattr(model, 'electrostatics')
            if epoch == start_epoch:
                deployed_redox = redox_module.redox
            redox_module.redox = dataclasses.replace(
                deployed_redox, kT=redox_kT(epoch, redox_kT_start, deployed_redox.kT, redox_kT_epochs))
        model.train()
        train_metrics = MetricTracker()
        train_elec_metrics = ElectrostaticMetricTracker()
        for stats in prior_stats.values():
            stats['sse'], stats['count'] = 0.0, 0
        total_loss = 0.0
        total_items_seen = 0

        batch_idx = -1
        optimizer.zero_grad(set_to_none=True)

        batches = _seeded_batches(train_loader, seed, epoch) if batch_seeded_rng else enumerate(train_loader)
        train_records = []
        for batch_idx, batch in batches:
            batch_loss = 0.0

            # --- GRADIENT ACCUMULATION (FP32/AMP) ---
            items = list(batch)

            if batched_graphs:
                if data_parallel is None or data_parallel.owns(batch_idx % grad_accum_steps):
                    # A data-parallel rank records its metric updates, replayed in batch order below.
                    recorders = None if data_parallel is None else (_CallRecorder(), _CallRecorder())
                    batch_loss = _batched_train_step(
                        model, items, device, _temperature_scale(epoch),
                        config['energy_weight'], force_weight, stress_weight,
                        sobolev_weight, sobolev_sigma,
                        baseline_energy, train_metrics if recorders is None else recorders[0], scaler,
                        len(items) * grad_accum_steps,
                        electrostatics=electrostatics,
                        electrostatic_metrics=train_elec_metrics if recorders is None else recorders[1],
                    )
                    if recorders is None:
                        total_items_seen += len(items)
                        total_loss = total_loss + batch_loss
                    else:
                        data_parallel.keep_gradients(batch_idx, trainable)
                        train_records.append(
                            (batch_idx, (batch_loss, len(items), recorders[0].calls, recorders[1].calls))
                        )
                if (batch_idx + 1) % grad_accum_steps == 0:
                    _optimizer_step()
                continue

            for item in items:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)
                if force_consistency_weight > 0.0:
                    item['pos'] = item['pos'].clone().detach().requires_grad_(True)

                # Standard Forward with optional AMP
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    temp_scale = _temperature_scale(epoch)
                    stress_target = (
                        bool(item['has_stress'].item())
                        and stress_weight > 0.0
                    )
                    # The local-linearisation term (Eq. 41) compares two forward
                    # passes.  With independent attention-dropout masks their
                    # difference is dominated by dropout variance rather than by
                    # curvature, so the RNG state is captured here and replayed for
                    # the second pass, giving both passes identical masks.
                    sobolev_active = sobolev_weight > 0.0 and sobolev_sigma > 0.0
                    rng_state_cpu = torch.get_rng_state() if sobolev_active else None
                    rng_state_dev = (
                        torch.cuda.get_rng_state(device)
                        if sobolev_active and device_type == 'cuda'
                        else None
                    )
                    p_E, p_F, p_S, aux = model(
                        item,
                        training=True,
                        temperature_scale=temp_scale,
                        detach_pos=force_consistency_weight <= 0.0,
                        compute_stress=stress_target,
                        return_electrostatics=electrostatic_observables,
                    )
                    n_ats = len(item['z'])

                    target_E = item['t_E'] - baseline_energy(item['z'])
                    has_e = bool(item.get('has_energy', True))
                    loss_e = ((p_E - target_E) / n_ats)**2 * float(has_e)
                    loss_f = torch.mean((p_F - item['t_F'])**2) * float(item.get('force_weight', 1.0))
                    loss_s = torch.tensor(0.0, device=device)
                    if stress_target:
                        loss_s = torch.mean(
                            (stress_to_voigt(p_S) - stress_to_voigt(item['t_S']))**2
                        )

                    loss_item = (config['energy_weight'] * loss_e) + \
                                (force_weight * loss_f) + \
                                (stress_weight * loss_s)

                    if electrostatic_observables:
                        elec_errors = _electrostatic_errors(
                            aux['electrostatics'], [item], electrostatics
                        )
                        loss_item = loss_item + _electrostatic_loss(
                            elec_errors, [item], electrostatics, p_E
                        )[0]
                        train_elec_metrics.update(elec_errors)

                    if aux_force_weight > 0.0 and 'force' in aux:
                        loss_item = loss_item + aux_force_weight * torch.mean((aux['force'] - item['t_F'])**2)
                    if aux_stress_weight > 0.0 and 'stress' in aux:
                        target_stress = item['t_S']
                        loss_item = loss_item + aux_stress_weight * torch.mean(
                            (aux['stress'] - torch.stack([
                                target_stress[0,0], target_stress[1,1], target_stress[2,2],
                                target_stress[0,1], target_stress[0,2], target_stress[1,2]
                            ]))**2
                        )

                    if sobolev_active:
                        # Draw the displacement first, then rewind the RNG so the
                        # perturbed pass replays exactly the dropout masks used above.
                        delta = torch.randn_like(item['pos']) * sobolev_sigma
                        pos_pert = item['pos'] + delta
                        perturbed = {**item, 'pos': pos_pert}
                        torch.set_rng_state(rng_state_cpu)
                        if rng_state_dev is not None:
                            torch.cuda.set_rng_state(rng_state_dev, device)
                        p_E_pert, _, _, _ = model(
                            perturbed,
                            training=True,
                            temperature_scale=temp_scale,
                            detach_pos=True,
                            compute_stress=False,
                        )
                        # E(r+d) - E(r) + F.d  vanishes to first order; with matched
                        # masks this measures curvature, not dropout variance.
                        fd = (p_E_pert - p_E) + (p_F.detach() * delta).sum()
                        loss_item = loss_item + sobolev_weight * fd.pow(2)

                    if force_consistency_weight > 0.0:
                        energy_grad = torch.autograd.grad(
                            p_E,
                            item['pos'],
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if energy_grad is not None:
                            consistency = (p_F + energy_grad).pow(2).mean()
                            loss_item = loss_item + force_consistency_weight * consistency

                # Normalize and Backward
                seed_loss = _electrostatic_seed_loss(model)
                loss_batch = (
                    loss_item if seed_loss is None else loss_item + seed_loss
                ) / (len(items) * grad_accum_steps)
                scaler.scale(loss_batch).backward()

                batch_loss += loss_item.item()

                with torch.no_grad():
                    # Pass the *shifted* energies. The composition baseline cancels
                    # in the difference, so this is analytically identical to
                    # comparing absolute energies but avoids differencing two
                    # O(10^3 eV) float32 numbers to extract an O(10^-1 eV) error.
                    train_metrics.update(
                        p_E, p_F, p_S, target_E, item['t_F'], item['t_S'],
                        stress_target, n_ats, has_energy=has_e,
                    )
                    if force_loss_ema is None:
                        force_loss_ema = loss_f.detach()
                    else:
                        force_loss_ema = 0.9 * force_loss_ema + 0.1 * loss_f.detach()
                total_items_seen += 1

            if (batch_idx + 1) % grad_accum_steps == 0:
                _optimizer_step()
            total_loss += batch_loss

        # Flush any residual gradients if the last batch didn't trigger a step
        if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
            _optimizer_step()
        if data_parallel is not None:
            for _batch, (loss, count, calls, elec_calls) in data_parallel.gather(train_records):
                total_loss = total_loss + loss
                total_items_seen += count
                _CallRecorder.replay(calls, train_metrics)
                _CallRecorder.replay(elec_calls, train_elec_metrics)

        avg_train_loss = total_loss / max(1, total_items_seen)
        if torch.is_tensor(avg_train_loss):
            avg_train_loss = float(avg_train_loss.detach().cpu())
        tr_e, tr_f, tr_s, tr_f_mse, tr_f_mae, tr_e_free, tr_e_bias = train_metrics.get_metrics()
        if redox_kT_start is not None:
            redox_module.redox = deployed_redox             # validation, checkpoints, deployment: the deployed kT
        # Validation
        model.eval()
        val_metrics = MetricTracker()
        val_elec_metrics = ElectrostaticMetricTracker()
        val_loss_accum = 0.0
        val_records = []

        for val_index, batch in enumerate(valid_loader):
            if batched_graphs:
                if data_parallel is None:
                    batch_val_metrics, batch_val_elec_metrics = val_metrics, val_elec_metrics
                elif data_parallel.owns(val_index):
                    batch_val_metrics, batch_val_elec_metrics = _CallRecorder(), _CallRecorder()
                else:
                    continue
                items = list(batch)
                bat = AtomisticDataset.collate_batched(items)
                for k, v in bat.items():
                    if isinstance(v, torch.Tensor):
                        bat[k] = v.to(device, non_blocking=True)
                n_at = bat['n_atoms'].to(device).to(bat['pos'].dtype)
                has_s = bat['has_stress'].to(device).to(bat['pos'].dtype)
                want_s = bool(has_s.any().item()) and stress_weight > 0.0
                # Validation and deployment both evaluate at T_att = 1 (D2).
                p_E, p_F, p_S, v_aux = model.forward_batched(
                    bat, training=False, compute_stress=want_s,
                    return_electrostatics=electrostatic_observables,
                )
                base = torch.stack(
                    [baseline_energy(it['z'].to(device)) for it in items]
                ).reshape(-1)
                target_E = bat['t_E'].reshape(-1) - base
                has_e = bat['has_energy'].to(device).to(bat['pos'].dtype)
                loss_e = energy_loss(p_E, target_E, n_at, has_e)
                dF2 = (p_F - bat['t_F']) ** 2
                per_f = torch.zeros_like(p_E).index_add(0, bat['batch'], dF2.sum(dim=1))
                loss_f = force_loss(per_f, n_at, bat['force_weight'].to(device).to(p_E.dtype))
                vloss = config['energy_weight'] * loss_e + force_weight * loss_f
                if want_s:
                    dS2 = ((p_S - bat['t_S']) ** 2).flatten(1)
                    voigt = dS2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)
                    vloss = vloss + stress_weight * (has_s * voigt / 6.0).sum()
                if electrostatic_observables:
                    elec_errors = _electrostatic_errors(v_aux['electrostatics'], items, electrostatics)
                    vloss = vloss + _electrostatic_loss(elec_errors, items, electrostatics, p_E).sum()
                    batch_val_elec_metrics.update(elec_errors)
                if data_parallel is None:
                    val_loss_accum += vloss.detach()
                S_all = p_S.detach() if want_s else torch.zeros(
                    len(items), 3, 3, device=device, dtype=bat['pos'].dtype
                )
                batch_val_metrics.update_batched(
                    p_E, p_F, S_all, target_E, bat['t_F'], bat['t_S'],
                    has_s if want_s else torch.zeros_like(has_s),
                    bat['batch'], bat['n_atoms'].to(device), has_energy=bat['has_energy'].to(device),
                )
                if data_parallel is not None:
                    val_records.append(
                        (val_index, (vloss.detach(), batch_val_metrics.calls, batch_val_elec_metrics.calls))
                    )
                continue
            for item in batch:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)

                # Keep grad tracking on so autograd can form forces/stresses; we
                # still avoid higher-order graphs with ``create_graph=False``
                # inside the model during validation.
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    stress_target = (
                        bool(item['has_stress'].item())
                        and stress_weight > 0.0
                    )
                    p_E, p_F, p_S, v_aux = model(
                        item,
                        training=False,
                        compute_stress=stress_target,
                        return_electrostatics=electrostatic_observables,
                    )
                    n_ats = len(item['z'])
                    target_E = item['t_E'] - baseline_energy(item['z'])
                    has_e = bool(item.get('has_energy', True))
                    loss_e = ((p_E - target_E) / n_ats)**2 * float(has_e)
                    loss_f = torch.mean((p_F - item['t_F'])**2) * float(item.get('force_weight', 1.0))
                    loss_s = torch.tensor(0.0, device=device)
                    if stress_target:
                        loss_s = torch.mean(
                            (stress_to_voigt(p_S) - stress_to_voigt(item['t_S']))**2
                        )
                    val_loss_accum += (
                        (config['energy_weight'] * loss_e)
                        + (force_weight * loss_f)
                        + (stress_weight * loss_s)
                    )
                    if electrostatic_observables:
                        elec_errors = _electrostatic_errors(
                            v_aux['electrostatics'], [item], electrostatics
                        )
                        val_loss_accum += _electrostatic_loss(
                            elec_errors, [item], electrostatics, p_E
                        )[0].detach()
                        val_elec_metrics.update(elec_errors)

                val_metrics.update(
                    p_E, p_F, p_S, target_E, item['t_F'], item['t_S'],
                    stress_target, n_ats, has_energy=has_e,
                )

        if data_parallel is not None:
            for _batch, (vloss, calls, elec_calls) in data_parallel.gather(val_records):
                val_loss_accum += vloss
                _CallRecorder.replay(calls, val_metrics)
                _CallRecorder.replay(elec_calls, val_elec_metrics)
            data_parallel.check_identical(list(model.parameters()) + list(model.buffers()))
        avg_val_loss = val_loss_accum / len(val_atoms)
        val_e, val_f, val_s, val_f_mse, val_f_mae, val_e_free, val_e_bias = val_metrics.get_metrics()
        avg_val_loss = float(avg_val_loss.detach().cpu())
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(avg_val_loss)
        history['train_energy_rmse'].append(float(tr_e))
        history['val_energy_rmse'].append(float(val_e))
        history['train_energy_rmse_offset_free'].append(float(tr_e_free))
        history['val_energy_rmse_offset_free'].append(float(val_e_free))
        history['train_energy_bias'].append(float(tr_e_bias))
        history['val_energy_bias'].append(float(val_e_bias))
        history['train_force_rmse'].append(float(tr_f))
        history['val_force_rmse'].append(float(val_f))
        history['train_stress_rmse'].append(float(tr_s))
        history['val_stress_rmse'].append(float(val_s))
        if electrostatic_observables:
            tr_elec, val_elec = train_elec_metrics.rmse(), val_elec_metrics.rmse()
            for name, _unit in ELECTROSTATIC_OBSERVABLES:
                history[f'train_{name}_rmse'].append(tr_elec[name])
                history[f'val_{name}_rmse'].append(val_elec[name])
        for kind, (block, items_k) in priors.items():
            # train: over the epoch's optimizer steps (train mode); val: now, in eval mode at T_att = 1
            _, val_prior_errors = _response_prior_loss(
                model, items_k, kind, device, electrostatics, 1.0, training=False,
            )
            stats = prior_stats[kind]
            tr_prior = math.sqrt(stats['sse'] / stats['count']) if stats['count'] else float('nan')
            val_prior = math.sqrt(
                sum(float(e.pow(2).sum()) for e in val_prior_errors) / sum(e.numel() for e in val_prior_errors)
            )
            history[f'train_{kind}_prior_rmse'].append(tr_prior)
            history[f'val_{kind}_prior_rmse'].append(val_prior)
            target = RESPONSE_PRIORS[kind][1]
            now = [
                round(float((e + torch.diagonal(item[target].to(e))).mean() if e.dim() == 1
                            else torch.diagonal(e + item[target].to(e)).mean()), 4)
                for e, item in zip(val_prior_errors, items_k)
            ]
            print(f"      {kind} prior RMSE train/val: {tr_prior:.4g}/{val_prior:.4g}; mean diagonal (eval) {now}")
        if scheduler_interval == 'epoch':
            scheduler.step()

        print(
            f"{epoch+1:5d} | "
            f"{avg_train_loss:10.4f} | {tr_e:10.2f} | {tr_e_free:9.2f} | {tr_f:12.6f} | {tr_f_mse:12.6f} | {tr_f_mae:12.6f} | {tr_s:10.4f} || "
            f"{avg_val_loss:10.4f} | {val_e:10.2f} | {val_e_free:10.2f} | {val_f:16.6f} | {val_f_mse:16.6f} | {val_s:12.6f}"
        )
        if electrostatic_observables:
            print(
                "      electrostatics RMSE train/val: "
                + " | ".join(
                    f"{name} {tr_elec[name]:.4g}/{val_elec[name]:.4g} {unit}"
                    for name, unit in ELECTROSTATIC_OBSERVABLES
                )
            )

        epochs_completed = epoch + 1
        if epoch + 1 >= early_stopping_min_epoch:
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                _save_checkpoint(
                    config['model_save_path'],
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    energy_shift_per_atom,
                    atomic_energy_map,
                    electrostatics=electrostatics_meta,
                )
                best_checkpoint_saved = True
                print(f"New best validation loss at epoch {best_epoch}: {best_val_loss:.6f}")
            else:
                epochs_without_improvement += 1

        if ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0:
            ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            _save_checkpoint(
                ckpt_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                energy_shift_per_atom,
                atomic_energy_map,
                electrostatics=electrostatics_meta,
            )

        if (
            early_stopping_patience > 0
            and epoch + 1 >= early_stopping_min_epoch
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}; best epoch was "
                f"{best_epoch} with validation loss {best_val_loss:.6f}."
            )
            break

    if not best_checkpoint_saved:
        _save_checkpoint(
            config['model_save_path'],
            epochs_completed,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            energy_shift_per_atom,
            atomic_energy_map,
            electrostatics=electrostatics_meta,
        )
        best_epoch = epochs_completed

    if config.get('save_last_checkpoint', True):
        root, extension = os.path.splitext(config['model_save_path'])
        last_path = config.get('last_model_save_path') or f"{root}_last{extension or '.pt'}"
        _save_checkpoint(
            last_path,
            epochs_completed,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            energy_shift_per_atom,
            atomic_energy_map,
            electrostatics=electrostatics_meta,
        )
    print(
        f"Training finished. Best checkpoint: {config['model_save_path']} "
        f"(epoch {best_epoch})."
    )

    if config.get('plot_training_curves', True) and rank == 0:
        try:
            plot_metric_history(history, save_dir=config.get('plot_dir', 'plots'))
        except Exception as error:
            print(f"[PLOTTING] Could not create training curves: {error}")
    if owns_process_group:
        dist.destroy_process_group()
    return history

if __name__ == "__main__": main()
