import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from .charge_state import (
    charge_state_changed,
    charge_state_from_atoms,
    charge_state_tensors,
    electrostatic_environment,
)
from .checkpoint import attach_checkpoint_electrostatics, load_checkpoint
from .model import LegacyTransformersACE, TransformersACE, TransformersACEV3, TransformersACEV4


def _load_checkpoint(path, map_location):
    return load_checkpoint(path, map_location=map_location)

class TransformersACECalculator(Calculator):
    """
    ASE calculator for Transformers-ACE.
    Uses standard ASE neighbor lists (highly compatible).
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path="model.pt", device=None, fragment_policy="auto", **kwargs):
        """``fragment_policy`` (electrostatic models only):

        ``"auto"`` -- fragments are the connected components of the bond graph
            at every call (trace_electrostatics' default).  A CHARGED structure
            whose bond graph splits during MD then has several fragments and no
            way to share its net charge, and the call raises.
        ``"initial"`` -- fragments are fixed from the FIRST structure this
            calculator sees without ``arrays['fragment_id']``, and written into
            ``atoms.arrays['fragment_id']`` / ``atoms.info['fragment_charges']``
            so they persist through the trajectory, like a fixed topology.  A
            charged molecule that dissociates keeps its net charge on the union
            of its parts, and the energy stays continuous (the dissociating
            bond's transfer vanishes smoothly with its envelope).  Physics limit,
            shared by every charge-equilibration model: once the parts are
            apart their charges freeze at the reference-charge shares, not at
            integers.  For integer products (redox), give the products their own
            fragment labels and integer charges -- a diabatic choice.
        """
        if fragment_policy not in ("auto", "initial"):
            raise ValueError(f"fragment_policy must be 'auto' or 'initial', got {fragment_policy!r}")
        self.fragment_policy = fragment_policy
        Calculator.__init__(self, **kwargs)
        
        # 1. Device Setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Transformers-ACE from {model_path} on {self.device}...")

        # 2. Load Model & Config
        try:
            checkpoint = _load_checkpoint(model_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if 'config' not in checkpoint:
            raise KeyError("Model file missing 'config'. Please retrain with updated train.py.")

        conf = checkpoint['config']

        self.atomic_energy_map = {int(k): float(v) for k, v in (conf.get('atomic_energies') or {}).items()}
        self.energy_shift_per_atom = float(conf.get('energy_shift_per_atom', 0.0)) if conf.get('energy_shift_per_atom') is not None else 0.0

        # Optional trace_electrostatics, rebuilt from what train.py saved.  It
        # needs float64 set BEFORE any e3nn module -- the model's included --
        # is constructed: e3nn bakes the dtype into generated code, and a later
        # .double() is a silent no-op (trace_electrostatics SPEC 3).  This is a
        # process-wide setting, exactly as in the training run.
        electrostatics = conf.get('electrostatics')
        if electrostatics:
            torch.set_default_dtype(torch.float64)
            print("Checkpoint carries trace_electrostatics: default dtype set to float64.")

        # Ensure cutoff is float
        self.r_max = float(conf['r_max'])

        # Checkpoints created before architecture versioning are v1. This keeps
        # every published model numerically tied to the implementation that was
        # used to train it while new checkpoints use the corrected v2 model.
        architecture_version = int(conf.get('architecture_version', 1))
        if architecture_version == 1:
            model_class = LegacyTransformersACE
        elif architecture_version == 2:
            model_class = TransformersACE
        elif architecture_version == 3:
            model_class = TransformersACEV3
        elif architecture_version == 4:
            model_class = TransformersACEV4
        else:
            raise ValueError(f"Unsupported architecture_version: {architecture_version}")
        radial_mlp_default = 64 if architecture_version == 1 else 32

        # 3. Initialize Architecture
        model_kwargs = dict(
            r_max=self.r_max,
            l_max=conf['l_max'],
            num_radial=conf['num_radial'],
            hidden_dim=conf['hidden_dim'],
            num_layers=conf['num_layers'],
            radial_basis_type=conf.get('radial_basis_type', 'bessel'),
            radial_trainable=conf.get('radial_trainable', False),
            envelope_exponent=conf.get('envelope_exponent', 5),
            gaussian_width=conf.get('gaussian_width', 0.5),
            descriptor_passes=conf.get('descriptor_passes', 1),
            descriptor_residual=conf.get('descriptor_residual', True),
            radial_mlp_hidden=conf.get('radial_mlp_hidden', radial_mlp_default),
            radial_mlp_layers=conf.get('radial_mlp_layers', 2),
            correlation_order=conf.get('correlation_order', 4),
            correlation_channels=conf.get('correlation_channels', 16),
            attention_num_heads=conf.get('attention_num_heads', conf.get('transformer_num_heads', 4)),
            attention_key_dim=conf.get('attention_key_dim', None),
            attention_ffn_hidden=conf.get('attention_ffn_hidden', conf.get('transformer_ffn_hidden', None)),
            attention_dropout=conf.get('attention_dropout', conf.get('transformer_dropout', 0.0)),
            attention_layer_scale_init=conf.get('attention_layer_scale_init', 1e-2),
            attention_distance_penalty=conf.get('attention_distance_penalty', True),
            use_aux_force_head=False,
            use_aux_stress_head=False,
        )
        if architecture_version == 4:
            model_kwargs.update(
                attention_num_shells=conf.get('attention_num_shells', 4),
                correlation_rank_initial=conf.get('correlation_rank_initial', None),
            )
        self.model = model_class(**model_kwargs)
        self.electrostatics = None
        self.spin = False
        self.spin_key = 'multiplicity'
        if electrostatics:
            if electrostatics.get('redox') is not None and fragment_policy == "initial":
                raise ValueError(
                    "fragment_policy='initial' assigns fixed fragment charges; a TRACE-Redox "
                    "checkpoint carries the net charge in its free sector and keeps every bound "
                    "fragment neutral. Use fragment_policy='auto'."
                )
            self.electrostatics = attach_checkpoint_electrostatics(self.model, electrostatics)
            self.spin = bool(dict(electrostatics.get('heads') or {}).get('spin', False))
            # The info key the training data used for 2S+1 ('spin' for OMol25).
            self.spin_key = electrostatics.get('spin_key', 'multiplicity')
        
        # 4. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Inputs are built in the model's own precision: float32 for every
        # existing checkpoint, float64 for one with electrostatics.
        self.dtype = next(self.model.parameters()).dtype
        self.atomic_energy_tensor = None
        if self.atomic_energy_map:
            max_z = max(self.atomic_energy_map)
            tensor = torch.zeros(max_z + 1, dtype=self.dtype, device=self.device)
            for z, val in self.atomic_energy_map.items():
                tensor[z] = val
            self.atomic_energy_tensor = tensor

    def _freeze_fragments(self, atoms):
        """``fragment_policy="initial"``: label the connected components once."""
        if 'fragment_id' in atoms.arrays:
            return
        cutoff = float(self.electrostatics.heads.r_max)
        i, j = neighbor_list('ij', atoms, cutoff)
        parent = list(range(len(atoms)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in zip(i.tolist(), j.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = [find(a) for a in range(len(atoms))]
        order = {root: k for k, root in enumerate(sorted(set(roots)))}
        labels = np.array([order[r] for r in roots], dtype=int)
        n_fragments = int(labels.max()) + 1 if len(labels) else 0
        charge = atoms.info.get('charge', atoms.info.get('total_charge', 0.0))
        if 'fragment_charges' not in atoms.info:
            if n_fragments > 1 and float(charge) != 0.0:
                raise ValueError(
                    f"fragment_policy='initial': the first structure has {n_fragments} "
                    f"fragments and net charge {charge}; say which fragment carries it "
                    "with atoms.info['fragment_charges'] (one entry per fragment, "
                    "labelled by connected component in order of lowest atom index)."
                )
            atoms.info['fragment_charges'] = [float(charge)] + [0.0] * (n_fragments - 1)
        atoms.arrays['fragment_id'] = labels

    def check_state(self, atoms, tol=1e-15):
        """ASE's change detection, plus the charge state for an electrostatic model.

        The electrostatic energy depends on the net charge, the fragments and
        the spin state, which live in ``atoms.info``/``atoms.arrays`` and which
        ASE's own comparison does not look at.
        """
        changes = Calculator.check_state(self, atoms, tol=tol)
        if (
            self.electrostatics is not None
            and self.atoms is not None
            and charge_state_changed(self.atoms, atoms)
        ):
            changes = list(changes) + ['charge_state']
        return changes

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Standard ASE setup
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Periodic-correct neighbor list. ASE shift S gives
        # r_j - r_i + S @ cell, so edges are stored as neighbor -> center.
        i, j, shifts = neighbor_list('ijS', atoms, self.r_max)
        edge_index = torch.stack(
            [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)], dim=0
        ).to(self.device)
        edge_shift = torch.tensor(shifts, dtype=self.dtype, device=self.device)
        
        # 2. Prepare Data
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        pos = torch.tensor(atoms.positions, dtype=self.dtype, device=self.device)
        cell = torch.tensor(atoms.cell.array, dtype=self.dtype, device=self.device)
        
        # Volume handling (Use 1.0 for non-periodic systems)
        if atoms.pbc.any():
            vol = atoms.get_volume()
        else:
            vol = 1.0

        data = {
            'z': z,
            'pos': pos,
            'cell': cell,
            'edge_index': edge_index,
            'edge_shift': edge_shift,
            'volume': torch.tensor(vol, dtype=self.dtype, device=self.device),
        }
        if self.electrostatics is not None and self.fragment_policy == "initial":
            self._freeze_fragments(atoms)
        if self.electrostatics is not None:
            # Periodicity (SPEC 7: an input, never inferred from the cell) and
            # the charge state -- info['charge'] / info['total_charge'],
            # arrays['fragment_id'] with info['fragment_charges'], and for
            # spin-resolved heads info['multiplicity'] /
            # info['fragment_multiplicities'] -- read exactly as in training.
            data.update(
                charge_state_tensors(
                    charge_state_from_atoms(atoms, spin=self.spin, spin_key=self.spin_key),
                    self.dtype, self.device,
                )
            )
            # Applied field, metals / electrodes, diabatic charge states
            # (trace_electrostatics SPEC 15, 20, 21), when the Atoms carry them.
            data.update(electrostatic_environment(atoms, self.dtype, self.device))
        
        # 3. Run Model
        calc_stress = 'stress' in properties

        pred_E, pred_F, pred_S, _ = self.model(
            data,
            training=False,
            compute_stress=calc_stress,
        )

        if self.atomic_energy_tensor is not None:
            if torch.max(z).item() >= self.atomic_energy_tensor.shape[0]:
                raise ValueError("Encountered atomic number without reference energy in atomic_energies")
            baseline = torch.sum(self.atomic_energy_tensor[z])
        else:
            baseline = torch.tensor(self.energy_shift_per_atom * len(atoms), device=self.device)

        pred_E = pred_E + baseline

        # 4. Store Results
        self.results['energy'] = pred_E.item()
        self.results['forces'] = pred_F.detach().cpu().numpy()
        
        if calc_stress:
            S_mat = pred_S.detach().cpu().numpy()
            # Convert 3x3 to Voigt (xx, yy, zz, yz, xz, xy)
            self.results['stress'] = np.array([
                S_mat[0,0], S_mat[1,1], S_mat[2,2],
                S_mat[1,2], S_mat[0,2], S_mat[0,1]
            ])


# Preserve the original calculator import for existing user scripts.
FlashACECalculator = TransformersACECalculator
