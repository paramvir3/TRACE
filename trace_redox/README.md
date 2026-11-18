# TRACE-Redox (`trace_redox`)

Free charge, the derivative discontinuity and the Fermi level for TRACE: the
SPEC 25 free-charge sector on top of the unchanged SPEC 1 bound sector of
`trace_electrostatics`. **The physics contract is [`SPEC.md`](SPEC.md)**: the
equations, the measured numbers, and what v1 does not do.

`trace_electrostatics` is not modified. Models and results built with it are
untouched, and the two packages can be run side by side on the same data.

## Two models in one package: what `RedoxConfig()` builds

**`RedoxConfig()` with its defaults builds the mean-field model** of SPEC §1–16:
atomic anchors (`anchor='atomic'`, the NIST IE / EA of the free atoms),
continuous atomic free charges in a Mermin ensemble at `kT`, certified
self-interaction removal (§8a) and, between molecules, the exact canonical
ensemble of molecular charge configurations (`fragments='bonds'`,
`fragment_ensemble='exact'`, §15). The defaults stay as they are so that every
earlier model reloads and every earlier result stays comparable. **They are not
the model of the benchmarks.**

Every current model (SPEC §0) is the formal-anchor model: integer charge states
on formal frames, the diabatic configuration Hamiltonian with one-electron
couplings, the bound sector eliminated exactly, and the first-principles overlap
tables. It is built only from explicit options.

### The released model: step 34b

The step 34b checkpoints (stage A `AuMgO_twosite_bf055_bondlaw_r5_s{42,43}`,
stage B `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}`; release assets with their
sha256 in `docs/electrostatics/MODELS.md`) were trained with this
`electrostatics` block, the same for both stages and both seeds (the whole
configurations: `experiments/benchmarks_4g/configs/<run_name>.yaml`, paths
relative to `experiments/benchmarks_4g`):

```yaml
electrostatics:
  enabled: true
  config:
    bond_screening: [2.0, 2.8]
  heads:
    bond_hardness_law: two_level
  partial_charge_weight: 0.0
  dielectric_prior:
    file: ../benchmarks_phonons/data/MgO_bulk_eps_prior_pbe.xyz
    weight: 0.05
  polarizability_prior:
    file: data/Au_atom_alpha_prior.xyz
    weight: 0.3
  redox:
    anchor: formal
    band_form: fermi
    band_initial_factor: 0.55
    bond_partition: true
    bond_pauli: true
    bond_pauli_level: almo
    bond_pauli_wall: true
    carrier_band: true
    carrier_induction: true
    carrier_overlap: gordon_kim
    carrier_penetration: true
    coupling_bound: harrison
    couplings: true
    diabatic_selection: amplitude
    diabatic_space: clusters
    exclusion: false
    fragment_ensemble: diabatic
    frame_induction_damping: true
    frame_pair: true
    frame_polarizability_anchor: true
    frame_polarizability_scope: ions
    max_charge: 3
    pauli_table: first_principles
    shift_gate: true
    spin_pairing: true
    state_polarizability: true
    state_polarizability_elements: [79, 8]
    state_polarizability_stiffness: onsite
    two_site: true
```

The same model in Python:

```python
from trace_redox import RedoxConfig

step34b = RedoxConfig(
    anchor="formal", max_charge=3, fragment_ensemble="diabatic",
    diabatic_selection="amplitude", diabatic_space="clusters",
    couplings=True, coupling_bound="harrison", shift_gate=True,
    bond_partition=True, bond_pauli=True, bond_pauli_level="almo", bond_pauli_wall=True,
    pauli_table="first_principles", carrier_overlap="gordon_kim",
    carrier_penetration=True, carrier_induction=True,
    carrier_band=True, band_form="fermi", band_initial_factor=0.55,
    state_polarizability=True, state_polarizability_elements=(79, 8),
    state_polarizability_stiffness="onsite",
    frame_induction_damping=True, frame_polarizability_anchor=True,
    frame_polarizability_scope="ions",
    frame_pair=True, spin_pairing=True, two_site=True,
    exclusion=False,
)
```

- Every other field takes its default: `kT` 0.1 eV, `diabatic_window` 2.5 eV,
  `coupling_cutoff` 4.0 Å, `band_cutoff` 3.5 Å, `attention_cutoff` 4.0 Å,
  `fragment_max_configurations` 512, `cluster_closure` true, and
  `bvk_mesh` (1, 1, 1) with `bvk_length` 0, i.e. one cell at Γ (SPEC §37 was not
  trained with).
- 25 fields differ from `RedoxConfig()`. The resolved `RedoxConfig`, every
  field, is stored in each checkpoint (SPEC §11), so the ASE calculator rebuilds
  this model from the file alone. The tables in `trace_redox/data/` are read at
  run time, not stored in the checkpoint; they must be the shipped files
  (`two_site.json` checks the sha256 of four of them and refuses a changed one).
- **`band_initial_factor` must be 0.55** for this model, and for any cold start
  with these options. It is the band unit's starting hopping factor f
  (τ = f V_ssσ, learned), anchored to PBE's conduction-band mass of MgO at the
  4G lattice constant (0.322–0.330 mₑ gives f = 0.55; SPEC §20). With the
  default 1.0 (Harrison's universal value), step 34, otherwise identical, stopped
  in epoch 2 on both seeds: on a 64-atom bulk MgO cell the cluster closure
  refused a cold-start state with two O 2p holes and two band electrons
  (529 configurations against the cap of 512). The checkpoints store 0.55, so
  loading them needs no option; a new training must set it.
- Step 34b **fails its pre-registered criteria** S2, S5 and S8 on both seeds
  and passes S1, S3, S4 and S7 (`docs/electrostatics/BENCHMARKS.md`).

## Use

```python
import torch
torch.set_default_dtype(torch.float64)          # before building any e3nn module

from trace_electrostatics import ElectrostaticConfig, ElectrostaticHeads
from trace_redox import RedoxConfig, attach_redox

heads = ElectrostaticHeads(irreps_in=str(model.attention_irreps), r_max=3.5)
attach_redox(model, heads, ElectrostaticConfig(), RedoxConfig(kT=0.1))
energy, forces = model(data)[:2]   # data: 'z', 'pos', 'edge_index', 'total_charge'
```

This attaches the mean-field model (`RedoxConfig(kT=0.1)` is the default
set); pass a formal-anchor configuration such as `step34b` above for the model
of the benchmarks.

`RedoxElectrostatics(heads, config, redox)` can also be called directly with
the `ElectrostaticModule` signature. It returns `(energy, RedoxResult)`, where
the result holds:
- total, bound and free charges, and the dipoles;
- charge-state populations and the integer-state energies;
- the electronegativity and gap of every unit;
- the electrochemical potential `mu_e`;
- solver diagnostics (iterations, residuals, duality gap).

## Settings (`RedoxConfig`)

The fields of the mean-field model and their defaults. The formal-anchor
options (above) are documented field by field in `trace_redox/module.py` and
section by section in SPEC §17–40.

| field | default | meaning |
|---|---|---|
| `kT` | 0.1 eV | electronic temperature of the Mermin ensemble |
| `max_charge` | 1 | highest cation state K (states −1 … +K, NIST anchors E₊ₖ = IE₁ + … + IE_k; up to 4) |
| `anchor` | `'atomic'` | NIST IE / EA anchors; `'gap'` is the SPEC 1 limit |
| `width` | `'hardness'` | free-charge Gaussian from the atomic hardness; `'bound'` reuses the bound σ |
| `hartree_fraction` | 0.5 | initial h: fraction of IE − EA that is classical Hartree energy |
| `chi_bound`, `log_gap_bound` | 2.0 eV, 1.0 | bounds on the learned shifts |
| `sic` | `'certified'` | certified self-interaction removal (SPEC §8a); `'none'` is v1 |
| `self_interaction` | 0.0 | uniform λ of SPEC 25.4 (only with `sic='none'`); must be below λ_max (≈ 0 in molecules, SPEC §7) |
| `fragments` | `'bonds'` | molecules (bond-graph components) carry integer total charges in the two-level certified ensemble (SPEC §12); `'none'` is the atom-level mean field |
| `fragment_window` | 2 | molecular charge window half-width, \|N\| ≤ max(w, ⌈\|Q\|⌉); widened automatically where an edge state is populated (§12.7) |
| `fragment_max_uncertain` | 10 | bonds allowed in their switching shell at once (≤ 2^k partitions, §12.6) |
| `fragment_ensemble` | `'exact'` | between molecules: the exact canonical configuration ensemble (§15; no averaging, no certificate); `'two-level'` is the averaged ensemble of §12 |
| `fragment_max_configurations` | 512 | configurations solved per partition at most (§15.3); more is refused |
| `bound_elimination` | `'condensed'` | exact elimination through K's SQE structure (§13); `'dense'` is the O((n_b + 3n)³) reference |
| `orbitals` | False | attention-Wannier orbitals (units spread over neighbours); refused with `fragments='bonds'` |
| `attention_cutoff` | 4.0 Å | radius of the charge-state and orbital attention |

## Tests and measurements

```bash
python -m pytest tests/redox -q -p no:cacheprovider
```

```bash
python experiments/redox_v1/measure_physics.py
```

- The tests: see SPEC §9 and §12.11 (Mac CPU, float64). "About 3 min" was
  written here for the first suite; the last recorded count is 1118 passed,
  2 xfailed (2026-11-15, CHANGELOG Round 46).
- The script measures λ_max and the Na···Cl free charge versus R (SPEC §7–8).

## Scope

- **Boundaries:** dense route, Gaussian kernel, the states {−1, 0, +1, …, +K}
  with K = `max_charge` ≤ 4.
  Open, 3-D periodic and 2-D slab (no net free charge on a slab) are all
  supported.
- **Default: certified self-interaction removal** (`sic='certified'`, SPEC
  §8a). A separated Na···Cl stays neutral and H₂⁺-like systems cost the exact
  ionisation energy; v1 (`sic='none'`) kept 0.30 e on Na at 100 Å.
- **Default between molecules: the exact canonical configuration ensemble**
  (`fragments='bonds'`, `fragment_ensemble='exact'`, SPEC §14–15).
  - Molecules are the components of a smooth chemical-bond graph (Cordero
    radii + 0.25 → 0.60 Å).
  - Each molecular charge configuration is solved exactly, with the
    inter-molecular Coulomb, every molecule's polarisation and Σ N_F = Q exact.
  - Averaging is kept where it is physics: the atomic mean field inside a
    bonded molecule.
  - Against the independent enumeration, energies agree to ≤ 1e-13 eV and hole
    marginals to 1e-8: H₂O···H₂O⁺ at 4–30 Å, H₂O···H₂S⁺, and Na···Cl across
    the ionic–neutral crossing.
  - The averaged `'two-level'` ensemble (§12) was off by up to 0.11 eV and
    smeared the hole over both molecules.
  - §14 proves the two-sided bound F_2L ≤ F_exact ≤ F_2L + ½ tr(S̃ Cov).
- **Remaining limitations of the mean-field defaults:** no coherent (hopping)
  sharing of one electron between molecules (the configuration set is its
  diabatic basis, §14.8); no spin; `orbitals` is refused with fragments; a
  structure whose two largest atoms' bond reach exceeds `attention_cutoff` is
  refused (§12.6). The formal-anchor model has the coherent sharing (one-electron
  couplings, §17.6) and the spin-pairing factors of the couplings (§29); its own
  limits are stated section by section in the SPEC.
- **Cost** (SPEC §13): the bound sector is eliminated through K's SQE
  structure (Woodbury over the bond block), exactly and never forming K. One
  Au₂/MgO training step of the mean-field model takes 1.7–2.0 s on a Mac CPU,
  against 10.3–11.3 s on the dense route, which is kept as
  `bound_elimination='dense'` for reference.
- **Training:** the `electrostatics.redox` YAML block (SPEC §11;
  `docs/electrostatics.md`, section 6).

Everything else is refused with an explicit error.
