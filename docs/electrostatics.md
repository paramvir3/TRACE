# `trace_electrostatics` user guide

Long-range electrostatics, induction and (optionally) many-body dispersion for
TRACE: a variational charge-equilibration model whose energy is added to the
TRACE energy before forces and stress are taken.

This guide covers how to use the package. The physics, the derivations and every
measured number are in [`trace_electrostatics/SPEC.md`](../trace_electrostatics/SPEC.md),
what changed recently is in
[`trace_electrostatics/CHANGELOG.md`](../trace_electrostatics/CHANGELOG.md), and
every public function and class, with its signature, is listed in the generated
[API reference](electrostatics_api.md).

Contents: 1 Overview · 2 Requirements · 3 Quick start · 4 Choosing settings ·
5 Observables · 6 Training · 7 Inference · 8 Performance · 9 Troubleshooting ·
10 Known limits · 11 Where the physics lives

---

## 1. Overview

**What the model computes.** Each atom carries a Gaussian charge cloud, an
induced dipole and, optionally, an induced or permanent quadrupole. Charge can
move only along bonds: every pair of atoms closer than the heads' `r_max` is a
bond with a *split charge* `t` (charge that has flowed from one atom to the
other). So the net charge of each fragment is fixed by construction, and a
fixed carrier `ρ` places that net charge. Readout heads predict the per-atom
electronegativity `χ`, hardness `η`, Gaussian width `σ` and dipole stiffness
`A_μ`, and the per-bond hardness `κ`, from the TRACE node features. The package
builds the quadratic energy of all these moments, with the full Coulomb
interaction between every pair of Gaussian multipoles (open boundaries) or its
lattice sum (periodic), and returns its **minimum**. Forces, stress,
polarizabilities and Born charges are exact derivatives of that minimum.

**The master equation** (SPEC 1.5.1, 17):

```
E  =  min_{m ∈ V}  ½ mᵀ K m + cᵀ m   (+ E_disp)

m = (t, μ̂, Θ̂)            split charges on bonds, reduced dipoles μ̂_i = μ_i/(e ℓ_i),
                          reduced quadrupoles Θ̂_i = Θ_i/(e ℓ_i²)   (Θ̂ only with quadrupoles on)
q = B diag(f) t + ρ       atomic charges; B = bond incidence, f = dissociation envelope
K                         Coulomb (Gram) matrix of the Gaussian multipoles + on-site
                          hardness η, A_μ, A_Θ + bond hardness κ; positive definite by
                          theorem under the default Gaussian kernel (SPEC 12, 14)
V                         bonds that join two explicitly labelled fragments are pinned
                          to zero (SPEC 17.4)
E_disp                    optional: zero-point energy of the same dipole block (SPEC 13)
```

The minimum is found by preconditioned projected conjugate gradients (CG),
converged to a relative residual `solve_rtol` (default `1e-10`), and it is
differentiated **implicitly**. Forces need no extra linear solve, and derivatives
of every order are those of the exact minimiser (SPEC 17). The older "k CG steps
differentiated through" energy is kept only as the opt-in
`solve_mode="fixed_k"`.

Units are **eV, Å, e** throughout: dipoles in e·Å, fields in V/Å (= eV/(e·Å)),
hardness in eV/e². `K_E = 14.399645 eV·Å/e²` is defined once, in
`trace_electrostatics/constants.py`.

---

## 2. Requirements

- Python ≥ 3.8 with torch, e3nn (the 0.4.x or 0.6.x series) and ASE (for the
  neighbour lists and the calculator). The environments of record, on which
  every number in this guide, the SPECs and the benchmark records was measured:
  - **CPU** (macOS; the examples below, all CPU training, every score and the
    test suites): Python 3.8.16, torch 2.2.0, e3nn 0.4.4, ase 3.23.0,
    numpy 1.24.2, and phonopy 2.17.2 for the phonon scores.
  - **GPU host** (Linux, an RTX 5060 Ti; GPU training): Python 3.12,
    torch 2.11+cu128, e3nn 0.6.0 (the electrostatics suite passes there with
    ase 3.29 and numpy 2.5; CHANGELOG Round 4).
  - `requirements.txt` pins a newer set (torch 2.12.0, ase 3.28.0, numpy 2.4.6);
    no run or score reported in this release used it.
  - PySCF is needed only by the first-principles table builders in
    `tools/electrostatics/`.
- **Double precision, set first.** Call `torch.set_default_dtype(torch.float64)`
  *before* constructing any e3nn module. That includes `ElectrostaticHeads` and
  the TRACE model itself. e3nn bakes the dtype into generated code at
  construction, and a later `.double()` is a silent no-op (measured: 2.5e-8
  equivariance error instead of 1.9e-15, SPEC 3). `ElectrostaticHeads`
  refuses to build under any other default dtype, and `attach_electrostatics`
  checks again.
- Both platforms are measured. On CUDA in float64 every route agrees with the
  CPU to ≤ 1e-14 relative (the matrix-free open route to the solve tolerance;
  CHANGELOG Round 4). The GPU of record runs float64 at about 1/64 of its
  float32 rate, so a GPU step is not always faster: one TRACE-Redox training
  step on the dense bound route took 26.6 s there against 10.3–11.3 s on the
  CPU (trace_redox SPEC §13). Timings per platform: section 6 (batched
  training) and section 8.
- `train.py` refuses `use_amp: true` when electrostatics is on.
- The package never imports `flashace`: it can be used and tested on its own.

---

## 3. Quick start

The examples in this guide run top to bottom in one Python session. Every block
after the setup block uses the setup helpers, and later sections reuse the
structures built here (`nh4`, `nacl`). `h` stands in for the TRACE node
features. In production the TRACE model supplies it (section 7), so the
numbers the examples print are those of untrained heads.

### 3.0 Setup

```python
import numpy as np
import torch
torch.set_default_dtype(torch.float64)       # FIRST: before any e3nn module exists

from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from trace_electrostatics import ElectrostaticConfig, ElectrostaticHeads, electrostatic_state


def graph(atoms, cutoff):
    """Positions and a neighbour list in TRACE's convention.

    edge_vec = pos[src] - pos[dst] + edge_shift @ cell, with EVERY pair out to
    `cutoff`, which must be >= the heads' r_max. ASE's 'ijS' list is exactly
    this with src = j, dst = i.
    """
    i, j, shift = neighbor_list("ijS", atoms, cutoff)
    pos = torch.tensor(atoms.positions, requires_grad=True)
    cell = torch.tensor(atoms.cell.array)
    edge_index = torch.tensor(np.stack([j, i]))
    edge_shift = torch.tensor(shift, dtype=torch.float64)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
    return pos, cell, edge_index, edge_shift, edge_vec, edge_vec.norm(dim=-1)


def features(n_atoms, heads, seed=0):
    """Stand-in for the TRACE node features h (a trained model supplies these)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_atoms, heads.irreps_in.dim, generator=generator)


torch.manual_seed(0)
heads = ElectrostaticHeads(r_max=4.0)   # irreps_in defaults to TRACE's 64x0e + 32x1o + 16x2e
```

`r_max` is the bond range: pairs beyond it carry no split charge. It must
satisfy two rules (section 4.8): it is **at most the model's neighbour cutoff**,
and **at least ~1.4× the first-shell bond length**. 4.0 Å covers the 2.82 Å
Na–Cl bond used below.

### 3.1 Molecule or cluster: open boundaries, net charge

```python
d = 1.03 / np.sqrt(3.0)                                # N-H 1.03 A, tetrahedral
nh4 = Atoms("NH4", positions=[[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]])

pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, heads.r_max)
state = electrostatic_state(
    features(len(nh4), heads), pos, edge_index, edge_vec, edge_len, heads,
    ElectrostaticConfig(),        # Gaussian kernel, converged solve, dense operator
    total_charge=+1.0,            # no pbc and no cell: open boundaries
)
forces = -torch.autograd.grad(state.energy, pos)[0]
q = state.solve_result.q
print(f"U = {state.energy.item():.6f} eV   sum q = {q.sum().item():.12f} e")
print("q =", q.detach().numpy().round(4), "  CG iterations:", state.solve_result.diagnostics.n_iter)
```

With no `pbc` and no cell (or ASE's all-zero gas-phase cell), the boundaries
are open, and the Coulomb sum runs over **all** pairs, not the neighbour list.
`total_charge` must be a Python number. It applies to the single fragment here.
With several fragments, see section 4.6.

### 3.2 3-D crystal: `pbc`, `cell`, `edge_shift`, and `matrix_free` for large cells

```python
nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)    # 8 atoms, first shell 2.82 A
nacl.rattle(0.05, seed=1)                              # the ideal cell has zero forces by symmetry

pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nacl, heads.r_max)
state = electrostatic_state(
    features(len(nacl), heads), pos, edge_index, edge_vec, edge_len, heads,
    ElectrostaticConfig(),
    pbc=(True, True, True), cell=cell, edge_shift=edge_shift,
)
forces = -torch.autograd.grad(state.energy, pos)[0]
print(f"U = {state.energy.item():.6f} eV/cell   max |F| = {forces.abs().max().item():.3e} eV/A")
```

Periodicity comes from `pbc`, never from the cell (SPEC 7). Always pass
`edge_shift`: a pair bonded through several periodic images is several bonds,
and dropping the shift makes the energy depend on the choice of cell (SPEC 7.2).
For large cells, switch to the matrix-free operator, which forms no `n × n`
Coulomb blocks (SPEC 16). Optionally use particle-mesh Ewald for its
reciprocal sum (SPEC 19):

```python
big = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(3)   # 216 atoms
big.rattle(0.05, seed=1)
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(big, heads.r_max)
h = features(len(big), heads)

for config in (ElectrostaticConfig(solver="matrix_free"),                     # direct k-sum
               ElectrostaticConfig(solver="matrix_free", reciprocal="pme", pme_order=8)):
    s = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, heads, config,
                            pbc=True, cell=cell, edge_shift=edge_shift)
    print(f"{config.reciprocal:5s}: U/atom = {s.energy.item() / len(big):.9f} eV, "
          f"{s.solve_result.diagnostics.n_iter} CG iterations")
```

The dense and matrix-free routes compute the same functional; each Ewald sum is
converged to `ewald_accuracy` (default `1e-6`), so they agree to that level. The
same holds for PME, whose mesh is sized for `ewald_accuracy`.

### 3.3 Slab: two periodic axes and vacuum

```python
slab = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 1))   # two (001) layers, 32 atoms
slab.center(vacuum=10.0, axis=2)     # vacuum along the third lattice vector
slab.pbc = (True, True, False)       # so the neighbour list does not wrap the vacuum axis
slab.rattle(0.05, seed=2)

pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(slab, heads.r_max)
state = electrostatic_state(
    features(len(slab), heads), pos, edge_index, edge_vec, edge_len, heads,
    ElectrostaticConfig(),                       # slabs need the dense route
    pbc=(True, True, False), cell=cell, edge_shift=edge_shift,
)
print(f"U = {state.energy.item():.6f} eV   slab route: {state.assembly.slab}")
```

A slab runs 3-D Ewald plus the Yeh–Berkowitz correction `2π K_E M_n²/V`, which
is applied to the operator so that the solve minimises the corrected energy
(SPEC 7). Requirements:
- The non-periodic lattice vector must be orthogonal to the periodic plane
  (checked).
- There must be vacuum. The code checks only that the atoms do not fill the
  cell. The convergence condition is a gap of at least the slab thickness, and
  Yeh and Berkowitz recommend a cell ≥ 3× the thickness. The example uses 8×.
- `solver="matrix_free"` refuses slabs.
- A single periodic axis (a wire) is refused.

---

## 4. Choosing settings

| decision | setting | default | change it when |
|---|---|---|---|
| multipole kernel | `ElectrostaticConfig(multipole_kernel=)` | `"gaussian"` | only to reproduce Thole results; Thole needs `ElectrostaticHeads(hardness_margin="legacy")` (4.1) |
| operator | `ElectrostaticConfig(solver=)` | `"dense"` (`train.py`: `"auto"`, 4.2) | large periodic cells or large clusters: `"matrix_free"`, or `"auto"` per structure (4.2) |
| periodic reciprocal sum | `ElectrostaticConfig(reciprocal=, pme_order=)` | `"ewald"`, 8 | large 3-D cells on the matrix-free route: `"pme"` (4.2) |
| quadrupoles | heads `quadrupole_blocks`, `allow_truncated_a_theta`; call `quadrupoles=True` | off | σ-holes, lone pairs, quadrupolar molecules (4.3) |
| dispersion | heads `dispersion=True`; config `dispersion=True`, `dispersion_range` | off | you want the long-range dispersion from the same dipoles (4.4) |
| spin | heads `spin=True`; call `fragment_spins` | off | open-shell systems (4.5) |
| fragments and net charges | call `fragment_id`, `fragment_charges` / `total_charge` | connected components, neutral | ions, redox states, charged fragments (4.6) |
| solve | `solve_rtol`, `solve_max_iter`, `solve_mode` | `1e-10`, `1000`, `"converged"` | rarely (4.7) |
| bond range | heads `r_max`, `envelope_onset` | `5.0` Å, `0.75` | always check: r_max ≥ ~1.4× first-shell bond length (4.8) |

"Heads" options go to `ElectrostaticHeads(...)`, "config" options to
`ElectrostaticConfig(...)`, and "call" options are keyword arguments of
`electrostatic_state` / `electrostatic_energy` (or keys of the flashace `data`
dict, section 7).

### 4.1 Kernel

The Gaussian kernel (default) treats every moment as a Gaussian density, so `K`
is a Coulomb Gram matrix and is positive definite for every geometry (SPEC 12,
14). Thole damping has only a *pair* guarantee (`thole_a < 1`): three or more
close sites can make `K` indefinite (SPEC 1.6). It also needs the older, √2
larger hardness floor (SPEC 17.3):

```python
thole_heads = ElectrostaticHeads(r_max=4.0, hardness_margin="legacy")
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, thole_heads.r_max)
state = electrostatic_state(
    features(len(nh4), thole_heads), pos, edge_index, edge_vec, edge_len, thole_heads,
    ElectrostaticConfig(multipole_kernel="thole", thole_a=0.39), total_charge=1.0,
)
print(f"Thole: U = {state.energy.item():.6f} eV")
```

`hardness_margin="legacy"` is necessary for Thole but not sufficient. On a
compact random cluster (minimum separation 0.14 Å) Thole still raised
`IndefiniteOperatorError` with the legacy floor, while the Gaussian kernel
solved it. Quadrupoles and `dispersion_range > 0` require the Gaussian kernel.

### 4.1a Charge transfer out of contact: the contact law (SPEC 23)

Without electrical training data (dipoles, α), the bond hardness κ is not pinned
by energies and forces. On water, a model trained on energies alone learned κ
flat in distance, so charge sloshed between molecules. Its polarizability per
molecule grew from 2.5 to 6.2 Å³ between monomer and tetramer, against
1.35–1.44 Å³ from PBE0.

`ElectrostaticHeads(charge_transfer_decay=3.0)` makes κ jump ×10³ when two
atoms are more than 0.5 Å beyond the sum of their covalent radii, and grow as
e^{3 r/Å} beyond that.
- Covalent and ionic contacts are left alone (O–H ×1.04, Na–Cl ×1.21).
- Hydrogen bonds and non-bonded pairs are stiffened (O···H ×3·10³).
- It needs the atomic numbers: pass `species=` to `electrostatic_state`; the
  model hook reads `data['z']`.

```python
ct_heads = ElectrostaticHeads(r_max=4.0, charge_transfer_decay=3.0)
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, ct_heads.r_max)
species = torch.tensor(nh4.numbers)
state = electrostatic_state(features(len(nh4), ct_heads), pos, edge_index, edge_vec, edge_len,
                            ct_heads, ElectrostaticConfig(), total_charge=1.0, species=species)
print("kappa range (eV/e^2):", state.assembly.params.kappa.min().item(), state.assembly.params.kappa.max().item())
```

Off by default, so existing checkpoints define the same energy. Recommended
for new models unless dipole and polarizability labels are trained.

### 4.1b A physical bond hardness: screening and the two-level anchor (SPEC 26)

Energies and forces leave the hardness of a bond that carries no charge
(t* ≈ 0) at its initial value. In a crystal, by SPEC 26's Theorem 1, that
value alone sets ε∞: a trained MgO model reached ε∞ = 1316. Two opt-in
settings make the bond hardness physical:

```yaml
electrostatics:
  config:
    bond_screening: [2.0, 2.8]        # Pauli screening: no direct transfer past an atom in between
  heads:
    bond_hardness_law: two_level      # kappa anchored at (dchi^2 + 4h^2)^(3/2) / (4h^2), learned within x10
```

On rocksalt MgO with untrained heads, these two settings take ε∞ from 454–596
to 10.6–12.6 (experiment ≈ 3). Whatever training does, ε∞ stays bounded.

### 4.2 Dense or matrix-free; Ewald or PME

- `solver="dense"` builds the pair blocks and `K` explicitly. It is the only route
  for slabs, dispersion and the dense-operator observables
  (`polarizability(route="minimiser")`, `born_charges_minimiser`,
  `dense_hessian`). Guards: open boundaries refuse more than `max_direct_atoms`
  (2048) atoms, and `K` may not exceed `max_operator_bytes` (4 GB). Raise them
  deliberately if the memory is there.
- `solver="matrix_free"` computes `K v` on the fly, for periodic cells (pair list
  plus reciprocal sum, SPEC 16) and for open boundaries (all pairs in bounded
  tiles, SPEC 18). It gives the same energy, forces, charges and default-route
  responses as the dense route. **Open boundaries above 2048 atoms still need
  `max_direct_atoms` raised**: that guard is checked before the solver choice,
  and its message wrongly says no matrix-free solver exists (section 9).
- `solver="auto"` picks the route per structure
  (`trace_electrostatics.config.resolve_solver`): dense for slabs and with
  dispersion (the matrix-free route refuses both), matrix-free for 3-D periodic
  cells, and for open boundaries dense up to `AUTO_DENSE_OPEN_MAX_ATOMS` (128)
  atoms and matrix-free above. Both routes compute the same functional, so the
  choice is one of cost.
  - **`train.py` defaults to `"auto"`** when `electrostatics.config.solver` is
    not given, and to `"dense"` when `polarizability_route` or
    `born_charge_method` is `minimiser` (those observables exist only on the
    dense operator). The resolved value is stored in the checkpoint.
    `ElectrostaticConfig()` itself keeps the default `"dense"`.
  - TRACE-Redox always evaluates on the dense route: `RedoxElectrostatics`
    replaces any `solver` by `"dense"` (its bound sector is eliminated through
    K's SQE structure, trace_redox SPEC §13).
- `reciprocal="pme"` (with `pme_order`, default 8) evaluates the periodic
  reciprocal sum by smooth particle-mesh Ewald instead of the direct k-sum.
  - It is valid only with `solver="matrix_free"`; the config raises for
    `solver="dense"`, and under open boundaries the setting is ignored.
  - The mesh is sized from `ewald_accuracy`.
  - The real-space cutoff is `ewald_real_cutoff` (default 10 Å, floored at the
    Gaussian widths), not the cost-balanced cutoff the direct k-sum uses
    (`ewald_balance`).
  - Measured accuracy against the direct sum, and cost: SPEC 19.

### 4.3 Quadrupoles

The heads build the on-site quadrupole stiffness `A_Θ`, and optionally the
permanent-quadrupole source `χ_Θ`. The call switches the sector on:

```python
quad_heads = ElectrostaticHeads(
    r_max=4.0,
    quadrupole_blocks=("A_Theta", "chi_theta"),
    allow_truncated_a_theta=True,     # l_max = 2 features: A_Theta from 0e + 2e only (warns)
)
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, quad_heads.r_max)
state = electrostatic_state(
    features(len(nh4), quad_heads), pos, edge_index, edge_vec, edge_len, quad_heads,
    ElectrostaticConfig(), total_charge=1.0, quadrupoles=True,
)
theta = state.solve_result.theta      # [n, 3, 3] traceless second moment, e*A^2 (Buckingham = 3/2 of it)
print(f"U = {state.energy.item():.6f} eV   max |Theta| = {theta.abs().max().item():.3e} e*A^2")
```

- At TRACE's `l_max = 2` there is no `4e` feature, so `A_Θ` is the truncated
  `0e + 2e` form. It is bounded and positive definite, but it is not a general
  quadrupole stiffness (SPEC 10.1). Without `allow_truncated_a_theta=True` the
  head raises `QuadrupoleIrrepsError`.
- `A_Theta` is mandatory: without it the sector is unbounded below.
- The on-site blocks `q_theta` and `mu_theta` are refused (SPEC 10).
- Heads built with quadrupole blocks still run with the sector **off** unless
  the call passes `quadrupoles=True`.
- Quadrupoles need the Gaussian kernel. They work with open and periodic
  boundaries, dense and matrix-free, Ewald and PME.

### 4.4 Dispersion

Dispersion is the zero-point energy of the model's own induced-dipole block,
with a per-atom effective electron count `N_i` from the heads (SPEC 13):

```python
from trace_electrostatics import electrostatic_energy

disp_heads = ElectrostaticHeads(r_max=4.0, dispersion=True)
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, disp_heads.r_max)
h = features(len(nh4), disp_heads)
config = ElectrostaticConfig(dispersion=True, dispersion_range=2.0)   # lambda = 2 A, Gaussian only

state = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, disp_heads, config, total_charge=1.0)
print(f"electrostatic {state.energy.item():.6f} eV   dispersion {state.dispersion.item():.6f} eV")
energy, result = electrostatic_energy(h, pos, edge_index, edge_vec, edge_len, disp_heads, config,
                                      total_charge=1.0)          # returns the SUM
```

- `electrostatic_state` keeps dispersion separate (`state.dispersion`).
  `electrostatic_energy` and the flashace model return the sum.
- `dispersion_range = λ` (default 0, the full zero-point energy) hands the short
  range to the local model by subtraction. Measured: intermolecular C6 unchanged
  to 1e-6 for λ ≤ 2 Å (SPEC 13).
- Periodic cells are folded onto a supercell whose plane spacings reach
  `dispersion_supercell_length` (10 Å), with the non-analytic Γ term on by
  default (SPEC 13.1).
- `dispersion_quadrupoles=True` (needs the quadrupole sector) is an opt-in
  approximation; on random parameters it added 149 % to `E_disp` (SPEC 13).
- Dense route only. Bond charge flow does not fluctuate (SPEC 13, "NOT YET").

### 4.5 Spin

The spin sector splits the charge sector into up and down channels. It adds a
per-fragment spin constraint and an on-site exchange term `½ J^x s²` with
`J^x > 0` (SPEC 9):

```python
spin_heads = ElectrostaticHeads(r_max=4.0, spin=True)
oh = Atoms("OH", positions=[[0, 0, 0], [0.97, 0, 0]])           # hydroxyl radical, a doublet
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(oh, spin_heads.r_max)
state = electrostatic_state(
    features(len(oh), spin_heads), pos, edge_index, edge_vec, edge_len, spin_heads,
    ElectrostaticConfig(), fragment_spins=[0.5],                 # S per fragment, NOT 2S
)
s = state.solve_result.s                                         # spin populations q_up - q_down
print("s =", s.detach().numpy().round(4), " sum s = 2S =", round(s.sum().item(), 12))
```

- `fragment_spins` holds the spin quantum number `S_F` of each fragment; the
  spin populations sum to `2 S_F`.
- Spin polarisation comes only from the spin-dependent electronegativity.
  Because the energy is quadratic in `s`, spin-state splittings are fitted, not
  derived (SPEC 9.1, 9.2).
- Spin works on the dense and matrix-free routes.

### 4.6 Fragments and net charges

Charge constraints are **per fragment**, never global: a single global
constraint is size-inconsistent (SPEC 1.3).
- **Default fragments** are the connected components of the neighbour graph.
- **`total_charge`** applies only when there is one fragment. With several
  fragments and a non-zero net charge you must pass `fragment_charges`, one per
  fragment.
- **Explicit fragment labels** (`fragment_id`) additionally **pin every bond
  that joins two labelled fragments**, so no charge crosses between them
  (SPEC 17.4).

That is how you state *where* an integer charge lives: an ion next to a
solvent molecule, one redox state of a donor–acceptor pair.

```python
na_water = Atoms("NaOH2", positions=[[0, 0, -2.3], [0, 0, 0], [0.757, 0, 0.586], [-0.757, 0, 0.586]])
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(na_water, heads.r_max)
h = features(len(na_water), heads)

labelled = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, heads, ElectrostaticConfig(),
                               fragment_id=torch.tensor([0, 1, 1, 1]), fragment_charges=[1.0, 0.0])
merged = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, heads, ElectrostaticConfig(),
                             total_charge=1.0)                  # one connected fragment
print("Na+ | H2O :", labelled.solve_result.q.detach().numpy().round(4))
print("[NaH2O]+  :", merged.solve_result.q.detach().numpy().round(4))
```

With labels the sodium keeps exactly +1 and polarises the water through the
Coulomb and dipole couplings. Without them the model shares the charge
continuously over the connected cluster.

**What charge equilibration cannot do on its own.** Inside a fragment the
model distributes charge continuously; it cannot localise an INTEGER charge (the
fractional-charge error shared by every charge-equilibration model). Where an
integer charge sits is therefore declared: by fragment labels (above), or, when
the charge moves during a process (redox, dissociation), by coupled diabatic
charge states (section 4.9).

For ahead-of-time deployment, supply both `fragment_id` and `fragment_charges`:
deriving the fragments costs a connected-components pass and a host
synchronisation.

### 4.7 Solve tolerance and solve mode

`solve_rtol` (default `1e-10`) is the relative residual at which CG stops. The
energy error is second order in it, the force error first order. At the default,
forces match finite differences to 1.2e-11 eV/Å, and the work of the forces
around a closed loop is −1.2e-9 eV, the quadrature floor. It is +1.2e-7 eV at
`1e-6` and +2.2e-5 eV at `1e-3` (SPEC 17.2). If the solve cannot reach the
tolerance within `solve_max_iter` (default 1000) it raises rather than return a
truncated iterate.

```python
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, heads.r_max)
h = features(len(nh4), heads)
for label, config in (("converged, rtol 1e-6 ", ElectrostaticConfig(solve_rtol=1e-6)),
                      ("converged, rtol 1e-10", ElectrostaticConfig()),
                      ("legacy fixed_k, k = 6", ElectrostaticConfig(solve_mode="fixed_k", n_cg_iter=6))):
    s = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, heads, config, total_charge=1.0)
    print(f"{label}: U = {s.energy.item():.12f} eV ({s.solve_result.diagnostics.n_iter} iterations)")
```

`solve_mode="fixed_k"` is kept bit for bit for reproducing old results. Its
responses are not those of the minimiser, and more `k` does not fix that (SPEC
17.1). The seed the heads learn changes only the iteration count of the
converged solve, never its energy (SPEC 17.2).

### 4.8 Bond range and the dissociation envelope

A split charge `t_b = f(r_b) τ_b` is switched off smoothly by the envelope `f`:
`f = 1` below `envelope_onset · r_max` (default 0.75), with a C² quintic switch
to zero at `r_max` (SPEC 17.4). Two rules follow:

1. **`r_max ≥ ~1.4×` the first-shell bond length.** Then the dominant bonds sit
   where `f = 1` and the envelope does not bias the Born charges. SPEC 17.4 has
   the measured failure when they do not. On rocksalt (first shell 2.80 Å),
   `r_max = 3.5` Å with onset 0.5 put the first shell inside the switching
   shell and gave `Z*_Na = −5.14`, against `+1.40` at `r_max = 5.0` Å with the
   default onset.
2. **`r_max ≤` the model's neighbour cutoff.** The neighbour list must contain
   every pair out to `r_max`, or bonds vanish while `f > 0`.
   `attach_electrostatics` and `train.py` both refuse a longer `r_max`.

`envelope_onset=None` restores the legacy full-range cosine, which is not
recommended: it put a fixed −0.9/Å into `d ln t/dr` at 2 Å of a 5 Å cutoff.

### 4.9 Charge transfer between fragments: coupled charge states

Each diabatic state fixes integer charges on explicit fragments; the adiabatic
energy is the lowest eigenvalue of `diag(E_a) + V` (empirical valence bond,
SPEC 21). With `V != 0` the energy and forces are smooth through the crossing
(gap `2|V|` there), and away from it the ground state is the lower state, with
integer charges to O(V²/ΔE).

```python
import functools
from trace_electrostatics import adiabatic_charge_states

donor_acceptor = Atoms("OHOH", positions=[[0, 0, 0], [0.97, 0, 0], [4.0, 0, 0], [4.97, 0.1, 0]])
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(donor_acceptor, heads.r_max)
h = features(len(donor_acceptor), heads)
state_fn = functools.partial(electrostatic_state, h, pos, edge_index, edge_vec, edge_len,
                             heads, ElectrostaticConfig())
ids = torch.tensor([0, 0, 1, 1])
result = adiabatic_charge_states(
    state_fn,
    [{"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},     # OH- ... OH
     {"fragment_id": ids, "fragment_charges": [0.0, -1.0]}],    # OH ... OH-
    coupling=0.05,                                               # eV; any differentiable f(pos)
)
print("E =", result.energy.item(), " weights =", result.weights.detach().numpy().round(4))
```

Here the coupling is an input: a constant, or any differentiable function of
the positions. Each state costs one solve. From a dataset or the ASE calculator:
`atoms.info['charge_states']` and `atoms.info['charge_state_coupling']`.

**Charge states in an extxyz file.** A list of dictionaries cannot be stored in
extxyz, so a training file uses the equivalent file form (both forms together
are refused):

```python
import numpy as np
from ase import Atoms
from flashace.charge_state import declared_charge_states

pair = Atoms("OH2OH", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0],
                                 [5.5, 0.3, 0.1], [6.45, 0.4, 0.2]])
pair.info["charge"] = -1
pair.arrays["charge_state_fragment"] = np.array([0, 0, 0, 1, 1])  # labels 0..F-1, all states
pair.info["charge_state_charges"] = np.array([0, -1, -1, 0])      # S x F, one row per state
pair.info["charge_state_coupling"] = 0.05                          # eV, or "through_bond"
print([s["fragment_charges"] for s in declared_charge_states(pair)])  # [[0.0, -1.0], [-1.0, 0.0]]
```

Every row must hold integers summing to the structure's net charge. For
example, Au2 on Al-doped MgO uses the fragments substrate / Au2 and the states
(0, 0) and (+1, −1).

**Learned through-bond coupling (SPEC 22).** `charge_state_coupling="through_bond"`
predicts V from the structure instead. A `TransferCouplingHead` assigns every
bond of the neighbour graph a hopping `h ≥ 0`, and every atom a tunnelling gap
`Δ_i = Σ h + softplus(γ_i) + 0.1 eV`. The donor–acceptor coupling is the bridge
Green's function

    V_DA = w_Dᵀ [H_DA + H_DB (Δ_B − H_BB)⁻¹ H_BA] w_A,

which is a sum over every tunnelling path of the products of `h/Δ` along it
(McConnell superexchange). The gap construction makes the bridge operator
positive definite for any parameters, so V is always defined and smooth.

- **Which states couple:** only states that differ by one electron moved
  between two fragments. The EVB element is `−V`.
- **Distance law:** V decays exponentially along a bridge. It is exactly zero
  between fragments with no path inside the head's `r_max`.

```python
from trace_electrostatics import ElectrostaticModule, TransferCouplingHead

module = ElectrostaticModule(heads, ElectrostaticConfig(), TransferCouplingHead.for_heads(heads))
data = {"charge_states": [{"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},
                          {"fragment_id": ids, "fragment_charges": [0.0, -1.0]}],
        "charge_state_coupling": "through_bond"}
energy, _ = module(h, pos, edge_index, edge_vec, edge_len, data=data)
```

In training, add `transfer_coupling: {}` to the `electrostatics:` block (section
6). In the calculator, set `atoms.info['charge_state_coupling'] = 'through_bond'`.
The head is untrained until data containing charge-transfer states trains it.
Its prior is 2.5 eV at 1.5 Å, decaying as exp(−2.2 r/Å), with a 1.5 eV gap.


### 4.10 Metals and constant-potential electrodes

`metallic` (bool per atom) turns each connected metallic region into an
equipotential conductor (a zero-hardness spanning star of bonds; SPEC 20).
`electrode_potential` (V per atom, NaN off-electrode) holds electrodes at fixed
potential DIFFERENCES through an external circuit, with total charge
conserved. All electrodes of a circuit must share one fragment.

```python
plate = [[i * 2.5, j * 2.5, 0.0] for i in range(4) for j in range(4)]
capacitor = Atoms("Na32", positions=plate + [[x, y, 6.0] for x, y, _ in plate])
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(capacitor, heads.r_max)
state = electrostatic_state(
    features(32, heads), pos, edge_index, edge_vec, edge_len, heads, ElectrostaticConfig(),
    metallic=torch.ones(32, dtype=torch.bool),
    electrode_potential=torch.tensor([0.0] * 16 + [1.0] * 16),      # 1 V across the gap
    fragment_id=torch.zeros(32, dtype=torch.long), fragment_charges=[0.0],
)
q = state.solve_result.q
print(f"Q(top) = {q[16:].sum().item():+.4f} e   Q(bottom) = {q[:16].sum().item():+.4f} e")
```

Measured (SPEC 20): equipotential to 1e-9 V; `u_top − u_bottom = ΔV` to
4e-9; charges exactly linear in ΔV; `dE/dV = −(charge delivered)`; the
image-charge law of an ideal conductor is recovered as a plate grows (0.8 % at
an 87.5 Å plate). From a dataset or the calculator: `atoms.arrays['metallic']`,
`atoms.arrays['electrode_potential']`.

---

## 5. Observables

All observables are derivatives of the same energy, taken from the state that
produced it. They are exact derivatives of the minimiser under the converged
solve.

| function | returns | units | notes |
|---|---|---|---|
| `polarization(q, pos, mu)` | `P = Σ(q_i r_i + μ_i)`, `[3]` | e·Å | about the origin of `pos`; origin-dependent for a charged system |
| `bond_polarization(result, assembly)` | `Ω P`, `[3]` | e·Å | bond-current polarization (SPEC 15); single-valued under PBC; equals `polarization` in open boundaries |
| `polarizability(result, assembly)` | `α`, `[3, 3]` | **e²Å²/eV** | `as_volume=True` gives Å³ |
| `polarizability_contributions(...)` | `(α_ct, α_ind)` | as above | charge-transfer and induced-dipole channels; they sum to `α` |
| `dielectric_tensor(result, assembly)` | `ε∞`, `[3, 3]` | – | 3-D periodic only; `I + 4π K_E α/Ω` |
| `born_charges(P, pos)` | `Z*[i, a, b] = ∂P_a/∂r_ib` | e | differentiates whatever `P` you pass |
| `born_charges_minimiser(assembly)` | `Z*` of the exact minimiser | e | dense route only |
| `observables.born_charges_from_energy(assembly)` | `Z* = −∂²U/∂E∂r` | e | all routes |
| `static_dielectric_tensor(eps_inf, Z, Φ, Ω)` | `ε₀` | – | `Φ` = force constants of the **full** model energy |
| `nonanalytic_force_constants(Z, eps_inf, Ω, q̂)` | LO–TO term, `[3n, 3n]` | eV/Å² | |
| `acoustic_sum_rule_residual(Z, Q)` | `Σ_i Z*_i − Q·I` | e | a check, never a correction |
| `hyperpolarizability(alpha, d)` | `β = d ⊗ α` | – | a constructed readout, not an energy derivative (below) |

**The `as_volume` unit trap.** `polarizability` returns the response in
**e²Å²/eV**, not Å³. A polarizability volume is `K_E = 14.4` times larger, so
comparing the raw number with a literature value is a 14× error (SPEC 0). Pass
`as_volume=True`, or use `to_polarizability_volume`, for Å³. `dielectric_tensor`
takes care of the conversion itself. Do not feed it a volume.

```python
from trace_electrostatics import (acoustic_sum_rule_residual, bond_polarization, born_charges,
    born_charges_minimiser, dielectric_tensor, polarizability, polarizability_contributions,
    polarization)

# Open boundaries: NH4+ (a fresh state, so the graph has not been consumed by a force call)
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nh4, heads.r_max)
state = electrostatic_state(features(len(nh4), heads), pos, edge_index, edge_vec, edge_len, heads,
                            ElectrostaticConfig(), total_charge=1.0)
result, assembly = state.solve_result, state.assembly

alpha = polarizability(result, assembly)                    # e^2 A^2 / eV
alpha_A3 = polarizability(result, assembly, as_volume=True)  # A^3 = K_E * alpha
alpha_ct, alpha_ind = polarizability_contributions(result, assembly)
P = polarization(result.q, pos, result.mu)                   # e*A
Z = born_charges(P, pos)                                     # [n, 3, 3], e
print("alpha (A^3):", alpha_A3.diagonal().detach().numpy().round(4))
print("sum_i Z_i - Q I:", acoustic_sum_rule_residual(Z, 1.0).abs().max().item())   # Q = +1, not 0

# Periodic: rocksalt
pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(nacl, heads.r_max)
state = electrostatic_state(features(len(nacl), heads), pos, edge_index, edge_vec, edge_len, heads,
                            ElectrostaticConfig(), pbc=True, cell=cell, edge_shift=edge_shift)
eps_inf = dielectric_tensor(state.solve_result, state.assembly)
Z = born_charges(bond_polarization(state.solve_result, state.assembly), pos)
Z_exact = born_charges_minimiser(state.assembly)             # dense route only
print("eps_inf diagonal:", eps_inf.diagonal().detach().numpy().round(3))
print("Z* routes agree to", (Z - Z_exact).abs().max().item())
```

(Untrained heads give arbitrary magnitudes; the identities are what the example
checks.)

- **Routes.** `polarizability` and `dielectric_tensor` take `route="fixed_k"`
  (the default) or `route="minimiser"`.
  - `"fixed_k"` is the second field derivative of *the energy the model
    returns*. Despite the name, under the default converged solve it is the
    exact minimiser's response, and it works on every route. The name dates
    from the legacy solve mode.
  - `"minimiser"` is `DᵀK⁻¹D` from the dense operator, and is dense-only.
  - Under the converged solve the two agree to the solve tolerance.
    `reciprocity_residual` in `trace_electrostatics.observables` uses the
    minimiser route to test Maxwell reciprocity.
- **Born charges sum to `Q·I`, not zero,** for a charged system. Nothing
  enforces an acoustic sum rule.
- **Periodic cells** use `bond_polarization` (SPEC 15). `Σ q_i r_i` is
  branch-dependent under PBC and misses the divergence-free bond currents. For
  the same reason `train.py` ignores dipole and polarizability labels on
  periodic structures.
- **Hyperpolarizability.** The energy is quadratic in the moments, so its third
  field derivative is zero. `hyperpolarizability(alpha, d)` builds
  `β_abc = d_a α_bc` from a learned vector `d`
  (`ElectrostaticHeads(hyperpolarizability=True)`). `d` enters no energy, force,
  dipole or `α`, so only β labels train it. On an energy-trained checkpoint β
  has the right symmetry and an arbitrary magnitude.

---

## 6. Training with `train.py`

Electrostatics is opt-in through an `electrostatics:` block in the training YAML.
If the block is absent, or has `enabled: false`, nothing changes: the default
dtype stays float32, dataset items carry exactly the keys they always did, and no
checkpoint key is added.

```yaml
electrostatics:
  enabled: true                   # required whenever the block is present
  config:                         # ElectrostaticConfig fields, passed through
    solve_rtol: 1.0e-10
  heads:                          # ElectrostaticHeads options, passed through
    r_max: 4.0                    # <= the model r_max (defaults to it); >= ~1.4x first-shell bond
    hidden_dim: 16
    key_dim: 8
    bond_radial_basis: 4
    charge_transfer_decay: 3.0    # SPEC 23 contact law: recommended unless alpha is trained
  dipole_weight: 1.0              # default 0 = off
  polarizability_weight: 0.1      # default 0 = off
  born_charge_weight: 0.1         # default 0 = off
  dielectric_weight: 0.1          # eps_inf on 3-D periodic cells; default 0 = off
  dielectric_prior:               # optional: eps_inf on bulk cells that carry NO energies
    file: bulk_MgO_eps.extxyz     #   3-D periodic, info['dielectric_tensor'] on each
    weight: 0.001                 #   one extra forward per optimizer step
  polarizability_prior:           # optional: alpha (A^3) on open structures with NO energies
    file: atoms_alpha.extxyz      #   open boundaries, info['polarizability'] on each
    weight: 0.3
  field_polarizability_prior:     # optional: the WHOLE model's alpha by finite fields (diagonal labels)
    file: Au2_alpha.extxyz        #   open boundaries; several configurations allowed (redox)
    weight: 0.3
  partial_charge_weight: 0.0      # reference partial charges; default 0 = off (see below)
  partial_charge_key: partial_charges   # atoms.arrays key, e.g. hirshfeld_charges
  transfer_coupling: {}           # optional: learned charge-state coupling (SPEC 22)
  polarizability_route: fixed_k   # or minimiser (observables route names)
  born_charge_method: energy      # born_charges_from_energy, or minimiser
```

Add the block to an ordinary training config and run:

```bash
python train.py --config config.yaml
```

**Batched training solves several molecules at once (SPEC 24).** With
`batched_graphs: true`, open-boundary structures go through one block-diagonal
solve per chunk of up to 96 atoms (CPU) or 768 atoms (CUDA), instead of one
solve per structure. Results are identical to the solve tolerance and the
route is automatic. The training cost per structure:

| | GPU | CPU |
|---|---|---|
| Ag3 | 77 → 6.5 ms | 18.7 → 3.0 ms |
| NaCl | 87 → 14 ms | 29 → 19 ms |

Periodic structures, fields, metals, charge states, spin heads and dispersion
keep the per-structure solve.

**A dielectric prior.** `dielectric_prior` gives ε∞ labels on 3-D periodic
structures that have no DFT energies or forces: for example bulk MgO with its
measured ε∞ = 2.95, when the training data are slabs. They are evaluated once per
optimizer step on a forward of their own and enter no energy, force or validation
loss. Their RMSE is printed and recorded each epoch (`*_dielectric_prior_rmse`).
- With TRACE-Redox it needs `anchor: formal`: the integer free charges leave the
  bound sector as the model's electronic response (trace_redox SPEC 17.7). The
  mean-field anchors refuse it.
- It is refused under data parallelism.
- `polarizability_prior` does the same for open-boundary structures with a
  polarizability label (Å³), for example measured atomic polarizabilities.
- With TRACE-Redox, a prior structure with one configuration in the window
  takes the bound sector's response. With several, the thermal and the coherent
  charge-transfer response are not in the bound sector: the priors then add
  the configuration ensemble's response from its exact record (trace_redox
  SPEC §30, and §36.10 for several clusters), and refuse only a structure
  whose record is not exact (for example the legacy core + carrier
  elimination, or a dressed SPEC 32 fold). **[Corrected 2026-11-16:** this
  item said that every prior structure must have exactly one configuration
  and that any other is refused; that was the rule before SPEC 30.**]**
- The per-structure `dielectric_weight` loss is allowed with TRACE-Redox only
  with `anchor: formal`. It reads the bound sector's ε∞ alone; the
  configuration ensemble's response is not added to it.
- With TRACE-Redox's `downfold_transfer: true` (trace_redox SPEC 32: the O 2p ↔
  metal s bond folded into the configuration energies with Harrison's fixed
  V_spσ), a structure with an O–metal contact within `downfold_cutoff` (4.5 Å)
  is refused by the priors: the fold's field dependence is not in the response
  record. Structures without such a contact (an Au atom, MgO bulk) are
  unaffected. The option needs `fragment_ensemble: diabatic` and
  `anchor: formal`, zeroes the learned O–metal couplings, and refuses a
  configuration whose folded charge-transfer state lies below it
  (`DownfoldError`): that state must then be explicit. A pair inside one
  covalent unit (water's O–H) is never folded. The pair's measured constants
  (`downfold_scale`, `downfold_free_axes`; SPEC 32) are fixed, not learned.
  `downfold_hop_ordering: pauli` orders a bonded metal pair's two-site states
  with the antibonding one lowest (the physical ordering, 2026-10-29;
  `bonding` is the legacy default), and every metal atom of a contact cluster
  holds a folded state for each donor (partner states, always on).
  `downfold_block: exact` dresses the whole explicit ensemble at once (one
  Löwdin fixed point per contact cluster on the explicit block, so folded states
  shared by several configurations are counted once; the dressed couplings enter
  the ensemble Hamiltonian); `per_state` (the default) dresses each
  configuration's energy by its own root; when a cluster's ground state is
  charge-transfer-dominated the dominant folded state is promoted to an
  explicit configuration (at most eight per solve, recorded as `promoted`).
  `frame_wall: true` adds the frame pairs' own Pauli wall (the frozen-density
  Thomas–Fermi kinetic overlap times one first-principles constant; tabulated
  for Mg²⁺–Au⁰, where the 4G data stop at 2.43 Å), a physics baseline the
  backbone trains its residual against.
  `frame_pair: true` (SPEC 39, 2026-11-13; off by default, nothing trained with
  it yet) is the general form and replaces `frame_wall` (legacy, one pair): for
  every pair of atoms of the nine library elements it adds the frame pair's own
  interaction, the exact electrostatics of the two frozen frame densities
  beyond the model's Gaussian kernel plus their electron-gas Pauli repulsion
  (Thomas–Fermi kinetic + exchange, scaled by the first-principles ratio as a
  function of distance where the pair has been computed). It needs `carrier_penetration: true` and
  `carrier_overlap: gordon_kim`, is refused together with `frame_wall` and
  with `carriers: covalent_units`; while SPEC 33's wall is on, the pair sum
  of the site that wall was measured on is taken out again, so nothing is
  counted twice. It is the same in
  every charge configuration and must be trained with, never switched on over
  frozen weights.
  `two_site: true` (SPEC 40, 2026-11-13; off by default, nothing trained with
  it yet) is the other half of `frame_pair` for a pair of half-filled s atoms
  (H, Ag, Au today): the pair's own first-principles hop, −t(R), takes the
  place of the learned coupling of that pair (one sign, falling with distance,
  so it can neither make nor unmake a wall), and the pair's neutral
  configuration is lowered by the first-principles relaxation ρ(R) so that it
  sits at the pair's no-hop level. Both functions are tabulated from the
  pair's ground state and its two σσ* excitations (PBE, TDA); no form is
  assumed and no constant is fitted. It needs `frame_pair: true`,
  `spin_pairing: true` and `diabatic_selection: amplitude`. Applied to Au–Au
  today (Ag–Ag and H–H are tabulated and withheld; pairs of two different s
  atoms have no points and keep the learned coupling).
  For Au–Au the pair's other levels are fixed the same way (SPEC 40.7,
  2026-11-14): the one-extra-electron, one-hole and ion-pair configurations
  each carry a first-principles entry, solved so that the ISOLATED pair's
  ground state is the first-principles one as Au₂, Au₂⁻ and Au₂⁺ (its
  electron affinity and ionisation energy are then PBE's on the exact atomic
  values, within 0.16 eV of the PBE0 references). The hop is the coupling of
  moves of an s electron only; any other move of the pair (a 5d electron
  leaving an Au⁺) keeps the learned coupling. These entries are the
  package's own fixed terms against first principles, so they are tied to
  what they were built with (the single model's `bond_partition`,
  `pauli_table`, `carrier_induction`, `frame_polarizability_anchor` and the
  other options listed in `two_site.ENTRY_OPTIONS`, the electrostatic
  `gaussian_a`, and the tables): a model that differs, or a
  `coupling_cutoff` below 4.0 Å, is refused; rebuild with
  `tools/electrostatics/build_two_site_entries.py`. The pair description
  covers 2.0–3.0 Å and is switched off by 4.0 Å: a dimer stretched beyond
  3.0 Å is not described.
  On the trained single model, switched on frozen as a check of the
  mechanism only, the charge module's Au–Au bond force on the compressed test
  dimer goes from −18.8 eV/Å to repulsive and close to DFT's +15.25 (the
  numbers are in SPEC 40.5 and 40.7). It must be trained with (a cold start),
  never switched on over frozen weights; no model has been trained with it.
  `bond_pauli: true` adds the bond-oriented Pauli repulsion of a bonded s-class
  atom against a frame anion (SPEC 33; measured on Au₂ over the O-top O of
  MgO(100): the σ bond shields the contact atom when it points away from the
  anion, by 0.76 eV at 2.26 Å, and not when the dimer tilts), a frame term with
  `bond_pauli_level` 'almo' (the orbitals relaxed, the default) or 'frz'.
  `bond_pauli_wall: true` adds the law's own reference, the lone s atom's
  interaction with the anion at the law's level (frozen for 'frz', relaxed
  for 'almo'; isotropic, every s-class atom), so wall + law is the dimer's
  own interaction at that level and the backbone learns the charge transfer.
  `downfold_decay` (1/Å, default 0)
  steepens the coupling beyond Harrison's 1/r² by e^(−κ(r − r₁)), anchored at
  `downfold_decay_anchor` (2.26 Å); the measured value is 0.45.
- `field_polarizability_prior` trains the whole model's static polarisability,
  α_ii = −[E(+h eᵢ) − 2E(0) + E(−h eᵢ)]/h² · K_E with h = 0.01 V/Å (seven batched
  forwards of the total energy per optimizer step).
  - With TRACE-Redox that includes the thermal and coherent charge-transfer
    response, so structures with several configurations are accepted. That is
    the point: a metal dimer's α∥ identifies the free sector's charge-transfer
    physics, which energies alone do not.
  - The labels must be diagonal: orient each structure along its principal
    axes. It needs only first derivatives in the weights.

**Validation.**
- Every key is validated, even with `enabled: false`: an unknown key at any
  level raises and lists the allowed ones. Float fields accept PyYAML's `1e-10`,
  which PyYAML reads as a string.
- `heads.irreps_in` is not allowed: the heads are built on
  `model.attention_irreps`.
- `use_amp: true` is refused.

**What happens when enabled.**
- `torch.set_default_dtype(torch.float64)` runs before any dataset tensor, model
  or e3nn module exists, and the dataset builds every tensor in float64.
- The heads are attached before the optimiser is built.
- The resolved construction arguments are stored under
  `checkpoint['config']['electrostatics']`.

**Notes on the YAML route:**
- `heads: {hardness_margin: legacy}` is accepted: it is the floor the Thole
  kernel needs, SPEC 17.3.
- Heads built with the quadrupole stiffness `A_Theta` train and run with the
  quadrupole sector **on**. The model hook switches it on whenever the heads
  carry it (section 4.3).

**ASE keys read by the dataset** (units eV, Å, e; the same reader,
`flashace.charge_state`, serves the calculator):

| key | meaning |
|---|---|
| `info['charge']` or `info['total_charge']` | net charge, e (0 if absent); both may be given if they agree |
| `arrays['fragment_id']` | integer fragment label per atom, `0..F-1`, every label used |
| `info['fragment_charges']` | `F` net charges; must sum to the net charge. Omitted with `F > 1`, it is filled only for a neutral structure (all zero), otherwise refused |
| `info['multiplicity']` / `info['fragment_multiplicities']` | `2S+1` of the structure / per fragment; read only for spin heads, which then require it. Passed to the package as `S_F = (m−1)/2`. `info['spin']` is not read: datasets use it for S, 2S and 2S+1 |
| `calc.results['dipole']` or `info['dipole']` | dipole, e·Å, about the Cartesian origin of `positions`; open boundaries only |
| `info['polarizability']` | polarizability **volume**, Å³, 3×3 (9 row-major or 6 Voigt) |
| `arrays['born_charges']` or `calc.results['born_effective_charges']` | Born charges, e, `[n,3,3]` or `[n,9]`, `Z[i,a,b] = ∂P_a/∂r_ib` |
| `arrays[partial_charge_key]` | reference partial charges, e, `[n]` (Hirshfeld, Mulliken, ...); must sum to the net charge within 1e-3 e, since the solve enforces `Σq = Q` exactly. Read always, trained on only with `partial_charge_weight > 0` |
| `calc.results['dielectric_tensor']` or `info['dielectric_tensor']` | clamped-ion `ε∞`, dimensionless, 3×3 (9 row-major or 6 Voigt); 3-D periodic cells only |

A dipole or polarizability on a structure with any periodic direction is ignored,
with a printed count. So is a dielectric tensor on a structure that is not 3-D
periodic. Random-rotation augmentation rotates `p`, `α`, `Z*` and `ε∞` with the
positions.

**Losses.** Each is evaluated only for structures that carry the reference, and
normalised like Eqs. (37)–(40). Unlabelled structures count in the batch
denominator, as stress-free structures do.

```
dipole          ((P − P_ref) / N_s)²,  mean over 3,   P = Σ_i (q_i r_i + μ_i)
polarizability  ((K_E α − α_ref) / N_s)², mean over 9   (α in e²Å²/eV → Å³)
Born charges    (Z* − Z*_ref)², mean over 9 N_s
dielectric      (ε∞ − ε∞_ref)², mean over 9         (ε∞ = I + 4π K_E α_cell/Ω; intensive)
```

All four are functions of the model parameters with their graph kept, so they
train both the heads and the trunk.
- **Per-structure state.** A forward with `return_electrostatics=True` puts one
  `flashace.model.ElectrostaticOutput` per structure in `aux['electrostatics']`.
  Its `state()` rebuilds the assembled operator with `electrostatic_state`,
  checked against the module's energy.
- **Born charges from a batch** are the complete derivative (equal to the
  single-structure one to 1e-16).
- **Solver-seed loss.** In training mode the module can also accumulate this
  loss (`solver_seed_loss_weight`). It trains only the solver's starting point
  and is added to the backward pass, not to the logged loss (SPEC 2.0.1).
  - `train.py` turns it OFF for new runs under the converged solve: the energy
    does not depend on the seed, and the trained seed saved only 3–9 % of CG
    iterations while costing 44 % of the electrostatic training overhead.
  - It stays on (1.0) for `solve_mode: fixed_k`, where the energy does depend
    on the seed.

**Batched charged systems.** `collate_batched` gives every per-structure quantity
a structure dimension:

| key | shape |
|---|---|
| `pbc` | `[B, 3]` |
| `total_charge` | `[B]` |
| `fragment_id` | `[N]`, labels local to each structure, `−1` = none supplied |
| `n_fragments` | `[B]` |
| `fragment_charges` | `[ΣF]` |
| `fragment_spins` | `[ΣF]`, with `n_fragment_spins` `[B]` |

`forward_batched` slices these per structure and hands each call exactly what
`forward` would hand that structure alone.

---

## 7. Inference

### 7.1 The ASE calculator

`TransformersACECalculator` takes these steps with an electrostatics checkpoint:
- Before building the model, it sets float64 as the process default.
- It rebuilds the heads and the config from
  `checkpoint['config']['electrostatics']`
  (`flashace.checkpoint.attach_checkpoint_electrostatics`), then loads the
  weights with a strict `load_state_dict`.
- It passes `pbc`, the cell and the charge state to the model. These are read
  with the same `flashace.charge_state` reader as in training, so the ASE keys
  of section 6 mean the same thing here.

```python
from flashace.calculator import TransformersACECalculator

calc = TransformersACECalculator(model_path="model.pt", device="cpu")   # written by train.py
hydroxide = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
hydroxide.info["charge"] = -1
hydroxide.calc = calc
e_anion = hydroxide.get_potential_energy()
hydroxide.info["charge"] = 0          # a charge-state change invalidates the cached result
e_radical = hydroxide.get_potential_energy()
print(f"E(OH-) = {e_anion:.6f} eV   E(OH) = {e_radical:.6f} eV")
print("forces (eV/A):", hydroxide.get_forces().round(4).tolist())
```

The calculator returns energy, forces and stress, with the electrostatic
energy included. A change of `info['charge']`, or of any other charge-state key,
invalidates the cached results, which ASE's own change detection would not do.
It also passes `info['external_field']` (V/Å), `arrays['metallic']`,
`arrays['electrode_potential']` and `info['charge_states']` /
`info['charge_state_coupling']` (sections 4.9, 4.10). The quadrupole sector is
on whenever the heads carry it.

**Charged reactive MD.** With the default `fragment_policy="auto"` fragments
are recomputed at every call, so a charged molecule that dissociates has two
fragments and nothing saying which one keeps the charge: the call raises.
`TransformersACECalculator(..., fragment_policy="initial")` fixes the
fragments of the first structure (written into `atoms.arrays['fragment_id']` and
`atoms.info['fragment_charges']`), so the charge stays on the union of the parts
and the energy is continuous through the dissociation. The separated parts
then carry fractional charges; for integer products use charge states
(section 4.9).

### 7.2 Attaching electrostatics to a model in Python

`flashace` models call an attached module inside `forward`, adding `U_elec` to
the energy before the autograd call. Forces and stress then follow from the
existing code, with the strained positions and cell. A model has no
electrostatics until `attach_electrostatics` is called, and
`detach_electrostatics(model)` removes it.

```python
from flashace.model import TransformersACEV3
from trace_electrostatics import attach_electrostatics

model = TransformersACEV3(r_max=5.0, l_max=2, num_radial=4, hidden_dim=16, num_layers=1,
                          correlation_order=2, correlation_channels=4, attention_num_heads=2)
model_heads = ElectrostaticHeads(irreps_in=model.attention_irreps, r_max=model.r_max)
module = attach_electrostatics(model, model_heads, ElectrostaticConfig())   # model.electrostatics

atoms = nh4.copy()
i, j, shift = neighbor_list("ijS", atoms, model.r_max)
data = {
    "z": torch.tensor(atoms.numbers), "pos": torch.tensor(atoms.positions),
    "cell": torch.zeros(3, 3), "edge_index": torch.tensor(np.stack([j, i])),
    "edge_shift": torch.tensor(shift, dtype=torch.float64), "volume": torch.tensor(1.0),
    "pbc": torch.tensor([False, False, False]), "total_charge": 1.0,
}
energy, forces, stress, aux = model(data, training=False, compute_stress=False)
print(f"E = {energy.item():.6f} eV (TRACE + electrostatics, untrained)")
```

The model reads `pbc`, `total_charge`, `fragment_id`, `fragment_charges`,
`fragment_spins` and `quadrupoles` from `data` when they are present.
`attach_electrostatics` refuses heads whose `r_max` exceeds `model.r_max`.

### 7.3 LAMMPS

The LAMMPS TorchScript and AOT exports **refuse** a model with electrostatics
attached (`transformers_ace.deploy.refuse_electrostatics`, `NotImplementedError`).
Their wrapper rebuilds the energy from the local layers only, and would otherwise
drop `U_elec`, its forces and its virial silently. Use the ASE calculator.

---

## 8. Performance

Numbers below are measured values from SPEC 16–18. PME numbers are in SPEC 19.

**Periodic, dense.** The dense route materialises the real-space image tensors,
the reciprocal blocks and `K`. It tops out near 500 atoms: 432 atoms took 16.9 s
and 7.0 GB for energy + forces (SPEC 16).

**Periodic, matrix-free.** Rocksalt supercells, CPU, 4 threads, energy + forces,
accuracy 1e-8 (SPEC 16):

| atoms | 1024 | 2000 | 3456 | 6750 |
|---|---|---|---|---|
| time (s) | 5.3 | 14.0 | 37.2 | 151.5 |
| peak RSS (GB) | 1.4 | 2.8 | 4.2 | 4.9 |

Scaling is ~N^1.5 up to 2000 atoms and ~N^1.8 over the whole range. This table
was measured with the legacy fixed-k solve. SPEC 16 does not say so, but SPEC 19
identifies its 3456-atom figure as the fixed-k default. The converged solve
needs more CG iterations, as the next paragraph shows.

**Iterations to `solve_rtol = 1e-10`** with the Woodbury bond preconditioner
(SPEC 17.6), rocksalt supercells:

| atoms | dense | matrix-free |
|---|---|---|
| 64 | 22 | – |
| 216 | 25 | 43 |
| 512 | – | 56 |

The matrix-free preconditioner factorises a dense atom-space matrix
(`n_atoms × n_atoms`) once per solve, up to 8000 atoms. Above that, the bond
sector is left unpreconditioned (SPEC 17.6).

**Open boundaries, matrix-free** (SPEC 18). Seconds per `K v` product, Gaussian
kernel, dipoles, 10 threads, coefficient tiles precomputed:

| atoms | 500 | 2000 | 8000 |
|---|---|---|---|
| s per product | 0.004 | 0.06 | 0.46 |

- A solve applies `K` roughly once per CG iteration.
- The memory-bounded mode is 6–8× slower and stays ≤ 0.5 GB at 8000 atoms.
- The dense route no longer fits at 8000 atoms.
- The cost is O(n²): there is no far-field approximation.

**Practical guidance.**
- Use `solver="dense"` for molecules and cells of up to a few hundred atoms. It
  is the only route with dispersion, slabs and the minimiser observables.
- Use `solver="matrix_free"` for larger 3-D cells and large clusters. For
  clusters above 2048 atoms, raise `max_direct_atoms`.
- Consider `reciprocal="pme"` for large 3-D cells on the matrix-free route
  (SPEC 19).
- `ewald_accuracy` (default `1e-6`) sets the lattice-sum accuracy on every
  periodic route. On the matrix-free direct k-sum, `ewald_balance` (default on)
  picks the real-space cutoff that minimises cost (SPEC 16).
- Keep `solve_rtol` at its default unless profiling says otherwise (section 4.7).

---

## 9. Troubleshooting

| error | cause | fix |
|---|---|---|
| `IndefiniteOperatorError` (`trace_electrostatics`) | CG met a direction with `pᵀKp ≤ 0`: the quadratic form is not positive definite, so the energy is unbounded below. In practice this means the Thole kernel, whose guarantee is pairwise only, or Thole without the legacy floor | use `multipole_kernel="gaussian"` (positive definite by theorem). With Thole use `ElectrostaticHeads(hardness_margin="legacy")`, which is necessary but not sufficient (section 4.1) |
| `SolverNotConvergedError` (`trace_electrostatics`) | the residual is still above `solve_rtol` after `solve_max_iter` iterations; the solve refuses to return a truncated iterate | raise `solve_max_iter`, loosen `solve_rtol`, or look for a badly conditioned `K` (very small `κ` or on-site stiffness) |
| `FloatingPointError: ... non-finite curvature` | NaN or inf in the operator, the linear term or the preconditioner | find the NaN upstream (features, positions, coincident atoms) |
| `ElectrostaticsAssertionError: hardness floor violated` / `sigma must be strictly positive` (`trace_electrostatics.diagnostics`) | `η_i < S_0 = √2 K_E/(σ_i√π)`, or NaN in `η` or `σ`. Head outputs satisfy the floor by construction, so this means parameters that did not come from `ElectrostaticHeads`, or NaN features: a NaN feature row raises `sigma must be strictly positive` | fix the NaN source. Do not turn off `check_hardness_floor`: it guards an energy that is unbounded below, and a second copy of the check runs anyway |
| `ValueError: total_charge=… was given for a system with N fragments` | net charge with several fragments (e.g. an ion outside the neighbour cutoff of the rest) | pass `fragment_charges` (one per fragment), or `fragment_id` + `fragment_charges` |
| `ValueError: fragment_charges has shape …` / `fragment_spins has shape …` | wrong length for the number of fragments | one entry per fragment, in label order |
| `TypeError: total_charge must be a Python number` | a tensor net charge | pass a float |
| `ValueError` from `flashace.charge_state` | inconsistent ASE keys: labels not `0..F−1`, fragment charges that do not sum to the net charge, a multiplicity that does not match the electron count's parity, a missing multiplicity for spin heads | fix the `Atoms` metadata (section 6 table) |
| `RuntimeError: ElectrostaticHeads must be constructed with float64 …` / `trace_electrostatics requires torch.set_default_dtype(torch.float64) …` | default dtype not float64 when the heads are built or attached | call `torch.set_default_dtype(torch.float64)` first, before the model and heads are built |
| `RuntimeError: electrostatics needs the model AND the heads built under float64` (`train.py`), or the checkpoint equivalent | model built before float64 was set | let `train.py` / the calculator set it; do not build the model yourself first |
| `ValueError: … max_direct_atoms` | dense open boundaries above 2048 atoms | `solver="matrix_free"` (no limit; SPEC 18), or raise `max_direct_atoms` |
| `ValueError: reciprocal='pme' is a periodic reciprocal sum` | PME requested for open boundaries | drop `reciprocal` |
| `EnvelopeShellWarning` | an atom's nearest bonded neighbour lies inside the envelope's switching shell, which biases charge transfer and Born charges | `ElectrostaticHeads(r_max ≥ ~1.4 × first-shell bond)` or a larger `envelope_onset` (section 4.8) |
| `ValueError` naming electrodes / counter-electrode / fragments | one electrode alone, electrodes in different fragments, or a potential on non-metallic atoms | section 4.10 |
| `DeprecationWarning: check_hardness_floor=False` | the floor is an invariant of the parameters and cannot be switched off | remove the flag |
| `ValueError: … max_operator_bytes` | dense `K` above 4 GB | `solver="matrix_free"`, or raise the limit deliberately |
| `ValueError` naming `pbc` / cell | a single periodic axis (wire); periodic `pbc` with an all-zero or singular cell; a slab whose vacuum vector is not orthogonal to the surface, or with no vacuum | use `(T,T,T)`, `(T,T,F)` or open; fix the cell |
| `NotImplementedError` on the matrix-free route | slabs, dispersion, `born_charges_minimiser`, `polarizability(route="minimiser")` | dense route, or the default-route observables |
| `ValueError: reciprocal='pme' is the matrix-free route's reciprocal sum` | PME with `solver="dense"` | add `solver="matrix_free"` |
| `QuadrupoleIrrepsError` / `ValueError: quadrupoles=True, but these heads carry … no A_Theta` / `quadrupoles=True requires … 'gaussian'` | quadrupole sector mis-specified | section 4.3 |
| `ValueError: config.dispersion=True, but these heads emit no effective electron count` | dispersion requested without dispersion heads | `ElectrostaticHeads(dispersion=True)` |
| `ValueError: the model's neighbour cutoff r_max=… is shorter than …` | heads `r_max` > model cutoff | build the heads with `r_max ≤ model.r_max` |
| `FixedKDegenerateWarning` / `FixedKDegenerateError` | `solve_mode="fixed_k"` only: the truncated map has no derivative at an exactly symmetric structure | use the default converged solve |
| `NotImplementedError: … trace_electrostatics attached` on export | LAMMPS / AOT export of an electrostatic model | use the ASE calculator (section 7.3) |

---

## 10. Known limits

From the CHANGELOG ("Known limits") and the SPEC:

- float64 is required; the GPU of record runs it at about 1/64 of its float32
  rate. GPU parity is measured on every route (CHANGELOG Round 4); there is no
  GPU performance study beyond the timings in sections 2, 6 and 8.
- Open boundaries are O(n²): no FMM or treecode.
- Dispersion and the dense-operator observables refuse on the matrix-free
  routes. So do slabs.
- Metals: a metallic region is a spanning TREE of zero-hardness bonds, so the
  in-plane polarizability of a periodic metal stays finite (physically it is
  infinite); electrode potentials are differences only (SPEC 20).
- No trained model yet, and no comparison against DFT/CCSD reference data. The
  training pipeline for it exists. **[Updated 2026-11-16:** models have since
  been trained with TRACE-Redox on this package and scored against the DFT
  labels of the 4G-HDNNP benchmarks (`docs/electrostatics/BENCHMARKS.md`).**]**
- The electrostatics cannot be exported through AOT or LAMMPS on torch 2.2. The
  export refuses rather than dropping it.
- Integer charge localisation is declared, by fragment labels or by coupled
  diabatic charge states whose coupling is an input (sections 4.6, 4.9). The
  spin state is an input; spin-state splittings are fitted, not derived (SPEC 9.2).
- Solve iterations grow slowly with size (57 / 80 / 98 at 512 / 1728 / 4096
  atoms): no multilevel preconditioner yet (SPEC 17.6).
- Quadrupoles at `l_max = 2` use the truncated `0e + 2e` stiffness. The on-site
  `q_theta` / `mu_theta` couplings are refused.
- Charge-flow dispersion is not implemented: bond transfers are clamped in
  `E_disp` (SPEC 13).
- The hyperpolarizability is a readout that energy, force, dipole and
  polarizability losses cannot train.
- Batched training (`forward_batched`) does not yet carry fields, electrodes or
  charge states per structure; the single-structure path and the calculator do.

---

## 11. Where the physics lives

`trace_electrostatics/SPEC.md` is the technical contract:

| topic | SPEC |
|---|---|
| units, the e²Å²/eV vs Å³ trap | 0 |
| charge parameterisation, per-fragment constraints, hardness floor | 1.2, 1.3, 17.3 |
| master equation | 1.5.1 |
| periodicity is an input; slabs; `edge_shift` | 7 |
| spin | 9 |
| quadrupoles | 10 |
| Gaussian kernel and positive definiteness | 12, 14 |
| dispersion | 13, 13.1 |
| bond-current polarization, dielectric response | 15 |
| matrix-free periodic route | 16 |
| converged solve, implicit derivatives, envelope, fragments, preconditioner | 17 |
| matrix-free open boundaries | 18 |
| particle-mesh Ewald | 19 |
| metals and electrodes | 20 |
| coupled charge states (redox) | 21 |

Sections 2 and 8 describe the legacy fixed-k solve.
