"""TRACE-Redox: the SPEC 25 free-charge sector on top of the unchanged SPEC 1 bound sector.

    F(R) = E_b0 + min_{c : sum c = Q} [ sum_u Phi_u(c_u) + b^T c + 1/2 c^T W_lam c ]

* ``E_b0``, ``u0``, ``W``, ``J`` -- the bound sector eliminated exactly onto
  free Gaussians of widths ``sigma_f`` (:mod:`.bound`); ``b = Omega u0``,
  ``W_lam = Omega W Omega^T - lam diag(J_uu)``, ``J_uu = (Omega J Omega^T)_uu``
  the unit's bare self-Coulomb (``Omega = I`` without orbitals).
* ``Phi_u`` -- the Mermin ensemble of the unit over the states n = -1, 0, +1
  (:mod:`.ensemble`) with the self-Coulomb-removed state energies

      Ehat_+1 =  chi_u + G_u / 2,   Ehat_0 = 0,   Ehat_-1 = -chi_u + G_u / 2,
      chi_u = chi_Z + L_chi tanh(d_chi / L_chi),   G_u = G_Z exp(L_G tanh(d_G / L_G)) > 0

  ``G_u`` is the first-order gap (the derivative discontinuity of the unit,
  SPEC 25.3 Corollary with the residual self-interaction included) and is
  POSITIVE for every parameter value: ``Ehat`` is convex on the integers, so no
  unit ever disproportionates internally.  The physical integer energies are
  ``E_un = Ehat_un + (1 - lam) J_uu n^2 / 2``.
* Anchors.  ``anchor='atomic'``: ``chi_Z = (IE_Z + EA_Z) / 2`` (Mulliken) and
  ``G_Z = U_Z - (1 - lam) J_Z`` with ``U_Z = IE_Z - EA_Z`` and ``J_Z`` the atom's own
  free-charge self-energy, so that an ISOLATED atom has ``E_+1 = IE_Z`` and
  ``E_-1 = -EA_Z`` exactly at initialisation (NIST values).
  ``anchor='gap'``: ``chi_Z = 0``, ``G_Z = 2 gap`` (a large gap is the exact
  SPEC 1 limit: the free sector is then empty for a neutral system).
* ``lam`` -- residual self-interaction removal (SPEC 25.4), the exact rewrite
  ``1/2 lam J {c}(1 - {c}) = 1/2 lam J (interp(n^2)(c) - c^2)``: the
  piecewise-linear part sits in ``Ehat`` (through ``E_un``), the concave part in
  ``W_lam``.  Convexity needs ``W_lam`` positive definite, ``lam < lam_max``
  (SPEC 25.2); checked, refused otherwise.

The minimisation is exact (:mod:`.free_solver`); forces are ``-dF/dR`` and every
parameter derivative is the implicit one, to second order.

Scope, refused loudly outside it: dense route (open, 3-D periodic, 2-D slab
without net free charge), Gaussian kernel; states {-1, 0, +1}; bound fragments neutral (no ``fragment_id`` /
``fragment_charges``); no spin, metals, SPEC 21 charge states, dispersion or
block-diagonal batches.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from trace_electrostatics.config import ElectrostaticConfig
from trace_electrostatics.kernels import gram_self_energy
from trace_electrostatics.energy import ELECTROSTATICS_ATTR

from .bound import BoundSectorError, assemble_neutral, eliminate_bound_sector, eliminate_core_carrier
from .constants import (
    ATOMIC_ELECTRON_AFFINITY_EV,
    ATOMIC_IONIZATION_EV,
    ATOMIC_SUCCESSIVE_IONIZATION_EV,
    MAX_CATION_CHARGE,
    MAX_Z,
)
from .diabatic import (
    ConfigurationSearchError, canonical_free_energy, configuration_energies, density_matrix, diabatic_hamiltonian,
    formal_ladder, search_configurations, select_configurations, switched_hamiltonian,
)
from .free_solver import FreeChargeSolution, differentiable_solution
from .heads import ChargeStateHead, FreeChargeWidth, IonicFreeChargeWidth, OrbitalHead, TransferCoupling
from .fragments import (
    FragmentEnsembleError,
    block_certificate,
    build_molecules,
    chemical_bonds,
    edge_populations,
    exact_partition,
    percolation_partitions,
    solve_fragment_ensemble,
)
from .sic import certified_removal


from .downfold import DownfoldError, PromotionRequest  # noqa: E402

MAX_PROMOTIONS = 4     #: SPEC 32: promotion passes per solve (each pass promotes the set that restores the explicit weight)

def shift_gate(n: int, edge_index: Tensor, edge_len: Tensor, r_cut: float, onset: float = 0.8) -> Tensor:
    """``g [n]``: g_u = 1 - prod over edges into u of (1 - s(r)), s = 1 for r <= onset r_cut and the C^2 quintic
    switch to 0 at r_cut (SPEC 24).  0 for an isolated atom, 1 with any neighbour inside the onset; a periodic image
    of the atom itself counts as a neighbour (the atom is not isolated).  The product is taken as exp(sum log(1 - s))
    with 1 - s clamped at 1e-300: exactly 1 - 1e-300 = 1.0 in float64 where some s = 1, and s is flat (zero gradient)
    there, so the clamp changes no derivative."""
    dtype, device = edge_len.dtype, edge_len.device
    x = ((edge_len - onset * r_cut) / ((1.0 - onset) * r_cut)).clamp(0.0, 1.0)
    s = 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    log_free = torch.log((1.0 - s).clamp_min(1e-300))
    total = torch.zeros(n, dtype=dtype, device=device).index_add(0, edge_index[1], log_free)
    return 1.0 - torch.exp(total)


#: Fields added after checkpoints existed whose default is not the statement those checkpoints trained: a saved config
#: without the field takes the value here (RedoxConfig.from_checkpoint).
LEGACY_FIELD_VALUES = {"state_polarizability_stiffness": "block", "frame_polarizability_scope": "all"}
#: SPEC 37: a band unit holding more than this (e) is occupied in the structure's leading configurations; its
#: capacity is per structure (SPEC 20), so a supercell is then not a replication and Born-von Karman sampling refuses.
BVK_BAND_OCCUPATION = 0.5
#: SPEC 38: passes that place the Fermi band unit's state window on the lowest configuration's filling
FERMI_REFERENCE_PASSES = 64
#: the switch each of those fields acts under: with it off the field does not reach the energy
_LEGACY_FIELD_SWITCH = {"state_polarizability_stiffness": "state_polarizability",
                        "frame_polarizability_scope": "frame_polarizability_anchor"}


def statement_changes(saved: Mapping[str, Any], config: "RedoxConfig") -> dict:
    """``{field: (trained, now)}`` over ``LEGACY_FIELD_VALUES``: where ``config`` states another energy than the one
    the checkpoint config ``saved`` trained (its stored value, or the legacy value when it predates the field).  For a
    run that loads a checkpoint's weights into a model built from its own YAML (train.py ``resume_from``)."""
    out = {}
    for name, legacy in LEGACY_FIELD_VALUES.items():
        if not getattr(config, _LEGACY_FIELD_SWITCH[name]):
            continue
        trained, now = saved.get(name, legacy), getattr(config, name)
        if trained != now:
            out[name] = (trained, now)
    return out


@dataclasses.dataclass(frozen=True)

class RedoxConfig:
    """Settings of the free-charge sector (the bound sector keeps its ElectrostaticConfig)."""

    #: Electronic temperature k T_e of the Mermin ensemble, eV (SPEC 25.6 d).
    kT: float = 0.1
    #: Highest cation state K: the states are -1, 0, +1, ..., +K (anchors E_+k = IE_1 + ... + IE_k).
    max_charge: int = 1
    #: 'atomic' (NIST IE / EA anchors) or 'gap' (G_Z = 2 gap, chi_Z = 0: the SPEC 1 limit).
    anchor: str = "atomic"
    gap: float = 50.0
    #: Free-charge width: 'hardness' (FreeChargeWidth) or 'bound' (the bound charge's sigma).
    width: str = "hardness"
    #: Initial fraction h of the atomic hardness that is classical Hartree energy.
    hartree_fraction: float = 0.5
    #: |electronegativity shift| <= chi_bound, eV.
    chi_bound: float = 2.0
    #: |log(G / G_Z)| <= log_gap_bound.
    log_gap_bound: float = 1.0
    #: Residual self-interaction removal lam of SPEC 25.4 (0 <= lam < lam_max).
    self_interaction: float = 0.0
    #: 'none' (v1 mean field) or 'certified': the largest on-site removal r with a
    #: closed-form convexity certificate (trace_redox.sic; SPEC 8a).  Excludes lam > 0.
    sic: str = "certified"
    sic_theta: float = 0.999
    sic_smoothing: float = 0.01
    #: 'none': every atom is its own unit (the mean field of §1).  'bonds': molecules
    #: (components of the chemical-bond graph) carry integer total charges in the
    #: two-level certified ensemble of SPEC §12; ``sic`` then acts inside each molecule.
    fragments: str = "bonds"
    #: Molecular charge window half-width (|N| <= max(fragment_window, |Q|)), §12.7.
    fragment_window: int = 2
    #: Bonds allowed in their switching shell at once (2^k partitions), §12.6.
    fragment_max_uncertain: int = 10
    #: 'exact': the exact canonical configuration ensemble between molecules (SPEC §15;
    #: no averaging, no block certificate).  'two-level': the averaged ensemble of §12.
    #: 'diabatic': integer charges on EVERY atom, no mean field anywhere; the canonical
    #: ensemble of the diabatic Hamiltonian over atomic configurations (SPEC §17).
    fragment_ensemble: str = "exact"
    #: 'diabatic': configurations within this many eV of the lowest one are kept (§17.3).
    diabatic_window: float = 2.5
    #: 'diabatic': which configurations enter H (SPEC 21).  'window': those within diabatic_window of the lowest
    #: (SPEC 17.3; F jumps when a coupled configuration crosses the edge).  'amplitude': the window set plus every
    #: configuration whose importance (its leading perturbative amplitude from the thermal reference) exceeds
    #: selection_amplitude_off, each coupling switched (C^2, log-scaled) to full at selection_amplitude_on: F is
    #: continuous.
    diabatic_selection: str = "window"
    #: 'diabatic': how the configuration space is built (SPEC 36).  'global' (the ensemble as shipped, kept): one
    #: Hamiltonian for the structure, importances from the whole structure's thermal reference; M identical,
    #: non-interacting clusters then do not have M times one cluster's energy (benchmarks_4g README R13).  'clusters':
    #: linked configurations only, one Hamiltonian per cluster and exact pair increments of the free energy
    #: (trace_redox/clusters.py): additive for non-interacting clusters, exact for two.
    diabatic_space: str = "global"
    #: SPEC 36.7 (diabatic_space='clusters'): every cluster's space is the product closure of its linked set, and a
    #: selected configuration that is a product of two clusters' configurations (a transfer between clusters through a
    #: reservoir, uncoupled) does not join them.  False: SPEC 36 as first built (legacy; one merged cluster lacks the
    #: products of its parts, experiments/benchmarks_4g/README.md R14 and R19).
    cluster_closure: bool = True
    #: SPEC 37, Born-von Karman sampling of the configuration fluctuations (trace_redox/bvk.py): the energy of a
    #: periodic cell is its M-fold supercell's energy per cell, E_M(C) = E_Gamma(S_M[C]) / |M|, so that a cluster's
    #: excitation meets its own images at M a instead of a (benchmarks_4g README R15, R16).  (1, 1, 1): the cell
    #: itself, the model as shipped.  Needs diabatic_space='clusters' (the copies must fluctuate independently) and
    #: must be 1 along a direction that is not periodic.  The supercell is evaluated whole.
    bvk_mesh: tuple = (1, 1, 1)
    #: SPEC 37: a length L / A instead of a mesh.  Each structure takes M_d = ceil(L / w_d) along its periodic
    #: directions (w_d: the spacing of the lattice planes), so cells of one crystal with equal M_d w_d have the same
    #: energy per cell.  0: off.  The mesh jumps when a width crosses L / integer, as a k-point spacing does.
    bvk_length: float = 0.0
    #: SPEC 22 (formal anchors): the bound split-charge variable of a bond is kept only inside a covalent unit of more
    #: than full_unit_atoms atoms; between carrier units charge flow is the free sector's (its integer transfers and
    #: couplings), so it is not counted twice.  Applied to forward and bound_state alike.
    bond_partition: bool = False
    #: SPEC 23 (formal anchors, single free charges): the exact electrostatics of overlapping first-principles charge-
    #: state densities (nucleus + electrons) replaces the Gaussian Coulomb of the carriers for every pair within 8 A, as
    #: a two-body term of the configuration energies; the band's carriers see it too (SPEC 20).
    carrier_penetration: bool = False
    #: SPEC 25: 'electrostatic' (SPEC 23 alone) or 'gordon_kim' (the same densities' electron-gas kinetic, exchange and
    #: correlation added: the Pauli half of the overlap from first principles; it replaces SPEC 18's fitted exclusion).
    carrier_overlap: str = "electrostatic"
    #: SPEC 25's K: 'gordon_kim' (fitted to the electron-gas term; default) or 'first_principles' (fitted to the
    #: frozen-density PBE interaction in the Watson-sphere convention over the data's contact distances; the overlay
    #: trace_redox/data/pauli_fp.npz from tools/electrostatics/build_pauli_fp.py; 2026-10-25, step 20)
    pauli_table: str = "gordon_kim"
    #: SPEC 24 (formal anchors): the learned charge-state shifts are gated by the environment, g_u = 1 - prod_v (1 - s),
    #: s = 1 inside 0.8 attention_cutoff and C^2 to 0 at it, so an isolated ion has its exact ladder energy.
    shift_gate: bool = False
    #: SPEC 26: the bound sector's moments feel each carrier through its true field (Gauss: the enclosed fraction of its
    #: first-principles density), exactly in the configuration energies (one-body + pair terms).
    carrier_induction: bool = False
    #: SPEC 34: the overlap damping of the frame-charge -> bound-dipole coupling on tabulated nearest-neighbour
    #: frame pairs (trace_redox/frame_damping.py); needs anchor='formal'.
    frame_induction_damping: bool = False
    #: SPEC 35: the frame-polarizability anchor, A_mu = K_E ell^2 / (g alpha_Z(f)) on the tabulated frame ions with
    #: one learned crystal factor g (trace_redox/polarizability_anchor.py); needs anchor='formal'.
    frame_polarizability_anchor: bool = False
    #: SPEC 35.1: the species the crystal factor g multiplies.  'ions' (default since 2026-11-12): the charged frame
    #: ions only; a neutral frame atom (Au0) keeps its tabulated free-atom value.  'all': SPEC 35 as first built
    #: (legacy; what 28b and the PBEsol phonon models trained; restored for checkpoints that predate the field).
    frame_polarizability_scope: str = "ions"
    selection_amplitude_off: float = 0.05
    selection_amplitude_on: float = 0.2
    #: 'diabatic': learned one-electron couplings tau_uv between configurations (§17.6), and their cutoff, A.
    couplings: bool = False
    coupling_cutoff: float = 4.0
    #: SPEC 27: None (unbounded, as trained before 2026-10-18) or 'harrison' (|tau| <= 3.24 hbar^2 / (m (r^2 + r0^2)))
    coupling_bound: str | None = None
    #: SPEC 31: None (learned couplings) or 'harrison': every pair but a frame cation-anion pair takes Harrison's
    #: universal two-centre value times a learned factor in (1/2, 2); the cation-anion pairs are learned and bounded
    #: (SPEC 27)
    coupling_anchor: str | None = None
    #: SPEC 32: fold the donor -> acceptor bond (a frame anion's electron into an open-shell s acceptor in its frame
    #: state, O2- -> Au0) out of the explicit ensemble and into each configuration's energy, as the exact local star
    #: root with Harrison's fixed V_sp sigma (trace_redox/downfold.py)
    downfold_transfer: bool = False
    #: SPEC 32: the O-s contact cutoff (A); the fold's 1/r^2 holds against the first-principles overlap to 3.6 A and
    #: the switch acts on the last 0.5 A.  The model's edge list must reach it.
    downfold_cutoff: float = 4.5
    #: SPEC 32: the pair's measured constants (MgO(100) / Au, theory note 2026-10-18 section 4c): the scale c on
    #: Harrison's V_sp sigma (0.71), the sigma* anisotropy kappa' of a bonded s-class acceptor (0.22), the
    #: crystal-field shift per extra cation neighbour of a hole's p component (1.0 eV), and the s-s bond cutoff (A)
    #: that defines a bonded acceptor.  The defaults (1, 0, 0) leave the plain term.
    downfold_scale: float = 1.0
    downfold_anisotropy: float = 0.0
    downfold_crystal_field: float = 0.0
    downfold_bond_cutoff: float = 3.2
    #: SPEC 32: a hole on an O-class donor couples through its dangling p axes only (the axes with the fewest frame-
    #: cation neighbours); measured 2026-10-29 (the 28-ion flat split).  Off: every component couples.
    downfold_free_axes: bool = False
    downfold_hop_ordering: str = "bonding"
    downfold_block: str = "per_state"
    downfold_decay: float = 0.0
    downfold_decay_anchor: float = 2.26
    frame_wall: bool = False
    #: SPEC 39 (2026-11-13): the frame pair's own interaction for EVERY element pair of the density library
    #: (trace_redox/frame_pair.py): the exact electrostatics of the two frozen frame densities beyond the model's Gaussian
    #: kernel plus their electron-gas Pauli repulsion (first-principles constant where computed), the same in every
    #: configuration.  SPEC 23 and SPEC 25 made absolute, so it needs carrier_penetration and carrier_overlap=
    #: 'gordon_kim'; refused with frame_wall (it holds that wall's Mg-Au pair); while bond_pauli_wall is on, the pair sum
    #: of the site SPEC 33's wall was measured on is taken out again (the wall is the atom's whole in-situ interaction).
    #: Trained with (the backbone learns the residual); never judged frozen.
    frame_pair: bool = False
    #: SPEC 40 (2026-11-13): the two-site sector of a pair of half-filled s frames (H, Ag, Au; trace_redox/two_site.py)
    #: from first principles: the covalent relaxation -rho_ab(R) on the pair's frame configuration alone (so that it sits
    #: at the pair's no-hop level, SPEC 39's term minus rho) and the hop tau_ab(R) = -t_ab(R) w(R) in place of the learned
    #: coupling of that pair (one sign, falling with distance).  Needs frame_pair (rho is defined against it),
    #: spin_pairing (one t for the neutral pair and for one carrier), couplings and diabatic_selection='amplitude' (the
    #: selection must see the hop).  Applied to Au-Au (Ag-Ag and H-H are tabulated, not applied).  For Au-Au the pair's
    #: other three levels carry first-principles entries too (SPEC 40.7: the isolated pair is right as Au2, Au2-, Au2+);
    #: they are tied to the fixed-term options they were built with, and a model with others is refused.  Never judged
    #: frozen; nothing trained with it.
    two_site: bool = False
    #: SPEC 33: the bond-oriented Pauli repulsion of a bonded s-class atom against a closed-shell anion (a frame term;
    #: measured on Au2 over the O-top O of MgO(100), 2026-11-04); level 'almo' (relaxed, the physical target) or 'frz'
    bond_pauli: bool = False
    bond_pauli_level: str = "almo"
    #: SPEC 33 (the wall part): the lone s atom's frozen interaction E1(R) with the anion, isotropic, every s-class atom
    #: (the reference of the bond-oriented law; needs bond_pauli=True)
    bond_pauli_wall: bool = False
    #: SPEC 33 per charge state (step 27, 2026-11-08): the law's coefficients follow the s atom's charge state (the
    #: anion's measured law for n = -1, the neutral's otherwise), entered in each configuration's energy as T(n) - T(0)
    #: on the formal ladder; the frame path keeps the neutral law.  Needs bond_pauli=True and anchor='formal'.
    bond_pauli_states: bool = False
    #: SPEC 28: each charge state's own polarisability (first-principles ratios), exact per selected configuration
    state_polarizability: bool = False
    #: SPEC 28: None (every element with reliable ratios) or the atomic numbers whose states take their own
    #: polarisabilities; every other element keeps its frame's (free-ion ratios of reduced cations in an oxide make
    #: K(n) indefinite with the frame's damping width, measured 2026-10-19)
    state_polarizability_elements: tuple | None = None
    #: SPEC 28.6 (2026-11-10): the stiffness a charge state scales.  'onsite': the atom's own dipole stiffness A_mu,j
    #: (extensive).  'block' (legacy, kept): the operator's dipole block P_jj = A_mu,j + the lattice sum of the atom
    #: with its own periodic images; the same as 'onsite' in an open system, not extensive in a periodic cell.  A
    #: checkpoint saved before this key existed trained 'block' and reloads with it (:meth:`from_checkpoint`).
    state_polarizability_stiffness: str = "onsite"
    #: SPEC 29: spin-pairing factors sqrt 2 on the couplings of moves that pair or unpair frontier electrons
    spin_pairing: bool = False
    #: 'learned': the bound sector keeps its electronegativity heads (chi, chi_Theta).  'none' (formal
    #: anchors only): it is linear response alone, electronegativity entering once, through the exact
    #: ladders; then E_b0 = 0 and b = 0 with no applied field (SPEC 17.8).
    bound_electronegativity: str = "learned"
    #: 'single': one Gaussian per unit for all of its free charge.  'core_carrier' (formal anchors only):
    #: the formal ion core on the compact width, the carrier n - f on a diffuse width set by the exact
    #: ladder's smallest curvature, so that no one-electron move has a negative second difference (SPEC 17.9).
    free_charge_model: str = "single"
    #: 'ionic': every element's free charge on its ionic (Shannon) width, or the QEq width where no ion is
    #: tabulated, with the learned factor.  'frame' (formal anchors only): ionic-frame elements keep that; neutral-
    #: frame elements (formal_frame == 0: metals, covalent elements) take the fixed ladder width sigma_car, so that
    #: no move between two of them can collapse (SPEC 17.10).
    free_width_rule: str = "ionic"
    #: Exclusion (Pauli) of charge-state densities by closed-shell frames (formal anchors only; SPEC 18): an extra
    #: electron on a cation- or neutral-frame unit pays K x density overlap with its closed-shell neighbours, a hole
    #: relieves it; zero at every frame.  Cutoff in A (within the model's neighbour list).
    exclusion: bool = False
    exclusion_cutoff: float = 5.0
    #: Relaxed exclusion (SPEC 18.4; needs exclusion=True): each charge state's frontier density polarises away from
    #: the closed shells as a rigid shell held by the unit's own polarisability (the bound sector's alpha_u), to second
    #: order in the displacement; Pi then counts what remains after that relaxation.
    exclusion_relaxation: bool = False
    #: Free carriers vs bound fluctuations (SPEC 19; formal diabatic only).  'atoms': every integer configuration of
    #: the atoms (steps 2-10).  'covalent_units': deviations from the frames are sign-definite inside every covalent
    #: unit (bonded open-shell atoms), so a unit holds carriers of one sign anywhere on its atoms, while electron-hole
    #: pairs inside a covalent bond are left to the bound sector; transfer between units is unrestricted.
    carriers: str = "atoms"
    #: With carriers='covalent_units': units of at most this many atoms keep every intra-unit configuration (solved
    #: exactly, the full CI of the minimal basis); larger units hold sign-definite carriers only.  A computational
    #: active-space size, not a fitted parameter.
    full_unit_atoms: int = 3
    #: The exclusion on bound fluctuations (SPEC 19.2; needs exclusion=True): its Parr-Pearson finite differences at
    #: each frame are added to the bound sector's electronegativity and hardness.
    exclusion_bound: bool = False
    #: Two-body exclusion between open-shell-frame units (SPEC 18.5; needs exclusion=True): two frontier densities
    #: repel when either is a closed shell (Au-Au-: sigma*^2, bond order 0), so bond strength follows charge state.
    exclusion_pairs: bool = False
    #: Anion densities for the exclusion (SPEC 18.6): 'learned' (one I_eff per anion element) or 'in_situ' (each
    #: anion's own in-situ removal energy from the configuration energies; anion units then carry holes too).
    exclusion_anions: str = "learned"
    #: With exclusion_anions='in_situ': anion units also carry hole states (O-, Cl0) in the exclusion.  Off: on step 9's
    #: weights it releases frame-frame O-O overlap and drives spurious O -> Au transfer in undoped cells (SPEC 18.6).
    exclusion_anion_holes: bool = False
    #: A band unit for delocalised carriers (SPEC 20; formal diabatic only): the extra-electron states of the
    #: cation-frame atoms are carried by one unit holding 0-2 electrons in the lowest orbital of the one-carrier
    #: tight-binding Hamiltonian over them (Harrison s-s hopping x a bounded learned factor), within band_cutoff (A).
    carrier_band: bool = False
    band_cutoff: float = 3.5
    #: What the band unit holds (SPEC 38).  'phi1' (SPEC 20, legacy, the default): 0-2 electrons in the lowest orbital.
    #: 'fermi': any filling N up to two per band site, with Fermi-Dirac occupations of every orbital at the model's kT
    #: and the Hartree-Fock average of the band's own interaction in that state (trace_redox/fermi.py); the band's
    #: charge shape then depends on N and couples to the atoms through one pair table per atom.  Needed for extensivity
    #: when a cell holds more than two excess electrons (a replicated doped cell) and for degenerate lowest levels.
    #: The unit-band hybridisation is not built with 'fermi': the band's row of the coupling matrix is zero.
    band_form: str = "phi1"
    #: The band unit's initial hopping factor f (tau = f V_ssSigma; learned, bounded to (1/20, 20)).  1 is Harrison's
    #: universal value.  2026-10-22: PBE's conduction-band mass of MgO at the 4G lattice constant (0.322-0.330 m_e)
    #: gives f = 0.55 for the nearest-neighbour s band (m* = hbar^2 / (2 f tau_H a^2)); Harrison's f = 1 gives 0.179.
    band_initial_factor: float = 1.0
    #: Configurations solved per partition at most (§15.4); more is refused.
    fragment_max_configurations: int = 512
    fragment_tol: float = 1e-9
    #: Bound-sector elimination: 'condensed' (exact, through K's SQE structure,
    #: SPEC §13) or 'dense' (the reference route, O((n_bonds + 3n)^3)).
    bound_elimination: str = "condensed"
    #: Radius of the charge-state and orbital attention, A (<= the model cutoff).
    attention_cutoff: float = 4.0
    hidden_dim: int = 64
    key_dim: int = 32
    n_radial: int = 8
    #: Attention-Wannier orbitals (SPEC 25.8 C); off: every atom is its own unit.
    orbitals: bool = False
    orbital_s_max: float = 0.45
    orbital_s_init: float = 0.02
    tol: float = 1e-11
    max_iter: int = 200

    @classmethod
    def from_checkpoint(cls, saved: Mapping[str, Any]) -> "RedoxConfig":
        """The config a checkpoint trained.  ``train.py`` stores every field, so a field missing from ``saved`` did
        not exist when the model was trained: it takes ``LEGACY_FIELD_VALUES`` (the statement in force before the
        field) where that differs from today's default, with a warning when the difference reaches the energy.  A
        field this package does not know is refused."""
        import warnings
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(saved) - known)
        if unknown:
            raise ValueError(f"the checkpoint's RedoxConfig has field(s) {unknown} that this trace_redox does not know, "
                             "so the trained energy functional cannot be reproduced")
        values = dict(saved)
        for name, legacy in LEGACY_FIELD_VALUES.items():
            if name in values:
                continue
            values[name] = legacy
            if name == "state_polarizability_stiffness" and values.get(_LEGACY_FIELD_SWITCH[name], False):
                warnings.warn(
                    "the checkpoint's RedoxConfig predates state_polarizability_stiffness: it trained SPEC 28 on the "
                    "operator's dipole block ('block'), which is kept so the trained energy is reproduced; in a "
                    "periodic cell that form is not extensive (SPEC 28.6; the default is now 'onsite')",
                    RuntimeWarning, stacklevel=2)
            if name == "frame_polarizability_scope" and values.get(_LEGACY_FIELD_SWITCH[name], False):
                warnings.warn(
                    "the checkpoint's RedoxConfig predates frame_polarizability_scope: it trained SPEC 35's crystal "
                    "factor on every tabulated species ('all'), which is kept so the trained energy is reproduced; the "
                    "default is now 'ions' (SPEC 35.1: a neutral frame atom keeps its free-atom polarizability)",
                    RuntimeWarning, stacklevel=2)
        return cls(**values)

    def __post_init__(self) -> None:
        if not self.kT > 0.0:
            raise ValueError(f"kT must be > 0, got {self.kT}")
        if not 0.0 <= self.self_interaction < 1.0:
            raise ValueError(f"self_interaction must lie in [0, 1), got {self.self_interaction}")
        if self.anchor not in ("atomic", "gap", "formal"):
            raise ValueError(f"anchor must be 'atomic', 'gap' or 'formal', got {self.anchor!r}")
        if self.anchor == "formal" and (self.fragments != "bonds" or self.fragment_ensemble != "diabatic"):
            raise ValueError("anchor='formal' (integer formal oxidation states, SPEC 17.5) needs "
                             "fragments='bonds' and fragment_ensemble='diabatic'")
        if self.width == "ionic":
            # Ionic sizes between sites (trace_redox.heads.IonicFreeChargeWidth) with exact isolated-ion
            # energies make the free Gram operator indefinite in an ionic solid: a convex ladder caps the
            # on-site curvature at IE - EA, below the Madelung field of compact charges (formal MgO:
            # lambda_min(J) = -6.6 eV/e^2 even at the cap; tests/redox/test_redox_free_width.py).  The
            # convex mean-field solver cannot hold such charges; they need integer configurations
            # (docs/ELECTROSTATICS_AUDIT_2026-10-09.md, step C2).
            raise ValueError(
                "width='ionic' is not available in the mean-field free sector: compact ionic charges with "
                "exact isolated-ion energies make W indefinite in an ionic solid; they need the integer-"
                "configuration (diabatic) treatment of audit step C2"
            )
        if self.width not in ("hardness", "bound"):
            raise ValueError(f"width must be 'hardness' or 'bound', got {self.width!r}")
        if self.chi_bound <= 0.0 or self.log_gap_bound <= 0.0 or self.gap <= 0.0:
            raise ValueError("chi_bound, log_gap_bound and gap must be positive")
        if not (isinstance(self.max_charge, int) and 1 <= self.max_charge <= MAX_CATION_CHARGE):
            raise ValueError(f"max_charge must be an integer in 1..{MAX_CATION_CHARGE}, got {self.max_charge!r}")
        if self.sic not in ("none", "certified"):
            raise ValueError(f"sic must be 'none' or 'certified', got {self.sic!r}")
        if self.sic == "certified" and self.self_interaction != 0.0:
            raise ValueError("sic='certified' replaces the uniform lam; set self_interaction=0")
        if not 0.0 < self.sic_theta < 1.0 or self.sic_smoothing < 0.0:
            raise ValueError("need 0 < sic_theta < 1 and sic_smoothing >= 0")
        if self.bound_elimination not in ("condensed", "dense"):
            raise ValueError(f"bound_elimination must be 'condensed' or 'dense', got {self.bound_elimination!r}")
        if self.carrier_penetration and (self.anchor != "formal" or self.fragment_ensemble != "diabatic"
                                         or self.free_charge_model != "single"):
            raise ValueError("carrier_penetration=True needs anchor='formal', fragment_ensemble='diabatic' and "
                             "free_charge_model='single' (SPEC 23)")
        if self.carrier_overlap not in ("electrostatic", "gordon_kim"):
            raise ValueError(f"carrier_overlap must be 'electrostatic' or 'gordon_kim', got {self.carrier_overlap!r}")
        if self.pauli_table not in ("gordon_kim", "first_principles"):
            raise ValueError(f"pauli_table must be 'gordon_kim' or 'first_principles', got {self.pauli_table!r}")
        if self.pauli_table == "first_principles" and self.carrier_overlap != "gordon_kim":
            raise ValueError("pauli_table='first_principles' replaces SPEC 25's K: it needs carrier_overlap='gordon_kim'")
        if self.carrier_overlap == "gordon_kim" and not self.carrier_penetration:
            raise ValueError("carrier_overlap='gordon_kim' needs carrier_penetration=True (SPEC 25)")
        if self.carrier_overlap == "gordon_kim" and self.exclusion:
            raise ValueError("carrier_overlap='gordon_kim' carries the Pauli repulsion itself: exclusion=True would count it "
                             "twice (SPEC 25)")
        if self.spin_pairing and not self.couplings:
            raise ValueError("spin_pairing needs couplings=True (it scales the configuration couplings; SPEC 29)")
        if self.bond_pauli_level not in ("almo", "frz"):
            raise ValueError(f"bond_pauli_level must be 'almo' or 'frz', got {self.bond_pauli_level!r}")
        if self.bond_pauli_wall and not self.bond_pauli:
            raise ValueError("bond_pauli_wall needs bond_pauli=True (the wall is the bond-oriented law's reference; SPEC 33)")
        if self.frame_pair and not (self.carrier_penetration and self.carrier_overlap == "gordon_kim"):
            raise ValueError("frame_pair=True is the frame entry of SPEC 23 and SPEC 25: it needs carrier_penetration=True "
                             "and carrier_overlap='gordon_kim' (SPEC 39)")
        if self.two_site and self.downfold_transfer:
            raise ValueError("two_site=True is not built with downfold_transfer=True: the fold's couplings are not "
                             "resolved by charge step (SPEC 40.6 ix)")
        if not isinstance(self.two_site, bool):
            raise ValueError(f"two_site must be True or False, got {self.two_site!r}")
        if self.two_site and self.diabatic_selection != "amplitude":
            raise ValueError("two_site=True needs diabatic_selection='amplitude': the 'window' selection does not see the "
                             "couplings, so a configuration reached by the first-principles hop enters only when it lies "
                             "within diabatic_window of the lowest one, and the energy steps by eV where it does "
                             "(gas-phase Au2: 1.6-2.0 eV; H2 never hops; SPEC 40.4)")
        if self.two_site and not (self.frame_pair and self.spin_pairing and self.couplings):
            raise ValueError("two_site=True needs frame_pair=True (the covalent relaxation is defined against SPEC 39's "
                             "frame term), spin_pairing=True (one hop for the neutral pair and for one carrier, SPEC 29) "
                             "and couplings=True (SPEC 40)")
        if self.frame_pair and self.frame_wall:
            raise ValueError("frame_pair=True holds every frame pair, the Mg-Au pair of frame_wall among them: with "
                             "frame_wall=True that pair would be counted twice (SPEC 39)")
        if self.bond_pauli_states and not self.bond_pauli:
            raise ValueError("bond_pauli_states needs bond_pauli=True (it makes SPEC 33's law depend on the charge state)")
        if self.bond_pauli_states and self.anchor != "formal":
            raise ValueError("bond_pauli_states needs anchor='formal' (it enters through the formal ladder)")
        if self.state_polarizability and not (self.fragment_ensemble == "diabatic" and self.anchor == "formal"):
            raise ValueError("state_polarizability needs fragment_ensemble='diabatic' and anchor='formal' (SPEC 28)")
        if self.state_polarizability_stiffness not in ("onsite", "block"):
            raise ValueError("state_polarizability_stiffness must be 'onsite' (the atom's own stiffness A_mu; SPEC 28.6) "
                             f"or 'block' (legacy: the operator's block P_jj), got {self.state_polarizability_stiffness!r}")
        if self.state_polarizability_elements is not None:
            elements = tuple(int(z) for z in self.state_polarizability_elements)
            if not self.state_polarizability:
                raise ValueError("state_polarizability_elements needs state_polarizability=True (SPEC 28)")
            if not elements or any(z < 1 for z in elements):
                raise ValueError(f"state_polarizability_elements must list atomic numbers >= 1, got {elements}")
            object.__setattr__(self, "state_polarizability_elements", elements)
        if self.coupling_bound not in (None, "harrison"):
            raise ValueError(f"coupling_bound must be None or 'harrison', got {self.coupling_bound!r}")
        if self.coupling_bound is not None and not self.couplings:
            raise ValueError("coupling_bound needs couplings=True (it bounds the learned couplings; SPEC 27)")
        if self.coupling_anchor not in (None, "harrison"):
            raise ValueError(f"coupling_anchor must be None or 'harrison', got {self.coupling_anchor!r}")
        if self.coupling_anchor is not None and not self.couplings:
            raise ValueError("coupling_anchor needs couplings=True (it anchors the couplings; SPEC 31)")
        if self.downfold_transfer and (self.fragment_ensemble != "diabatic" or self.anchor != "formal"):
            raise ValueError("downfold_transfer needs fragment_ensemble='diabatic' and anchor='formal' (frame states; "
                             "SPEC 32)")
        if not float(self.downfold_cutoff) > 0.0:
            raise ValueError(f"downfold_cutoff must be positive, got {self.downfold_cutoff}")
        if not float(self.downfold_scale) > 0.0:
            raise ValueError(f"downfold_scale must be positive, got {self.downfold_scale}")
        if float(self.downfold_anisotropy) < 0.0:
            raise ValueError(f"downfold_anisotropy must be >= 0, got {self.downfold_anisotropy}")
        if float(self.downfold_crystal_field) < 0.0:
            raise ValueError(f"downfold_crystal_field must be >= 0, got {self.downfold_crystal_field}")
        if not float(self.downfold_bond_cutoff) > 0.0:
            raise ValueError(f"downfold_bond_cutoff must be positive, got {self.downfold_bond_cutoff}")
        if self.downfold_hop_ordering not in ("bonding", "pauli"):
            raise ValueError(f"downfold_hop_ordering must be 'bonding' (legacy) or 'pauli', got {self.downfold_hop_ordering!r}")
        if self.downfold_block not in ("per_state", "exact"):
            raise ValueError(f"downfold_block must be 'per_state' or 'exact', got {self.downfold_block!r}")
        if not float(self.downfold_decay) >= 0.0:
            raise ValueError(f"downfold_decay must be >= 0 (1/A), got {self.downfold_decay}")
        if not float(self.downfold_decay_anchor) > 0.0:
            raise ValueError(f"downfold_decay_anchor must be positive (A), got {self.downfold_decay_anchor}")
        if self.carrier_induction and not self.carrier_penetration:
            raise ValueError("carrier_induction=True needs carrier_penetration=True (the same density library; SPEC 26)")
        if self.frame_induction_damping and self.anchor != "formal":
            raise ValueError("frame_induction_damping=True needs anchor='formal' (the frame charges it damps; SPEC 34)")
        if self.frame_polarizability_anchor and self.anchor != "formal":
            raise ValueError("frame_polarizability_anchor=True needs anchor='formal' (the frame states it reads; SPEC 35)")
        if self.frame_polarizability_scope not in ("ions", "all"):
            raise ValueError(f"frame_polarizability_scope must be 'ions' or 'all', got {self.frame_polarizability_scope!r}")
        if self.shift_gate and self.anchor != "formal":
            raise ValueError("shift_gate=True needs anchor='formal' (SPEC 24)")
        if self.bond_partition and (self.anchor != "formal" or self.fragment_ensemble != "diabatic"):
            raise ValueError("bond_partition=True needs anchor='formal' and fragment_ensemble='diabatic' (SPEC 22)")
        if self.diabatic_selection not in ("window", "amplitude"):
            raise ValueError(f"diabatic_selection must be 'window' or 'amplitude', got {self.diabatic_selection!r}")
        if self.diabatic_selection == "amplitude":
            if self.fragment_ensemble != "diabatic":
                raise ValueError("diabatic_selection='amplitude' needs fragment_ensemble='diabatic'")
            if not 0.0 < self.selection_amplitude_off < self.selection_amplitude_on:
                raise ValueError("need 0 < selection_amplitude_off < selection_amplitude_on, got "
                                 f"{self.selection_amplitude_off}, {self.selection_amplitude_on}")
        if self.diabatic_space not in ("global", "clusters"):
            raise ValueError(f"diabatic_space must be 'global' or 'clusters', got {self.diabatic_space!r}")
        if self.diabatic_space == "clusters":
            if self.fragment_ensemble != "diabatic":
                raise ValueError("diabatic_space='clusters' needs fragment_ensemble='diabatic' (SPEC 36)")
            if self.downfold_transfer:
                raise ValueError("diabatic_space='clusters' is not built with downfold_transfer (SPEC 36, limit L5): "
                                 "the fold dresses one global block")
        try:
            mesh = tuple(self.bvk_mesh)
            whole = len(mesh) == 3 and all(not isinstance(m, bool) and float(m) == int(m) and int(m) >= 1 for m in mesh)
        except (TypeError, ValueError):
            whole = False
        if not whole:
            raise ValueError(f"bvk_mesh must be three positive integers, got {self.bvk_mesh!r}")
        object.__setattr__(self, "bvk_mesh", tuple(int(m) for m in mesh))
        if isinstance(self.bvk_length, bool) or not float(self.bvk_length) >= 0.0:
            raise ValueError(f"bvk_length must be a length >= 0 (0: off), got {self.bvk_length!r}")
        object.__setattr__(self, "bvk_length", float(self.bvk_length))
        if self.bvk_mesh != (1, 1, 1) or self.bvk_length > 0.0:
            if self.bvk_mesh != (1, 1, 1) and self.bvk_length > 0.0:
                raise ValueError("give bvk_mesh or bvk_length, not both (SPEC 37)")
            if self.fragment_ensemble != "diabatic" or self.diabatic_space != "clusters":
                raise ValueError("Born-von Karman sampling (bvk_mesh, bvk_length) needs fragment_ensemble='diabatic' "
                                 "and diabatic_space='clusters' (SPEC 37): only there do the copies of a cluster "
                                 "fluctuate independently")
        if self.fragments not in ("none", "bonds"):
            raise ValueError(f"fragments must be 'none' or 'bonds', got {self.fragments!r}")
        if self.exclusion and self.anchor != "formal":
            raise ValueError("exclusion=True needs anchor='formal' (it reads the formal frames and the exact ladder)")
        if self.exclusion_relaxation and not self.exclusion:
            raise ValueError("exclusion_relaxation=True needs exclusion=True")
        if self.carrier_band and (self.anchor != "formal" or self.fragment_ensemble != "diabatic"):
            raise ValueError("carrier_band=True needs anchor='formal' and fragment_ensemble='diabatic'")
        if self.band_form not in ("phi1", "fermi"):
            raise ValueError(f"band_form must be 'phi1' or 'fermi', got {self.band_form!r}")
        if self.band_form == "fermi" and not self.carrier_band:
            raise ValueError("band_form='fermi' needs carrier_band=True (SPEC 38)")
        if self.band_form == "fermi" and self.downfold_transfer:
            raise ValueError("band_form='fermi' is not built with downfold_transfer (SPEC 38)")
        f = float(self.band_initial_factor)
        if not (math.isfinite(f) and 1.0 / 20.0 < f < 20.0):
            raise ValueError(f"band_initial_factor must lie in the open range (1/20, 20) of exp(ln 20 tanh s), got {f}")
        if self.exclusion_anions not in ("learned", "in_situ"):
            raise ValueError(f"exclusion_anions must be 'learned' or 'in_situ', got {self.exclusion_anions!r}")
        if self.exclusion_anion_holes and self.exclusion_anions != "in_situ":
            raise ValueError("exclusion_anion_holes needs exclusion_anions='in_situ'")
        if self.exclusion_anions == "in_situ":
            if not self.exclusion:
                raise ValueError("exclusion_anions='in_situ' needs exclusion=True")
            if self.exclusion_bound or self.exclusion_relaxation:
                raise ValueError("exclusion_anions='in_situ' is not implemented with exclusion_bound or "
                                 "exclusion_relaxation")
        if self.exclusion_pairs and not self.exclusion:
            raise ValueError("exclusion_pairs=True needs exclusion=True")
        if self.exclusion_bound and not self.exclusion:
            raise ValueError("exclusion_bound=True needs exclusion=True")
        if int(self.full_unit_atoms) < 1:
            raise ValueError("full_unit_atoms must be >= 1")
        if self.carriers not in ("atoms", "covalent_units"):
            raise ValueError(f"carriers must be 'atoms' or 'covalent_units', got {self.carriers!r}")
        if self.carriers == "covalent_units" and (self.anchor != "formal" or self.fragment_ensemble != "diabatic"):
            raise ValueError("carriers='covalent_units' needs anchor='formal' and fragment_ensemble='diabatic'")
        if self.carriers == "covalent_units" and self.frame_pair:
            raise ValueError("frame_pair=True is not defined for carriers='covalent_units': a bond inside a unit is held by "
                             "the backbone with no hop to pair the frame repulsion with, and the rule for pairs inside a "
                             "unit is not decided (SPEC 39.6 v)")
        if self.free_width_rule not in ("ionic", "frame"):
            raise ValueError(f"free_width_rule must be 'ionic' or 'frame', got {self.free_width_rule!r}")
        if self.free_width_rule == "frame" and self.anchor != "formal":
            raise ValueError("free_width_rule='frame' needs anchor='formal' (it reads the formal frames)")
        if self.free_charge_model not in ("single", "core_carrier"):
            raise ValueError(f"free_charge_model must be 'single' or 'core_carrier', got {self.free_charge_model!r}")
        if self.free_charge_model == "core_carrier" and self.anchor != "formal":
            raise ValueError("free_charge_model='core_carrier' needs anchor='formal' (integer configurations and frames)")
        if self.bound_electronegativity not in ("learned", "none"):
            raise ValueError(f"bound_electronegativity must be 'learned' or 'none', got {self.bound_electronegativity!r}")
        if self.bound_electronegativity == "none" and self.anchor != "formal":
            raise ValueError(
                "bound_electronegativity='none' needs anchor='formal': only the exact ladders then carry "
                "electronegativity (with atomic or mean-field anchors the bound drive is the charge physics)"
            )
        if self.fragments == "bonds":
            if self.orbitals:
                raise ValueError("fragments='bonds' with orbitals=True is not supported (an orbital may span two molecules)")
            if self.self_interaction != 0.0:
                raise ValueError("fragments='bonds' uses the certified removals; set self_interaction=0")
            if self.fragment_window < 1 or self.fragment_max_uncertain < 0:
                raise ValueError("need fragment_window >= 1 and fragment_max_uncertain >= 0")
            if self.fragment_ensemble not in ("exact", "two-level", "diabatic"):
                raise ValueError(
                    f"fragment_ensemble must be 'exact', 'two-level' or 'diabatic', got {self.fragment_ensemble!r}")
            if not self.diabatic_window > 0.0:
                raise ValueError(f"diabatic_window must be > 0, got {self.diabatic_window}")
            if self.couplings and self.fragment_ensemble != "diabatic":
                raise ValueError("couplings=True needs fragment_ensemble='diabatic' (configurations to couple)")
            if not 0.0 < self.coupling_cutoff <= self.attention_cutoff:
                raise ValueError("need 0 < coupling_cutoff <= attention_cutoff (both read the model's neighbour list)")
            if self.fragment_max_configurations < 1:
                raise ValueError("need fragment_max_configurations >= 1")


@dataclasses.dataclass
class RedoxResult:
    """Everything one evaluation produced; tensors carry the graph."""

    energy: Tensor             #: [] total electrostatic + redox energy, eV
    energy_bound: Tensor       #: [] E_b0, the SPEC 1 minimum with no free charge
    energy_free: Tensor        #: [] optimal free-sector energy (bound response included)
    q: Tensor                  #: [n] total atomic charges q_b + f, e
    q_bound: Tensor            #: [n] bound charges B t, e (zero net per fragment)
    free_charge: Tensor        #: [n] free charge on the atoms, f = Omega^T c, e
    unit_charge: Tensor        #: [U] free charge per unit c, e
    populations: Tensor        #: [U, K + 2] populations of the states (-1, 0, +1, ..., +K)
    states: Tensor             #: [K + 2] the charge states n (electrons removed)
    state_energies: Tensor     #: [U, K + 2] integer-state energies E_un, eV (+inf: state absent, Z < n)
    electronegativity: Tensor  #: [U] chi_u, eV
    gap: Tensor                #: [U] G_u = g_1, the first gap (between n = 0 and +1 / -1), eV (> 0)
    sigma_free: Tensor         #: [n] free-charge Gaussian widths, A
    sic_removal: Tensor | None  #: [U] certified on-site removal r_u (eV / e^2), or None
    mu: Tensor                 #: [n, 3] physical dipoles, e A
    t: Tensor                  #: [n_bonds] bond variables
    rho: Tensor                #: [n] zeros (bound fragments neutral)
    theta_hat: Tensor | None   #: [n, 5] reduced quadrupoles, or None
    mu_e: Tensor               #: [] electrochemical potential of the free charge, eV / e (NaN if all units are saturated)
    nu: Tensor                 #: [U] unit potentials, eV / e
    omega: Tensor | None       #: [U, n] orbitals, or None (units = atoms)
    #: fragments='bonds' (SPEC §12): [(partition weight, molecules' atom lists,
    #: [(charge window, populations)] per molecule)], or None (one molecule / mean field).
    #: In that mode ``nu`` is the pi-weighted molecular potential, ``populations``
    #: the p- and pi-weighted atomic state populations, and ``duality_gap`` is NaN.
    molecules: list | None
    iterations: int
    residual_charge: float
    residual_total: float
    duality_gap: float
    #: SPEC 30: the configuration ensemble's response record (diabatic formal sector with several configurations):
    #: {'H': diabatic Hamiltonian, 'M': [m, 3] configuration dipoles, 'kT', 'alpha_bound': [3, 3] D^T K^-1 D,
    #: 'exact': bool, 'reason': str, and with SPEC 28 'alpha_shift': [m, 3, 3] each configuration's bound alpha minus
    #: alpha_bound}; None with one configuration or outside the diabatic sector
    #: SPEC 36.10: on several clusters 'H' is None and 'terms' holds one entry per free energy of the cluster product,
    #: {'weight', 'kind' ('reference' / 'cluster' / 'pair'), 'H', 'rows' (into 'M' and 'alpha_shift')}; at a mesh
    #: (SPEC 37) 'alpha_bound' is per cell and 'scale' = 1 / copies multiplies the ensemble part.  Read it with
    #: trace_redox.response.record_polarizability.
    ensemble: dict | None = None
    #: SPEC 37: the Born-von Karman mesh of this evaluation.  Beyond (1, 1, 1) the energies are per cell and the
    #: per-atom fields (q, q_bound, free_charge, sigma_free, mu, rho, theta_hat) are the home cell's; the unit, bond,
    #: state and ensemble fields (unit_charge, populations, t, nu, omega, molecules, ensemble, ...) stay the
    #: supercell's, atoms in the order of trace_redox.bvk.tile (copy c, atom i at c n + i; the home cell is copy 0).
    bvk_mesh: tuple = (1, 1, 1)


def carrier_widths(species: Tensor, sigma_core: Tensor, max_charge: int) -> Tensor:
    """``[n]`` carrier widths sigma_car = sqrt(2) K_E / (sqrt(pi) U_min), from the exact ladder (SPEC 17.9).

    U_min is the smallest curvature E_Z(n+1) + E_Z(n-1) - 2 E_Z(n) over the unit's interior states (both
    neighbours allowed).  Then the Gram self-energy of the carrier, S0(sigma_car) = U_min, never exceeds the
    exact curvature of a state the carrier can move through.  No parameter; ``sigma_core`` gives the dtype.
    """
    from trace_electrostatics.constants import K_E as _K_E
    states, E, L, K = formal_ladder(species, max_charge)
    E = E.to(sigma_core)
    curvature = E[:, 2:] + E[:, :-2] - 2.0 * E[:, 1:-1]                    # at states[1:-1]
    interior = states[1:-1].to(sigma_core).unsqueeze(0)
    allowed = (interior > L.to(sigma_core).unsqueeze(1)) & (interior < K.to(sigma_core).unsqueeze(1)) \
        & torch.isfinite(curvature)
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("core + carrier widths need at least one interior state per unit (max_charge >= 1)")
    U_min = torch.where(allowed, curvature, torch.full_like(curvature, float("inf"))).min(dim=1).values
    if bool((U_min <= 0).any()):
        raise ValueError("a non-convex ladder (curvature <= 0) has no carrier width")
    return math.sqrt(2.0) * _K_E / (math.sqrt(math.pi) * U_min)


def without_bound_drive(assembly: Any) -> Any:
    """The assembly with the bound sector's own electronegativity (chi, chi_Theta) set to zero (SPEC 17.8).

    chi and chi_Theta enter U_b only linearly, so K, the free-charge couplings A and J, and the response W
    are unchanged; with no applied field the bound solution y0, its energy E_b0 and its potential b vanish.
    """
    params = assembly.params
    chi_theta = None if params.chi_theta is None else torch.zeros_like(params.chi_theta)
    return dataclasses.replace(assembly, params=dataclasses.replace(params, chi=torch.zeros_like(params.chi),
                                                                    chi_theta=chi_theta))


def lambda_max(W: Tensor, J_self: Tensor) -> Tensor:
    """``lam_max = lambda_min(D^{-1/2} W D^{-1/2})``, ``D = diag(J_uu)`` (SPEC 25.2)."""
    scale = J_self.rsqrt()
    return torch.linalg.eigvalsh(scale.unsqueeze(1) * W * scale.unsqueeze(0)).min()


def _saturated(b, W, e_hat, states, index):
    """``Q`` at an end of the charge capacity: unit u in state ``index[u]``, no entropy."""
    c = states[index]
    energy = e_hat.gather(1, index.unsqueeze(1)).sum() + b @ c + 0.5 * (c @ (W @ c))
    populations = torch.zeros_like(e_hat)
    populations[torch.arange(int(index.shape[0]), device=index.device), index] = 1.0
    nan = torch.tensor(float("nan"), dtype=b.dtype, device=b.device)
    return FreeChargeSolution(
        z=c, mu_e=nan, nu=torch.full_like(c, float("nan")), charge=c, populations=populations,
        energy=energy, iterations=0, residual_charge=0.0, residual_total=0.0, duality_gap=0.0,
    )


class RedoxElectrostatics(nn.Module):
    """Drop-in for ``ElectrostaticModule`` on ``flashace`` models (forward signature and result)."""

    def __init__(self, heads: nn.Module, config: ElectrostaticConfig | None = None,
                 redox: RedoxConfig | None = None) -> None:
        super().__init__()
        config = ElectrostaticConfig() if config is None else config
        if not isinstance(config, ElectrostaticConfig):
            raise TypeError("config must be an ElectrostaticConfig")
        config.validate_dtype()
        if config.dispersion:
            raise NotImplementedError("trace_redox v1 does not carry the dispersion term")
        if config.multipole_kernel != "gaussian":
            raise NotImplementedError("trace_redox needs multipole_kernel='gaussian'")
        if getattr(heads, "spin", False):
            raise NotImplementedError("trace_redox v1 does not support spin heads")
        self.heads = heads
        if redox.frame_polarizability_anchor:                                 # SPEC 35: the crystal factor g = exp(.)
            self.log_alpha_scale = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.config = dataclasses.replace(config, solver="dense")
        self.redox = RedoxConfig() if redox is None else redox
        #: builders, probes and tests only: True evaluates the package's FIXED terms alone, with every learned level
        #: shift and every learned coupling at zero (SPEC 40.7: the entries of an s1 pair are defined against them).
        #: Not a parameter and not in the state dict; refused in training mode.
        self.fixed_terms_only = False
        r = self.redox
        invariant_dim = int(heads.scalar_dim + heads.non_scalar_norm.irreps_out.dim)
        self.charge_states = ChargeStateHead(
            invariant_dim, hidden_dim=r.hidden_dim, key_dim=r.key_dim,
            r_cut=r.attention_cutoff, n_radial=r.n_radial,
            n_tokens=1 + r.max_charge + (1 if r.anchor == "formal" else 0),     # formal: a -2 token too
        )
        if r.anchor == "formal":
            self.width = IonicFreeChargeWidth(fallback_qeq=True)               # compact ionic sizes (SPEC 17.5)
        else:
            self.width = (
                FreeChargeWidth(r.self_interaction, h_init=r.hartree_fraction)
                if r.width == "hardness" else None
            )
        self.coupling = (
            TransferCoupling(invariant_dim, hidden_dim=r.hidden_dim, n_radial=r.n_radial, r_cut=r.coupling_cutoff,
                             bound=r.coupling_bound, anchor=r.coupling_anchor)
            if r.couplings else None
        )
        from .exclusion import Exclusion
        self.exclusion = Exclusion(r_cut=r.exclusion_cutoff, relax=r.exclusion_relaxation) if r.exclusion else None
        from .band import BandUnit
        self.band = BandUnit(r_cut=r.band_cutoff, initial_factor=r.band_initial_factor) if r.carrier_band else None
        self._band_last = None
        self.orbitals = (
            OrbitalHead(invariant_dim, hidden_dim=r.hidden_dim, key_dim=r.key_dim,
                        r_cut=r.attention_cutoff, n_radial=r.n_radial, s_max=r.orbital_s_max,
                        s_init=r.orbital_s_init)
            if r.orbitals else None
        )
        ie = torch.tensor(ATOMIC_IONIZATION_EV, dtype=torch.float64)
        ea = torch.tensor(ATOMIC_ELECTRON_AFFINITY_EV, dtype=torch.float64)
        self.register_buffer("mulliken", 0.5 * (ie + ea), persistent=False)
        self.register_buffer("hardness", ie - ea, persistent=False)
        self.register_buffer(
            "ionization_series", torch.tensor(ATOMIC_SUCCESSIVE_IONIZATION_EV, dtype=torch.float64),
            persistent=False,
        )
        for name, p in self.named_parameters():
            if p.dtype != torch.float64:
                raise TypeError(
                    f"parameter {name} is {p.dtype}; trace_redox runs in float64 "
                    "(torch.set_default_dtype(torch.float64) before construction)"
                )

    # -- the flashace seams (duck-typed by flashace.model) ---------------------

    def take_solver_seed_loss(self):
        """No seed loss: both solves are exact (dense Cholesky, dual Newton)."""
        return None

    def state(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "observables (dipole / polarizability / Born-charge losses) are not defined for "
            "trace_redox v1: its energy is not a single SPEC 1 state (with anchor='formal' the "
            "electronic response is the bound sector's: bound_state, SPEC 17.7)"
        )

    # -- evaluation -------------------------------------------------------------

    def _inputs(self, pos: Tensor, data: Mapping[str, Any] | None, pbc: Any) -> dict:
        data = {} if data is None else data
        for key in ("fragment_id", "fragment_charges", "fragment_spins", "charge_states",
                    "metallic", "electrode_potential"):
            if data.get(key) is not None:
                raise NotImplementedError(f"data[{key!r}] is not supported by trace_redox v1")
        total = data.get("total_charge", 0.0)
        if torch.is_tensor(total):
            if total.numel() != 1:
                raise ValueError("total_charge must be one number per structure")
            total = float(total.detach().cpu())
        quadrupoles = data.get("quadrupoles")
        if quadrupoles is None:
            quadrupoles = "A_Theta" in tuple(getattr(self.heads, "quadrupole_blocks", ()) or ())
        field = data.get("external_field")
        if field is not None:
            field = torch.as_tensor(field, dtype=pos.dtype, device=pos.device).reshape(3)
        species = data.get("z")
        if species is not None:
            species = torch.as_tensor(species, device=pos.device).long()
            if bool(((species < 1) | (species > MAX_Z)).any()):
                raise ValueError(f"atomic numbers must lie in 1..{MAX_Z}")
        return {
            "total_charge": float(total),
            "pbc": data.get("pbc") if pbc is None else pbc,
            "species": species,
            "quadrupoles": bool(quadrupoles),
            "external_field": field,
        }

    def free_widths(self, species: Tensor) -> Tensor:
        """``[n]`` free-charge widths (A): the width module's, and with free_width_rule='frame' the fixed ladder
        width for neutral-frame elements (SPEC 17.10)."""
        sigma = self.width(species)
        if self.redox.free_width_rule != "frame":
            return sigma
        from .constants import formal_frame
        neutral = formal_frame(species).to(sigma.device) == 0
        return torch.where(neutral, carrier_widths(species, sigma, int(self.redox.max_charge)), sigma)

    def bound_state(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        pbc: Any = None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        data: Mapping[str, Any] | None = None,
    ):
        """The bound sector alone (no free charge) as a ``trace_electrostatics.ElectrostaticState``.

        Its energy is :attr:`RedoxResult.energy_bound` of :meth:`forward` on the same inputs (the same
        input resolution, every bound fragment neutral, the dense exact solve of ``assemble_neutral``).
        It is the source of the model's electronic response: an applied field shifts every integer
        configuration n by -d_n . E - 1/2 E . alpha_b . E with the same alpha_b (W and b do not depend
        on E), so -d2F/dE2 = alpha_b + Var_p(d) / kT.  With anchor='formal' the second term is weighted
        by the populations above the ground configuration, ~exp(-gap / kT).
        """
        from trace_electrostatics.energy import assemble, state_from_assembly

        inputs = self._inputs(pos, data, pbc)
        config = self.config if self.config.solver == "dense" else dataclasses.replace(self.config, solver="dense")
        assembly = assemble(
            h, pos, edge_index, edge_vec, edge_len, self._heads_for(inputs["species"]), config,
            cell=cell, edge_shift=edge_shift, pbc=inputs["pbc"], total_charge=0.0,
            quadrupoles=inputs["quadrupoles"], species=inputs["species"],
            external_field=inputs["external_field"],
        )
        if self.redox.bound_electronegativity == "none":
            assembly = without_bound_drive(assembly)
        assembly = self._partition_bonds(assembly, inputs["species"], edge_index, edge_len)
        assembly = self._with_exclusion_anchors(assembly, inputs["species"], edge_index, edge_len, edge_vec)
        return state_from_assembly(assembly, config)

    def _heads_for(self, species):
        """The heads as the assembly sees them: SPEC 35's anchored readout when the flag is on, else the heads."""
        if not self.redox.frame_polarizability_anchor or species is None:
            return self.heads
        from .polarizability_anchor import AnchoredHeads
        return AnchoredHeads(self.heads, species, self.log_alpha_scale, self.redox.frame_polarizability_scope)

    def _damp_frame_induction(self, elim, assembly, species: Tensor, sigma_free: Tensor, pos: Tensor, cell, pbc):
        """SPEC 34: the elimination with the frame's overlap-damped drive on the bound dipoles (frame_damping.py)."""
        from trace_electrostatics.solver import gaussian_widths
        from .constants import formal_frame
        from .frame_damping import frame_induction_damping
        dtype = elim.y0.dtype
        w_mu, _ = gaussian_widths(assembly.params, assembly.config.gaussian_a)
        return frame_induction_damping(elim, species, formal_frame(species).to(pos.device), sigma_free.to(dtype),
                                       w_mu.to(dtype), assembly.params.ell.to(dtype), pos.to(dtype), cell, pbc)

    def band_diagnostics(self, h: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
                         data: Mapping[str, Any]) -> dict:
        """The band unit's one-carrier quantities (eps, tau, lambda1, phi, J11, sites) for one evaluation (SPEC 20)."""
        if self.band is None:
            raise ValueError("carrier_band is off")
        self.forward(h, pos, edge_index, edge_vec, edge_len, data=data)
        return self._band_last

    def _with_exclusion_anchors(self, assembly: Any, species: Tensor, edge_index: Tensor, edge_len: Tensor,
                                edge_vec: Tensor) -> Any:
        """The assembly with the exclusion's Parr-Pearson anchors in the bound sector's chi and eta (SPEC 19.2);
        unchanged unless ``exclusion_bound``.  Used by forward and bound_state alike, so energy_bound and the
        bound state stay the same functional."""
        if not self.redox.exclusion_bound:
            return assembly
        from .constants import formal_frame
        from .exclusion import exclusion_bound_anchors
        params = assembly.params
        dtype, device = params.chi.dtype, params.chi.device
        frames = formal_frame(species).to(device)
        states = torch.arange(int(frames.min()) - 1, int(frames.max()) + 2, dtype=dtype, device=device)
        Pi = self.exclusion(species, states, edge_index, edge_len.to(dtype), frames, edge_vec=edge_vec.to(dtype),
                            alpha=assembly.alpha_volume)
        chi_P, eta_P = exclusion_bound_anchors(Pi, states, frames)
        return dataclasses.replace(assembly, params=dataclasses.replace(params, chi=params.chi + chi_P,
                                                                        eta=params.eta + eta_P))

    def _partition_bonds(self, assembly: Any, species: Tensor, edge_index: Tensor, edge_len: Tensor) -> Any:
        """SPEC 22: the assembly with f_b -> 0 on every bond that is not inside one covalent unit of more than
        ``full_unit_atoms`` atoms (SPEC 19 units; with carriers='atoms' every atom is its own unit, so every bond).
        The envelope scales the incidence, the seed and the periodic bond dipole, so every consumer sees it; the
        bond's own hardness stays, decoupled.  Unchanged unless ``bond_partition``."""
        if not self.redox.bond_partition:
            return assembly
        from .constants import formal_frame
        n = int(species.shape[0])
        device = assembly.pos.device
        if self.redox.carriers == "covalent_units":
            from .fragments import covalent_units
            frames = formal_frame(species).to(device)
            bi, bj, _ = chemical_bonds(species, edge_index, edge_len, coverage=float(self.redox.attention_cutoff))
            units = covalent_units(frames, bi, bj, n, full_unit_atoms=int(self.redox.full_unit_atoms))
        else:
            units = torch.arange(n, device=device)
        size = torch.zeros(int(units.max()) + 1 if n else 0, dtype=torch.long, device=device).index_add(
            0, units, torch.ones_like(units))
        i, j = assembly.bond_index[0], assembly.bond_index[1]
        dtype = assembly.envelope.dtype if assembly.envelope is not None else assembly.pos.dtype
        keep = ((units[i] == units[j]) & (size[units[i]] > 1)).to(dtype)
        if assembly.incidence.is_sparse:
            raise ValueError("bond_partition needs the dense incidence of the dense route")
        return dataclasses.replace(
            assembly,
            incidence=assembly.incidence * keep.unsqueeze(0),
            t_seed=assembly.t_seed * keep,
            bond_dipole=None if assembly.bond_dipole is None else assembly.bond_dipole * keep.unsqueeze(-1),
            envelope=None if assembly.envelope is None else assembly.envelope * keep,
        )

    def _state_ratio_table(self, species: Tensor, states: Tensor) -> Tensor:
        """SPEC 28's ratios rho = alpha(Z, s) / alpha(Z, frame), with elements outside
        ``state_polarizability_elements`` (if given) held at their frame (rho = 1)."""
        from .polarizability import ratio_table
        table = ratio_table(species, states)
        elements = self.redox.state_polarizability_elements
        if elements is None:
            return table
        keep = torch.isin(species.to(torch.long), torch.tensor(elements, dtype=torch.long, device=species.device))
        return torch.where(keep.unsqueeze(1).to(table.device), table, torch.ones_like(table))

    def _state_stiffness(self, elim) -> Tensor | None:
        """SPEC 28.6: what a charge state scales.  'onsite': each atom's own dipole stiffness A_mu ``[n, 3, 3]``;
        'block' (legacy): None, and SPEC 28 takes the operator's dipole block, as it did before 2026-11-10."""
        if self.redox.state_polarizability_stiffness == "block":
            return None
        return elim.assembly.params.A_mu

    def _ensemble_record(self, elim, species, states, inputs, pos, cell, omega, induction):
        """SPEC 30: the response record of the last diabatic solution (trace_redox/response.py), or None with one
        configuration.  Exact for the bound-sector elimination, with SPEC 28's state-dependent K(n) through
        ``alpha_shift [m, 3, 3]`` (each configuration's bound alpha minus alpha_bound); inexact with core_carrier.
        Built only while ``self.record_ensemble`` is True (train.py's response priors set it around their forward), so
        ordinary energies and forces pay nothing for it."""
        from .band import atom_charges
        from .bound import _variational_indices
        from .response import free_dipoles
        if not getattr(self, "record_ensemble", False):        # opt-in: the response priors switch it on
            return None
        last = getattr(self, "_ensemble_last", None)
        if last is None or int(last["configs"].shape[0]) < 2:
            return None
        r = self.redox
        if r.free_charge_model == "core_carrier":
            return {"exact": False, "reason": "core_carrier: the elimination's variables are not the bound sector's "
                                              "(SPEC 30)"}
        terms = None
        if last.get("H") is None:                     # SPEC 36.10: several clusters, one record term per free energy
            # F = E0 + sum_c (F_c - E0) + sum_{c<c'} (F_cc' - F_c - F_c' + E0) is linear in its terms, so the
            # response is the same combination of theirs.  Rows: the reference, every cluster's configurations,
            # every pair space's; each row gets its own dipole (and SPEC 28 shift) below, as in SPEC 30.
            clusters, pairs = last["clusters"], last["pair_index"]
            n_c, n_p = len(clusters), len(pairs)
            blocks = [last["reference"].unsqueeze(0)] + list(clusters) + list(last["pair_configs"])
            bounds = [0]
            for block in blocks:
                bounds.append(bounds[-1] + int(block.shape[0]))
            configs = torch.cat([block.to(torch.long) for block in blocks], dim=0)
            E0 = last["hamiltonians"][0][0, 0].reshape(1, 1)
            weights = [1.0 - n_c + n_p] + [2.0 - n_c] * n_c + [1.0] * n_p
            kinds = ["reference"] + ["cluster"] * n_c + ["pair"] * n_p
            hams = [E0] + list(last["hamiltonians"]) + list(last["pair_hamiltonians"])
            terms = [{"weight": w, "kind": kind, "H": Ht,
                      "rows": torch.arange(bounds[k], bounds[k + 1], device=configs.device)}
                     for k, (w, kind, Ht) in enumerate(zip(weights, kinds, hams))]
            H, kT = None, last["kT"]
        else:
            configs, H, kT = last["configs"], last["H"], last["kT"]
        dtype = elim.y0.dtype
        n = int(elim.A.shape[0])
        cfg = configs.to(dtype)
        q_atoms = atom_charges(configs, omega, dtype)                                 # [m_c, n]
        Y = elim.y0.unsqueeze(1) - elim.operator.solve(elim.A.transpose(0, 1) @ q_atoms.transpose(0, 1))
        if induction is not None:                                                     # SPEC 26: the full source
            N, Qc = induction
            S = int(states.numel())
            k_idx = (configs[:, :n].to(torch.long) - int(round(float(states[0]))))
            flat = torch.arange(n, device=configs.device).unsqueeze(0) * S + k_idx
            Qsel = Qc.to(dtype)[torch.arange(n, device=configs.device).unsqueeze(0), k_idx]
            if hasattr(N, "solve_contracted"):                                         # InductionColumns
                Y = Y - N.solve_contracted([flat[c] for c in range(int(configs.shape[0]))],
                                           [Qsel[c] for c in range(int(configs.shape[0]))])
            else:
                Y = Y - torch.stack([N[:, flat[c]] @ Qsel[c] for c in range(int(configs.shape[0]))], dim=1)
        D = elim.assembly.dipole_derivative().to(dtype)[_variational_indices(elim.assembly)]   # [m, 3]
        M = free_dipoles(configs, last["K"], pos.to(dtype), cell, inputs["pbc"], omega=omega) + Y.transpose(0, 1) @ D
        alpha_bound = D.transpose(0, 1) @ elim.operator.solve(D)
        record = {"H": H, "M": M, "kT": kT, "alpha_bound": 0.5 * (alpha_bound + alpha_bound.transpose(0, 1)),
                  "exact": True, "reason": ""}
        if terms is not None:
            record["terms"] = terms
        if r.state_polarizability:                    # SPEC 28: K(n) per configuration, exact by Woodbury
            from .constants import formal_frame
            from .polarizability import state_polarizability_response
            dM, dalpha = state_polarizability_response(configs, elim, formal_frame(species).to(configs.device),
                                                       self._state_ratio_table(species, states), states, D, omega=omega,
                                                       induction=induction, stiffness=self._state_stiffness(elim))
            record["M"] = M + dM
            record["alpha_shift"] = dalpha
        fold = getattr(self, "_downfold_last", None)
        if r.downfold_transfer and fold is not None and int(fold.get("dressed", 0)) > 0:
            record["exact"] = False                       # SPEC 32: the fold's field dependence is not in the record
            record["reason"] = ("SPEC 32: the downfolded transfer's field dependence (d lambda / dF) is not in the "
                                "response record")
        return record

    def _fermi_band(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, L_units: Tensor, K_units: Tensor,
                    frames: Tensor, T: Tensor | None, edge_index: Tensor, edge_len: Tensor, pairs,
                    units: Tensor | None, frames_c: Tensor | None, Q: float):
        """SPEC 38: the Fermi band unit at its reference filling N0, or None when no atom is a band site.

        The unit's state axis holds the fillings N0 - K .. N0 - L only, so N0 must be the filling of the lowest
        configuration.  It starts at the number of electrons outside every frame, N_x = sum(frames) - Q, cut to what
        the band holds; a detached pass (the solver's own search, window 0) finds the lowest configuration, and N0
        moves to its filling until that is N0 itself.  The energy of a filling does not depend on N0: the passes only
        place the window.  A cycle takes its smallest filling; more than ``FERMI_REFERENCE_PASSES`` passes are refused."""
        r = self.redox
        kT = float(r.kT)
        total = int(round(Q))
        args = (L_units, K_units, frames, None if T is None else T.detach(), edge_index, edge_len)
        with torch.no_grad():
            e_d, b_d, W_d = e_hat.detach(), b.detach(), W.detach()
            pairs_d = None if pairs is None else (pairs[0], pairs[1], pairs[2].detach())
            carriers = self.band._carriers(e_d, states, b_d, W_d, L_units, frames.to(torch.long), edge_index, edge_len,
                                           pairs_d)
            if carriers is None:
                return None
            sites, onsite = carriers[0], carriers[3]
            capacity = 2 * int(sites.numel()) if bool(torch.isfinite(onsite).all()) else 1
            units_x = None if units is None else torch.cat([units, (units.max() + 1).reshape(1)])
            frames_x = None if frames_c is None else torch.cat(
                [frames_c, torch.zeros(1, dtype=frames_c.dtype, device=frames_c.device)])
            N0 = min(max(int(frames.sum()) - total, 0), capacity)
            seen = []
            for _ in range(FERMI_REFERENCE_PASSES):
                trial = self.band.augment_fermi(e_d, states, b_d, W_d, *args, pairs=pairs_d, kT=kT, reference=N0)
                pairs_x = trial["pairs"] if pairs_d is None else tuple(
                    torch.cat([x, y]) for x, y in zip(pairs_d, trial["pairs"]))
                low, E_low = search_configurations(trial["e_hat"], states, trial["b"], trial["W"], trial["K"],
                                                   total + N0, window=0.0, cap=int(r.fragment_max_configurations),
                                                   min_charge=trial["L"], units=units_x, frames=frames_x, pairs=pairs_x)
                moved = int(low[int(torch.argmin(E_low))][-1])
                if moved == 0:
                    break
                seen.append(N0)
                N0 = N0 - moved
                if N0 in seen:                             # a cycle of the descent: its smallest filling
                    N0 = min(seen[seen.index(N0):])
                    break
            else:
                raise ConfigurationSearchError(
                    f"SPEC 38: the band's reference filling did not settle in {FERMI_REFERENCE_PASSES} passes "
                    f"(last {N0}, visited {seen[-4:]})")
        aug = self.band.augment_fermi(e_hat, states, b, W, L_units, K_units, frames, T, edge_index, edge_len,
                                      pairs=pairs, kT=kT, reference=N0)
        aug["passes"] = len(seen) + 1
        return aug

    def _diabatic_solution(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor,
                           L_units: Tensor | None, K_units: Tensor, Q: float,
                           T: Tensor | None = None, units: Tensor | None = None,
                           frames: Tensor | None = None, pairs=None, diag_correction=None,
                           electrons: Tensor | None = None, selection_correction=None,
                           selection_bound: Tensor | None = None, exclude=None,
                           block_correction=None, reservoir: Tensor | None = None,
                           observable=None, steps=None) -> FreeChargeSolution:
        """The canonical ensemble over integer configurations (SPEC 17.3-17.5): search, exact energies
        (with the graph), F = -kT ln Tr exp(-H / kT), charges from rho's diagonal.  With
        ``diabatic_space='clusters'`` (SPEC 36) the cluster product ensemble instead; ``reservoir [U]`` marks the units
        that link no clusters (the band unit).  ``observable(configs) -> [M, D]`` is averaged over the ensemble with
        the graph (``FreeChargeSolution.expectation``; SPEC 38: the band unit's state populations)."""
        r = self.redox
        dtype, device = b.dtype, b.device
        n_units = int(b.shape[0])
        kT = float(r.kT)
        amplitude = r.diabatic_selection == "amplitude" and T is not None
        a_off, a_on = float(r.selection_amplitude_off), float(r.selection_amplitude_on)
        if r.diabatic_space == "clusters":
            if selection_correction is not None or exclude is not None or block_correction is not None:
                raise NotImplementedError("diabatic_space='clusters' is not built with the downfolded transfer (SPEC 36)")
            return self._cluster_solution(e_hat, states, b, W, L_units, K_units, Q, T, units=units, frames=frames,
                                          pairs=pairs, diag_correction=diag_correction, electrons=electrons,
                                          reservoir=reservoir, observable=observable, steps=steps)
        configs = select_configurations(e_hat, states, b, W, K_units, int(round(Q)),
                                        window=float(r.diabatic_window),
                                        cap=int(r.fragment_max_configurations), min_charge=L_units,
                                        units=units, frames=frames,
                                        pairs=None if pairs is None else (pairs[0], pairs[1], pairs[2].detach()),
                                        T=T if amplitude else None, amplitudes=(a_off, a_on), kT=kT,
                                        search=search_configurations, electrons=electrons,
                                        correction=selection_correction, bound=selection_bound, steps=steps)
        if exclude is not None:                            # SPEC 32: a folded state is never explicit beside its partner
            with torch.no_grad():
                E_sel = configuration_energies(configs, e_hat.detach(), states, b.detach(), W.detach(),
                                               pairs=None if pairs is None else (pairs[0], pairs[1], pairs[2].detach()))
                if selection_correction is not None:
                    E_sel = E_sel + selection_correction(configs)
            configs = exclude(configs, E_sel)
        from . import nanprobe
        nanprobe.watch("diabatic.e_hat", e_hat)
        nanprobe.watch("diabatic.W", W)
        def build_hamiltonian(configs_, extra):
            E_ = configuration_energies(configs_, e_hat, states, b, W, pairs=pairs)
            if diag_correction is not None:              # SPEC 28: many-body, exact on the selected set
                E_ = E_ + diag_correction(configs_)
            for idx, energy, _ in extra:                 # a promoted folded state keeps its own energy
                E_ = E_.index_put((torch.tensor([idx], device=E_.device),), energy.reshape(1).to(E_.dtype))
            nanprobe.watch("diabatic.E", E_)
            # one-electron moves only (Slater-Condon), with their Jordan-Wigner signs (§17.2, §17.6); SPEC 21
            # switches each by its importance (the leading perturbative amplitude from the thermal reference)
            if T is None:
                H_ = torch.diag(E_)
            elif amplitude:
                H_ = switched_hamiltonian(E_, configs_, K_units, T, a_off, a_on, kT, electrons=electrons, steps=steps)
            else:
                H_ = diabatic_hamiltonian(E_, configs_, K_units, T, electrons=electrons, steps=steps)
            if extra:
                # a promoted state couples only as it did in the block it came from (the fold's couplings to the
                # configurations explicit before it, the folded hops among promoted states): the module's own
                # T-couplings of its charge row are removed — they would count an s-s hop twice and add O-O hops the
                # folded space never had (7.2 and 6.0 eV on seed 42's dump, 2026-10-30)
                prom_idx = torch.tensor([idx for idx, _, _ in extra], device=H_.device)
                keep = torch.ones(int(H_.shape[0]), dtype=H_.dtype, device=H_.device).index_fill(0, prom_idx, 0.0)
                mask = keep.unsqueeze(0) * keep.unsqueeze(1) + torch.diag(1.0 - keep)
                H_ = H_ * mask
            for idx, _, couplings in extra:              # ... and the fold's couplings as its explicit couplings
                n_old = int(couplings.numel())
                rows = torch.full((n_old,), idx, dtype=torch.long, device=H_.device)
                cols = torch.arange(n_old, device=H_.device)
                H_ = H_.index_put((rows, cols), couplings.to(H_.dtype), accumulate=True)
                H_ = H_.index_put((cols, rows), couplings.to(H_.dtype), accumulate=True)
            return E_, H_

        promoted, extra = [], []
        for attempt in range(MAX_PROMOTIONS + 1):
            E, H = build_hamiltonian(configs, extra)
            if block_correction is None:
                break
            try:                                         # SPEC 32, exact block: Sigma on the whole explicit block
                H = H + block_correction(configs, H, promoted)
                break
            except PromotionRequest as request:          # the charge-transfer states must be explicit: promote them
                if attempt == MAX_PROMOTIONS:
                    raise DownfoldError(f"{request}; {MAX_PROMOTIONS} promotion passes did not make the block "
                                        f"explicit-dominated") from None
                for j, (row, o_, k_) in enumerate(zip(request.rows, request.os, request.kinds)):
                    # its couplings: to every configuration explicit before this pass, then to the states promoted
                    # earlier in this pass (their mutual hops)
                    vec = torch.cat([request.couplings[j], request.mutual[j, :j]])
                    extra.append((int(configs.shape[0]), request.energies[j], vec))
                    promoted.append((row, o_, k_))
                    new_row = torch.tensor(row, dtype=configs.dtype, device=configs.device).unsqueeze(0)
                    configs = torch.cat([configs, new_row], dim=0)
        self._ensemble_last = {"H": H, "configs": configs, "K": K_units, "kT": kT, "promoted": list(promoted),
                               "space": "global"}
        F = canonical_free_energy(H, kT)
        p = torch.diagonal(density_matrix(H, kT))
        occupied = torch.round(configs.to(dtype) - states[0]).to(torch.long)
        populations = torch.zeros(n_units, int(states.numel()), dtype=dtype, device=device)
        populations = populations.index_put(
            (torch.arange(n_units, device=device).repeat(configs.shape[0]), occupied.reshape(-1)),
            p.detach().repeat_interleave(n_units), accumulate=True)
        charge = p @ configs.to(dtype)
        nan = torch.tensor(float("nan"), dtype=dtype, device=device)
        return FreeChargeSolution(
            z=charge, mu_e=nan, nu=torch.full_like(charge, float("nan")), charge=charge,
            populations=populations, energy=F, iterations=int(configs.shape[0]),
            residual_charge=0.0, residual_total=abs(float(charge.sum()) - Q), duality_gap=float("nan"),
            expectation=None if observable is None else p @ observable(configs).to(dtype),
        )

    def _cluster_solution(self, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, L_units: Tensor | None,
                          K_units: Tensor, Q: float, T: Tensor | None = None, units: Tensor | None = None,
                          frames: Tensor | None = None, pairs=None, diag_correction=None,
                          electrons: Tensor | None = None, reservoir: Tensor | None = None,
                          observable=None, steps=None) -> FreeChargeSolution:
        """SPEC 36: the cluster product ensemble (trace_redox/clusters.py) with this module's exact energies (the
        quadratic form, the pair tables, SPEC 28's correction) and its Hamiltonian rule (SPEC 21's switch, or every
        coupling at full strength).  As in the shipped ensemble the selection sees the quadratic form and the pair
        tables only."""
        from . import nanprobe
        from .clusters import cluster_free_energy, linked_configurations
        r = self.redox
        dtype, device = b.dtype, b.device
        kT = float(r.kT)
        amplitude = r.diabatic_selection == "amplitude" and T is not None
        a_off, a_on = float(r.selection_amplitude_off), float(r.selection_amplitude_on)
        n0, clusters = linked_configurations(
            e_hat, states, b, W, K_units, int(round(Q)), window=float(r.diabatic_window),
            cap=int(r.fragment_max_configurations), T=T if amplitude else None, amplitudes=(a_off, a_on), kT=kT,
            min_charge=L_units, units=units, frames=frames,
            pairs=None if pairs is None else (pairs[0], pairs[1], pairs[2].detach()),
            electrons=electrons, reservoir=reservoir, search=search_configurations,
            couplings=T, closure=bool(r.cluster_closure), steps=steps)
        nanprobe.watch("diabatic.e_hat", e_hat)
        nanprobe.watch("diabatic.W", W)

        def energy(configs_):
            E_ = configuration_energies(configs_, e_hat, states, b, W, pairs=pairs)
            if diag_correction is not None:              # SPEC 28: many-body, exact for every configuration asked
                E_ = E_ + diag_correction(configs_)
            nanprobe.watch("diabatic.E", E_)
            return E_

        def hamiltonian(E_, configs_):
            if T is None:
                return torch.diag(E_)
            if amplitude:
                return switched_hamiltonian(E_, configs_, K_units, T, a_off, a_on, kT, electrons=electrons, steps=steps)
            return diabatic_hamiltonian(E_, configs_, K_units, T, electrons=electrons, steps=steps)

        def bare(E_, configs_):
            return diabatic_hamiltonian(E_, configs_, K_units, T, electrons=electrons, steps=steps)

        out = cluster_free_energy(n0, clusters, energy, hamiltonian, kT, K_units, L_units, states, units=units,
                                  frames=frames, observable=observable, reservoir=reservoir,
                                  closure=bool(r.cluster_closure), bare=bare if amplitude else None,
                                  cap=int(r.fragment_max_configurations))
        clusters = out.clusters                          # the closed sets (SPEC 36.7), linked rows first
        configs = torch.cat([n0.unsqueeze(0)] + [S[1:] for S in clusters], dim=0)
        if len(clusters) == 1:
            H = out.hamiltonians[0]
        elif not clusters:
            H = (out.energy * torch.ones(1, 1, dtype=dtype, device=device))
        else:
            H = None                                     # no single Hamiltonian: readers of "H" must ask for "space"
        self._ensemble_last = {"H": H, "configs": configs, "K": K_units, "kT": kT, "promoted": [], "space": "clusters",
                               "reference": n0, "clusters": clusters, "hamiltonians": out.hamiltonians,
                               "pair_sizes": out.pair_sizes, "pair_index": out.pair_index,
                               "pair_hamiltonians": out.pair_hamiltonians, "pair_configs": out.pair_configs}
        nan = torch.tensor(float("nan"), dtype=dtype, device=device)
        return FreeChargeSolution(
            z=out.charge, mu_e=nan, nu=torch.full_like(out.charge, float("nan")), charge=out.charge,
            populations=out.populations, energy=out.energy, iterations=int(configs.shape[0]),
            residual_charge=0.0, residual_total=abs(float(out.charge.sum()) - Q), duality_gap=float("nan"),
            expectation=out.expectation,
        )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        pbc: Any = None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, RedoxResult]:
        """The energy of one structure and its record.  With a Born-von Karman mesh beyond (1, 1, 1) (SPEC 37) the
        energy per cell of the structure's supercell; otherwise the cell itself."""
        r = self.redox
        graph = dict(pbc=pbc, cell=cell, edge_shift=edge_shift, data=data)
        if r.bvk_mesh == (1, 1, 1) and r.bvk_length == 0.0:
            return self._forward_cell(h, pos, edge_index, edge_vec, edge_len, **graph)
        from .bvk import UNIT_MESH, mesh_for
        mesh = mesh_for(cell, (data or {}).get("pbc") if pbc is None else pbc, r.bvk_mesh, r.bvk_length)
        if mesh == UNIT_MESH:
            return self._forward_cell(h, pos, edge_index, edge_vec, edge_len, **graph)
        return self._forward_bvk(mesh, h, pos, edge_index, edge_vec, edge_len, **graph)

    def _forward_bvk(self, mesh: tuple, h: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor,
                     edge_len: Tensor, *, pbc: Any, cell: Tensor | None, edge_shift: Tensor | None,
                     data: Mapping[str, Any] | None) -> tuple[Tensor, RedoxResult]:
        """SPEC 37: E_M(C) = E_Gamma(S_M[C]) / |M|.  The supercell is tiled from the cell's graph (trace_redox.bvk),
        evaluated as any structure, and the record is cut to the home cell (``RedoxResult.bvk_mesh``)."""
        from .bvk import tile
        data = {} if data is None else dict(data)
        big = tile(mesh, h, pos, edge_index, edge_vec, edge_len, cell, edge_shift, data.get("z"))
        copies, n = big["copies"], int(pos.shape[0])
        data["total_charge"] = data.get("total_charge", 0.0) * copies
        if big["species"] is not None:
            data["z"] = big["species"]
        energy, full = self._forward_cell(big["h"], big["pos"], big["edge_index"], big["edge_vec"], big["edge_len"],
                                          pbc=pbc, cell=big["cell"], edge_shift=big["edge_shift"], data=data)
        if (self.band is not None and self.redox.carrier_band and self.redox.band_form == "phi1"
                and self._band_last is not None):                        # SPEC 38's unit holds any filling
            held = abs(float(full.unit_charge[-1].detach()))             # the band unit is the last unit
            if held > BVK_BAND_OCCUPATION:
                raise NotImplementedError(
                    f"Born-von Karman sampling of a structure whose band unit holds {held:.2f} e: the band unit takes "
                    "at most two electrons per STRUCTURE (SPEC 20), so the supercell is not a replication of the cell "
                    "(benchmarks_4g README R16, L3); refused until the band is extensive")
        for name in ("_frame_wall_last", "_frame_pair_last", "_bond_pauli_last", "_bond_pauli_wall_last"):   # diagnostics: per cell
            if getattr(self, name, None) is not None:
                setattr(self, name, getattr(self, name) / copies)
        home = {name: getattr(full, name)[:n] for name in ("q", "q_bound", "free_charge", "sigma_free", "mu", "rho")}
        if full.theta_hat is not None:
            home["theta_hat"] = full.theta_hat[:n]
        energy = energy / copies
        ensemble = full.ensemble
        if ensemble is not None and ensemble.get("exact", False):       # SPEC 36.10: the response per cell
            ensemble = dict(ensemble, alpha_bound=ensemble["alpha_bound"] / copies, scale=1.0 / copies)
        return energy, dataclasses.replace(full, energy=energy, energy_bound=full.energy_bound / copies,
                                           energy_free=full.energy_free / copies, bvk_mesh=tuple(mesh),
                                           ensemble=ensemble, **home)

    def _forward_cell(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        pbc: Any = None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, RedoxResult]:
        inputs = self._inputs(pos, data, pbc)
        r = self.redox
        species = inputs["species"]
        if species is None and (r.anchor in ("atomic", "formal") or r.width == "hardness"):
            raise ValueError("trace_redox needs the atomic numbers (data['z']) for its anchors and widths")
        assembly = assemble_neutral(
            h, pos, edge_index, edge_vec, edge_len, self._heads_for(species), self.config,
            pbc=inputs["pbc"], cell=cell, edge_shift=edge_shift,
            quadrupoles=inputs["quadrupoles"], species=species,
            external_field=inputs["external_field"],
            elimination=r.bound_elimination,       # R25: the limit bounds what this route holds
        )
        if r.bound_electronegativity == "none":
            assembly = without_bound_drive(assembly)
        assembly = self._partition_bonds(assembly, species, edge_index, edge_len)
        assembly = self._with_exclusion_anchors(assembly, species, edge_index, edge_len, edge_vec)
        dtype, device = assembly.pos.dtype, assembly.pos.device
        n = assembly.n_atoms
        sigma_free = self.free_widths(species) if self.width is not None else assembly.params.sigma
        # Formal mode: compact ionic widths, so the Ewald free-charge route (sigma_self = sigma_free: the
        # same physics, on-site correction exactly zero); the plane-wave route would need ~1.4e5 k-vectors.
        ensemble = None                                                       # SPEC 30 response record
        if r.free_charge_model == "core_carrier":
            from .constants import formal_frame
            sigma_carrier = carrier_widths(species, sigma_free.to(dtype), int(r.max_charge))
            frames = formal_frame(species).to(device)
            elim = eliminate_core_carrier(assembly, sigma_free.to(dtype), sigma_carrier, frames,
                                          method=r.bound_elimination)
        else:
            sigma_carrier, frames = None, None
            elim = eliminate_bound_sector(assembly, sigma_free.to(dtype), method=r.bound_elimination,
                                          sigma_self=sigma_free.to(dtype) if r.anchor == "formal" else None)
        if r.frame_induction_damping:
            elim = self._damp_frame_induction(elim, assembly, species, sigma_free.to(dtype), pos.to(dtype), cell, inputs["pbc"])
        lam = float(r.self_interaction)

        invariants = self.heads.node_invariants(h).to(dtype)
        edge_len = edge_len.to(dtype)
        raw = self.charge_states(invariants, edge_index, edge_len)           # [n, 1 + K]: (+1, -1, +2, ...)
        d_chi = 0.5 * (raw[:, 0] - raw[:, 1])
        d_gap = 0.5 * (raw[:, 0] + raw[:, 1])
        own_self = torch.diagonal(elim.J)                                     # J_Z, unspread
        if r.anchor == "atomic":
            chi_anchor = self.mulliken.to(dtype)[species]
            gap_anchor = self.hardness.to(dtype)[species] - (1.0 - lam) * own_self
            if bool((gap_anchor.detach() <= 0).any()):
                worst = int(torch.argmin(gap_anchor.detach()))
                raise BoundSectorError(
                    f"atom {worst} (Z = {int(species[worst])}): IE - EA = "
                    f"{float(self.hardness[species[worst]]):.3f} eV is below (1 - lam) J = "
                    f"{float((1.0 - lam) * own_self[worst]):.3f} eV, so its integer energies are "
                    "not convex (it would disproportionate). Use width='hardness'."
                )
        else:
            chi_anchor = torch.zeros(n, dtype=dtype, device=device)
            gap_anchor = torch.full((n,), 2.0 * float(r.gap), dtype=dtype, device=device)
        chi = chi_anchor + r.chi_bound * torch.tanh(d_chi / r.chi_bound)
        gap = gap_anchor * torch.exp(r.log_gap_bound * torch.tanh(d_gap / r.log_gap_bound))

        if self.orbitals is not None:
            omega = self.orbitals(invariants, edge_index, edge_len)           # [U, n]
            b = omega @ elim.u0
            W = omega @ elim.W @ omega.transpose(0, 1)
            J_units = omega @ elim.J @ omega.transpose(0, 1)
        else:
            omega = None
            b, W, J_units = elim.u0, elim.W, elim.J
        J_self = torch.diagonal(J_units)
        # SPEC §12: molecules from the chemical-bond graph.  Every partition, including
        # one molecule spanning the system, goes through the same two-level machinery:
        # special-casing that molecule onto the present path put a jump (measured
        # 4.2e-5 eV) where a bond enters its switch.  B2 then holds to the bound
        # 0 <= E_none - E_bonds <= (1 - c)/2 sum_u r_u Var_u (§12.8).
        partitions = None
        if r.fragments == "bonds":
            if species is None:
                raise ValueError("fragments='bonds' needs the atomic numbers (data['z'])")
            bi, bj, bs = chemical_bonds(species, edge_index, edge_len, coverage=float(r.attention_cutoff))
            # C5 (2026-11-10): the diabatic ensemble (SPEC 17) carries integer charges on every atom and never reads
            # the partitions, so their 2^k enumeration and its limit do not apply to it (the doped Au2/MgO 2x2 cell
            # has 12 uncertain connections).  The coverage check of chemical_bonds stays for every ensemble.
            partitions = [] if r.fragment_ensemble == "diabatic" else percolation_partitions(
                n, bi, bj, bs, int(r.fragment_max_uncertain))
        removal = None
        if r.sic == "certified" and partitions is None:
            removal = certified_removal(W, J_units, theta=r.sic_theta, smoothing=r.sic_smoothing)
            W = W - torch.diag(removal)
        if lam > 0.0:
            W = W - lam * torch.diag(J_self)
        W = 0.5 * (W + W.transpose(0, 1))
        # The convex solvers need W positive definite; configuration energies do not (SPEC 17.3).
        info = 0 if r.fragment_ensemble == "diabatic" else torch.linalg.cholesky_ex(W.detach())[1]
        if int(info) != 0:
            try:
                limit = float(lambda_max(W.detach() + lam * torch.diag(J_self.detach()), J_self.detach()))
            except torch.linalg.LinAlgError:          # the diagnostic itself on an ill-conditioned W
                limit = float("nan")
            raise BoundSectorError(
                f"W - lam D is not positive definite (lam = {lam}, lam_max = {limit:.4f} here); "
                "SPEC 25.2 requires lam < lam_max"
            )

        # Levels eps_k = Ehat_k - Ehat_{k-1}, strictly increasing (Ehat convex on the
        # integers) for every parameter value: eps_0 = chi - g_1/2, eps_1 = chi + g_1/2,
        # eps_k = eps_{k-1} + g_k with g_k > 0.  Anchors (lam = 0 shown):
        # g_k = IE_k - IE_{k-1} - J, so an isolated ion has E_+k = IE_1 + ... + IE_k.
        K = int(r.max_charge)
        states = torch.arange(-1, K + 1, dtype=dtype, device=device)
        level = chi + 0.5 * gap
        energy_k = level
        columns = [-(chi - 0.5 * gap), torch.zeros_like(chi), level]          # n = -1, 0, +1
        valid = [torch.ones_like(chi, dtype=torch.bool)] * 3
        inf = torch.full_like(chi, float("inf"))
        for k in range(2, K + 1):
            if r.anchor == "atomic":
                series = self.ionization_series.to(dtype)[species]
                exists = torch.isfinite(series[:, k - 1])                     # the atom has >= k electrons
                step = series[:, k - 1] - series[:, k - 2] - (1.0 - lam) * own_self
                if bool((exists & (step.detach() <= 0)).any()):
                    worst = int(torch.nonzero(exists & (step.detach() <= 0))[0])
                    raise BoundSectorError(
                        f"atom {worst} (Z = {int(species[worst])}): IE_{k} - IE_{k - 1} is below "
                        f"(1 - lam) J; its states +{k - 1}, +{k} would not be convex"
                    )
                step = torch.where(exists, step, torch.ones_like(step))      # finite placeholder: no NaN gradient
            else:
                exists = torch.ones_like(chi, dtype=torch.bool)
                step = torch.full_like(chi, 2.0 * float(r.gap))
            g_k = step * torch.exp(r.log_gap_bound * torch.tanh(raw[:, k] / r.log_gap_bound))
            level = level + g_k
            energy_k = energy_k + level
            columns.append(torch.where(exists, energy_k, inf))
            valid.append(exists)
        e_hat = torch.stack(columns, dim=1)                                    # [U, K + 2]
        valid = torch.stack(valid, dim=1)
        state_energies = e_hat + 0.5 * (1.0 - lam) * J_self.unsqueeze(1) * (states * states).unsqueeze(0)
        if removal is not None:
            # the removed on-site Coulomb, back in the populations: 1/2 r_u <n_u^2>
            e_hat = e_hat + 0.5 * removal.unsqueeze(1) * (states * states).unsqueeze(0)

        Q = inputs["total_charge"]
        fermi_atoms = None                                 # SPEC 38: the atoms' charges with a Fermi band unit
        if assembly.slab and Q != 0.0:
            raise BoundSectorError(
                "a 2-D slab with a net free charge needs the charged-slab terms of the "
                "Yeh-Berkowitz correction (Ballenegger, Arnold & Cerda, JCP 131, 094107 (2009)), "
                "which the bound sector does not carry either; refused"
            )
        n_units = int(b.shape[0])
        top = valid.sum(dim=1) - 1                                             # index of each unit's highest state
        capacity = float(states[top].sum())
        if r.anchor != "formal" and (Q < -n_units - 1e-12 or Q > capacity + 1e-12):
            raise ValueError(
                f"total charge {Q} e is outside [-{n_units}, {capacity:g}]: the capacity of these "
                f"{n_units} units with states -1..+{K} (an atom of atomic number Z holds at most +Z)"
            )
        molecules = None
        if r.anchor == "formal":
            # SPEC 17.5: exact isolated-ion energies E_Z(n) + learned bounded shifts, minus the ISOLATED
            # self-energy only (own-image interactions are lattice energy); integer configurations.
            if abs(Q - round(Q)) > 1e-9:
                raise ValueError(f"anchor='formal' needs an integer total charge, got {Q}")
            states, E_Z, L_units, K_units = formal_ladder(species, K)
            states, E_Z = states.to(dtype=dtype, device=device), E_Z.to(dtype=dtype, device=device)
            if self.band is not None and r.carrier_band and int(round(float(states[0]))) > -2:
                # SPEC 24: the band holds up to two electrons (state -2) whatever elements are present; the atoms'
                # own ladders get +inf there (L_units is unchanged, so no atom visits it)
                pad = int(round(float(states[0]))) + 2
                states = torch.cat([torch.arange(-2, -2 + pad, dtype=dtype, device=device), states])
                E_Z = torch.cat([torch.full((E_Z.shape[0], pad), float("inf"), dtype=dtype, device=device), E_Z], dim=1)
            bound = float(r.chi_bound)
            delta = bound * torch.tanh(raw / bound)                              # tokens +1, -1, +2..+K, -2
            if self.fixed_terms_only:
                if self.training:
                    raise RuntimeError("fixed_terms_only is for builders, probes and tests: it switches every learned "
                                       "level shift and coupling off and must not be on in training mode")
                # builders, probes and tests: the package's fixed terms alone, no learned level shift (SPEC 40.7: the
                # entries of an s1 pair are defined against them; they do not depend on any weight, tested)
                delta = torch.zeros_like(delta)
            if r.shift_gate:
                # SPEC 24: the environment's correction vanishes without an environment (exact isolated ions)
                delta = delta * shift_gate(n, edge_index, edge_len, float(r.attention_cutoff)).unsqueeze(1)
            token = {1: 0, -1: 1, -2: K + 1, **{k: k for k in range(2, K + 1)}}
            shift = torch.stack([delta[:, token[int(s)]] if int(s) != 0 else torch.zeros_like(delta[:, 0])
                                 for s in states.tolist()], dim=1)
            state_energies = E_Z + shift
            if sigma_carrier is None:
                S0 = gram_self_energy(sigma_free.to(dtype))
                e_hat = state_energies - 0.5 * S0.unsqueeze(1) * (states * states).unsqueeze(0)
            else:
                # the isolated ion's own Coulomb energy for core f and carrier c = n - f (SPEC 17.9):
                # 1/2 f^2 S0(core) + f c k(sqrt(s_core^2 + s_car^2)) + 1/2 c^2 S0(carrier)
                from trace_electrostatics.kernels import kernel_self
                s_core = sigma_free.to(dtype)
                f = frames.to(dtype).unsqueeze(1)
                c = states.unsqueeze(0) - f
                self_energy = (0.5 * f * f * gram_self_energy(s_core).unsqueeze(1)
                               + f * c * kernel_self(torch.sqrt(s_core ** 2 + sigma_carrier ** 2)).unsqueeze(1)
                               + 0.5 * c * c * gram_self_energy(sigma_carrier).unsqueeze(1))
                e_hat = state_energies - self_energy
            if self.exclusion is not None and r.exclusion:     # the configuration decides, not the submodule
                from .constants import formal_frame
                anion_removal = None
                if r.exclusion_anions == "in_situ":
                    from .exclusion import in_situ_removal, smooth_removal
                    anion_removal = smooth_removal(in_situ_removal(e_hat, states, b, W, formal_frame(species).to(device)))
                e_hat = e_hat + self.exclusion(species, states, edge_index, edge_len.to(dtype),
                                               formal_frame(species).to(device), edge_vec=edge_vec.to(dtype),
                                               alpha=assembly.alpha_volume, anion_removal=anion_removal,
                                               anion_holes=bool(r.exclusion_anion_holes))
            nan_n = torch.full((n_units,), float("nan"), dtype=dtype, device=device)
            chi, gap = nan_n, nan_n
            T = self.coupling(invariants, edge_index, edge_len, species=species) if self.coupling is not None and r.couplings else None
            steps = None
            if T is not None and self.fixed_terms_only:    # the package's fixed terms hold no learned coupling
                T = torch.zeros_like(T)
            if r.two_site and T is not None:               # SPEC 40: the first-principles hop of every s1-s1 pair,
                from .two_site import two_site             # the coupling of its s moves only (SPEC 40.6 ix)
                T, steps = two_site().resolve_hops(T, species, edge_index, edge_len.to(dtype), float(self.coupling.r_cut))
            units = frames_c = None
            if r.carriers == "covalent_units":
                from .constants import formal_frame
                from .fragments import covalent_units
                frames_c = formal_frame(species).to(device)
                bi, bj, _ = chemical_bonds(species, edge_index, edge_len, coverage=float(r.attention_cutoff))
                units = covalent_units(frames_c, bi, bj, n_units, full_unit_atoms=int(r.full_unit_atoms))
            pairs = None
            if r.exclusion_pairs:
                from .constants import formal_frame
                pairs = self.exclusion.pair_terms(species, states, edge_index, edge_len.to(dtype),
                                                  formal_frame(species).to(device))
            if r.carrier_induction:
                # SPEC 26: exact one-body + pair corrections of the induction for the carriers' true fields
                from .constants import formal_frame
                from .induction import induction_terms
                from trace_electrostatics.solver import gaussian_widths
                f_atoms = formal_frame(species).to(device)
                w_mu, _ = gaussian_widths(assembly.params, assembly.config.gaussian_a)
                Q_c = states.unsqueeze(0) - f_atoms.to(dtype).unsqueeze(1)
                Q_c = torch.where(torch.isfinite(e_hat), Q_c, torch.zeros_like(Q_c))
                # N_c stands for K^-1 V (InductionColumns: never formed; the dense tensor on the reference route)
                E1_c, pairs_c, N_c = induction_terms(elim, species, states, f_atoms, sigma_free.to(dtype),
                                                     w_mu.to(dtype), assembly.params.ell.to(dtype), pos.to(dtype),
                                                     Q_c, cell=cell, pbc=inputs["pbc"])
                e_hat = e_hat + torch.where(torch.isfinite(e_hat), E1_c, torch.zeros_like(E1_c))
                pairs = pairs_c if pairs is None else tuple(torch.cat([x, y]) for x, y in zip(pairs, pairs_c))
            if r.carrier_penetration:
                from .constants import formal_frame
                from .penetration import library as penetration_library
                from .penetration import penetration_pairs, penetration_self_terms
                pen_lib = penetration_library(r.pauli_table)
                pp = penetration_pairs(species, states, formal_frame(species).to(device), sigma_free.to(dtype),
                                       pos.to(dtype), cell=cell, pbc=inputs["pbc"], model=r.carrier_overlap,
                                       lib=pen_lib)
                pairs = pp if pairs is None else tuple(torch.cat([x, y]) for x, y in zip(pairs, pp))
                # an atom and its own periodic images (cells shorter than the cutoff): one-body, same state
                own = penetration_self_terms(species, states, formal_frame(species).to(device), sigma_free.to(dtype),
                                             pos.to(dtype), cell=cell, pbc=inputs["pbc"], model=r.carrier_overlap,
                                             lib=pen_lib)
                e_hat = e_hat + torch.where(torch.isfinite(e_hat), own, torch.zeros_like(own))
            if r.two_site:
                # SPEC 40: the covalent relaxation, the frame-frame entry of the state-pair slot of every s1-s1 pair
                from .constants import formal_frame
                from .two_site import two_site
                rp = two_site().relaxation_pairs(species, states, formal_frame(species).to(device), pos.to(dtype),
                                                 cell, inputs["pbc"], config=r,
                                                 gaussian_a=float(self.config.gaussian_a))
                if rp is not None:
                    pairs = rp if pairs is None else tuple(torch.cat([x, y]) for x, y in zip(pairs, rp))
            if r.bond_pauli_states:
                # SPEC 33 per charge state (step 27): each state's law minus the neutral's, a one-body ladder term; the
                # frame path below keeps the neutral law, so the sum is the per-state law up to a constant per structure
                from .bond_pauli import bond_pauli
                if int(e_hat.shape[0]) != int(species.numel()):
                    raise ValueError("bond_pauli_states needs atom carriers (one ladder row per atom)")
                T_bp = bond_pauli(r.bond_pauli_level).state_terms(species, states, pos.to(dtype), cell, inputs["pbc"])
                k0 = int((states == 0).nonzero()[0])
                dT = T_bp - T_bp[:, k0:k0 + 1]
                e_hat = e_hat + torch.where(torch.isfinite(e_hat), dT, torch.zeros_like(dT))
            e_run, b_run, W_run, L_run, K_run, T_run = e_hat, b, W, L_units, K_units, T
            Q_run, band_charges = Q, None                  # SPEC 38: the solver's total and the band's charge shapes
            if self.band is not None and r.carrier_band:
                from .constants import formal_frame
                if r.band_form == "fermi":
                    if omega is not None:
                        raise NotImplementedError("band_form='fermi' needs atom carriers (SPEC 38)")
                    aug = self._fermi_band(e_hat, states, b, W, L_units, K_units, formal_frame(species).to(device), T,
                                           edge_index, edge_len, pairs, units, frames_c, Q)
                else:
                    aug = self.band.augment(e_hat, states, b, W, L_units, K_units, formal_frame(species).to(device), T,
                                            edge_index, edge_len, pairs=pairs)
                self._band_last = aug
                if aug is not None:
                    e_run, b_run, W_run, L_run, K_run, T_run = (aug[k] for k in ("e_hat", "b", "W", "L", "K", "T"))
                    if aug.get("pairs") is not None:
                        pairs = aug["pairs"] if pairs is None else tuple(
                            torch.cat([x, y]) for x, y in zip(pairs, aug["pairs"]))
                    if r.band_form == "fermi":             # state s is the filling N0 - s, of charge s - N0
                        band_charges = aug["charges"]
                        Q_run = Q + aug["reference"]
                    else:
                        omega = aug["omega"]
                    if units is not None:
                        units = torch.cat([units, (units.max() + 1).reshape(1)])
                    if frames_c is not None:
                        frames_c = torch.cat([frames_c, torch.zeros(1, dtype=frames_c.dtype, device=device)])
            diag_correction = None
            if r.state_polarizability:
                from .constants import formal_frame
                from .polarizability import state_polarizability_corrections
                rho_table = self._state_ratio_table(species, states)
                f_sp = formal_frame(species).to(device)
                band_omega = None if self.band is None or not r.carrier_band or self._band_last is None else self._band_last["omega"]
                if band_charges is not None:
                    band_omega = band_charges

                with_induction = (N_c, Q_c) if r.carrier_induction else None      # SPEC 26: the full source
                own_stiffness = self._state_stiffness(elim)                       # SPEC 28.6

                def diag_correction(cfgs):
                    return state_polarizability_corrections(cfgs, elim, f_sp, rho_table, states, omega=band_omega,
                                                            induction=with_induction, stiffness=own_stiffness)
            selection_correction = selection_bound = exclude = None
            if r.downfold_transfer:
                # SPEC 32: every O-s move leaves the explicit ensemble (the pairs' couplings are zeroed, a selected
                # folded partner is dropped) and dresses each configuration's energy by the exact block root
                from .constants import formal_frame
                from .downfold import acceptor_mask, donor_mask, downfold_moves, drop_folded_partners
                from .downfold import dressing as downfold_dressing
                n_at = int(species.numel())
                fold = downfold_moves(species, edge_index, edge_vec.to(dtype), edge_len.to(dtype),
                                      r_cut=float(r.downfold_cutoff), bond_cutoff=float(r.downfold_bond_cutoff),
                                      beta=float(r.downfold_crystal_field), free_axes_only=bool(r.downfold_free_axes),
                                      units=units, decay=float(r.downfold_decay), anchor=float(r.downfold_decay_anchor))
                if T_run is not None:
                    cls = donor_mask(species).to(device).unsqueeze(1) & acceptor_mask(species).to(device).unsqueeze(0)
                    cls = cls | cls.transpose(0, 1)
                    keep = torch.ones_like(T_run)
                    keep[:n_at, :n_at] = (~cls).to(T_run.dtype)
                    T_run = T_run * keep
                # the fold's s-s hops keep their gradient: a detached T made the forces inconsistent with the energy
                # (4.7 % on an Au, 2026-10-30) and gave the coupling parameters no gradient through the hops
                T_fold = T_run if T_run is not None else torch.zeros(n_at, n_at, dtype=dtype, device=device)
                inner_correction = diag_correction
                frames_fold = formal_frame(species).to(device)
                electrons_fold = species.to(torch.long)

                def explicit_energy(cfgs, with_inner=True, _e=e_run, _b=b_run, _W=W_run, _p=pairs,
                                    _c=inner_correction):
                    E_c = configuration_energies(cfgs, _e, states, _b, _W, pairs=_p)
                    return E_c if (_c is None or not with_inner) else E_c + _c(cfgs)

                def diag_correction(cfgs, strict=True, with_inner=True, _inner=inner_correction, _fold=fold,
                                    _L=L_run, _K=K_run, _e=e_run, _b=b_run, _W=W_run, _p=pairs):
                    plain = configuration_energies(cfgs, _e, states, _b, _W, pairs=_p)
                    inner = torch.zeros_like(plain) if (_inner is None or not with_inner) else _inner(cfgs)
                    energy_fn = (explicit_energy if with_inner
                                 else (lambda c, _f=explicit_energy: _f(c, with_inner=False)))
                    dressed, record = downfold_dressing(cfgs, energy_fn, _fold, T_fold, frames_fold, _L, _K,
                                                        electrons_fold, strict=strict, base=plain + inner,
                                                        scale=float(r.downfold_scale),
                                                        anisotropy=float(r.downfold_anisotropy),
                                                        hop_sign=(1 if r.downfold_hop_ordering == "pauli" else -1))
                    if strict:
                        self._downfold_last = record
                    return inner + dressed

                block_correction, final_diag = None, diag_correction
                if r.downfold_block == "exact":
                    # the fold on the whole explicit block (2026-10-29): the final solve adds only the inner
                    # correction to the energies and the block's Sigma to the Hamiltonian; the selection keeps the
                    # per-state dressing (its screen), a stated approximation
                    from .downfold import dressing_exact as downfold_dressing_exact

                    def final_diag(cfgs, _inner=inner_correction):
                        return (torch.zeros(int(cfgs.shape[0]), dtype=dtype, device=device) if _inner is None
                                else _inner(cfgs))

                    def block_correction(cfgs, H0, promoted=None, _inner=inner_correction, _fold=fold, _L=L_run,
                                         _K=K_run, _e=e_run, _b=b_run, _W=W_run, _p=pairs):
                        plain = configuration_energies(cfgs, _e, states, _b, _W, pairs=_p)
                        inner = torch.zeros_like(plain) if _inner is None else _inner(cfgs)
                        sigma, record = downfold_dressing_exact(cfgs, explicit_energy, _fold, T_fold, frames_fold,
                                                                _L, _K, electrons_fold, H0, strict=True,
                                                                base=plain + inner, scale=float(r.downfold_scale),
                                                                anisotropy=float(r.downfold_anisotropy),
                                                                hop_sign=(1 if r.downfold_hop_ordering == "pauli" else -1),
                                                                promoted=promoted)
                        self._downfold_last = record
                        return sigma

                def selection_correction(cfgs):
                    # the selection's dressing takes Delta without SPEC 28 (meV-scale there); the solve is exact
                    with torch.no_grad():
                        return diag_correction(cfgs, strict=False, with_inner=False)

                selection_bound = torch.zeros(int(e_run.shape[0]), dtype=dtype, device=device)
                selection_bound[:n_at] = fold.atom_bound

                def exclude(cfgs, E_sel, _fold=fold, _L=L_run, _K=K_run):
                    return drop_folded_partners(cfgs, E_sel, _fold, _L, _K)
            reservoir = None
            if int(e_run.shape[0]) > int(e_hat.shape[0]):  # SPEC 36: the band unit links no clusters
                reservoir = torch.zeros(int(e_run.shape[0]), dtype=torch.bool, device=device)
                reservoir[int(e_hat.shape[0]):] = True
            sol = self._diabatic_solution(e_run, states, b_run, W_run, L_run, K_run, Q_run, T_run, units=units,
                                          frames=frames_c, pairs=pairs,
                                          diag_correction=final_diag if r.downfold_transfer else diag_correction,
                                          electrons=species.to(torch.long) if r.spin_pairing else None,
                                          selection_correction=selection_correction, selection_bound=selection_bound,
                                          exclude=exclude,
                                          block_correction=block_correction if r.downfold_transfer else None,
                                          reservoir=reservoir,
                                          observable=None if band_charges is None else band_charges.onehot,
                                          steps=steps)
            if band_charges is not None:
                # the atoms' charges with each filling's own shape, the band unit's charge -<N>; omega stays None
                fermi_atoms = sol.charge[:-1] + sol.expectation @ band_charges.shape.to(dtype)
                sol = dataclasses.replace(sol, charge=torch.cat([sol.charge[:-1],
                                                                 sol.charge[-1:] - float(aug["reference"])]),
                                          residual_total=abs(float(sol.charge.sum()) - Q_run))
                self._band_last = dict(aug, populations=sol.expectation.detach())
            ensemble = self._ensemble_record(elim, species, states, inputs, pos, cell,
                                             omega if band_charges is None else band_charges,
                                             (N_c, Q_c) if r.carrier_induction else None)
        elif abs(Q - capacity) <= 1e-12:
            sol = _saturated(b, W, e_hat, states, top)
        elif abs(Q + n_units) <= 1e-12:
            sol = _saturated(b, W, e_hat, states, torch.zeros_like(top))
        elif partitions is not None and r.fragment_ensemble == "diabatic":
            # SPEC §17: integer charges on every atom.  E(n) is the closed form of a fixed
            # molecule (the bound sector equilibrated to each configuration through W); the
            # free energy is the canonical one of H = diag(E(n)) (+ couplings, §17.5).  The
            # removal of §8a, if any, cancels exactly for integer charges.
            if abs(Q - round(Q)) > 1e-9:
                raise ValueError(f"fragment_ensemble='diabatic' needs an integer total charge, got {Q}")
            K_units = torch.round(states[top]).to(torch.long)
            T = self.coupling(invariants, edge_index, edge_len, species=species) if self.coupling is not None and r.couplings else None
            sol = self._diabatic_solution(e_hat, states, b, W, None, K_units, Q, T,
                                          electrons=species.to(torch.long) if r.spin_pairing and species is not None else None)
        elif partitions is not None and r.fragment_ensemble == "exact":
            # SPEC §15: the exact canonical configuration ensemble.  Every configuration
            # is solved on the atom-level certified operator (convex, §8a); the two-level
            # ensemble of §12 only selects and smoothly weights the configurations.
            if abs(Q - round(Q)) > 1e-9:
                raise ValueError(f"fragments='bonds' needs an integer total charge, got {Q}")
            Qi = int(round(Q))
            kT = float(r.kT)
            removal = None
            W_x, e_hat_x = W, e_hat
            if r.sic == "certified":
                removal = certified_removal(W, J_units, theta=r.sic_theta, smoothing=r.sic_smoothing)
                W_x = W - torch.diag(removal)
                W_x = 0.5 * (W_x + W_x.transpose(0, 1))
                e_hat_x = e_hat + 0.5 * removal.unsqueeze(1) * (states * states).unsqueeze(0)
            M = None
            energy_free, charge_free, populations_free = 0.0, 0.0, 0.0
            molecules, residual, n_solves = [], 0.0, 0
            for frags, weight in partitions:
                if len(frags) == 1:
                    one = differentiable_solution(b, W_x, e_hat_x, states, Q, kT, tol=r.tol, max_iter=r.max_iter)
                    F_P, c_P, pop_P = one.energy, one.charge, one.populations.detach()
                    molecules.append((float(weight), [frags[0].tolist()],
                                      [([Qi], torch.ones(1, dtype=dtype, device=device))]))
                    residual = max(residual, one.residual_charge, one.residual_total)
                    n_solves += 1
                else:
                    if M is None:
                        M = torch.cholesky_inverse(torch.linalg.cholesky(W))
                        M = 0.5 * (M + M.transpose(0, 1))
                    D_list, _, S_P = block_certificate(W, M, frags)
                    windows: dict = {}
                    for _attempt in range(64):
                        mols = build_molecules(frags, D_list, e_hat, J_units, states, top, Q, int(r.fragment_window),
                                               r.sic == "certified", r.sic_theta, r.sic_smoothing, windows)
                        screen = solve_fragment_ensemble([(torch.ones((), dtype=dtype, device=device), frags)], [mols],
                                                         S_P, b, states, Q, kT, tol=r.fragment_tol, inner_tol=r.tol)
                        # this partition enters with weight pi: its edge test is pi-weighted (§12.7)
                        wider = edge_populations(screen, threshold=1e-8 / max(float(weight), 1e-300))
                        if not wider:
                            break
                        windows.update(wider)
                    else:
                        raise FragmentEnsembleError("molecular charge windows did not settle in 64 enlargements")
                    F_P, c_P, pop_P, margs, n_conf, res = exact_partition(
                        mols, screen.screen[0], b, W_x, e_hat_x, states, Qi, kT, r.tol, int(r.fragment_max_configurations))
                    molecules.append((float(weight), [m.atoms.tolist() for m in mols],
                                      [(m.window, mg) for m, mg in zip(mols, margs)]))
                    residual = max(residual, res)
                    n_solves += n_conf
                energy_free = energy_free + weight * F_P
                charge_free = charge_free + weight * c_P
                populations_free = populations_free + float(weight) * pop_P
            nan = torch.tensor(float("nan"), dtype=dtype, device=device)
            sol = FreeChargeSolution(
                z=charge_free, mu_e=nan, nu=torch.full_like(charge_free, float("nan")), charge=charge_free,
                populations=populations_free, energy=energy_free, iterations=n_solves,
                residual_charge=residual, residual_total=residual, duality_gap=float("nan"),
            )
        elif partitions is not None:      # fragment_ensemble == 'two-level' (§12)
            if abs(Q - round(Q)) > 1e-9:
                raise ValueError(f"fragments='bonds' needs an integer total charge, got {Q}")
            M = torch.cholesky_inverse(torch.linalg.cholesky(W))
            M = 0.5 * (M + M.transpose(0, 1))
            certified = [(frags, weight, block_certificate(W, M, frags)) for frags, weight in partitions]
            S_bar = sum(weight * S_P for _, weight, (_, _, S_P) in certified)
            windows: dict = {}
            for _attempt in range(64):
                built = [build_molecules(frags, D_list, e_hat, J_units, states, top, Q, int(r.fragment_window),
                                         r.sic == "certified", r.sic_theta, r.sic_smoothing, windows)
                         for frags, _, (D_list, _, _) in certified]
                fsol = solve_fragment_ensemble([(w, f) for f, w in partitions], built, S_bar, b, states, Q,
                                               float(r.kT), tol=r.fragment_tol, inner_tol=r.tol)
                wider = edge_populations(fsol)
                if not wider:
                    break
                windows.update(wider)        # §12.7: never truncate a populated charge state
            else:
                raise FragmentEnsembleError("molecular charge windows did not settle in 64 enlargements")
            molecules = [(float(w), [m.atoms.tolist() for m in mols], [(m.window, pop.detach()) for m, pop in zip(mols, pops)])
                         for w, mols, pops in fsol.partitions]
            sol = FreeChargeSolution(
                z=fsol.charge, mu_e=fsol.mu_e, nu=fsol.potentials, charge=fsol.charge,
                populations=fsol.atomic_populations, energy=fsol.energy, iterations=fsol.iterations,
                residual_charge=fsol.residual, residual_total=fsol.residual, duality_gap=float("nan"),
            )
        else:
            sol = differentiable_solution(b, W, e_hat, states, Q, float(r.kT), tol=r.tol, max_iter=r.max_iter)

        energy_free = sol.energy + elim.constant                              # constant: 0, or SPEC 17.9's frame terms
        energy = elim.e_b0 + energy_free
        if r.frame_wall:                                 # the frame pairs' own Pauli wall (trace_redox.wall, 2026-10-31)
            from .wall import frame_wall
            wall = frame_wall().energy(species, pos.to(dtype), cell, inputs["pbc"])
            self._frame_wall_last = float(wall.detach())
            energy = energy + wall
        if r.frame_pair:                                 # SPEC 39: the frame pair's own interaction, every library pair
            from .frame_pair import frame_pair
            table = frame_pair()
            fpe = table.energy(species, sigma_free.to(dtype), pos.to(dtype), cell, inputs["pbc"])
            if r.bond_pauli and r.bond_pauli_wall:       # SPEC 33's wall is the s atom's in-situ interaction with its whole
                fpe = fpe - table.site_reference(species, pos.to(dtype), cell, inputs["pbc"], r.bond_pauli_level)   # site
            self._frame_pair_last = float(fpe.detach())
            energy = energy + fpe
        if r.bond_pauli:                                 # SPEC 33: the bond-oriented Pauli term (trace_redox.bond_pauli)
            from .bond_pauli import bond_pauli
            bp = bond_pauli(r.bond_pauli_level).energy(species, pos.to(dtype), cell, inputs["pbc"])
            self._bond_pauli_last = float(bp.detach())
            energy = energy + bp
            if r.bond_pauli_wall:                        # SPEC 33, the wall part: the lone atom's E1(R), isotropic
                from .bond_pauli import bond_pauli_wall
                bw = bond_pauli_wall(r.bond_pauli_level).energy(species, pos.to(dtype), cell, inputs["pbc"])
                self._bond_pauli_wall_last = float(bw.detach())
                energy = energy + bw
        if fermi_atoms is not None:
            free_atoms = fermi_atoms
        else:
            free_atoms = sol.charge if omega is None else omega.transpose(0, 1) @ sol.charge
        y = elim.bound_moments(free_atoms)
        t, mu_hat, theta_hat = elim.split_moments(y)
        zero_rho = torch.zeros(n, dtype=dtype, device=device)
        q_bound = assembly.charges(t, zero_rho)
        result = RedoxResult(
            energy=energy, energy_bound=elim.e_b0, energy_free=energy_free,
            q=q_bound + free_atoms, q_bound=q_bound, free_charge=free_atoms, unit_charge=sol.charge,
            populations=sol.populations[:n], states=states, state_energies=state_energies,
            electronegativity=chi, gap=gap, sigma_free=sigma_free, sic_removal=removal,
            mu=assembly.dipoles(mu_hat), t=t, rho=zero_rho, theta_hat=theta_hat,
            mu_e=sol.mu_e, nu=sol.nu, omega=omega, molecules=molecules,
            iterations=sol.iterations, residual_charge=sol.residual_charge,
            residual_total=sol.residual_total, duality_gap=sol.duality_gap, ensemble=ensemble,
        )
        return energy, result


def attach_redox(model: nn.Module, heads: nn.Module, config: ElectrostaticConfig | None = None,
                 redox: RedoxConfig | None = None) -> RedoxElectrostatics:
    """Attach TRACE-Redox in place of trace_electrostatics (same attribute, same seams)."""
    module = RedoxElectrostatics(heads, config, redox)
    cutoff = getattr(model, "r_max", None)
    if cutoff is not None:
        for name, value in (("heads.r_max", float(heads.r_max)),
                            ("attention_cutoff", float(module.redox.attention_cutoff))):
            if float(cutoff) < value:
                raise ValueError(
                    f"the model's neighbour cutoff {float(cutoff)} A is shorter than {name} = "
                    f"{value} A; neighbours would leave the list while still weighted"
                )
    setattr(model, ELECTROSTATICS_ATTR, module)
    return module


