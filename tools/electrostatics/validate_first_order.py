"""Validation of first_order_exchange.hl_hp_energies against exact properties and an independent reference.

    python tools/electrostatics/validate_first_order.py
    (python: an interpreter with PySCF installed)

With HF monomer determinants in the dimer-centred basis (ghost functions), for He2 and Ne2:
  (a) symmetry: exchanging the monomers leaves E_HL and E_HP unchanged;
  (b) variational bound: E_HL >= E_HF(dimer), since E_HL is the energy of one single determinant and the dimer HF
      energy is the minimum over all of them;
  (c) the supermolecular counterpoise HF interaction Delta E_HF = E_dimer - E_A - E_B equals E^(1) + delta_HF with
      E^(1) = E_HL - E_A - E_B = E_elst^(1) + E_exch^(1); for rare gases delta_HF (induction and exchange-induction)
      is a small negative correction (ESTABLISHED: Jeziorski, Moszynski and Szalewicz 1994), so |delta_HF / E^(1)|
      must be small.
Pass criteria (fixed before running): (a) 1e-10 hartree; (b) E_HL - E_dimer >= -1e-10; (c) |delta_HF / E^(1)| < 0.10
and delta_HF < 0.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from first_order_exchange import hl_hp_energies  # noqa: E402


def run(symbol, R, basis):
    from pyscf import gto, scf
    def mol_of(spec):
        return gto.M(atom=spec, basis=basis, unit="Angstrom", verbose=0)

    dimer = mol_of(f"{symbol} 0 0 0; {symbol} 0 0 {R}")
    mono_a = mol_of(f"{symbol} 0 0 0; ghost-{symbol} 0 0 {R}")
    mono_b = mol_of(f"ghost-{symbol} 0 0 0; {symbol} 0 0 {R}")
    energies, occs = {}, {}
    for key, mol in (("A", mono_a), ("B", mono_b), ("AB", dimer)):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        energies[key] = mf.kernel()
        occs[key] = [mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in (0, 1)]
    mf = scf.UHF(dimer)
    h = mf.get_hcore()
    e_nuc = dimer.energy_nuc()
    e_a_nuc, e_b_nuc = mono_a.energy_nuc(), mono_b.energy_nuc()
    e_hl, e_hp = hl_hp_energies(dimer, mf, h, occs["A"], occs["B"])
    e_hl2, e_hp2 = hl_hp_energies(dimer, mf, h, occs["B"], occs["A"])
    E_HL, E_HP = e_hl + e_nuc, e_hp + e_nuc
    # monomer energies with their own Hamiltonians (ghost basis, one nucleus each)
    E_A, E_B, E_AB = energies["A"], energies["B"], energies["AB"]
    e1 = E_HL - E_A - E_B
    elst = E_HP - E_A - E_B
    exch = E_HL - E_HP
    dHF = (E_AB - E_A - E_B) - e1
    ok_a = abs(e_hl - e_hl2) < 1e-10 and abs(e_hp - e_hp2) < 1e-10
    ok_b = E_HL - E_AB >= -1e-10
    ok_c = abs(dHF / e1) < 0.10 and dHF < 0
    uh = 1e6
    print(f"{symbol}2 R = {R} A ({basis}): E_elst(1) {elst * uh:+.3f}  E_exch(1) {exch * uh:+.3f}  E(1) {e1 * uh:+.3f}  "
          f"Delta E_HF {(E_AB - E_A - E_B) * uh:+.3f}  delta_HF {dHF * uh:+.3f} micro-hartree;  "
          f"(a) {'PASS' if ok_a else 'FAIL'} (b) {'PASS' if ok_b else 'FAIL'} (E_HL - E_HF = {(E_HL - E_AB) * uh:+.3f}) "
          f"(c) {'PASS' if ok_c else 'FAIL'} (delta_HF / E(1) = {dHF / e1:+.3f})", flush=True)
    del e_a_nuc, e_b_nuc


if __name__ == "__main__":
    run("He", 5.6 * 0.529177210903, "aug-cc-pvtz")
    run("Ne", 3.1, "aug-cc-pvtz")
