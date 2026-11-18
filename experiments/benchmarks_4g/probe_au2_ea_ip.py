"""The vertical electron affinity and ionisation energy of gas-phase Au2 through a trained model, with SPEC 39 + 29 +
40 switched on (review of 2026-11-14; README, "SPEC 40 reviewed"; frozen weights, a probe: nothing trained, nothing
written into a run directory).  Adapted from the reviewer's check.

    PYTHONPATH=../.. python probe_au2_ea_ip.py RUN_DIR [2.30,2.47,2.60,2.80]

Three variants: as trained; SPEC 39 + 29 + 40; the same with the covalent relaxation removed (the hop alone).  Each
with the learned parts as trained and switched off (module.fixed_terms_only: no learned level shift and no learned
coupling; until 2026-11-14 midday the shift alone was zeroed), the condition under which SPEC 40.6 (iii) and 40.7 read the pair.  Printed per distance: EA(Au2) =
E(Au2) - E(Au2-), IP(Au2) = E(Au2+) - E(Au2), the levels of the one-carrier and ionic configurations above the
covalent one, read from the module's own Hamiltonian, and the combination no one-body shift can move,

    I = E(a+ b-) + E(a0 b0) - E(a+ b0) - E(a- b0)      (configuration levels).

References (README of 2026-11-02, PBE0 dimer minus atom on the exact atomic values, stated spread 0.2 eV; not
recomputed here): EA 1.85 / 2.03 / 2.17 / 2.37 eV and IP 9.43 / 9.22 / 9.09 / 8.91 eV at 2.30 / 2.47 / 2.60 / 2.80 A.
"""
import dataclasses
import os
import sys
import warnings

warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
import trace_redox.module as M  # noqa: E402
from ase import Atoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.two_site import TwoSite, two_site  # noqa: E402

RUN = sys.argv[1]
RS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2.30, 2.47, 2.60, 2.80]
REF_EA = {2.30: 1.85, 2.47: 2.03, 2.60: 2.17, 2.80: 2.37}
REF_IP = {2.30: 9.43, 2.47: 9.22, 2.60: 9.09, 2.80: 8.91}

STATE = {"zero": False, "no_rho": False}
_relaxation_pairs = TwoSite.relaxation_pairs
TwoSite.relaxation_pairs = lambda self, *a, **k: None if STATE["no_rho"] else _relaxation_pairs(self, *a, **k)

calc = TransformersACECalculator(model_path=os.path.join(RUN, "model.pt"), device="cpu")
torch.set_num_threads(2)
mod = calc.model.electrostatics
base = mod.redox
ts = two_site()
print(f"=== {os.path.basename(os.path.normpath(RUN))}: space {base.diabatic_space}, selection {base.diabatic_selection}, "
      f"band {base.carrier_band}/{base.band_form}, pauli_table {base.pauli_table}, induction {base.carrier_induction}, "
      f"SPEC 28 {base.state_polarizability}, shift_gate {base.shift_gate}")


def run(R, q):
    at = Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]], pbc=False)
    at.info = {"charge": float(q)}
    at.calc = calc
    calc.reset()
    E = float(at.get_potential_energy())
    last = mod._ensemble_last
    H = None if last["H"] is None else last["H"].detach().clone()
    cfg = [tuple(int(v) for v in row[:2]) for row in last["configs"].tolist()]
    return E, H, cfg


def level(H, cfg, n):
    return float(H[cfg.index(n), cfg.index(n)]) if (H is not None and n in cfg) else float("nan")


VARIANTS = [("as trained", {}, False),
            ("SPEC 39 + 29 + 40", dict(frame_pair=True, spin_pairing=True, two_site=True), False),
            ("SPEC 39 + 29 + the hop of 40, no relaxation", dict(frame_pair=True, spin_pairing=True, two_site=True), True)]
for name, changes, no_rho in VARIANTS:
    for zero in (False, True):
        STATE["zero"], STATE["no_rho"] = zero, no_rho
        mod.redox = dataclasses.replace(base, **changes)
        calc.model.eval()
        mod.fixed_terms_only = bool(zero)        # the package's fixed terms: no learned shift, no learned coupling
        print(f"\n  {name}; learned parts {'OFF (fixed terms only)' if zero else 'as trained'}")
        print("    R     EA(Au2)  error   IP(Au2)  error | E(-1,0)-E(0,0)  E(+1,0)-E(0,0)  E(+1,-1)-E(0,0)        I    | rho(R)")
        for R in RS:
            E0, H0, c0 = run(R, 0)
            Em, Hm, cm = run(R, -1)
            Ep, Hp, cp = run(R, 1)
            l00, lpm = level(H0, c0, (0, 0)), level(H0, c0, (1, -1))
            # levels of different charge sectors on one scale: each sector's Hamiltonian holds absolute energies
            lm, lp = level(Hm, cm, (-1, 0)), level(Hp, cp, (1, 0))
            rho = float(ts.relaxation(79, 79, torch.tensor([R]))[0])
            print(f"    {R:.2f}  {E0 - Em:+7.3f} {E0 - Em - REF_EA.get(R, float('nan')):+7.3f}  {Ep - E0:+7.3f} {Ep - E0 - REF_IP.get(R, float('nan')):+7.3f} | "
                  f"{lm - l00:+13.4f}  {lp - l00:+14.4f}  {lpm - l00:+15.4f}  {lpm + l00 - lp - lm:+9.4f} | {rho:.3f}", flush=True)
STATE["zero"], STATE["no_rho"] = False, False
mod.fixed_terms_only = False
