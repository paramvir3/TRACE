"""The model's own two-site sector for a gas-phase Au2 (2026-11-01): at bond lengths R, the ensemble Hamiltonian the
module builds (open boundaries, Q = 0), its configurations, the diagonal Delta_ion = H[ion, ion] - H[cov, cov], the
actual coupling |H[cov, ion]| (SPEC 21's switch and SPEC 29's spin factor included as the model applies them), and the
ground state's weights; against the measured two-site Hubbard U(R), t(R) of tools/electrostatics/au2_hubbard.py, read in
the model's own form: the symmetric ionic combination couples to the covalent state with sqrt(2) |H_cov,ion| in the
model and with 2 t in the exact singlet, so the model reproduces the physical singlet when |H_cov,ion| = sqrt(2) t
(with spin_pairing off tau = sqrt 2 t; with SPEC 29 on, H = sqrt 2 tau and tau = t) and Delta_ion = U.  Usage: python probe_dimer_hubbard.py MODEL.pt [...]"""
import sys, math, dataclasses
import numpy as np
import torch
sys.path.insert(0, ".")
from ase import Atoms
from flashace.calculator import TransformersACECalculator
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
UT = {2.30: (0.900, 1.409), 2.47: (0.995, 1.249), 2.60: (1.063, 1.146), 2.80: (1.164, 1.009)}   # TDA-PBE U, t (eV)
UT_CC = {2.30: (0.745, 1.501), 2.47: (0.875, 1.317), 2.60: (0.997, 1.191), 2.80: (1.224, 1.012)}  # EOM-CCSD
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu")
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False)
    sp = bool(mod.redox.spin_pairing)
    print(f"=== {model.split('/')[-2][-30:]}: spin_pairing {sp}", flush=True)
    for R, (U, t) in UT.items():
        at = Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]], pbc=False); at.info = {"charge": 0.0}
        at.calc = calc; calc.reset(); at.get_potential_energy()
        last = mod._ensemble_last; H = last["H"].detach().numpy(); cfg = last["configs"].tolist()
        rows = {tuple(c[:2]): i for i, c in enumerate(cfg)}
        cov, ion, rev = rows.get((0, 0)), rows.get((1, -1)), rows.get((-1, 1))
        w, V = np.linalg.eigh(H); g = V[:, 0] ** 2
        d_ion = H[ion, ion] - H[cov, cov] if None not in (cov, ion) else float("nan")
        c_ion = abs(H[cov, ion]) if None not in (cov, ion) else float("nan")
        # the physical singlet in the model's form: coupling to the symmetric ionic combination sqrt(2) c (model) = 2 t
        # the H element carries SPEC 29's sqrt 2 when spin_pairing is on (tau = t), so its target is sqrt 2 t either way
        # (the first version used t with spin_pairing on; every reported run had it off, 2026-11-01)
        c_target = math.sqrt(2.0) * t
        p_ion = (g[ion] if ion is not None else 0) + (g[rev] if rev is not None else 0)
        Wp = math.sqrt(U * U + 16 * t * t); Eg = (U - Wp) / 2; r = Eg / (2 * t); p_phys = r * r / (1 + r * r)
        print(f"  R {R:.2f}: {len(cfg)} configurations {cfg if len(cfg) <= 6 else str(cfg[:6]) + '...'}; Delta_ion {d_ion:+.3f} eV "
              f"(U: PBE {U:.3f}, CCSD {UT_CC[R][0]:.3f}); |H_cov,ion| {c_ion:.3f} eV (target {c_target:.3f}: sqrt 2 t, PBE t {t:.3f}); "
              f"ground ionic weight {p_ion:.3f} (physical singlet {p_phys:.3f})", flush=True)
