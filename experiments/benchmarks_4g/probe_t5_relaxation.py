# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_t5_relaxation.py RUN_DIR [...]
"""Where does a model's T5 come from?  For each (doping, geometry) reference of eval_aumgo.py (mean substrate fixed,
mean Au positions as the start): the model energy at the start and after the BFGS relaxation, the relaxation energy,
the Au displacement, the Au-Au bond, the lower Au's height and nearest O / Mg, the free and total charges on the two
Au at start and end, and convergence.  T5 = [E_rel(wet) - E_rel(non-wet)]_doped - [same]_undoped."""
import os, sys, warnings; warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references, AU

refs = references()
for run in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu"); torch.set_num_threads(3)
    cap = []
    calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    print(f"=== {os.path.basename(run.rstrip('/'))}")
    E = {}
    for key, atoms in refs.items():
        a = atoms.copy(); a.info = dict(atoms.info); a.set_constraint(FixAtoms(indices=list(range(108)))); a.calc = calc
        a.calc.reset(); cap.clear(); e0 = a.get_potential_energy(); r0 = cap[-1]
        q0 = (r0.free_charge[AU].detach().numpy(), r0.q[AU].detach().numpy())
        p0 = a.positions[AU].copy()
        opt = BFGS(a, logfile=None); conv = opt.run(fmax=0.005, steps=400)
        a.calc.reset(); cap.clear(); e1 = a.get_potential_energy(); r1 = cap[-1]
        q1 = (r1.free_charge[AU].detach().numpy(), r1.q[AU].detach().numpy())
        disp = np.linalg.norm(a.positions[AU] - p0, axis=1)
        low = AU[int(np.argmax(a.positions[AU, 2]))]
        sym = np.array(a.get_chemical_symbols())
        dO = a.get_distances(low, np.where(sym == "O")[0], mic=True).min()
        dMg = a.get_distances(low, np.where(sym == "Mg")[0], mic=True).min()
        dAA = a.get_distance(AU[0], AU[1], mic=True)
        fmax = float(np.abs(a.get_forces()[AU]).max())
        E[key] = e1
        print(f"  {key[0]:8s} {key[1]:11s}: E_start {e0:+.4f}  E_relax {e1:+.4f}  relaxation {1e3*(e1-e0):+8.1f} meV  "
              f"|dR_Au| {disp[0]:.3f} {disp[1]:.3f} A  Au-Au {dAA:.3f}  low-Au to O {dO:.3f} Mg {dMg:.3f}  "
              f"steps {opt.nsteps} conv {bool(conv)} fmax {fmax:.4f}")
        print(f"      free charge Au start {np.round(q0[0], 3)} end {np.round(q1[0], 3)};  total q start {np.round(q0[1], 3)} end {np.round(q1[1], 3)}")
    for d in ("doped", "undoped"):
        print(f"  {d}: E(wet) - E(non-wet) relaxed {1e3*(E[(d,'wetting')] - E[(d,'non-wetting')]):+.1f} meV")
