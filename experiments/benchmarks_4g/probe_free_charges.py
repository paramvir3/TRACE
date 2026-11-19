# Frozen test probe (2026-10-08): where the free (redox) charges sit, 6 doped / 6 undoped / 6 Al->Mg test structures.
# Usage (repo root): python experiments/benchmarks_4g/probe_free_charges.py RUN/model.pt
"""Where do the free charges sit?  Trained redox checkpoint, test structures (doped, undoped, Al->Mg)."""
import sys, warnings; sys.path.insert(0,'.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator
ck=sys.argv[1]
test=read('experiments/benchmarks_4g/data/AuMgO_test.xyz',':')
doped=[a for a in test if a.info['system'].startswith('Al')][:6]; undoped=[a for a in test if a.info['system'].startswith('Au2Mg54')][:6]
alch=[]
for a in doped:
    b=a.copy(); b.info=dict(a.info); n=b.numbers.copy(); n[n==13]=12; b.numbers=n; alch.append(b)
calc=TransformersACECalculator(model_path=ck, device="cpu")
cap=[]; calc.model.electrostatics.register_forward_hook(lambda m,i,o: cap.append(o[1]))
for tag,group in (("doped",doped),("undoped",undoped),("doped geom, Al->Mg",alch)):
    rows=[]
    for a in group:
        b=a.copy(); b.info=dict(a.info); b.calc=calc; cap.clear(); b.get_potential_energy(); r=cap[0]; z=torch.tensor(b.numbers)
        dAl=np.min(b.get_distances(np.where(b.numbers==79)[0][0], np.where(a.numbers==13)[0], mic=True)) if (a.numbers==13).any() else np.nan
        # nearest-neighbour O of Al
        f=r.free_charge.detach(); qb=r.q_bound.detach()
        rows.append([float(f[z==13].sum()) if (z==13).any() else 0.0, float(f[z==8].sum()), float(f[z==12].sum()), float(f[z==79].sum()),
                     float(r.q[z==79].sum()), float(f[z==8].min()), float(r.energy_free)])
    m=np.array(rows).mean(0)
    print(f"{tag:20s} free: Al {m[0]:+.3f}  O {m[1]:+.3f} (most negative O {m[5]:+.3f})  Mg {m[2]:+.3f}  Au2 {m[3]:+.4f} | total q(Au2) {m[4]:+.4f} | E_free {m[6]:+.3f} eV")
