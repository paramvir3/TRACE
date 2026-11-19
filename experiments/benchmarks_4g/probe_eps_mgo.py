# Frozen test probe (2026-10-08): epsilon_inf of bulk MgO from a checkpoint's bound sector.
# Usage (repo root): python experiments/benchmarks_4g/probe_eps_mgo.py RUN/model.pt [...]
"""eps_inf of bulk MgO from the trained models' bound (SPEC 1) sector."""
import sys, warnings; sys.path.insert(0,'.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.build import bulk
from flashace.calculator import TransformersACECalculator
from trace_electrostatics.energy import electrostatic_state
from trace_electrostatics.observables import dielectric_tensor
atoms=bulk('MgO','rocksalt',a=4.212,cubic=True).repeat((2,2,2))
atoms.info['charge']=0.0
for ck in sys.argv[1:]:
    calc=TransformersACECalculator(model_path=ck, device="cpu")
    mod=calc.model.electrostatics
    cap=[]
    hook=mod.register_forward_hook(lambda m,i,o,**kw: None)
    hook.remove()
    def pre(m, args, kwargs):
        cap.append((args, kwargs)); return None
    h2=mod.register_forward_pre_hook(pre, with_kwargs=True)
    a=atoms.copy(); a.info=dict(atoms.info); a.calc=calc
    # AMENDMENT 2026-10-08: the energy call only captures the inputs (pre-hook). A TRACE-Redox model may
    # refuse bulk MgO in its FREE sector (W singular); eps_inf below is the bound sector's and is unaffected.
    try: a.get_potential_energy()
    except Exception as err:
        if not cap: raise
        print(f'  (model energy refused after input capture: {type(err).__name__}: {str(err)[:90]})')
    h2.remove()
    args,kwargs=cap[0]
    h,pos,ei,vec,L=[x.detach() for x in args[:5]]
    data=kwargs.get('data',{})
    keys={k:kwargs[k] for k in ('pbc','cell','edge_shift') if k in kwargs}
    pos=pos.clone().requires_grad_(True)
    st=electrostatic_state(h,pos,ei,vec,L,mod.heads,mod.config,species=data.get('z'),**keys)
    eps=dielectric_tensor(st.solve_result,st.assembly).detach()
    q=st.solve_result.q.detach(); z=torch.tensor(a.numbers)
    print(f"{ck.split('/')[-2]:22s} eps_inf diag {[round(float(x),3) for x in torch.diagonal(eps)]}  q(Mg) {float(q[z==12].mean()):+.3f}  q(O) {float(q[z==8].mean()):+.3f}  (module {type(mod).__name__})")
