"""Verification for trace_redox/SPEC.md §12.0 (run from the repo root).
Anatomy of the polyatomic error: one water, total free charge N in [0, 1], mean field.
E_F(N) = the model's own F for a single water at total charge N.  Deficit of the symmetric
split: D = 2 E(1/2) - E(0) - E(1).  Compare with kappa_F / 4, kappa_F = 1 / (1^T W^-1 1)."""
import sys, math, warnings; sys.path.insert(0,'.'); sys.path.insert(0,'tests/redox')
import torch; torch.set_default_dtype(torch.float64); warnings.simplefilter("ignore")
from redox_helpers import make_heads, edges, neighbour_list
from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
heads=make_heads()
a=math.radians(104.52)/2
pos=torch.tensor([[0.,0,0],[0.9572*math.sin(a),0.9572*math.cos(a),0.],[-0.9572*math.sin(a),0.9572*math.cos(a),0.]])
z=torch.tensor([8,1,1]); h=torch.randn(3,heads.irreps_in.dim,generator=torch.Generator().manual_seed(2))
ei=neighbour_list(pos); v,L=edges(pos,ei)
for sic in ("none","certified"):
    for kT in (0.005,):
        torch.manual_seed(1); m=RedoxElectrostatics(heads,ElectrostaticConfig(),RedoxConfig(sic=sic,kT=kT,fragments="none"))
        E={}
        for N in (0.0,0.25,0.5,0.75,1.0):
            e,r=m(h,pos,ei,v,L,data={"z":z,"total_charge":N}); E[N]=float(e)
            if N==1.0: c1=r.unit_charge
        a_=assemble_neutral(h,pos,ei,v,L,heads,m.config,species=z)
        el=eliminate_bound_sector(a_,m.width(z))
        W=el.W - (torch.diag(r.sic_removal) if r.sic_removal is not None else 0)
        one=torch.ones(3)
        kappa=float(1/(one@torch.linalg.solve(W,one)))
        kappa_bare=float(1/(one@torch.linalg.solve(el.W,one)))
        D=2*E[0.5]-E[0.0]-E[1.0]
        second=[E[0.0]-2*E[0.25]+E[0.5], E[0.25]-2*E[0.5]+E[0.75], E[0.5]-2*E[0.75]+E[1.0]]
        print(f"sic={sic:9s}: E(0)={E[0.0]:.4f} E(1/2)={E[0.5]:.4f} E(1)={E[1.0]:.4f}  split deficit D={D:+.4f} eV  "
              f"kappa_F/4={kappa/4:.4f} (screened W after atom SIC)  kappa_F(no SIC)/4={kappa_bare/4:.4f}  "
              f"curvature (2nd diff/0.25^2) = {[round(s/0.0625,3) for s in second]}  hole at N=1: {[round(x,3) for x in c1.tolist()]}")
