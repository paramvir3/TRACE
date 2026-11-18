"""Verification for trace_redox/SPEC.md §12.9 (run from the repo root).
Na...Cl, Q = 0, kT = 0.01: default fragments='bonds' (sic certified) versus the exact
enumerated ensemble over integer atomic charges (full W, no mean field) and the atom-level
certified mean field (fragments='none').  Beyond the bond switch (R > 1.66+1.02+0.60 = 3.28 A)
the two atoms are separate molecules, so B1 says 'bonds' tends to the exact ensemble as theta -> 1."""
import sys, math, warnings; sys.path.insert(0,'.'); sys.path.insert(0,'tests/redox')
import torch; torch.set_default_dtype(torch.float64); warnings.simplefilter("ignore")
from redox_helpers import make_heads, edges
from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
kT=0.01
heads=make_heads()
h=torch.randn(2,heads.irreps_in.dim,generator=torch.Generator().manual_seed(4))
z=torch.tensor([11,17])
def mod(**kw):
    torch.manual_seed(1); return RedoxElectrostatics(heads,ElectrostaticConfig(),RedoxConfig(kT=kT,**kw))
m_ref=mod(sic="none",fragments="none"); m_b=mod(); m_n=mod(fragments="none")
for a,b in zip(m_ref.parameters(), m_b.parameters()): assert torch.equal(a,b)
states=torch.tensor([-1.,0.,1.])
print(f"{'R':>6} {'F_exact':>10} {'bonds-exact':>12} {'none(cert)-exact':>17} {'v1-exact':>10} | q_Na exact / bonds / none(cert)")
for R in (2.36,3.0,4.0,6.0,8.0,9.0,9.43,10.0,12.0,15.0,30.0,100.0):
    pos=torch.tensor([[0.,0,0],[R,0,0]])
    ei=torch.tensor([[0,1],[1,0]]); v,L=edges(pos,ei)
    E_v1,res=m_ref(h,pos,ei,v,L,data={"z":z,"total_charge":0.0})
    asm=assemble_neutral(h,pos,ei,v,L,heads,m_ref.config,species=z)
    el=eliminate_bound_sector(asm,res.sigma_free)
    W,b,J=el.W.detach(),el.u0.detach(),torch.diagonal(el.J).detach()
    e_hat=(res.state_energies-0.5*J.unsqueeze(1)*states.unsqueeze(0)**2).detach()
    confs=[];qs=[]
    for n in (-1.,0.,1.):
        nn=torch.tensor([n,-n])
        confs.append(float(e_hat[0,int(n)+1]+e_hat[1,int(-n)+1]+b@nn+0.5*nn@(W@nn))); qs.append(n)
    confs=torch.tensor(confs); F=float(el.e_b0)+float(-kT*torch.logsumexp(-confs/kT,0))
    q_ex=float(torch.softmax(-confs/kT,0)@torch.tensor(qs))
    E_b,rb=m_b(h,pos,ei,v,L,data={"z":z,"total_charge":0.0})
    E_n,rn=m_n(h,pos,ei,v,L,data={"z":z,"total_charge":0.0})
    print(f"{R:6.2f} {F:10.4f} {float(E_b)-F:+12.2e} {float(E_n)-F:+17.2e} {float(E_v1)-F:+10.2e} | "
          f"{q_ex:.3f} / {float(rb.q[0]-rb.q_bound[0]):.3f} / {float(rn.q[0]-rn.q_bound[0]):.3f}")
