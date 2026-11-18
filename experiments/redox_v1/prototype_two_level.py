"""PROTOTYPE (verification only, NOT the package): the two-level certified fragment ensemble
of trace_redox/SPEC.md §12, for a pair of molecules with Q = +1.

Usage: python experiments/redox_v1/prototype_two_level.py [water|h2s]

Compares, at separations 4 / 6 / 10 / 30 A (untrained heads, kT = 0.01 eV):
  * the fragment-exact target: integer molecular charges, mean field inside each
    molecule, full Coulomb, enumerated configuration by configuration;
  * the two-level functional with the block certificate (§12.3-12.4);
  * the current mean field (sic='none').
"""
import sys, math, warnings; sys.path.insert(0,'.'); sys.path.insert(0,'tests/redox')
import torch; torch.set_default_dtype(torch.float64); warnings.simplefilter("ignore")
from redox_helpers import make_heads, edges, neighbour_list
from trace_electrostatics.config import ElectrostaticConfig
from trace_electrostatics.kernels import gram_self_energy
from trace_redox import RedoxConfig, RedoxElectrostatics, ensemble
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
from trace_redox.free_solver import differentiable_solution
kT=0.01; Q=1.0
heads=make_heads()
a=math.radians(104.52)/2
w1=torch.tensor([[0.,0,0],[0.9572*math.sin(a),0.9572*math.cos(a),0.],[-0.9572*math.sin(a),0.9572*math.cos(a),0.]])
hw=torch.randn(3,heads.irreps_in.dim,generator=torch.Generator().manual_seed(2))
b2=math.radians(92.1)/2
s1=torch.tensor([[0.,0,0],[1.336*math.sin(b2),1.336*math.cos(b2),0.],[-1.336*math.sin(b2),1.336*math.cos(b2),0.]])
hs=torch.randn(3,heads.irreps_in.dim,generator=torch.Generator().manual_seed(5))
SECOND=sys.argv[1] if len(sys.argv)>1 else "water"
states=torch.tensor([-1.,0.,1.])
def inv_sqrt(A):
    e,V=torch.linalg.eigh(A); return V@torch.diag(e.rsqrt())@V.T
def newton_max(f, x, iters=200, cap=1.0):
    """maximise a smooth concave f by damped Newton with autograd Hessians (small problems);
    steps capped in max-norm (the dead zone makes the mu-curvature ~exp(-gap/kT))."""
    for it in range(iters):
        x=x.detach().requires_grad_(True); val=f(x); (g,)=torch.autograd.grad(val,x,create_graph=True)
        H=torch.stack([torch.autograd.grad(g[i],x,retain_graph=True)[0] for i in range(len(x))])
        H=0.5*(H+H.T).detach()
        evals,evecs=torch.linalg.eigh(-H)
        evals=evals.clamp_min(1e-12*float(evals.abs().max()))
        step=evecs@((evecs.T@g.detach())/evals)
        n=float(step.abs().max())
        if n>cap: step=step*(cap/n)
        slope=float(g.detach()@step)
        if float(g.detach().abs().max())<1e-10: return x.detach(), float(val)
        t=1.0
        while t>1e-12:
            if float(f((x+t*step).detach()))>=float(val)+1e-4*t*slope: break
            t*=0.5
        x=(x+t*step).detach()
    raise RuntimeError(f"no convergence, |g| = {float(g.abs().max()):.2e}")
for R in (4.0, 6.0, 10.0, 30.0):
    if SECOND=="h2s":
        pos=torch.cat([w1, s1+torch.tensor([R,0,0])]); z=torch.tensor([8,1,1,16,1,1]); h=torch.cat([hw,hs])
    else:
        pos=torch.cat([w1, w1+torch.tensor([R,0,0])]); z=torch.tensor([8,1,1,8,1,1]); h=torch.cat([hw,hw])
    ei=neighbour_list(pos); v,L=edges(pos,ei)
    torch.manual_seed(1); m=RedoxElectrostatics(heads,ElectrostaticConfig(),RedoxConfig(sic="none",kT=kT,fragments="none"))
    E_mf,res=m(h,pos,ei,v,L,data={"z":z,"total_charge":Q})
    asm=assemble_neutral(h,pos,ei,v,L,heads,m.config,species=z); el=eliminate_bound_sector(asm,res.sigma_free)
    W,b,J=el.W.detach(),el.u0.detach(),torch.diagonal(el.J).detach()
    e_hat=(res.state_energies-0.5*J.unsqueeze(1)*states.unsqueeze(0)**2).detach()
    Eb0=float(el.e_b0)
    A,B=torch.arange(3),torch.arange(3,6); frag=torch.tensor([0,0,0,1,1,1])
    # ---- reference: integer fragment charges, full W, enumerated
    confs=[]
    for NA in (-1,0,1,2):
        NB=Q-NA
        def dual(x):
            zz,mu=x[:6],x[6:]
            nu=mu[frag]-b-W@zz
            return -0.5*zz@(W@zz)+mu[0]*NA+mu[1]*NB+ensemble.grand_potential(e_hat,states,nu,kT).sum()
        try:
            _,val=newton_max(dual,torch.zeros(8)); confs.append(val)
        except RuntimeError: pass
    confs=torch.tensor(confs); F_ref=Eb0+float(-kT*torch.logsumexp(-confs/kT,0))
    # ---- two-level with the block certificate
    M=torch.linalg.inv(W); K=inv_sqrt(M[A][:,A])@M[A][:,B]@inv_sqrt(M[B][:,B])
    th=0.999/(1.0+float(torch.linalg.norm(K)))      # strict: S >= 0.001 W > 0
    DA=th*torch.linalg.inv(M[A][:,A]); DB=th*torch.linalg.inv(M[B][:,B])
    S=W-torch.block_diag(DA,DB)
    assert float(torch.linalg.eigvalsh(S).min())>-1e-10
    def Omega(F,D,psi):
        es=[]
        for N in (-1.0,0.0,1.0,2.0):
            sol=differentiable_solution(-psi, D, e_hat[F], states, N, kT)
            es.append(sol.energy)
        es=torch.stack(es); return -kT*torch.logsumexp(-es/kT,0)
    def two_level(x):
        zz,mu=x[:6],x[6]
        psi=mu-b-S@zz
        return -0.5*zz@(S@zz)+mu*Q+Omega(A,DA,psi[A])+Omega(B,DB,psi[B])
    # initial mu: bisection on the monotone fragment charge sum(N_F(mu)) = Q at z = 0
    def total(mu):
        with torch.no_grad():
            tot=0.0
            for F,D in ((A,DA),(B,DB)):
                psi=(mu-b)[F]
                es=torch.stack([differentiable_solution(-psi,D,e_hat[F],states,N,kT).energy for N in (-1.,0.,1.,2.)])
                p=torch.softmax(-es/kT,0); tot+=float(p@torch.tensor([-1.,0.,1.,2.]))
            return tot
    lo,hi=-30.0,30.0
    for _ in range(60):
        mid=0.5*(lo+hi)
        if total(mid)<Q: lo=mid
        else: hi=mid
    x0=torch.zeros(7); x0[6]=0.5*(lo+hi)
    xopt,val=newton_max(two_level,x0); F_two=Eb0+val
    with torch.no_grad():
        psi=xopt[6]-b-S@xopt[:6]
        holeB=[]
        for F,D in ((A,DA),(B,DB)):
            es=torch.stack([differentiable_solution(-psi[F],D,e_hat[F],states,N,kT).energy for N in (-1.,0.,1.,2.)])
            holeB.append(float(torch.softmax(-es/kT,0)@torch.tensor([-1.,0.,1.,2.])))
    print(f"R={R:5.1f} A: theta={th:.3f}  F_ref={F_ref:.4f}  two-level={F_two:.4f} (err {F_two-F_ref:+.4f})  "
          f"MF={float(E_mf):.4f} (err {float(E_mf)-F_ref:+.4f})  MF hole on A={float(res.unit_charge[:3].sum()):.3f}  two-level <N_A>,<N_B>={holeB[0]:.3f},{holeB[1]:.3f}")
