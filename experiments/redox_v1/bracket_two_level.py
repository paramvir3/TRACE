"""Verification for trace_redox/SPEC.md §14 (run from the repo root).

For one partition (two molecules, Q = +1, kT = 0.01 eV, sic='none', untrained heads):
  F_2L      the two-level canonical free energy (the module's solve, rebuilt here);
  gap       1/2 tr(S Cov_p(y)) at the two-level optimum, from the conditionals y_F(N_F);
  G_restr   the EXACT energy functional evaluated on the two-level trial ensemble (p, y);
  F_ex      the exact classical canonical target (configuration-resolved, enumerated).
Claims tested: G_restr - F_2L = gap (identity) and F_2L <= F_ex <= F_2L + gap (bracket)."""
import sys, math, warnings; sys.path.insert(0, '.'); sys.path.insert(0, 'tests/redox'); warnings.simplefilter("ignore")
import torch; torch.set_default_dtype(torch.float64)
import test_redox_fragments as T
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
from trace_redox.fragments import block_certificate, build_molecules, solve_fragment_ensemble, _conditional
from redox_helpers import edges, neighbour_list
kT, Q = 0.01, 1
for second in ("water", "h2s"):
    for R in (4.0, 6.0, 10.0, 30.0):
        pos, z, h = T._pair(second, R)
        module = T._module(sic="none", kT=kT, fragments="bonds")
        E_mod, res = T._energy(module, pos, z, h, float(Q))
        ei = neighbour_list(pos); vec, L = edges(pos, ei)
        el = eliminate_bound_sector(assemble_neutral(h, pos, ei, vec, L, T.HEADS, module.config, species=z), res.sigma_free)
        W, b = el.W.detach(), el.u0.detach()
        J = el.J.detach(); states = res.states
        e_hat = (res.state_energies - 0.5 * torch.diagonal(J).unsqueeze(1) * states.unsqueeze(0) ** 2).detach()
        top = torch.isfinite(e_hat).sum(1) - 1
        frags = (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
        M = torch.linalg.inv(W); M = 0.5 * (M + M.T)
        D_list, theta, S = block_certificate(W, M, frags)
        mols = build_molecules(frags, D_list, e_hat, J, states, top, Q, 2, False, 0.999, 0.01)
        sol = solve_fragment_ensemble([(torch.tensor(1.0), frags)], [mols], S, b, states, Q, kT)
        F_2L = float(el.e_b0) + float(sol.energy)
        psi = sol.potentials
        # conditionals and the canonical joint distribution over (N_A, Q - N_A)
        cond = []
        for mol in mols:
            cond.append({N: _conditional(mol, N, psi[mol.atoms], states, kT, 1e-12, False) for N in mol.window})
        pA = dict(zip(mols[0].window, sol.partitions[0][2][0].tolist()))
        configs = [(NA, Q - NA, p) for NA, p in pA.items() if (Q - NA) in cond[1] and p > 0]
        ys = torch.stack([torch.cat((cond[0][NA][1], cond[1][NB][1])) for NA, NB, _ in configs])
        ps = torch.tensor([p for _, _, p in configs]); ps = ps / ps.sum()
        cbar = ps @ ys
        Sigma = (ys - cbar).T @ ((ys - cbar) * ps.unsqueeze(1))
        gap = 0.5 * float(torch.trace(S @ Sigma))
        # A_F(y_F(N)) = e_F(N; psi) + psi_F . y_F(N)
        A = lambda F, N: float(cond[F][N][0] + psi[mols[F].atoms] @ cond[F][N][1])
        entropy = kT * float((ps * torch.log(ps)).sum())
        G_2L = sum(float(p) * (A(0, NA) + A(1, NB)) for (NA, NB, _), p in zip(configs, ps)) + float(b @ cbar) + 0.5 * float(cbar @ S @ cbar) + entropy
        G_restr = sum(float(p) * (A(0, NA) + A(1, NB) + float(b @ y) + 0.5 * float(y @ S @ y)) for (NA, NB, _), p, y in zip(configs, ps, ys)) + entropy
        F_ex = T._molecule_exact(T._module(sic="none", kT=kT, fragments="none"), pos, z, h, float(Q), kT)
        inside = F_2L - 1e-9 <= F_ex <= F_2L + gap + 1e-9
        print(f"{second:5s} R={R:4.0f}  theta={float(theta[0]):.3f}  module-rebuild {float(E_mod)-F_2L:+.1e}  "
              f"G_2L-F_2L {float(el.e_b0)+G_2L-F_2L:+.1e}  identity {(G_restr-G_2L)-gap:+.1e} | "
              f"F_ex-F_2L = {F_ex-F_2L:.4f}  gap = {gap:.4f}  ({'inside' if inside else 'OUTSIDE'})  upper-bound error {F_2L+gap-F_ex:+.4f}")

# ---- the ionic-neutral crossing: Na...Cl, two one-atom molecules, Q = 0
print('Na...Cl, Q = 0')
from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics
from redox_helpers import make_heads
heads=make_heads()
h=torch.randn(2,heads.irreps_in.dim,generator=torch.Generator().manual_seed(4)); z=torch.tensor([11,17])
torch.manual_seed(1); mod=RedoxElectrostatics(heads,ElectrostaticConfig(),RedoxConfig(kT=kT,sic="none",fragments="none"))
states=torch.tensor([-1.,0.,1.])
for R in (8.0,9.0,9.43,10.0,12.0):
    pos=torch.tensor([[0.,0,0],[R,0,0]]); ei=torch.tensor([[0,1],[1,0]]); v,L=edges(pos,ei)
    _,res=mod(h,pos,ei,v,L,data={"z":z,"total_charge":0.0})
    el=eliminate_bound_sector(assemble_neutral(h,pos,ei,v,L,heads,mod.config,species=z),res.sigma_free)
    W,b,J=el.W.detach(),el.u0.detach(),el.J.detach()
    e_hat=(res.state_energies-0.5*torch.diagonal(J).unsqueeze(1)*states.unsqueeze(0)**2).detach()
    top=torch.isfinite(e_hat).sum(1)-1
    frags=(torch.tensor([0]),torch.tensor([1]))
    M=torch.linalg.inv(W); D,theta,S=block_certificate(W,0.5*(M+M.T),frags)
    mols=build_molecules(frags,D,e_hat,J,states,top,0,2,False,0.999,0.01)
    sol=solve_fragment_ensemble([(torch.tensor(1.0),frags)],[mols],S,b,states,0,kT)
    F2=float(el.e_b0)+float(sol.energy)
    pA=dict(zip(mols[0].window,sol.partitions[0][2][0].tolist()))
    confs=[(n,-n,p) for n,p in pA.items() if -n in mols[1].window]
    ys=torch.tensor([[float(a),float(b_)] for a,b_,_ in confs]); ps=torch.tensor([p for *_,p in confs]); ps=ps/ps.sum()
    cbar=ps@ys; Sig=(ys-cbar).T@((ys-cbar)*ps.unsqueeze(1)); gap=0.5*float(torch.trace(S@Sig))
    Es=torch.tensor([float(e_hat[0,n+1]+e_hat[1,-n+1]+b@torch.tensor([float(n),float(-n)])+0.5*torch.tensor([float(n),float(-n)])@W@torch.tensor([float(n),float(-n)])) for n in (-1,0,1)])
    Fex=float(el.e_b0)+float(-kT*torch.logsumexp(-Es/kT,0))
    print(f"R={R:5.2f} theta={float(theta[0]):.3f}  F_ex-F_2L = {Fex-F2:.2e}  gap = {gap:.2e}  {'inside' if F2-1e-9<=Fex<=F2+gap+1e-9 else 'OUTSIDE'}  upper-bound error {F2+gap-Fex:+.2e}  <q_Na> 2L {float(cbar[0]):.3f}")
