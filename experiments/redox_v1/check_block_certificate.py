"""Verification for trace_redox/SPEC.md §12.3 (run from the repo root).
Block certificate: W - blockdiag(D_F) >= 0 with D_F = theta_F (M_FF)^-1, M = W^-1,
theta_F = 1 / (1 + sum_{G != F} ||M_FF^-1/2 M_FG M_GG^-1/2||_Frobenius)."""
import torch; torch.set_default_dtype(torch.float64)
def inv_sqrt(A):
    e,V=torch.linalg.eigh(A); return V@torch.diag(e.rsqrt())@V.T
g=torch.Generator().manual_seed(0); worst=1e9; tight=[]
for trial in range(3000):
    n=int(torch.randint(2,30,(1,),generator=g)); m=int(torch.randint(1,min(n,8)+1,(1,),generator=g))
    A=torch.randn(n,n,generator=g); W=A@A.T+1e-2*torch.eye(n)
    lab=torch.randint(0,m,(n,),generator=g); frags=[torch.nonzero(lab==k).squeeze(1) for k in range(m) if (lab==k).any()]
    M=torch.linalg.inv(W); D=torch.zeros(n,n)
    for F in frags:
        s=sum(torch.linalg.norm(inv_sqrt(M[F][:,F])@M[F][:,G]@inv_sqrt(M[G][:,G])) for G in frags if G is not F)
        th=1.0/(1.0+float(s))
        D[F[:,None],F[None,:]]=th*torch.linalg.inv(M[F][:,F])
    lam=float(torch.linalg.eigvalsh(W-D).min())/float(torch.linalg.eigvalsh(W).max())
    worst=min(worst,lam)
    # also: D_F <= W_FF (never more than the fragment's own Coulomb)
    for F in frags:
        assert float(torch.linalg.eigvalsh(W[F][:,F]-D[F][:,F]).min())>=-1e-10*float(W.abs().max())
print("min relative eigenvalue of W - blockdiag(D) over 3000 random W and partitions:", worst)
# separated limit: block-diagonal W -> theta = 1, D = W_FF
W=torch.block_diag(torch.tensor([[4.,1],[1,3]]),torch.tensor([[5.,2,1],[2,4,1],[1,1,6]]))
M=torch.linalg.inv(W); F=[torch.arange(2),torch.arange(2,5)]
print("separated: D_A == W_AA:", torch.allclose(torch.linalg.inv(M[F[0]][:,F[0]]),W[:2,:2]))
