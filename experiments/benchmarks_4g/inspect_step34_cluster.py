"""Step 34 failure dumps (2026-11-15): configuration vectors, charge totals, spectrum and couplings of the 529-row cluster (raised cap).  Repo root: python experiments/benchmarks_4g/inspect_step34_cluster.py"""
import sys, os, dataclasses, collections, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
import replay_redox_failure as rrf
from trace_redox.constants import formal_frame
from trace_redox.diabatic import density_matrix
D="experiments/benchmarks_4g/runs/step34_failure_2026-11-15"
for s in (42,43):
    dump=torch.load(f"{D}/redox_failure_s{s}_local.pt",map_location="cpu",weights_only=False)
    m=rrf.build(dump); a,k=dump["args"],dump["kwargs"]
    m.redox=dataclasses.replace(m.redox,fragment_max_configurations=20000)
    with torch.no_grad(): E,res=m(*a,**k)
    L=m._ensemble_last; C=L["configs"]; n0=L["reference"]; H=L["H"]; z=k["data"]["z"]; n=int(z.numel())
    print(f"== s{s}: Q {k['data'].get('charge', k.get('charge'))} configs {tuple(C.shape)} K {L['K'][n:].tolist() if L['K'] is not None else None}")
    print(" n0 tail (band units):", n0[n:].tolist(), " n0 atoms sum:", int(n0[:n].sum()), " frame sum:", int(formal_frame(z).sum()))
    print(" n0 atoms differ from frame:", int((n0[:n]!=formal_frame(z)).sum()))
    tot=C.sum(1); print(" total of rows (unique):", sorted(set(tot.tolist())))
    w,V=torch.linalg.eigh(H); print(f" lowest eigenvalues: {(w[:4]-w[0]).tolist()} ; diag min {float(torch.diagonal(H).min()):.4f}, w0 {float(w[0]):.4f}, diag[n0] {float(H[0,0]):.4f}")
    off=H-torch.diag(torch.diagonal(H)); nz=off[off.abs()>1e-12].abs()
    print(f" couplings: {int(nz.numel())} nonzero, |T| max {float(nz.max()):.4f} median {float(nz.median()):.4f} eV")
    p=torch.diagonal(density_matrix(H,float(L['kT'])))
    print(f" p(n0) {float(p[0]):.4f}; charge on O (mean) {float(res.charge[:n][z==8].mean()) if hasattr(res,'charge') else 'na'}")
    print(" rows 0..3 band tails:", [C[i,n:].tolist() for i in range(4)], " row 0 == n0:", bool((C[0]==n0).all()))
