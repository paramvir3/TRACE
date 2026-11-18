"""Which configurations a cluster's ensemble holds, and their weights (frozen weights; 2026-10-19; reported).

Usage (repo root): python experiments/benchmarks_4g/probe_cluster_ensemble.py DATA.xyz CHARGE N CKPT [...]

For N seeded random structures of the given total charge: the selected configurations, their energies above the
lowest (diagonal of the diabatic H), and their populations, as trained and with the overlap stack (SPEC 22, 23,
25, 26; frozen).  Summarised: the mean population of configurations with more atoms away from their frames than
the total charge requires, |Q - sum f| ("ionic admixture" beyond the minimal carrier set), and the mean energy of
the lowest such configuration above the lowest overall.
Hypothesis under test (U5's undiagnosed cation-force failure): Proposition 8's inverted ionic gap puts multi-ion
configurations near the cation's ground state."""
import dataclasses
import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402

STACK = dict(bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
             carrier_induction=True)


def main():
    path, charge, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    frames = [a for a in read(path, ":") if round(float(a.info.get("charge", 0))) == charge]
    pick = np.random.default_rng(0).choice(len(frames), min(count, len(frames)), replace=False)
    for ck in sys.argv[4:]:
        calc = TransformersACECalculator(model_path=ck, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = mod.redox
        print(f"=== {ck.split('/')[-2]}  {len(pick)} structures of charge {charge:+d}")
        for label, kw in (("as trained", {}), ("overlap stack", STACK),
                          ("stack, zero shifts", dict(STACK, chi_bound=1e-9))):
            mod.redox = dataclasses.replace(base, **kw)
            p_multi, gaps, n_cfg, example = [], [], [], None
            for i in pick:
                a = frames[i].copy()
                a.info = {"charge": float(charge)}
                a.calc = calc
                calc.reset()
                a.get_potential_energy()
                last = mod._ensemble_last
                cfg, H = last["configs"], last["H"].detach()
                p = torch.diagonal(density_matrix(H, float(last["kT"]))).tolist()
                E = (torch.diagonal(H) - torch.diagonal(H).min()).tolist()
                f = formal_frame(torch.as_tensor(a.numbers)).to(cfg.device)
                n = int(f.numel())
                charged = (cfg[:, :n] != f.unsqueeze(0)).sum(1).tolist()  # atoms away from their frames
                minimal = abs(charge - int(f.sum()))                       # the fewest that hold the total charge
                multi = [k for k, c in enumerate(charged) if c > minimal]
                p_multi.append(sum(p[k] for k in multi))
                gaps.append(min((E[k] for k in multi), default=float("nan")))
                n_cfg.append(len(E))
                if example is None:
                    example = sorted(zip(E, cfg.tolist(), p))[:6]
            print(f"  {label:13s}: configurations {np.mean(n_cfg):.1f}; population of multi-ion configurations "
                  f"{np.mean(p_multi):.3f} (max {np.max(p_multi):.3f}); lowest multi-ion gap {np.nanmean(gaps):+.3f} eV "
                  f"(min {np.nanmin(gaps):+.3f})")
            print("      first structure, lowest six: " + "; ".join(f"{c} {e:+.3f} eV p {q:.3f}" for e, c, q in example))
        mod.redox = base


if __name__ == "__main__":
    main()
