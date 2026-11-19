"""Which charge states of the gold pair the selected ensembles hold on the 4G test slabs, with SPEC 39 + 29 + 40 on
(review of SPEC 40.7, 2026-11-14; frozen weights, a probe).  SPEC 40.7 gives entries to (0,0), (-1,0), (+1,0), (+1,-1)
and their mirrors; a pair state outside these (Au- Au-, a charge beyond +-1) keeps the frozen level.

    PYTHONPATH=../.. python probe_doped_pair_states.py RUN_DIR [n=40]

Per doping class: for each structure the ensemble's weight (the diagonal of its density matrix, 'global' space) on
each class of gold-pair state.
"""
import dataclasses, os, sys, warnings
warnings.simplefilter("ignore")
RUN = sys.argv[1]
OPTS = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
sys.argv = [sys.argv[0], "", ""]
import numpy as np, torch
import probe_auau_wall_terms as P
from ase.io import read
from trace_redox.diabatic import density_matrix
N = int(OPTS.get("n", 40))
m = P.Model(RUN)
m.mod.redox = dataclasses.replace(m.base, frame_pair=True, spin_pairing=True, two_site=True, diabatic_space="global", band_form="phi1")
test = read(os.path.join(P.HERE, "data", "AuMgO_test.xyz"), ":")
def cls(qa, qb):
    key = tuple(sorted((int(qa), int(qb))))
    return {(0, 0): "neutral (0,0)", (-1, 0): "one electron (-1,0)", (0, 1): "one hole (+1,0)", (-1, 1): "ion pair (+1,-1)",
            (-1, -1): "both negative (-1,-1)"}.get(key, f"other {key}")
for name, members in (("undoped", [a for a in test if 13 not in a.numbers][:N]), ("doped", [a for a in test if 13 in a.numbers][:N])):
    tot = {}
    worst = {}
    for a in members:
        m.evaluate(a, forces=False)
        last = m.mod._ensemble_last
        H = last["H"].detach(); cfg = last["configs"]
        p = torch.diagonal(density_matrix(H, float(last["kT"])))
        au = np.where(a.numbers == 79)[0]
        w = {}
        for k in range(int(cfg.shape[0])):
            c = cls(cfg[k, int(au[0])], cfg[k, int(au[1])])
            w[c] = w.get(c, 0.0) + float(p[k])
        for c, v in w.items():
            tot[c] = tot.get(c, 0.0) + v / len(members)
            worst[c] = max(worst.get(c, 0.0), v)
    print(f"{name} ({len(members)} test structures): mean weight (largest in one structure) of each class of gold-pair state")
    for c in sorted(tot, key=lambda c: -tot[c]):
        print(f"    {c:26s} {tot[c]:.4f}  ({worst[c]:.4f})")
