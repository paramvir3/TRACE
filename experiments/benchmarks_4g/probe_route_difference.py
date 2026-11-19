"""Which setting makes the 35-67 meV between the trained route of the single model ('clusters' space, 'fermi' band,
Harrison coupling bound) and the 'global' / 'phi1' / no-bound route on undoped cells (README, 2026-11-13 evening;
frozen weights, a probe).

    PYTHONPATH=../.. python probe_route_difference.py RUN_DIR [threads=2]

The two undoped Table II references: the energy with one setting changed at a time, and with all three.
"""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
RUN = sys.argv[1]
sys.argv = [sys.argv[0], "", ""] + [a for a in sys.argv[2:] if a.startswith("threads=")]
import probe_auau_wall_terms as P  # noqa: E402
from eval_aumgo import references  # noqa: E402

m = P.Model(RUN)
VARIANTS = [("trained (clusters, fermi, bound)", {}),
            ("no coupling bound", dict(coupling_bound=None)),
            ("band 'phi1'", dict(band_form="phi1")),
            ("space 'global'", dict(diabatic_space="global")),
            ("global + phi1", dict(diabatic_space="global", band_form="phi1")),
            ("global + phi1 + no bound", dict(diabatic_space="global", band_form="phi1", coupling_bound=None))]
ref = references()
print(f"=== {m.name}: undoped references; E - E(trained route) in meV")
base = {}
for name, changes in VARIANTS:
    row = []
    for g in ("wetting", "non-wetting"):
        try:
            m.mod.redox = dataclasses.replace(m.base, **changes)
            E = m.evaluate(ref[("undoped", g)], forces=False)["E"]
            base.setdefault(g, E)
            row.append(f"{g}: {1e3 * (E - base[g]):+9.3f}")
        except Exception as err:                                   # a refused combination is a result too
            row.append(f"{g}: refused ({str(err)[:70]})")
    print(f"  {name:34s} " + "   ".join(row), flush=True)
