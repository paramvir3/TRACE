"""How much the sign convention of the configuration couplings matters on the slabs (README, pre-registered
2026-11-13 evening; frozen weights, a probe: nothing trained, nothing written into a run directory).

    PYTHONPATH=../.. python probe_sign_rule.py RUN_DIR [threads=2] [n=15]

Undoped structures only, in the 'global' space with the 'phi1' band and no coupling bound (exactly the trained model
there when no coupling sits above the bound's knee: the difference from the trained route is printed per structure).
Each structure with the package's Jordan-Wigner signs (as trained) and with the 'singlet rule': no string, +T f for
every move and -T f where the donor holds an even and the acceptor an odd number of electrons (the move of a hole).
Reported: wet_u on the two Table II references and the class offset on the first n flat and n upright undoped test
structures, under both rules.
"""
import dataclasses
import os
import sys
import warnings

warnings.simplefilter("ignore")
RUN = sys.argv[1]
OPTS = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
sys.argv = [sys.argv[0], "", ""] + [a for a in sys.argv[2:] if a.startswith("threads=")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import probe_auau_wall_terms as P  # noqa: E402
import trace_redox.diabatic as D  # noqa: E402
from ase.io import read  # noqa: E402
from eval_aumgo import geometry_class, references  # noqa: E402

N_EACH = int(OPTS.get("n", 15))
m = P.Model(RUN)
ORIGINAL = D.one_electron_moves
STATE = {"rule": "package", "z": None, "moves": 0, "flipped": 0}


def moves(configurations, max_charge):
    rows, cols, donor, acceptor, sign = ORIGINAL(configurations, max_charge)
    if STATE["rule"] == "package":
        return rows, cols, donor, acceptor, sign
    n = configurations.to(torch.long)
    z = STATE["z"].to(n.device)
    U = int(n.shape[1])
    if U > int(z.numel()):                                  # the band unit counts -charge electrons (as in SPEC 29)
        z = torch.cat([z, torch.zeros(U - int(z.numel()), dtype=torch.long, device=n.device)])
    N = z.unsqueeze(0) - n
    hole = ((N[rows, donor] % 2) == 0) & ((N[rows, acceptor] % 2) == 1)
    STATE["moves"] += int(rows.numel())
    STATE["flipped"] += int((torch.where(hole, -1, 1) != sign.to(torch.long)).sum())
    return rows, cols, donor, acceptor, torch.where(hole, -torch.ones_like(sign), torch.ones_like(sign))


D.one_electron_moves = moves


def energies(atoms):
    STATE["z"] = torch.tensor(atoms.numbers, dtype=torch.long)
    out = {}
    m.set({})
    STATE["rule"] = "package"
    out["trained"] = m.evaluate(atoms, forces=False)["E"]
    m.mod.redox = dataclasses.replace(m.base, diabatic_space="global", band_form="phi1", coupling_bound=None)
    out["package"] = m.evaluate(atoms, forces=False)["E"]
    STATE["rule"] = "singlet"
    out["singlet"] = m.evaluate(atoms, forces=False)["E"]
    out["configs"] = int(m.mod._ensemble_last["configs"].shape[0])
    STATE["rule"] = "package"
    return out


print(f"=== {m.name}: undoped structures; energies in eV, differences in meV")
ref = references()
R = {g: energies(ref[("undoped", g)]) for g in ("wetting", "non-wetting")}
for g, r in R.items():
    print(f"  reference {g:12s}: global - trained {1e3 * (r['package'] - r['trained']):+.3f} meV; singlet - package "
          f"{1e3 * (r['singlet'] - r['package']):+.2f} meV; configurations {r['configs']}", flush=True)
wet = {k: 1e3 * (R["wetting"][k] - R["non-wetting"][k]) for k in ("trained", "package", "singlet")}
print(f"  wet_u: trained route {wet['trained']:+.1f}, package signs (global) {wet['package']:+.1f}, singlet rule {wet['singlet']:+.1f} meV;"
      f" change {wet['singlet'] - wet['package']:+.1f} meV")
test = [a for a in read(os.path.join(P.HERE, "data", "AuMgO_test.xyz"), ":") if 13 not in a.numbers]
groups = {"flat": [a for a in test if geometry_class(a) == "wetting"][:N_EACH],
          "upright": [a for a in test if geometry_class(a) == "non-wetting"][:N_EACH]}
shift, drift = {}, {}
for name, members in groups.items():
    rows = [energies(a) for a in members]
    shift[name] = np.array([1e3 * (r["singlet"] - r["package"]) for r in rows])
    drift[name] = np.array([1e3 * (r["package"] - r["trained"]) for r in rows])
    print(f"  {name:8s} ({len(rows)} test structures): singlet - package {shift[name].mean():+.2f} +- {shift[name].std(ddof=1) / np.sqrt(len(rows)):.2f} meV"
          f" (range {shift[name].min():+.2f} to {shift[name].max():+.2f}); global - trained: largest {np.abs(drift[name]).max():.3f} meV", flush=True)
print(f"  change of the class offset (flat - upright): {shift['flat'].mean() - shift['upright'].mean():+.2f} meV")
print(f"  moves examined under the singlet rule: {STATE['moves']}, of which the sign differs from the package's: {STATE['flipped']}")
