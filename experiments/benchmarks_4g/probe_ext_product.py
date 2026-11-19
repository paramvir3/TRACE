"""Extensivity diagnosis 3 (2026-11-10, after X1).  What part of E(M cells) / M - E(1 cell) is the selected space, what
part the SPEC 21 switch, and what part the lattice (the image term)?

Frozen weights; nothing in the package is changed.  ``trace_redox.module.select_configurations`` is wrapped in this
process: in a replicated cell it can return the PRODUCT SPACE of the one-cell selection (every copy takes any of the
cell's selected configurations independently; m^M configurations) instead of the shipped global selection.

    python probe_ext_product.py MODEL.pt FORM [nx ny ...]         FORM = checkpoint | onsite | block

Per undoped geometry and replication it prints E / M - E(1x1) and max|dF| for
    shipped            the global selection, the SPEC 21 switch (what the package does)
    product            the product space, the SPEC 21 switch
    product, unsw.     the product space, every one-electron coupling at full strength (diabatic_selection='window'
                       changes only the Hamiltonian here, the space is given), against the 1x1 cell evaluated the same way
The last line is the model's own finite-cell (image) effect: the same Hamiltonian diagonalised exactly in a space that
contains every product state.  Writes nothing into the run directory."""
import dataclasses
import itertools
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
import trace_redox.module as M
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

ONLY = sys.argv.pop() if sys.argv[-1] in ("non-wetting", "wetting") else None
model_path, form = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
if form != "checkpoint":
    mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form)
trained = mod.redox
print(f"model {model_path.split('/')[-2]}\nevaluated with state_polarizability_stiffness = {mod.redox.state_polarizability_stiffness!r}, "
      f"kT = {mod.redox.kT}, window = {mod.redox.diabatic_window}, amplitudes = ({mod.redox.selection_amplitude_off}, "
      f"{mod.redox.selection_amplitude_on})\n")
refs = references()
NAT = 110
STATE = {"mode": "shipped", "cell": None, "selected": None, "n": None}
original = M.select_configurations


def select(e_hat, states, b, W, K, Q, **kw):
    if STATE["mode"] == "given":                                  # the 1x1 cell with its own shipped selection
        STATE["n"] = int(STATE["cell"].shape[0])
        return STATE["cell"].to(b.device)
    configs = original(e_hat, states, b, W, K, Q, **kw)
    STATE["selected"] = configs.clone()
    if STATE["mode"] == "shipped":
        STATE["n"] = int(configs.shape[0])
        return configs
    cell = STATE["cell"]                                          # [m, U1], the 1x1 selection
    U = int(b.shape[0]); copies = (U - (int(cell.shape[1]) - NAT)) // NAT
    extra = cell[:, NAT:]
    assert bool((extra == extra[0]).all()), "a unit beyond the atoms is active in the cell's set"
    rows = []
    for choice in itertools.product(range(int(cell.shape[0])), repeat=copies):
        rows.append(torch.cat([cell[a, :NAT] for a in choice] + [extra[0]]))
    out = torch.stack(rows).to(configs.dtype).to(configs.device)
    assert int(out.shape[1]) == U and bool((out.sum(1) == int(round(Q))).all())
    have = {tuple(r) for r in out.tolist()}
    STATE["missing"] = sum(tuple(r) not in have for r in configs.tolist())   # shipped configurations outside the product space
    STATE["n"] = int(out.shape[0])
    return out


M.select_configurations = select


def evaluate(key, rep, mode, switched=True):
    mod.redox = trained if switched else dataclasses.replace(trained, diabatic_selection="window")
    STATE["mode"] = mode
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); mod._ensemble_last = None
    t0 = time.time()
    e = a.get_potential_energy(); f = a.get_forces()
    last = mod._ensemble_last
    p = torch.diagonal(M.density_matrix(last["H"].detach(), float(last["kT"])))
    return dict(e=e, f=f, n=STATE["n"], p0=float(p.max()), seconds=time.time() - t0, H=last["H"].detach(),
                configs=last["configs"].to(torch.long))


for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
    if ONLY and key[1] != ONLY:
        continue
    one = evaluate(key, (1, 1, 1), "shipped")
    one_full_e = one["e"]
    STATE["cell"] = STATE["selected"].clone()
    cell = STATE["cell"]
    active = sorted(set((cell != cell[0]).nonzero()[:, 1].tolist()))
    H = one["H"]; d = torch.diagonal(H); order = torch.argsort(d)
    print(f"== {key}: 1x1 selects {one['n']} configurations on {int(cell.shape[1])} units; active units {active}; "
          f"largest weight {one['p0']:.4f}")
    print("   diagonal energies above the lowest (eV): " + " ".join(f"{float(d[i] - d[order[0]]):.4f}" for i in order))
    off = H - torch.diag(d)
    print("   couplings to the lowest (eV):            " + " ".join(f"{float(off[order[0], i]):+.4f}" for i in order))
    for i in order[1:]:
        ch = (cell[i] != cell[order[0]]).nonzero().flatten().tolist()
        print(f"      +{float(d[i] - d[order[0]]):.4f} eV: " + ", ".join(f"unit {u}: {int(cell[order[0], u])} -> {int(cell[i, u])}" for u in ch))
    one_unsw = evaluate(key, (1, 1, 1), "given", switched=False)
    offu = one_unsw["H"] - torch.diag(torch.diagonal(one_unsw["H"]))
    print("   the same, unswitched (eV):               " + " ".join(f"{float(offu[order[0], i]):+.4f}" for i in order))
    print(f"   1x1 unswitched - switched: {1e3 * (one_unsw['e'] - one['e']):+.4f} meV")
    if bool((cell[:, NAT:] != cell[0, NAT:]).any()):
        # the band unit is ONE unit for the whole cell (README R3, cause 1; the comment said R12 until 2026-11-10), so its states have no product space: the
        # comparison is made on the atoms' configurations, with the 1x1 cell evaluated on the same reduced set
        keep = (cell[:, NAT:] == cell[order[0], NAT:]).all(1)
        STATE["cell"] = cell = cell[keep]
        print(f"   a unit beyond the atoms is active: the comparison below keeps the {int(keep.sum())} configurations that leave it "
              f"in the reference state, in the 1x1 cell too")
        one = evaluate(key, (1, 1, 1), "given", switched=True)
        one_unsw = evaluate(key, (1, 1, 1), "given", switched=False)
        print(f"   1x1 on the reduced set: {1e3 * (one['e'] - one_full_e):+.4f} meV against the shipped selection; "
              f"unswitched - switched {1e3 * (one_unsw['e'] - one['e']):+.4f} meV")
    for rep in reps:
        copies = rep[0] * rep[1]
        rows = (("shipped", "shipped", True, one), ("product", "product", True, one),
                ("product, unsw.", "product", False, one_unsw))
        for label, mode, switched, base in rows:
            try:
                r = evaluate(key, rep, mode, switched=switched)
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                print(f"   {rep[0]}x{rep[1]} {label:15s}: REFUSED {type(err).__name__}: {str(err)[:160]}")
                continue
            df = np.abs(r["f"] - np.tile(base["f"], (copies, 1))).max()
            note = f"  shipped configurations outside the product space: {STATE.get('missing')}" if mode == "product" else ""
            print(f"   {rep[0]}x{rep[1]} {label:15s}: configurations {r['n']:5d}  E/cell - E(1x1) {1e3 * (r['e'] / copies - base['e']):+11.4f} meV  "
                  f"max|dF| {df:.2e} eV/A  largest weight {r['p0']:.4f}  ({r['seconds']:.0f} s){note}", flush=True)
