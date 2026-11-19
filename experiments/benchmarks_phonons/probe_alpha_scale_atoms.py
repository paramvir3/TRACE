"""Which atoms of the undoped slabs pull SPEC 35's crystal factor g up?  (README, pre-registered 2026-11-13; frozen
weights, a probe: nothing trained, nothing written into a run directory.)

The force term of one structure in train.py's step is  w |F - F_ref|^2 / (3 N)  summed over atoms (train.force_loss),
so its derivative in log g = log_alpha_scale splits by atom exactly:

    c_i = forces_weight * force_weight * 2 (F_i - F_i^ref) . (dF_i / dlog g) / (3 N),

with dF / dlog g by a central difference in log g (h = 1e-4).  Checked per structure against the autograd value stored
by probe_alpha_scale_terms.py (column grad_f of its npz).  Atom groups: the two Au; the substrate layer facing the
dimer (O and Mg apart); layers 2-5; the far, bare surface layer (O and Mg apart).

    PYTHONPATH=. python experiments/benchmarks_phonons/probe_alpha_scale_atoms.py RUN_DIR [n=all] [threads=2]
                 [ref=TERMS.npz] [out=FILE.npz] [model=model.pt] [h=1e-4] [only=I,J,...] [atoms=PER_ATOM.npz]
"""
import math
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
import yaml

opt = dict(a.split("=", 1) for a in sys.argv[2:])
torch.set_num_threads(int(opt.get("threads", 2)))
torch.manual_seed(0)
import train                                                      # noqa: E402
from ase.io import read                                           # noqa: E402
from flashace.calculator import TransformersACECalculator         # noqa: E402

run = sys.argv[1].rstrip("/")
config = yaml.safe_load(open(run + "/config.yaml"))
settings = train.parse_electrostatics_config(config)
train.configure_default_dtype(settings)
path = run + "/" + opt.get("model", "model.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
calc = TransformersACECalculator(model_path=path, device="cpu")
torch.set_num_threads(int(opt.get("threads", 2)))
model = calc.model
model.eval()
s = model.electrostatics.log_alpha_scale
w_f = float(config.get("forces_weight", 1.0))
H = float(opt.get("h", 1e-4))
print(f"run {run.split('/')[-1]}, checkpoint epoch {checkpoint['epoch']}; g = {math.exp(float(s)):.6f}; forces_weight {w_f}; "
      f"step in log g {H}; threads {torch.get_num_threads()}", flush=True)

frames = read(config["train_file"], index=":")
slabs = [i for i, a in enumerate(frames) if 79 in a.numbers and 13 not in a.numbers]
n_use = len(slabs) if opt.get("n", "all") == "all" else int(opt["n"])
pick = slabs if n_use >= len(slabs) else sorted(np.random.default_rng(20261113).choice(slabs, n_use, replace=False).tolist())
if "only" in opt:                                                # the check's follow-up: named structures, any step h
    pick = [int(x) for x in opt["only"].split(",")]
dataset = train.AtomisticDataset([frames[i] for i in pick], config["r_max"], dtype=torch.float64, electrostatics=True)
stored = None
if "ref" in opt:
    z = np.load(opt["ref"], allow_pickle=True)
    stored = {int(r[0]): float(r[9]) for r in z["rows"]}           # column grad_f, by index in the training file

GROUPS = ["Au", "L1 O (facing)", "L1 Mg (facing)", "L2", "L3", "L4", "L5", "L6 O (bare)", "L6 Mg (bare)"]


LAYER = {}                                                        # the last structure's layer of every substrate atom, and its side


def groups(a):
    """Group label of every atom; the layers counted from the one facing the dimer (minimum image along z)."""
    au = np.where(a.numbers == 79)[0]
    sub = np.where(a.numbers != 79)[0]
    L = float(a.cell[2, 2])
    zc = float(a.positions[au, 2].mean())
    dz = (a.positions[sub, 2] - zc + 0.5 * L) % L - 0.5 * L
    side = 1.0 if dz[int(np.argmin(np.abs(dz)))] > 0 else -1.0
    depth = (side * dz) % L                                          # the facing layer first, the far surface last
    order = np.argsort(depth)
    cuts = np.where(np.diff(depth[order]) > 1.0)[0] + 1
    layers = np.split(order, cuts)
    assert len(layers) == 6 and all(len(x) == 18 for x in layers), [len(x) for x in layers]
    LAYER.clear(); LAYER.update({int(j): k for k, idx in enumerate(layers, start=1) for j in sub[idx]}); LAYER["side"] = side
    out = np.empty(len(a), dtype=object)
    out[au] = "Au"
    for k, idx in enumerate(layers, start=1):
        for j in sub[idx]:
            el = "O" if a.numbers[j] == 8 else "Mg"
            out[j] = f"L1 {el} (facing)" if k == 1 else (f"L6 {el} (bare)" if k == 6 else f"L{k}")
    return out, float(depth[order][:18].mean())


def forces(item):
    bat = train.AtomisticDataset.collate_batched([item])
    _, F, _, _ = model.forward_batched(bat, training=False, temperature_scale=1.0, compute_stress=False)
    return F.detach(), bat


DUMP = []
keep = s.detach().clone()
tot = {g: 0.0 for g in GROUPS}
tot_z = {g: 0.0 for g in GROUPS}
count = {g: 0 for g in GROUPS}
err2 = {g: 0.0 for g in GROUPS}
resp2 = {g: 0.0 for g in GROUPS}
per_structure, worst, t0 = [], 0.0, time.time()
for k, i in enumerate(pick):
    item, a = dataset[k], frames[i]
    assert [int(z) for z in item["z"]] == a.numbers.tolist()
    lab, gap = groups(a)
    F0, bat = forces(item)
    out = []
    for sign in (+1.0, -1.0):
        with torch.no_grad():
            s.copy_(keep + sign * H)
        out.append(forces(item)[0])
    with torch.no_grad():
        s.copy_(keep)
    dF = ((out[0] - out[1]) / (2 * H)).numpy()
    res = (F0 - bat["t_F"]).numpy()
    N, fw = len(a), float(bat["force_weight"])
    c_xyz = w_f * fw * 2.0 * res * dF / (3.0 * N)                    # [n, 3]
    c = c_xyz.sum(1)
    for g in GROUPS:
        m = lab == g
        tot[g] += float(c[m].sum()); tot_z[g] += float(c_xyz[m, 2].sum()); count[g] += int(m.sum())
        err2[g] += float((res[m] ** 2).sum()); resp2[g] += float((dF[m] ** 2).sum())
    if "atoms" in opt:                                             # per-atom record for analyse_surface_normal_force.py
        for j in range(N):
            DUMP.append([i, j, int(a.numbers[j]), LAYER.get(j, 0), LAYER["side"], *a.positions[j], *res[j], *dF[j], fw, N])
    total = float(c.sum())
    ref = stored.get(i) if stored else None
    rel = abs(total - ref) / max(abs(ref), 1e-30) if ref is not None else float("nan")
    worst = max(worst, rel) if ref is not None else worst
    per_structure.append((i, total, ref if ref is not None else float("nan"), gap))
    if k < 3 or (k + 1) % 25 == 0 or k + 1 == len(pick) or "only" in opt:
        print(f"   [{k + 1}/{len(pick)}] {time.time() - t0:.0f} s; structure {i}: sum over atoms {total:+.6e}"
              + (f", stored autograd {ref:+.6e} (relative gap {rel:.1e})" if ref is not None else "")
              + f"; facing layer {gap:.2f} A from the dimer's centre", flush=True)

M = len(pick)
grand = sum(tot.values())
print(f"== {M} undoped training slabs; the force term's d/d log g, mean per structure {grand / M:+.4e} "
      f"(as a share of a 436-structure step: {grand / 436:+.4e}); check against the stored values: largest relative gap {worst:.1e} "
      f"({'pass' if worst <= 1e-4 else 'FAIL'})")
print(f"   {'group':16s} {'atoms':>6s} {'sum / structure':>16s} {'share':>8s} {'per atom':>12s} {'z part':>8s} | rms force error (eV/A) | rms dF/dlog g (eV/A)")
for g in GROUPS:
    n = count[g] / M
    print(f"   {g:16s} {n:6.0f} {tot[g] / M:+16.4e} {tot[g] / grand:+8.1%} {tot[g] / count[g]:+12.3e} {tot_z[g] / tot[g] if tot[g] else float('nan'):+8.0%} | "
          f"{math.sqrt(err2[g] / (3 * count[g])):.4f} | {math.sqrt(resp2[g] / (3 * count[g])):.4f}")
surf = tot["L1 O (facing)"] + tot["L1 Mg (facing)"] + tot["L6 O (bare)"] + tot["L6 Mg (bare)"]
contact = tot["Au"] + tot["L1 O (facing)"] + tot["L1 Mg (facing)"]
inner = tot["L2"] + tot["L3"] + tot["L4"] + tot["L5"]
facing_pa = (tot["L1 O (facing)"] + tot["L1 Mg (facing)"]) / (count["L1 O (facing)"] + count["L1 Mg (facing)"])
bare_pa = (tot["L6 O (bare)"] + tot["L6 Mg (bare)"]) / (count["L6 O (bare)"] + count["L6 Mg (bare)"])
print(f"== shares: the two surface layers {surf / grand:+.1%}; Au + the facing layer {contact / grand:+.1%}; layers 2-5 {inner / grand:+.1%}")
print(f"   per atom, bare far surface / facing layer: {bare_pa:+.3e} / {facing_pa:+.3e} = {bare_pa / facing_pa:+.3f}; "
      f"O alone: {tot['L6 O (bare)'] / count['L6 O (bare)']:+.3e} / {tot['L1 O (facing)'] / count['L1 O (facing)']:+.3e}")
if "out" in opt:
    np.savez(opt["out"], groups=np.array(GROUPS), total=np.array([tot[g] for g in GROUPS]), z_part=np.array([tot_z[g] for g in GROUPS]),
             count=np.array([count[g] for g in GROUPS]), per_structure=np.array(per_structure), log_alpha_scale=float(keep))
if "atoms" in opt:
    np.savez(opt["atoms"], rows=np.array(DUMP, dtype=float), forces_weight=w_f, log_alpha_scale=float(keep),
             columns="structure atom Z layer side x y z res_x res_y res_z dF_x dF_y dF_z force_weight n_atoms")
print(time.strftime("done %Y-%m-%d %H:%M %Z"))
