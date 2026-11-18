"""Which loss term holds SPEC 35's crystal factor g = exp(log_alpha_scale) where it is?  (the single model, 2026-11-12)

For a checkpoint and its run's config.yaml, the gradient with respect to log_alpha_scale of every term of train.py's
optimizer step, with train.py's own loss functions and normalisation.  One step is

    (1 / grad_accum_steps) sum over the window's structures of [energy_weight * energy_loss + forces_weight * force_loss]
    + sum over the priors of weight * prior loss,            then the global norm is clipped and AdamW steps,

so the expected data gradient of a step is the dataset mean of the per-structure terms.  Printed: the per-structure
terms by class (slabs without Al, slabs with Al, bulk cells), split into the energy and the force term; the priors'
gradients; the balance; the per-window noise; and the optimizer's own Adam moments of the parameter when the checkpoint
carries them.  Two checks of the probe itself: the set's energy and force RMSE in train.py's metric (to be compared with
the training log's validation line of the checkpoint's epoch when set=valid), and autograd against a central difference
in log_alpha_scale on the first structure of each class.  Nothing is trained or written to a run directory.

    PYTHONPATH=. python experiments/benchmarks_phonons/probe_alpha_scale_terms.py RUN_DIR [set=train|valid] [n=all]
                 [threads=4] [fd=1] [out=FILE.npz] [model=model.pt]
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
torch.set_num_threads(int(opt.get("threads", 4)))
torch.manual_seed(0)
import train                                                      # noqa: E402
from ase.io import read                                           # noqa: E402
from flashace.calculator import TransformersACECalculator         # noqa: E402
from flashace.optim import build_optimizer                        # noqa: E402

run = sys.argv[1].rstrip("/")
which = opt.get("set", "train")
config = yaml.safe_load(open(run + "/config.yaml"))
settings = train.parse_electrostatics_config(config)
train.configure_default_dtype(settings)
path = run + "/" + opt.get("model", "model.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
calc = TransformersACECalculator(model_path=path, device="cpu")
model = calc.model
model.eval()
s = model.electrostatics.log_alpha_scale
redox = model.electrostatics.redox
accum = int(config.get("grad_accum_steps", 1))
w_e, w_f = float(config["energy_weight"]), float(config.get("forces_weight", 1.0))
print(f"run {run.split('/')[-1]}, checkpoint epoch {checkpoint['epoch']}; log_alpha_scale {float(s):+.6f} (g = {math.exp(float(s)):.6f}); "
      f"scope {redox.frame_polarizability_scope!r}; energy_weight {w_e}, forces_weight {w_f}, grad_accum_steps {accum}, "
      f"clip_grad_norm {config.get('clip_grad_norm')}, threads {torch.get_num_threads()}", flush=True)

# ---- the optimizer's own record of this parameter (Adam moments), if the checkpoint carries it
try:
    optimizer = build_optimizer(model, config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    group = next(g for g in optimizer.param_groups if any(p is s for p in g["params"]))
    state = optimizer.state[s]
    b1, b2 = group["betas"]
    t = int(state["step"])
    m, v = float(state["exp_avg"]), float(state["exp_avg_sq"])
    m_hat, v_hat = m / (1.0 - b1 ** t), v / (1.0 - b2 ** t)
    print(f"== optimizer state of log_alpha_scale: group {group['name']!r}, lr {group['lr']:.3e}, weight decay {group['weight_decay']:.1e}, "
          f"betas {group['betas']}, eps {group['eps']:.1e}, step {t}")
    print(f"   Adam first moment m_hat {m_hat:+.4e} (mean clipped gradient, memory ~{1 / (1 - b1):.0f} steps), sqrt(v_hat) {math.sqrt(v_hat):.4e} "
          f"(its rms, memory ~{1 / (1 - b2):.0f} steps); m_hat / sqrt(v_hat) = {m_hat / math.sqrt(v_hat):+.4f} "
          f"-> the next step changes log g by {-group['lr'] * m_hat / (math.sqrt(v_hat) + group['eps']):+.3e} (lr = {group['lr']:.3e}); "
          f"weight decay adds {-group['lr'] * group['weight_decay'] * float(s):+.3e}")
except Exception as error:                                         # the record is a bonus; the measurement does not need it
    print(f"== optimizer state not read: {type(error).__name__}: {error}")

# ---- the priors, exactly as _optimizer_step adds them
print("== the priors of one optimizer step: weight x d loss / d log g", flush=True)
prior_grad = {}
for kind in ("dielectric", "polarizability"):
    block = getattr(settings, kind + "_prior", None)
    if not block:
        continue
    items = train._response_prior_items(block, kind, config["r_max"], torch.float64, settings, False)
    target = float(items[0][train.RESPONSE_PRIORS[kind][1]][0, 0])
    loss, errors = train._response_prior_loss(model, items, kind, "cpu", settings, 1.0, training=False)
    (g,) = torch.autograd.grad(float(block["weight"]) * loss, s, allow_unused=True)
    prior_grad[kind] = 0.0 if g is None else float(g)
    print(f"   {kind:14s} prior (weight {block['weight']}): model {float(errors[0][0, 0]) + target:.4f} against the label {target:.4f}; "
          f"loss {float(loss):.6f}; gradient {prior_grad[kind]:+.6e}", flush=True)
P = sum(prior_grad.values())

# ---- the data terms, structure by structure, through the trainer's batched forward and loss functions
frames = read(config["train_file" if which == "train" else "valid_file"], index=":")
n_use = len(frames) if opt.get("n", "all") == "all" else int(opt["n"])
pick = list(range(len(frames))) if n_use >= len(frames) else sorted(np.random.default_rng(20261112).choice(len(frames), n_use, replace=False).tolist())
dataset = train.AtomisticDataset([frames[i] for i in pick], config["r_max"], dtype=torch.float64, electrostatics=True)
reference = {int(k): float(e) for k, e in checkpoint["config"]["atomic_energies"].items()}


def kind_of(numbers):
    return "slab+Al" if 13 in numbers else ("slab" if 79 in numbers else "bulk")


def terms(item, graph=True):
    """(weighted energy loss, weighted force loss, energy error per atom, force MSE, has_energy, force_weight) of one structure."""
    bat = train.AtomisticDataset.collate_batched([item])
    E, F, _, _ = model.forward_batched(bat, training=graph, temperature_scale=1.0, compute_stress=False)
    n_at = bat["n_atoms"].to(E.dtype)
    base = sum(reference[int(z)] for z in item["z"])
    target = bat["t_E"].reshape(-1) - base
    has_e = bat["has_energy"].to(E.dtype)
    loss_e = train.energy_loss(E, target, n_at, has_e)
    per = torch.zeros_like(E).index_add(0, bat["batch"], ((F - bat["t_F"]) ** 2).sum(dim=1))
    fw = bat["force_weight"].to(E.dtype)
    loss_f = train.force_loss(per, n_at, fw)
    return w_e * loss_e, w_f * loss_f, float((E - target) / n_at), float(per / (3.0 * n_at)), float(has_e), float(fw)


rows, classes, t0, fd_done = [], [], time.time(), set()
n_fd = int(opt.get("fd", 1))
for k, i in enumerate(pick):
    item = dataset[k]
    cls = kind_of(set(int(z) for z in item["z"]))
    le, lf, err_e, mse_f, has_e, fw = terms(item)
    (ge,) = torch.autograd.grad(le, s, retain_graph=True, allow_unused=True)
    (gf,) = torch.autograd.grad(lf, s, allow_unused=True)
    ge, gf = (0.0 if ge is None else float(ge)), (0.0 if gf is None else float(gf))
    rows.append((i, len(item["z"]), has_e, fw, err_e, mse_f, float(le), float(lf), ge, gf))
    classes.append(cls)
    if n_fd and cls not in fd_done:                                # autograd against a central difference in log g
        fd_done.add(cls)
        keep, h, values = s.detach().clone(), 1e-4, []
        for sign in (+1.0, -1.0):
            with torch.no_grad():
                s.copy_(keep + sign * h)
            a, b = terms(item)[:2]
            values.append((float(a), float(b)))
        with torch.no_grad():
            s.copy_(keep)
        fe, ff = (values[0][0] - values[1][0]) / (2 * h), (values[0][1] - values[1][1]) / (2 * h)
        print(f"   check, structure {i} ({cls}, {len(item['z'])} atoms): energy term autograd {ge:+.6e} against central difference {fe:+.6e}; "
              f"force term {gf:+.6e} against {ff:+.6e}", flush=True)
    if k < 3 or (k + 1) % 25 == 0 or k + 1 == len(pick):
        shown = f"{1000 * err_e:+.2f} meV/atom" if has_e else "not labelled"
        print(f"   [{k + 1}/{len(pick)}] {time.time() - t0:.0f} s; structure {i} ({cls}): energy error {shown}, "
              f"force rms {math.sqrt(mse_f):.4f} eV/A, d/dlog g energy {ge:+.3e}, force {gf:+.3e}", flush=True)

r = np.array(rows)
classes = np.array(classes)
if "out" in opt:
    np.savez(opt["out"], rows=r, classes=classes, prior_dielectric=prior_grad.get("dielectric", 0.0),
             prior_polarizability=prior_grad.get("polarizability", 0.0), log_alpha_scale=float(s),
             columns="index n_atoms has_energy force_weight energy_error_per_atom force_mse loss_e loss_f grad_e grad_f")
n, has, err, mse, ge, gf = r[:, 1], r[:, 2], r[:, 4], r[:, 5], r[:, 8], r[:, 9]
total = ge + gf
N = len(r)
w = n * has
mean_e = (err * w).sum() / max(w.sum(), 1.0)
print(f"== {which} set, {N} structures, train.py's metrics: E RMSE {1000 * math.sqrt((err ** 2 * w).sum() / max(w.sum(), 1.0)):.2f} meV/atom "
      f"(offset-free {1000 * math.sqrt((w * (err - mean_e) ** 2).sum() / max(w.sum(), 1.0)):.2f}, mean {1000 * mean_e:+.2f}); "
      f"force RMSE {math.sqrt(mse.mean()):.6f} eV/A")
print("== the data terms of one optimizer step, d/d log g: the dataset mean of the per-structure terms (class sums / all structures)")
print(f"   {'class':8s} {'n':>4s} {'energy term':>13s} {'force term':>13s} {'both':>13s} | per structure: mean, std of (energy + force) | force rms eV/A | energy rms meV/atom")
for cls in ("slab", "slab+Al", "bulk"):
    sel = classes == cls
    if not sel.any():
        continue
    e_rms = 1000 * math.sqrt((err[sel] ** 2 * has[sel]).sum() / max(has[sel].sum(), 1.0))
    print(f"   {cls:8s} {int(sel.sum()):4d} {ge[sel].sum() / N:+13.4e} {gf[sel].sum() / N:+13.4e} {total[sel].sum() / N:+13.4e} | "
          f"{total[sel].mean():+.4e}, {total[sel].std():.4e} | {math.sqrt(mse[sel].mean()):.4f} | {e_rms:.2f}")
D = total.mean()
sem = total.std(ddof=1) / math.sqrt(N)
print(f"   {'all':8s} {N:4d} {ge.mean():+13.4e} {gf.mean():+13.4e} {D:+13.4e} | std over structures {total.std():.4e}; standard error of the mean {sem:.2e}")
window = total.std() / math.sqrt(accum)
print("== the balance of one optimizer step (before clipping)")
print(f"   data, expected (D)                : {D:+.4e}  (+ wants g down, - wants g up)")
for kind, g in prior_grad.items():
    print(f"   {kind:14s} prior          : {g:+.4e}")
print(f"   sum (D + priors)                  : {D + P:+.4e}   = {(D + P) / abs(prior_grad.get('dielectric', float('nan'))):+.3f} of |dielectric prior|")
print(f"   D / |dielectric prior|            : {D / abs(prior_grad.get('dielectric', float('nan'))):+.3f}")
print(f"   per-window noise of the data term : {window:.4e} (std over structures / sqrt({accum})) = {window / abs(prior_grad.get('dielectric', float('nan'))):.2f} of |dielectric prior|")
order = np.argsort(-np.abs(total))[:8]
print("   largest per-structure terms: " + "; ".join(f"[{int(r[j, 0])}] {classes[j]} {total[j]:+.2e} (E {ge[j]:+.1e}, F {gf[j]:+.1e})" for j in order))
