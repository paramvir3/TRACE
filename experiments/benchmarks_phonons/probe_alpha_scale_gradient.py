"""Why does SPEC 35's crystal factor g = exp(log_alpha_scale) not move in training?

For a checkpoint of the PBEsol bulk-only family and its run's config.yaml, with train.py's own prior-loss functions:
the gradient of each term of one optimizer step with respect to log_alpha_scale (the dielectric prior, the
polarizability prior, each with its weight; the force loss of training cells, each divided by grad_accum_steps as in
train.py), and a scan of g on the frozen weights: eps of the prior cell, the Au atom's polarizability, both losses.
Nothing is trained or written to a run directory.

    PYTHONPATH=. python experiments/benchmarks_phonons/probe_alpha_scale_gradient.py RUN_DIR [n_cells=16] [threads=2] [scope=ions]
"""
import math
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
import yaml

opt = dict(a.split("=", 1) for a in sys.argv[2:])
torch.set_num_threads(int(opt.get("threads", 2)))
import train                                                      # noqa: E402
from ase.io import read                                           # noqa: E402
from flashace.calculator import TransformersACECalculator         # noqa: E402

run = sys.argv[1].rstrip("/")
config = yaml.safe_load(open(run + "/config.yaml"))
settings = train.parse_electrostatics_config(config)
train.configure_default_dtype(settings)
calc = TransformersACECalculator(model_path=run + "/model_last.pt", device="cpu")
model = calc.model
model.eval()
if "scope" in opt:                                              # SPEC 35.1 on the frozen weights (the checkpoint's own is 'all')
    import dataclasses
    model.electrostatics.redox = dataclasses.replace(model.electrostatics.redox, frame_polarizability_scope=opt["scope"])
print(f"frame_polarizability_scope {model.electrostatics.redox.frame_polarizability_scope!r}")
s = model.electrostatics.log_alpha_scale
print(f"run {run.split('/')[-1]}; log_alpha_scale {float(s):+.6f} (g = {math.exp(float(s)):.6f}); "
      f"prior weights: dielectric {settings.dielectric_prior['weight']}, polarizability {settings.polarizability_prior['weight']}")
items = {k: train._response_prior_items(getattr(settings, k + "_prior"), k, config["r_max"], torch.float64, settings, False)
         for k in ("dielectric", "polarizability")}
targets = {k: float(items[k][0][train.RESPONSE_PRIORS[k][1]][0, 0]) for k in items}


def prior(kind):
    loss, errors = train._response_prior_loss(model, items[kind], kind, "cpu", settings, 1.0, training=False)
    return loss, float(errors[0][0, 0]) + targets[kind]


print("== one optimizer step, the gradient of each term with respect to log_alpha_scale")
total = 0.0
for kind in ("dielectric", "polarizability"):
    w = float(getattr(settings, kind + "_prior")["weight"])
    loss, value = prior(kind)
    (g,) = torch.autograd.grad(w * loss, s, allow_unused=True)
    g = 0.0 if g is None else float(g)
    total += g
    print(f"   {kind:14s} prior: model {value:.4f} against the label {targets[kind]:.4f}; loss {float(loss):.6f}; "
          f"weight x d loss / d log g = {g:+.6f}")
print(f"   the two priors together: {total:+.6f}")

# the force loss of training cells, as train.py accumulates it: |dF|^2 / (3 N) / grad_accum_steps per structure
accum = int(config.get("grad_accum_steps", 1))
frames = read(config["train_file"], index=":")
rng = np.random.default_rng(20261112)
pick = sorted(rng.choice(len(frames), int(opt.get("n_cells", 16)), replace=False).tolist())
dataset = train.AtomisticDataset([frames[i] for i in pick], config["r_max"], dtype=torch.float64, electrostatics=True)
grads = []
for k in range(len(pick)):                                        # one structure through the trainer's batched forward
    bat = train.AtomisticDataset.collate_batched([dataset[k]])
    E, F, _, _ = model.forward_batched(bat, training=True, temperature_scale=1.0, compute_stress=False)
    loss = ((F - bat["t_F"].to(F)) ** 2).sum() / (3.0 * F.shape[0]) / accum * float(config.get("forces_weight", 1.0))
    (g,) = torch.autograd.grad(loss, s, allow_unused=True)
    grads.append(0.0 if g is None else float(g))
grads = np.array(grads)
print(f"   force loss, {len(grads)} training cells (each / {accum}): mean {grads.mean():+.3e}, std {grads.std():.3e}, "
      f"largest |.| {np.abs(grads).max():.3e}; a window of {accum}: mean {accum * grads.mean():+.3e}, std {math.sqrt(accum) * grads.std():.3e}")

print("== frozen weights, g scanned (log_alpha_scale set by hand; nothing else changes)")
keep = s.detach().clone()
for g_value in (1.0, math.exp(float(keep)), 0.98, 0.95, 0.92, 0.90, 0.85):
    with torch.no_grad():
        s.fill_(math.log(g_value))
    le, eps = prior("dielectric")
    la, alpha = prior("polarizability")
    we, wa = float(settings.dielectric_prior["weight"]), float(settings.polarizability_prior["weight"])
    print(f"   g = {g_value:.4f}: eps(prior cell) {eps:.4f}; alpha(Au atom) {alpha:.4f} A^3; weighted losses: dielectric "
          f"{we * float(le):.6f}, polarizability {wa * float(la):.6f}, sum {we * float(le) + wa * float(la):.6f}")
with torch.no_grad():
    s.copy_(keep)
