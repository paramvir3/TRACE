"""Step 34 stage A (2026-11-15): the eps prior cell's eps_inf (train.py's dielectric prior, eval mode, as the epoch
line prints it) and the band's reference filling N0 for stage A's epoch-1 models, with the band unit's hopping
factor f set to 0.55 (SPEC 20's PBE-mass anchor), step 33's learned values, and left as trained.  Frozen weights:
a diagnostic of the starting point, not a prediction of a run.

    python experiments/benchmarks_4g/probe_step34_eps_band_factor.py      (repo root, the Mac)
"""
import math, os, sys, warnings
warnings.simplefilter("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import torch, yaml
torch.set_num_threads(4)
import train
from flashace.calculator import TransformersACECalculator
RUNS = os.path.join(REPO, "experiments", "benchmarks_4g", "runs")
for seed in (42, 43):
    run = os.path.join(RUNS, f"AuMgO_twosite_bondlaw_r5_s{seed}")
    config = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    settings = train.parse_electrostatics_config(config)
    train.configure_default_dtype(settings)
    model = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu").model
    model.eval()
    band = model.electrostatics.band
    block = settings.dielectric_prior
    items = train._response_prior_items(block, "dielectric", config["r_max"], torch.float64, settings, False)
    target = items[0][train.RESPONSE_PRIORS["dielectric"][1]]
    trained = float(band.log_factor)
    for label, f in (("as trained", None), ("0.55 (SPEC 20)", 0.55), ("0.64 (step 33 s43)", 0.6397), ("0.73 (step 33 s42)", 0.7319)):
        with torch.no_grad():
            band.log_factor.fill_(trained if f is None else math.atanh(math.log(f) / math.log(20.0)))
        fac = math.exp(math.log(20.0) * math.tanh(float(band.log_factor)))
        try:
            _, errors = train._response_prior_loss(model, items, "dielectric", "cpu", settings, 1.0, training=False)
            e = errors[0]
            eps = float((e + torch.diagonal(target.to(e))).mean() if e.dim() == 1 else torch.diagonal(e + target.to(e)).mean())
            N0 = (model.electrostatics._band_last or {}).get("reference")
            print(f"s{seed} f {fac:.4f} [{label}]: eps_inf (mean diagonal) {eps:.4f} (label {float(torch.diagonal(target).mean()):.4f}); band N0 {N0}", flush=True)
        except Exception as err:                                             # noqa: BLE001
            print(f"s{seed} f {fac:.4f} [{label}]: {type(err).__name__}: {str(err)[:140]}", flush=True)
    with torch.no_grad():
        band.log_factor.fill_(trained)
