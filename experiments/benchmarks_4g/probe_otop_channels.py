"""The model's two charge-transfer channels at the upright Au2's O-top contact (2026-10-27; a probe, frozen weights).

Usage (repo root): python experiments/benchmarks_4g/probe_otop_channels.py MODEL.pt [...]

PBE's relaxation at the contact is -1.98 eV (28 ions) and the model's -0.22 / -0.14 eV (probe_t5_first_order.py,
step 17).  The model can transfer charge there by two channels: O -> Au (the Au nearest the slab takes an electron
from an O: Au- + O-) and Au -> surface (Au+ + an electron on a cation or in the band).  For the undoped upright
reference this prints each channel's lowest configuration: its energy above the frame configuration (the solver's
exact diagonal), its population, the coupling |T| to the frame along the one-electron move, and the second-order
estimate -T^2 / dE (a check on scale only; the ensemble is exact).
"""
import dataclasses
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import probe_t5_first_order as Q  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402


def main():
    Q._install()
    refs = Q.references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        print(f"=== {os.path.basename(os.path.dirname(model))}", flush=True)
        for geom in ("non-wetting", "wetting"):
            at = refs[("undoped", geom)]
            s = Q.evaluate(calc, at, 0.0)
            z, n = s["z"], int(s["z"].numel())
            fr = formal_frame(z)
            configs, H = s["configs"], s["H"]
            E = torch.diagonal(H)
            p = torch.diagonal(density_matrix(H, s["kT"]))
            is_frame = [(configs[c, :n] == fr).all() and (configs[c, n:] == 0).all() for c in range(len(configs))]
            f0 = [c for c, ok in enumerate(is_frame) if ok][0]
            au = Q.AU[int(torch.argmax(torch.as_tensor(at.positions[Q.AU, 2])))] if geom == "non-wetting" else Q.AU[0]
            pos = torch.as_tensor(at.positions)
            ox = [j for j in range(n) if int(z[j]) == 8]
            d_o = torch.linalg.norm(pos[ox] - pos[au], dim=1)
            o_near = ox[int(torch.argmin(d_o))]
            T = s["args"]["T"]
            print(f"  undoped {geom}: {len(configs)} configurations; Au {au} nearest O {o_near} at {float(d_o.min()):.3f} A; "
                  f"T(Au,O) {float(T[au, o_near]) if T is not None else float('nan'):+.3f} eV", flush=True)
            classes = {}
            for c in range(len(configs)):
                ch = (configs[c, :n] != fr).nonzero().flatten().tolist()
                band = int(configs[c, n:].abs().sum()) if configs.shape[1] > n else 0
                key = tuple(sorted((int(z[j]), int(configs[c, j]) - int(fr[j])) for j in ch)) + (("band", band),)
                classes.setdefault(key, []).append(c)
            rows = sorted(classes.items(), key=lambda kv: min(float(E[c]) for c in kv[1]))
            for key, cs in rows[:10]:
                c = min(cs, key=lambda i: float(E[i]))
                ch = (configs[c, :n] != fr).nonzero().flatten().tolist()
                t = float(H[f0, c])
                dE = float(E[c] - E[f0])
                est = -t * t / dE if dE > 0 and t != 0 else 0.0
                print(f"    {str(key):44s} n {len(cs):4d} p {float(sum(p[i] for i in cs)):.4f} lowest dE {dE:+7.3f} eV "
                      f"at {ch[:4]}; H(frame, it) {t:+.3f}; -t^2/dE {est:+.3f}", flush=True)


if __name__ == "__main__":
    main()
