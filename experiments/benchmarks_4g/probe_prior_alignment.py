"""Frozen check 2 of the Au2 two-site (Hubbard) prior (pre-registered 2026-11-01 13:54, benchmarks README): does the
prior's own update direction move the wetting energies toward DFT?

L = sum_R [(Delta_ion - U)^2 + (|H_cov,ion| - sqrt2 t)^2] (eV^2) on the gas-phase dimer at R = 2.30 / 2.47 / 2.60 /
2.80 A (TDA-PBE U, t; spin_pairing off).  Its update direction in the run's optimizer geometry: Muon's polar factor
(exact, by SVD; the run's Newton-Schulz quintic approximates it) times sqrt(max(1, rows / cols)) on the hidden matrices,
AdamW's g / (sqrt(v_hat) + eps) with the checkpoint's second moments elsewhere, fresh momenta, no weight decay; and the
Euclidean -grad L.  Reported: the first-order changes of wet_u, wet_d (E[wetting] - E[non-wetting], the four
unrelaxed T5 references) and T5 along each direction, scaled to the step that lowers L by 10 %, and the modules that
carry wet_u's change.  First order on frozen weights only.
Usage (repo root): python experiments/benchmarks_4g/probe_prior_alignment.py MODEL.pt [...]"""
import math, os, sys, warnings
from collections import defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase import Atoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from flashace.optim import get_muon_param_groups  # noqa: E402
from eval_aumgo import references  # noqa: E402
UT = {2.30: (0.900, 1.409), 2.47: (0.995, 1.249), 2.60: (1.063, 1.146), 2.80: (1.164, 1.009)}
DFT = {"wet_u": 928.9, "wet_d": -2.7, "T5": -931.6}
CAP = {}


def patch(calc):
    orig = calc.model.forward

    def fwd(data, **kw):
        kw["return_electrostatics"] = True                  # keeps the energy's graph through the force derivative
        out = orig(data, **kw)
        CAP["E"] = out[0]
        return out
    calc.model.forward = fwd


def energy(calc, atoms, charge=0.0):
    a = atoms.copy(); a.info = {"charge": float(charge)}; a.calc = calc; calc.reset(); a.get_potential_energy()
    return CAP["E"].reshape(())


def grads(y, params):
    g = torch.autograd.grad(y, params, allow_unused=True)
    return [torch.zeros_like(p) if x is None else x.detach() for p, x in zip(params, g)]


def main():
    refs = references()
    for model in sys.argv[1:]:
        run = os.path.dirname(model)
        cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
        calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(3)
        patch(calc)
        mod = calc.model.electrostatics
        if os.environ.get("SP"):
            # SPEC 29 on (frozen weights): the neutral pair's coupling carries sqrt 2, the ion resonances 1; the prior's
            # target on the Hamiltonian element is unchanged (|H_cov,ion| = sqrt 2 tau = sqrt 2 t)
            import dataclasses
            mod.redox = dataclasses.replace(mod.redox, spin_pairing=True)
            print("  [spin_pairing on]", flush=True)
        named = [(n, p) for n, p in calc.model.named_parameters() if p.requires_grad]
        names, params = [n for n, _ in named], [p for _, p in named]
        # the run's optimizer groups, checked against the checkpoint's state (sizes and shapes)
        groups = get_muon_param_groups(calc.model, learning_rate=1e-4, weight_decay=float(cfg.get("weight_decay", 0.0)),
                                       parameter_mode=str(cfg.get("muon_parameter_mode", "hidden")),
                                       include_patterns=cfg.get("muon_include", ()) or (),
                                       exclude_patterns=cfg.get("muon_exclude", ()) or ())
        ck = torch.load(model, map_location="cpu", weights_only=False)
        osd = ck["optimizer_state_dict"]
        assert [len(g["params"]) for g in groups] == [len(g["params"]) for g in osd["param_groups"]], "group sizes differ"
        kind, v_hat = {}, {}
        missing = 0
        for g, gs in zip(groups, osd["param_groups"]):
            for p, idx in zip(g["params"], gs["params"]):
                st = osd["state"].get(idx)              # no state: the tensor never had a gradient in training
                missing += st is None
                if g.get("use_muon"):
                    assert st is None or tuple(st["momentum_buffer"].shape) == tuple(p.shape)
                    kind[id(p)] = "muon"
                else:
                    assert st is None or tuple(st["exp_avg_sq"].shape) == tuple(p.shape)
                    # AdamW's first step from no state: v_hat = g^2, so the update is g / (|g| + eps)
                    v_hat[id(p)] = (None if st is None else st["exp_avg_sq"] / (1.0 - gs["betas"][1] ** float(st["step"])),
                                    gs["eps"])
                    kind[id(p)] = "adam"
        print(f"=== {os.path.basename(run)[-36:]}: {len(params)} trainable tensors; groups "
              f"{[len(g['params']) for g in groups]} match the checkpoint ({missing} without optimizer state)", flush=True)
        # the prior and its gradient
        L, gL = 0.0, [torch.zeros_like(p) for p in params]
        for R, (U, t) in UT.items():
            energy(calc, Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]], pbc=False))
            last = mod._ensemble_last; H = last["H"]; cfgl = last["configs"].tolist()
            rows = {tuple(x[:2]): i for i, x in enumerate(cfgl)}
            c0, i = rows[(0, 0)], rows[(1, -1)]
            term = (H[i, i] - H[c0, c0] - U) ** 2 + (H[c0, i].abs() - math.sqrt(2.0) * t) ** 2
            L += float(term)
            gL = [a + b for a, b in zip(gL, grads(term, params))]
        # the wetting energies' gradients
        E, gE = {}, {}
        for key, at in refs.items():
            e = energy(calc, at)
            E[key] = float(e); gE[key] = grads(e, params)
        gX = {"wet_u": [a - b for a, b in zip(gE[("undoped", "wetting")], gE[("undoped", "non-wetting")])],
              "wet_d": [a - b for a, b in zip(gE[("doped", "wetting")], gE[("doped", "non-wetting")])]}
        gX["T5"] = [a - b for a, b in zip(gX["wet_d"], gX["wet_u"])]
        X0 = {"wet_u": E[("undoped", "wetting")] - E[("undoped", "non-wetting")],
              "wet_d": E[("doped", "wetting")] - E[("doped", "non-wetting")]}
        X0["T5"] = X0["wet_d"] - X0["wet_u"]

        def direction(geom):
            out = []
            for p, g in zip(params, gL):
                if geom == "euclid" or kind.get(id(p)) is None:
                    out.append(-g)
                elif kind[id(p)] == "muon":
                    G = g.reshape(g.shape[0], -1) if g.ndim == 4 else g
                    Uu, S, Vh = torch.linalg.svd(G, full_matrices=False)
                    P = (Uu @ Vh) * math.sqrt(max(1.0, G.shape[-2] / G.shape[-1])) if float(S.max()) > 0 else torch.zeros_like(G)
                    out.append(-P.reshape_as(g))
                else:
                    v, eps = v_hat[id(p)]
                    out.append(-g / ((g.abs() if v is None else v.sqrt()) + eps))
            return out

        def evaluate():
            Lv = 0.0
            for R, (U, t) in UT.items():
                energy(calc, Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]], pbc=False))
                last = mod._ensemble_last; H = last["H"].detach(); cl = last["configs"].tolist()
                rw = {tuple(x[:2]): i for i, x in enumerate(cl)}
                a, b = rw[(0, 0)], rw[(1, -1)]
                Lv += float((H[b, b] - H[a, a] - U) ** 2 + (H[a, b].abs() - math.sqrt(2.0) * t) ** 2)
            Ev = {key: float(energy(calc, at)) for key, at in refs.items()}
            wu = Ev[("undoped", "wetting")] - Ev[("undoped", "non-wetting")]
            wd = Ev[("doped", "wetting")] - Ev[("doped", "non-wetting")]
            return Lv, {"wet_u": wu, "wet_d": wd, "T5": wd - wu}

        print(f"  prior L = {L:.4f} eV^2; as trained wet_u {X0['wet_u'] * 1e3:+.1f}, wet_d {X0['wet_d'] * 1e3:+.1f}, "
              f"T5 {X0['T5'] * 1e3:+.1f} meV (DFT {DFT['wet_u']:+.1f} / {DFT['wet_d']:+.1f} / {DFT['T5']:+.1f})", flush=True)
        for geom in ("optimizer", "euclid"):
            d = direction(geom)
            dL = sum(float((a * b).sum()) for a, b in zip(gL, d))
            scale = -0.1 * L / dL
            res = {x: scale * sum(float((a * b).sum()) for a, b in zip(gX[x], d)) * 1e3 for x in gX}
            print(f"  [{geom}] step lowering L by 10 %: d wet_u {res['wet_u']:+.2f}, d wet_d {res['wet_d']:+.2f}, "
                  f"d T5 {res['T5']:+.2f} meV", flush=True)
            if os.environ.get("FINITE") and geom == "optimizer":
                # the finite steps along the same direction: is the first order the regime? (artifact check)
                for mult in (1.0, 5.0):
                    with torch.no_grad():
                        for p, x in zip(params, d):
                            p.add_(x, alpha=mult * scale)
                    Lf, Xf = evaluate()
                    with torch.no_grad():
                        for p, x in zip(params, d):
                            p.sub_(x, alpha=mult * scale)
                    print(f"     finite step x{mult:g}: L {L:.3f} -> {Lf:.3f} eV^2 (first order {L * (1 - 0.1 * mult):.3f}); "
                          + ", ".join(f"d {x} {(Xf[x] - X0[x]) * 1e3:+.2f} (first order {mult * res[x]:+.2f})" for x in Xf)
                          + " meV", flush=True)
                L0c, X0c = evaluate()
                assert abs(L0c - L) < 1e-9 and all(abs(X0c[x] - X0[x]) < 1e-9 for x in X0), "weights not restored"
            by = defaultdict(float)
            for n, a, b in zip(names, gX["wet_u"], d):
                by[".".join(n.split(".")[:3])] += scale * float((a * b).sum()) * 1e3
            top = sorted(by.items(), key=lambda kv: -abs(kv[1]))[:6]
            print("     wet_u's change by module: " + "; ".join(f"{k} {v:+.2f}" for k, v in top), flush=True)
        byL = defaultdict(float)
        tot = sum(float((g * g).sum()) for g in gL)
        for n, g in zip(names, gL):
            byL[".".join(n.split(".")[:3])] += float((g * g).sum()) / tot
        print("  |grad L|^2 by module: " + "; ".join(f"{k} {v:.2f}" for k, v in sorted(byL.items(), key=lambda kv: -kv[1])[:6]),
              flush=True)


if __name__ == "__main__":
    main()
