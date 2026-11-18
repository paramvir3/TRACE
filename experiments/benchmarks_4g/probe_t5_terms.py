"""T5 split exactly into the terms of the configuration energy (frozen weights; 2026-10-19; reported).

Usage: python probe_t5_terms.py MODEL.pt [...]        (env T5_STACK=17: apply step 17's redox settings, frozen)

The canonical free energy satisfies F = Tr(rho H) - T S = sum_n p_n E(n) + Tr(rho H_off) - T S identically
(p_n = rho_nn), and the module's energy is E_q = e_b0 + F.  E(n) is a sum of distinct terms, so

    E_q = e_b0 + sum_terms <E_term> + E_coh - T S,      <E_term> = sum_n p_n E_term(n),

exactly, and T5 = (E_dw - E_dn) - (E_uw - E_un) splits the same way.  Terms (trace_redox SPEC 0):
  ladder   sum_u E_Z(n_u)                    exact isolated ions (NIST)
  shifts   sum_u g_u Delta_u(n_u)            learned
  self     -1/2 sum_u S0_u n_u^2             isolated Gaussian self energy
  screened b^T n + 1/2 n^T W n               Coulomb + induction of the bound sector (with SPEC 18's anchors, if on)
  exclusion18  SPEC 18's one-body exclusion energies (legacy; on in steps 9-14)
  overlap  sum O_uv + own images             SPEC 23 penetration + 25 Gordon-Kim
  ind26    SPEC 26 one- and two-body terms
  alpha28  SPEC 28 Woodbury correction
  band     the band unit's one-body energies and pair corrections (SPEC 20)
  coherent Tr(rho H_off): the hops
  entropy  F - Tr(rho H)
  elim_const  the elimination's constant, added outside the ensemble (SPEC 17.9's frame terms; SPEC 34's damped frame
           drive when frame_induction_damping is on; named 2026-11-13, absent when zero)
  frame    the configuration-independent frame terms added outside the ensemble: the frame pair's Pauli wall (Round 31),
           SPEC 33's bond-oriented law and its wall (2026-11-05; T5_BOP=1 / T5_BOP=wall switch them on, frozen)
           and SPEC 39's frame-pair interaction (2026-11-13) when its flag is on
Checked per structure: the ensemble terms sum to H's diagonal (1e-9) and, with the frame terms, to the module's energy (1e-9).
"""
import dataclasses
import os
import sys
import warnings

import torch
import yaml

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import trace_redox.band as BD  # noqa: E402
import trace_redox.exclusion as EX  # noqa: E402
import trace_redox.induction as I  # noqa: E402
import trace_redox.module as M  # noqa: E402
import trace_redox.penetration as P  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics.kernels import gram_self_energy  # noqa: E402
from trace_redox.diabatic import canonical_free_energy, configuration_energies, density_matrix, formal_ladder  # noqa: E402
from eval_aumgo import references  # noqa: E402

STORE = {}


def _wrap(name, f, method=False):
    if method:
        def g(self, *a, **k):
            out = f(self, *a, **k)
            STORE[name] = out
            return out
    else:
        def g(*a, **k):
            out = f(*a, **k)
            STORE[name] = out
            return out
    return g


def _install():
    sol = M.RedoxElectrostatics._diabatic_solution

    def capture(self, e_hat, states, b, W, L, K, Q, T=None, **kwargs):
        STORE.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), pairs=kwargs.get("pairs"),
                     diag=kwargs.get("diag_correction"))
        return sol(self, e_hat, states, b, W, L, K, Q, T=T, **kwargs)

    M.RedoxElectrostatics._diabatic_solution = capture
    M.eliminate_bound_sector = _wrap("elim", M.eliminate_bound_sector)
    # 2026-11-13: SPEC 34 replaces the elimination after it is built; its constant is added outside the ensemble
    M.RedoxElectrostatics._damp_frame_induction = _wrap("elim_damped", M.RedoxElectrostatics._damp_frame_induction, method=True)
    P.penetration_pairs = _wrap("O", P.penetration_pairs)
    P.penetration_self_terms = _wrap("own", P.penetration_self_terms)
    I.induction_terms = _wrap("ind", I.induction_terms)
    BD.BandUnit.augment = _wrap("aug", BD.BandUnit.augment, method=True)
    EX.Exclusion.forward = _wrap("excl", EX.Exclusion.forward, method=True)


def pair_energy(configs, states, U, pairs):
    if pairs is None:
        return torch.zeros(int(configs.shape[0]), dtype=torch.float64)
    zero_e = torch.zeros(U, int(states.numel()), dtype=torch.float64)
    return configuration_energies(configs, zero_e, states, torch.zeros(U, dtype=torch.float64),
                                  torch.zeros(U, U, dtype=torch.float64), pairs=tuple(x.detach() for x in pairs))


def terms(calc, atoms):
    for k in ("O", "own", "ind", "aug", "excl", "elim_damped"):
        STORE.pop(k, None)
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    calc.reset()
    cap = []
    hook = calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    a.get_potential_energy()
    hook.remove()
    mod, r = calc.model.electrostatics, cap[-1]
    last = mod._ensemble_last
    configs, H, kT = last["configs"], last["H"].detach(), float(last["kT"])
    states, e_hat = STORE["states"], STORE["e_hat"]
    U, n = int(e_hat.shape[0]), len(a)
    first = int(round(float(states[0])))
    z = torch.as_tensor(a.numbers)
    _, E_Z, _, _ = formal_ladder(z, int(mod.redox.max_charge))
    pad = int(states.numel()) - int(E_Z.shape[1])
    E_Z = torch.cat([torch.full((n, pad), float("inf")), E_Z], 1)
    shift = r.state_energies.detach()[:n] - E_Z
    sigma = mod.free_widths(z) if mod.width is not None else STORE["elim"].assembly.params.sigma
    S0 = gram_self_energy(sigma.detach().to(torch.float64))
    k = (configs - first).clamp(0, int(states.numel()) - 1)
    rows = torch.arange(n)
    st = states.unsqueeze(0).expand(int(configs.shape[0]), -1)
    out = {}
    out["ladder"] = torch.stack([E_Z[rows, k[c, :n]].sum() for c in range(len(configs))])
    out["shifts"] = torch.stack([torch.nan_to_num(shift[rows, k[c, :n]], posinf=0.0).sum() for c in range(len(configs))])
    nq = configs[:, :n].to(torch.float64)
    out["self"] = -0.5 * (S0.unsqueeze(0) * nq * nq).sum(1)
    out["screened"] = configs.to(torch.float64) @ STORE["b"] + 0.5 * torch.einsum("ci,ij,cj->c", configs.to(torch.float64),
                                                                                 STORE["W"], configs.to(torch.float64))
    one_body_known = torch.zeros(len(configs))
    if "own" in STORE:
        out["overlap"] = torch.stack([STORE["own"][rows, k[c, :n]].sum() for c in range(len(configs))])
        one_body_known = one_body_known + out["overlap"]
    if "ind" in STORE:
        E1 = STORE["ind"][0].detach()
        out["ind26"] = torch.stack([E1[rows, k[c, :n]].sum() for c in range(len(configs))])
        one_body_known = one_body_known + out["ind26"]
    if "excl" in STORE:
        X = STORE["excl"].detach()                                            # SPEC 18 one-body term [n, S]
        out["exclusion18"] = torch.stack([torch.nan_to_num(X[rows, k[c, :n]], posinf=0.0).sum() for c in range(len(configs))])
        one_body_known = one_body_known + out["exclusion18"]
    # every unit's ladder entry as the solve saw it, minus the parts named above, is the band's (and must be 0 on atoms)
    seen_atoms = torch.stack([torch.nan_to_num(e_hat[rows, k[c, :n]], posinf=0.0).sum() for c in range(len(configs))])
    named = out["ladder"] + out["shifts"] + out["self"] + one_body_known
    residue = seen_atoms - named                                           # unnamed one-body terms on atoms: must be 0
    assert float(residue.abs().max()) <= 1e-9 * (1.0 + float(seen_atoms.abs().max())), \
        f"an unnamed one-body term on the atoms: {float(residue.abs().max()):.3e} eV"
    out["band"] = torch.stack([e_hat[n:, k[c, n:]].sum() if U > n else torch.tensor(0.0) for c in range(len(configs))])
    if "O" in STORE:
        out["overlap"] = out.get("overlap", 0.0) + pair_energy(configs, states, U, STORE["O"])
    if "ind" in STORE:
        out["ind26"] = out["ind26"] + pair_energy(configs, states, U, STORE["ind"][1])
    if STORE.get("aug") is not None and STORE["aug"].get("pairs") is not None:
        out["band"] = out["band"] + pair_energy(configs, states, U, STORE["aug"]["pairs"])
    if STORE["diag"] is not None:
        with torch.no_grad():
            out["alpha28"] = STORE["diag"](configs)
    total = sum(out.values())
    diag = torch.diagonal(H)
    assert float((total - diag).abs().max()) <= 1e-9 * (1.0 + float(diag.abs().max())), \
        f"terms do not sum to H's diagonal: {float((total - diag).abs().max()):.3e}"
    rho = density_matrix(H, kT)
    p = torch.diagonal(rho)
    # 2026-11-13: the per-configuration terms, kept for probe_auau_ionic_gap.py (a record only; nothing above changes)
    STORE["per_config"] = {"out": {key: val.detach() for key, val in out.items()}, "configs": configs, "p": p.detach(), "H": H}
    F = float(canonical_free_energy(H, kT))
    TrRhoH = float((rho * H).sum())
    res = {key: float(p @ val) for key, val in out.items()}
    res["coherent"] = TrRhoH - float(p @ diag)
    res["entropy"] = F - TrRhoH
    elim = STORE.get("elim_damped", STORE["elim"])                 # the elimination the module used (SPEC 34's when on)
    res["e_b0"] = float(elim.e_b0)
    const = float(getattr(elim, "constant", 0.0))                  # energy_free = sol.energy + elim.constant (module.py)
    if const != 0.0:
        res["elim_const"] = const
    # the frame terms added outside the ensemble (configuration independent): the frame pair's Pauli wall (Round 31),
    # SPEC 33's bond-oriented law and its wall; read only when their flags are on (the module sets the floats then)
    rc = mod.redox
    frame = 0.0
    for flag, attr in (("frame_wall", "_frame_wall_last"), ("frame_pair", "_frame_pair_last"), ("bond_pauli", "_bond_pauli_last"),
                       ("bond_pauli_wall", "_bond_pauli_wall_last")):
        if getattr(rc, flag, False):
            frame += float(getattr(mod, attr))
    if frame != 0.0:
        res["frame"] = frame
    E_q = sum(res.values())
    assert abs(E_q - float(r.energy)) <= 1e-9 * (1.0 + abs(float(r.energy))), (E_q, float(r.energy))
    return res


def main():
    _install()
    refs = references()
    order = ["e_b0", "ladder", "shifts", "self", "screened", "exclusion18", "overlap", "ind26", "alpha28", "band", "coherent",
             "entropy", "frame"]
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        if os.environ.get("T5_BOP"):                 # frozen check: SPEC 33's law (+ wall with T5_BOP=wall) switched on
            mod.redox = dataclasses.replace(mod.redox, bond_pauli=True, bond_pauli_wall=os.environ["T5_BOP"] == "wall")
        if os.environ.get("T5_STACK") == "17":
            run = os.path.join(HERE, "runs", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s42")
            block = yaml.safe_load(open(os.path.join(run, "config.yaml")))["electrostatics"]["redox"]
            names = {f.name for f in dataclasses.fields(mod.redox)}
            mod.redox = dataclasses.replace(mod.redox, **{key: (tuple(v) if isinstance(v, list) else v)
                                                          for key, v in block.items() if key in names})
        R = {key: terms(calc, at) for key, at in refs.items()}
        keys = [k for k in order if any(k in R[s] for s in R)]
        t5 = {k: 1e3 * ((R[("doped", "wetting")].get(k, 0) - R[("doped", "non-wetting")].get(k, 0))
                        - (R[("undoped", "wetting")].get(k, 0) - R[("undoped", "non-wetting")].get(k, 0))) for k in keys}
        wu = {k: 1e3 * (R[("undoped", "wetting")].get(k, 0) - R[("undoped", "non-wetting")].get(k, 0)) for k in keys}
        wd = {k: 1e3 * (R[("doped", "wetting")].get(k, 0) - R[("doped", "non-wetting")].get(k, 0)) for k in keys}
        label = os.path.basename(os.path.dirname(model)) + (" + step 17 settings (frozen)" if os.environ.get("T5_STACK") == "17" else "")
        print(f"=== {label}   (meV; T5 = wet_d - wet_u; DFT T5 -931.6, wet_d -2.7, wet_u +928.9)")
        print(f"  {'term':9s} {'T5':>9s} {'wet_d':>9s} {'wet_u':>9s}")
        for k in keys:
            print(f"  {k:9s} {t5[k]:+9.1f} {wd[k]:+9.1f} {wu[k]:+9.1f}")
        print(f"  {'sum (E_q)':9s} {sum(t5.values()):+9.1f} {sum(wd.values()):+9.1f} {sum(wu.values()):+9.1f}")


if __name__ == "__main__":
    main()
