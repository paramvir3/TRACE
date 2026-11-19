# Diagnostic probe (2026-10-09; reported, not a criterion): the Al -> Au transfer margin of a FORMAL checkpoint.
# Usage (repo root): python experiments/benchmarks_4g/probe_transfer_margin.py RUN/model.pt [RUN2/model.pt ...]
"""Is the electron-transfer configuration of Au2/MgO:Al approaching the ground configuration during training?

For each of the first 6 doped test structures, through the model's own forward (anchor='formal',
fragment_ensemble='diabatic'), with E(n) = sum_u e_hat_u(n_u) + b.n + 1/2 n.W.n exactly as the module:

- E*: the lowest configuration found by the module's search;
- margin = min E(Au2 net charge <= -1) - min E(Au2 net charge >= 0) over the searched configurations
  (negative: transfer is the ground state).  If one class has no configuration inside the window, the
  margin is only bounded: reported as '< -window' or '> +window';
- margin_move (an upper bound when the ground state has no transfer): the best direct one-electron move
  from any donor unit to an Au, minus E*;
- p(Au-) at kT = 0.1 and 0.5 eV: Boltzmann weight of the Au-charged configurations among those searched;
- the learned shifts delta_u(s) = e_hat_u(s) - E_Z(s) + 1/2 S0_u s^2 (bounded by chi_bound), means per element.

Nothing here is trained or fitted; it reads the checkpoint as deployed (kT of the checkpoint for the search).

Screening sensitivity (CHECKPOINT:SCALE): the same search with the bound response scaled,
    W(s) = J_pair + diag(S0) + s (W - J_pair - diag(S0)),
J_pair the Ewald pair part at the free widths (trace_electrostatics.lattice.free_charge_lattice_blocks);
in linear response eps_inf - 1 is proportional to s, so s = (3 - 1) / (eps_model - 1) puts MgO at eps_inf = 3.
b (the bound sector's own charges) is kept.  W - J_pair - diag(S0) must be negative semidefinite (it is
-A K^-1 A^T); its extreme eigenvalues are printed as the check of the decomposition.
"""
import math
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from ase.io import read  # noqa: E402

import trace_redox.module as module_code  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics.kernels import gram_self_energy  # noqa: E402
from trace_electrostatics.ewald import EwaldParams  # noqa: E402
from trace_electrostatics.lattice import free_charge_lattice_blocks  # noqa: E402
from trace_redox.diabatic import configuration_energies, formal_ladder, search_configurations  # noqa: E402


def scaled_operator(atoms, sigma, S0, W, scale):
    """W(s) and the extreme eigenvalues of the screening part W - J_pair - diag(S0)."""
    from scipy.special import erfcinv
    cell = torch.tensor(atoms.cell.array)
    r_cut = max(10.0, 1.01 * math.sqrt(2.0) * float(sigma.max()) * float(erfcinv(1e-9)))
    ewald = EwaldParams.auto_tune(cell, r_cut, accuracy=1e-10)
    ones = torch.ones(len(atoms))
    J = free_charge_lattice_blocks(torch.tensor(atoms.positions), cell, sigma, sigma, sigma, ones, ewald)["j"]
    bare = J + torch.diag(S0)
    D = W - bare
    D = 0.5 * (D + D.T)
    lam = torch.linalg.eigvalsh(D)
    return bare + scale * D, (float(lam[0]), float(lam[-1]))


def margins(configs, E, au):
    charged = configs[:, au].sum(dim=1) <= -1
    if bool(charged.any()) and bool((~charged).any()):
        m = float(E[charged].min()) - float(E[~charged].min())
    else:
        m = -math.inf if bool(charged.any()) else math.inf
    E_star = float(E.min())
    p = {kT: float(torch.softmax(-(E - E_star) / kT, dim=0)[charged].sum()) for kT in (0.1, 0.5)}
    return m, p


def probe(checkpoint, structures, scale=None):
    calc = TransformersACECalculator(model_path=checkpoint, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    r = mod.redox
    assert r.anchor == "formal" and r.fragment_ensemble == "diabatic", "a formal diabatic checkpoint is needed"
    captured = {}
    original = module_code.search_configurations

    def capture(e_hat, states, b, W, K, Q, **kw):
        out = original(e_hat, states, b, W, K, Q, **kw)
        captured.update(e_hat=e_hat.detach(), states=states.detach(), b=b.detach(), W=W.detach(), K=K, Q=Q,
                        L=kw.get("min_charge"), configs=out[0], E=out[1].detach(), pairs=kw.get("pairs"),
                        units=kw.get("units"), frames=kw.get("frames"))
        return out

    widths = []
    hook = mod.width.register_forward_hook(lambda m, i, o: widths.append(o.detach()))
    # the exclusion term (SPEC 18) is part of e_hat; capture it so the learned shifts can be separated from it
    exclusions = []
    if getattr(mod, "exclusion", None) is not None:
        mod.exclusion.register_forward_hook(lambda m, i, o: exclusions.append(o.detach()))
    module_code.search_configurations = capture
    rows = []
    try:
        for atoms in structures:
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.calc = calc
            captured.clear()
            widths.clear()
            exclusions.clear()
            a.get_potential_energy()
            z = torch.tensor(a.numbers)
            e_hat, states, b, W = captured["e_hat"], captured["states"], captured["b"], captured["W"]
            configs, E = captured["configs"], captured["E"]
            K, L = captured["K"], captured["L"]
            _, E_Z, _, _ = formal_ladder(z, int(r.max_charge))
            # the free-charge widths the module used (free_width_rule='frame' puts neutral-frame elements on the
            # ladder width, not the learned ionic width that mod.width returns)
            assert getattr(r, "free_charge_model", "single") == "single", "core + carrier self-energy: not handled"
            sigma = mod.free_widths(z).detach().to(widths[-1])
            S0 = gram_self_energy(sigma)
            n_atoms = int(z.numel())
            extra = int(e_hat.shape[0]) - n_atoms                  # a band unit (SPEC 20) is appended after the atoms
            pad = lambda mask: torch.cat([mask, torch.zeros(extra, dtype=torch.bool)]) if extra else mask
            shift = e_hat[:n_atoms] - E_Z.to(e_hat) + 0.5 * S0.unsqueeze(1) * (states * states).unsqueeze(0)
            if exclusions:
                Pi = exclusions[-1].to(e_hat)
                shift = torch.where(torch.isfinite(Pi), shift - Pi, torch.full_like(shift, float("nan")))
            # self-checks of the reconstruction: the energies are the module's; neutral states carry no shift;
            # every finite shift lies inside the tanh bound
            again = configuration_energies(configs, e_hat, states, b, W, pairs=captured["pairs"])
            assert float((again - E).abs().max()) <= 1e-9 * (1.0 + float(E.abs().max()))
            zero = int(torch.nonzero(states == 0).flatten()[0])
            assert float(shift[:, zero].abs().max()) <= 1e-9
            finite = torch.isfinite(shift)
            assert float(shift[finite].abs().max()) <= float(r.chi_bound) + 1e-9
            k0 = int(torch.argmin(E))
            E_star, n_star = float(E[k0]), configs[k0]
            au = pad(z == 79)
            al_mask = pad(z == 13)
            margin_window, p_au = margins(configs, E, au)
            scaled = None
            if scale is not None and not extra:                    # the scaled operator is atom-level only
                W_s, (lam_lo, lam_hi) = scaled_operator(a, sigma, S0, W, scale)
                best_run = None
                for start in (None, n_star):
                    c_s, E_s = search_configurations(e_hat, states, b, W_s, K, int(captured["Q"]),
                                                     window=float(r.diabatic_window),
                                                     cap=int(r.fragment_max_configurations), start=start, min_charge=L,
                                                     units=captured["units"], frames=captured["frames"],
                                                     pairs=captured["pairs"])
                    if best_run is None or float(E_s.min()) < float(best_run[1].min()):
                        best_run = (c_s, E_s)
                c_s, E_s = best_run
                k_s = int(torch.argmin(E_s))
                m_s, p_s = margins(c_s, E_s, au)
                scaled = dict(al=c_s[k_s][al_mask].tolist(), au=c_s[k_s][au].tolist(), margin=m_s, p_au=p_s,
                              lam=(lam_lo, lam_hi))
            # direct moves donor d -> Au a (one electron): n_d + 1 <= K_d, n_a - 1 >= L_a
            best = math.inf
            lower = L if L is not None else torch.full_like(K, -1)
            for a_idx in torch.nonzero(au).flatten().tolist():
                if int(n_star[a_idx]) - 1 < int(lower[a_idx]):
                    continue
                for d in range(len(z)):
                    if d == a_idx or int(n_star[d]) + 1 > int(K[d]):
                        continue
                    n = n_star.clone()
                    n[d] += 1
                    n[a_idx] -= 1
                    e = float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W, pairs=captured["pairs"])[0])
                    best = min(best, e - E_star)
            valid = torch.isfinite(E_Z.to(e_hat))
            means = {}
            for zz, s in ((13, 2), (13, 3), (79, -1), (79, 1), (12, 1), (8, -1)):
                col = int(torch.nonzero(states == s).flatten()[0])
                sel = (z == zz) & valid[:, col]
                if bool(sel.any()):
                    means[(zz, s)] = float(shift[sel, col].mean())
            rows.append(dict(al=n_star[al_mask].tolist(), au=n_star[au].tolist(), n_configs=int(configs.shape[0]),
                             margin_window=margin_window, margin_move=best, p_au=p_au, shifts=means, scaled=scaled))
    finally:
        module_code.search_configurations = original
        hook.remove()
    return rows


def main():
    test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    doped = [a for a in test if a.info["system"].startswith("Al")][:6]
    names = {(13, 2): "Al2+", (13, 3): "Al3+", (79, -1): "Au-", (79, 1): "Au+", (12, 1): "Mg+", (8, -1): "O-"}
    for argument in sys.argv[1:]:
        checkpoint, _, scale = argument.partition(":")
        scale = float(scale) if scale else None
        epoch = torch.load(checkpoint, map_location="cpu").get("epoch")
        print(f"=== {checkpoint} (epoch {epoch}){'' if scale is None else f'; screening scaled by s = {scale}'}")
        rows = probe(checkpoint, doped, scale)
        for k, row in enumerate(rows):
            m = row["margin_window"]
            window = ("< -window" if m < 0 else "> +window") if math.isinf(m) else f"{m:+.3f}"
            print(f"  doped {k}: lowest Al {row['al']} Au {row['au']}; {row['n_configs']} configs; margin {window} eV "
                  f"(direct move from the lowest {row['margin_move']:+.3f}); p(Au-) kT 0.1: {row['p_au'][0.1]:.2e}, "
                  f"kT 0.5: {row['p_au'][0.5]:.2e}")
            sc = row["scaled"]
            if sc is not None:
                m = sc["margin"]
                text = ("< -window" if m < 0 else "> +window") if math.isinf(m) else f"{m:+.3f}"
                print(f"      scaled: lowest Al {sc['al']} Au {sc['au']}; margin {text} eV; p(Au-) kT 0.1: "
                      f"{sc['p_au'][0.1]:.2e}; screening eigenvalues [{sc['lam'][0]:+.3f}, {sc['lam'][1]:+.2e}]")
        keys = sorted({k for row in rows for k in row["shifts"]})
        print("  mean learned shifts (eV, |.| <= chi_bound): " + ", ".join(
            f"{names[k]} {sum(row['shifts'][k] for row in rows) / len(rows):+.3f}" for k in keys))


if __name__ == "__main__":
    main()
