"""Extensivity, L3 (2026-11-10): the band unit's one-carrier spectrum on the doped benchmark cells, and what a
closed-shell determinant of its lowest orbitals would give per cell.  Frozen weights; nothing in the package is changed
(``BandUnit.augment`` and ``percolation_partitions`` are wrapped in this process only).

    python probe_ext_band_spectrum.py MODEL.pt [nx ny ...]

Per doping, geometry and replication it prints
  * the reference configuration (who holds the excess electrons), the band unit's charge, the number of excess
    electrons  N_x = sum of frames - Q  (formal anchors: electrons that are in no closed-shell frame state);
  * the one-carrier levels of h = diag(eps) - tau (SPEC 20), their gaps and near-degeneracies;
  * E_det(2m)/copies for m = 1 .. : the restricted frozen-orbital determinant of the m lowest orbitals,
        E_det = 2 tr(P h) + sum_uv (2 P_uu P_vv - P_uv^2) V_uv,   P = sum_k phi_k phi_k^T,
    with V the band's own interaction (SPEC 20: W between sites, the exact second-electron cost on a site).  m = 1 is
    SPEC 20's E_b(2) = 2 lambda_1 + J_11 and is checked against the module's value.
It also checks that the diabatic ensemble does not use the percolation partitions (a 2x2 doped cell is refused by their
2^k enumeration).  Writes nothing into the run directory."""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
import trace_redox.module as M
from trace_redox.band import _band_pair_corrections
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(2, len(sys.argv) - 1, 2)] or [(1, 1, 1), (2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space; band factor {float(mod.band.factor):.4f}")
CAP, REC = {}, {}
original_augment = mod.band.augment


def augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=None):
    out = original_augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=pairs)
    CAP.update(e_hat=e_hat.detach(), states=states, W=W.detach(), L=L, frames=frames, pairs=pairs, out=out)
    return out


mod.band.augment = augment
mod.register_forward_hook(lambda module, inputs, output: REC.update(record=output[1]))
original_partitions = M.percolation_partitions
refs = references()


def evaluate(key, rep, partitions=True):
    M.percolation_partitions = original_partitions if partitions else (lambda *a, **k: [])
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); CAP.clear(); REC.clear()
    return a, a.get_potential_energy(), a.get_forces()


def band_interaction():
    """V [B, B] of SPEC 20, rebuilt from the arrays handed to BandUnit.augment (the lines of augment itself)."""
    e_hat, states, W, L, f = CAP["e_hat"], CAP["states"], CAP["W"], CAP["L"], CAP["frames"].to(torch.long)
    sites = CAP["out"]["sites"]; U, S = int(e_hat.shape[0]), int(e_hat.shape[1]); first = int(round(float(states[0])))
    rows = torch.arange(U); col = lambda n: (n - first).clamp(0, S - 1)
    e_f, e_m1 = e_hat[rows, col(f)], e_hat[rows, col(f - 1)]
    e_m2 = torch.where(f - 2 >= L, e_hat[rows, col(f - 2)], torch.full_like(e_f, float("inf")))
    diag = torch.diagonal(W)
    V = W[sites][:, sites].clone()
    onsite = (e_m2 - 2.0 * e_m1 + e_f + diag)[sites]
    V = V - torch.diag(torch.diagonal(V)) + torch.diag(onsite)
    if CAP["pairs"] is not None and int(CAP["pairs"][0].numel()):
        V = V + _band_pair_corrections(CAP["pairs"], f, first, sites, U, S)[1].detach()
    return V


for doping in ("doped", "undoped"):
    for geometry in ("wetting", "non-wetting"):
        key = (doping, geometry)
        one = None
        for rep in reps:
            copies = rep[0] * rep[1]; t0 = time.time()
            try:
                a, e, f = evaluate(key, rep)
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                print(f"== {key} {rep[0]}x{rep[1]}: REFUSED with the partitions: {type(err).__name__}: {str(err)[:110]}")
                try:
                    a, e, f = evaluate(key, rep, partitions=False)
                except Exception as err2:                                    # noqa: BLE001
                    print(f"   and without them: {type(err2).__name__}: {str(err2)[:160]}"); continue
                print("   evaluated with the partitions replaced by an empty list")
            else:
                if rep == (1, 1, 1):
                    _, e2, f2 = evaluate(key, rep, partitions=False)
                    print(f"== {key}: partitions replaced by an empty list: dE {e2 - e:+.3e} eV, max|dF| {np.abs(f2 - f).max():.3e} eV/A")
                    a, e, f = evaluate(key, rep)
            z = a.get_atomic_numbers(); n = len(a)
            out, rec = CAP["out"], REC["record"]
            sites = out["sites"]; B = int(sites.numel())
            frames = CAP["frames"].to(torch.long)
            unit = rec.unit_charge.detach()
            n_x = int(frames.sum()) - 0
            au = [f"{float(unit[i]):+.2f}" for i in range(n) if z[i] == 79]
            if rep == (1, 1, 1):
                one = dict(e=e, f=f)
            print(f"== {key} {rep[0]}x{rep[1]}: E/cell - E(1x1) {1e3 * (e / copies - one['e']):+.3f} meV; band sites {B}; "
                  f"excess electrons N_x {n_x}; band unit charge {float(unit[-1]):+.3f}; Au charges {' '.join(au)}  ({time.time() - t0:.0f} s)")
            eps, tau = out["eps"].detach(), out["tau"].detach()
            h = torch.diag(eps) - tau
            ev, vec = torch.linalg.eigh(h)
            zs = torch.as_tensor(z)[sites]
            al = zs == 13
            line = f"   eps: Mg {float(eps[~al].min()):+.3f} .. {float(eps[~al].max()):+.3f} eV"
            if bool(al.any()):
                line += "; Al " + " ".join(f"{float(x):+.3f}" for x in eps[al])
            print(line + f"; lambda_1 {float(ev[0]):+.4f} (module {float(out['lambda1']):+.4f}); J_11 {float(out['J11']):.4f} eV")
            k = min(B, 2 * copies + 4)
            print("   levels above lambda_1 (eV): " + " ".join(f"{float(ev[i] - ev[0]):.4f}" for i in range(k)))
            close = [(i, float(ev[i + 1] - ev[i])) for i in range(k - 1) if float(ev[i + 1] - ev[i]) < 1e-3]
            print(f"   gaps below 1 meV among them: {close if close else 'none'}")
            w1 = vec[:, 0] ** 2
            print(f"   phi_1: weight on Al {float(w1[al].sum()):.3f}; participation 1/sum w^2 = {1.0 / float((w1 * w1).sum()):.1f} of {B} sites")
            V = band_interaction()
            if not bool(torch.isfinite(V).all()):
                print("   V has non-finite on-site entries: determinant energies skipped"); continue
            rows = []
            for m in range(1, min(B, 2 * copies) + 1):
                P = vec[:, :m] @ vec[:, :m].T
                rho = torch.diagonal(P)
                e_det = 2.0 * float(ev[:m].sum()) + float((2.0 * torch.outer(rho, rho) - P * P).mul(V).sum())
                rows.append(e_det)
            j11 = float((torch.outer(w1, w1) * V).sum())
            print(f"   check: 2 lambda_1 + J_11 from V here {2 * float(ev[0]) + j11:+.6f} eV; module {2 * float(out['lambda1']) + float(out['J11']):+.6f} eV")
            print("   E_det(2m) (eV), m = 1..: " + " ".join(f"{x:+.4f}" for x in rows))
            print("   E_det(2m) / copies (eV): " + " ".join(f"{x / copies:+.4f}" for x in rows), flush=True)
