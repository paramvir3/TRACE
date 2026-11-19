# Written on 2026-10-29 (the measurement scripts of the theory note's section 4c); run from the repo root.
"""Shared folded states between explicit configurations (the fold's per-state self-energy approximation): on the
undoped references, every explicit configuration's leaves (its moved configuration = the Q state), the Q states
reached from more than one explicit configuration, the diagonal dressing per configuration and the perturbative
off-diagonal self-energy Sigma_ij = sum_q v_iq v_jq / (E_i - E_q) over shared Q states (at the lower state's
energy).  Usage: python shared_q.py MODEL.pt"""
import sys, dataclasses, collections, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import probe_t5_first_order as Q
from flashace.calculator import TransformersACECalculator
from trace_redox.constants import formal_frame
Q._install()
refs = Q.references()
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000, downfold_transfer=True,
                                    downfold_hop_ordering="pauli", downfold_scale=0.85, downfold_anisotropy=0.0,
                                    downfold_crystal_field=0.0, downfold_free_axes=False)
    for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
        at = refs[key]; sym = at.get_chemical_symbols()
        s = Q.evaluate(calc, at, 0.0)
        rec = mod._downfold_last; configs = s["configs"]; z = s["z"]; fr = formal_frame(z)
        E = s.get("energies")
        print(f"=== {model.split('/')[-2][-3:]} {key[1]}: {len(configs)} configurations, {rec['leaves']} leaves; dressing per configuration (eV): "
              + ", ".join(f"{d:+.3f}" for d in rec["dressing"]))
        labels = []
        for c in range(len(configs)):
            cfg = configs[c]
            labels.append(" + ".join(f"{sym[k]}{k}{int(cfg[k] - fr[k]):+d}" for k in range(len(fr)) if cfg[k] != fr[k]) or "frame")
            print(f"   config {c}: {labels[-1]}")
        qstate = collections.defaultdict(list)      # Q configuration -> [(c, o, g, kind, Delta, v)]
        for c, o, g, k, d, v in rec["all_leaves"] or []:
            cfg = configs[c].clone()
            if k in (0, 1, 2, 3):
                cfg[o] += 1; cfg[g] -= 1
            else:
                cfg[o] -= 1; cfg[g] += 1
            qstate[(tuple(cfg.tolist()), o, k)].append((c, o, g, k, d, v))      # a folded state: hole configuration, O, p component
        shared = {q: L for q, L in qstate.items() if len({c for c, *_ in L}) > 1}
        print(f"   Q states {len(qstate)}, shared between explicit configurations: {len(shared)}")
        sigma = collections.defaultdict(float)
        for q, L in shared.items():
            by_c = collections.defaultdict(list)
            for c, o, g, k, d, v in L:
                by_c[c].append((d, v))
            cs = sorted(by_c)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    ci, cj = cs[i], cs[j]
                    vi = sum(v for _, v in by_c[ci]); vj = sum(v for _, v in by_c[cj])
                    di = min(d for d, _ in by_c[ci]); dj = min(d for d, _ in by_c[cj])
                    # E_q - E_i = Delta_i ; the same Q state seen from j sits at Delta_j = E_q - E_j
                    sigma[(ci, cj)] += -vi * vj / max(min(di, dj), 1e-3)
        for (ci, cj), val in sorted(sigma.items()):
            print(f"   Sigma({ci},{cj}) perturbative {val:+.3f} eV   [{labels[ci]} | {labels[cj]}]")
        if not sigma:
            print("   no shared Q state")

        # --- the exact P + Q block against the per-state dressing (ground-state level) ---
        H = s["H"].detach().double().numpy()                      # the ensemble's Hamiltonian: dressed diagonal, T off
        lam = np.array(rec["dressing"], dtype=float)
        C = H.shape[0]
        H_und = H.copy(); H_und[np.diag_indices(C)] -= lam        # undressed explicit block
        qs = list(qstate)                                          # Q states (deduplicated moved configurations)
        qi = {q: i for i, q in enumerate(qs)}
        NQ = len(qs); Eq = np.full(NQ, np.nan); V = np.zeros((C, NQ))
        E_c = np.diag(H_und)
        leaf_of = {}                                               # (q) -> (o, kind, g) for the hop structure
        for q, L in qstate.items():
            for c, o, g, k, d, v in L:
                e = E_c[c] + d
                Eq[qi[q]] = e if np.isnan(Eq[qi[q]]) else Eq[qi[q]]
                V[c, qi[q]] += v
                leaf_of.setdefault(qi[q], set()).add((o, k, g))
        t_hop = 1.909                                              # the model's |T_AuAu| f on these references
        HQ = np.diag(Eq)
        nhop = 0
        for i in range(NQ):
            for j in range(i + 1, NQ):
                (ci, oi, ki), (cj, oj, kj) = qs[i], qs[j]
                if oi != oj or ki != kj or ki >= 3:
                    continue
                diff = [a for a in range(len(ci)) if ci[a] != cj[a]]
                if len(diff) == 2 and all(abs(ci[a] - cj[a]) == 1 for a in diff) and sum(ci) == sum(cj):
                    HQ[i, j] = HQ[j, i] = +t_hop; nhop += 1           # the electron on the other s atom: the Pauli ordering
        print(f"   hops inside Q: {nhop}")
        full = np.block([[H_und, V], [V.T, HQ]])
        e_und = np.linalg.eigvalsh(H_und)[0]; e_ps = np.linalg.eigvalsh(H)[0]; e_ex = np.linalg.eigvalsh(full)[0]
        print(f"   ground state (eV, relative to the frame's undressed energy {E_c[0]:+.3f}): undressed {e_und - E_c[0]:+.3f}, "
              f"per-state dressed {e_ps - E_c[0]:+.3f}, exact P+Q {e_ex - E_c[0]:+.3f}; per-state error {e_ps - e_ex:+.3f} eV; "
              f"kT {s['kT']:.3f} eV; NQ {NQ}")
        w = np.linalg.eigh(full)[1][:, 0]; print(f"   exact ground-state weights: explicit {np.round(w[:C] ** 2, 3).tolist()}, folded total {float((w[C:] ** 2).sum()):.3f}")
        # --- validation: with P restricted to one configuration, the block must reproduce the fold's own dressing ---
        for c in range(C):
            reach = [i for i in range(NQ) if any(cc == c for cc, *_ in qstate[qs[i]])]
            # the partner states reachable by hops from c's states (zero coupling, same (o, kind), electron moved)
            grow = True
            while grow:
                grow = False
                for i in range(NQ):
                    if i not in reach and any(HQ[i, j] != 0 for j in reach):
                        reach.append(i); grow = True
            idx = np.array(reach)
            blk = np.block([[np.array([[H_und[c, c]]]), V[c:c + 1, idx]], [V[c:c + 1, idx].T, HQ[np.ix_(idx, idx)]]])
            root = np.linalg.eigvalsh(blk)[0] - H_und[c, c]
            print(f"   validation config {c}: single-configuration block root {root:+.4f} vs the fold's dressing {lam[c]:+.4f} eV  "
                  f"-> {'OK' if abs(root - lam[c]) < 2e-3 else 'MISMATCH'} ({len(idx)} folded states)")
