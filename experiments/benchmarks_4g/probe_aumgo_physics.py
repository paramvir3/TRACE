"""Polarization vs charge transfer in Au2 / MgO(001):Al -- what the trained model does, and DFT.

Usage: python probe_aumgo_physics.py [--states] RUN_DIR [N_PER_CLASS]

--states: the model was trained with the SPEC 21 charge states of
annotate_charge_states.py; the free evaluation then declares them (it is a
different energy function without them), the charges reported are those of the
DOMINANT diabatic state, and the dominant state's Au2 charge (0, -1 or +1) is
tallied per class.  The diabatic energies dE_CT are evaluated as before, and
for the reverse transfer (-1, +1) too.

1. Channels.  Over test structures: the RMS induced dipole |mu| per element
   (the dipole channel) and the RMS split-charge transfer |f t| over bonds (the
   charge-flow channel), so that "is polarization on" is a measured fact.
2. Gauss's law around the dopants.  For every structure, atoms are ordered by
   their minimum-image distance to the nearest dopant site (the three Al in a
   doped slab; the SAME three Mg sites, indices 8/56/104, in an undoped slab,
   whose geometry is the same to 0.002 A).  The cumulative net charge
   Q(<R) = sum of q_i over atoms within R is reported for TRACE and for the
   DFT Hirshfeld charges, as doped minus undoped:
       dQ(<R) = <Q(<R)>_doped - <Q(<R)>_undoped.
   A dielectric (linear polarization of bound charge) screens the dopant
   LOCALLY: dQ(<R) returns to ~0 within a few A.  A donor that gives an
   electron to a distant acceptor leaves dQ(<R) > 0 out to the acceptor, and
   the acceptor (Au2, beyond 10 A) carries the opposite charge.
3. The charge-separated state.  The slab and the Au2 are made two fragments
   (SPEC 1.3 per-fragment constraints) and the energy is evaluated with the
   fragment charges (0, 0) and (+1, -1):
       dE_CT = E(substrate +1, Au2 -1) - E(0, 0),
   the model's energy of an integer electron transfer substrate -> Au2 with
   the polarizable environment relaxed in each state.  Doped minus undoped
   tells whether the dopant makes that redox state cheap in the model.
Writes RUN_DIR/probe_aumgo_physics.json.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_aumgo as ev  # noqa: E402  (float64, threads, the calculator import)
from eval_aumgo import with_states  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

DOPANT_SITES = [8, 56, 104]          # Al in every doped structure; Mg at the same sites when undoped
AU = [108, 109]
R_EDGES = [2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 30.0]


class Capture:
    """Wraps the calculator's model to keep the electrostatic record of each call."""

    def __init__(self, model):
        self.model, self.last = model, None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, data, **kwargs):
        kwargs["return_electrostatics"] = True
        out = self.model(data, **kwargs)
        self.last = out[3]["electrostatics"][0]
        return out


def evaluate(calc, atoms, fragments=None, states=False):
    a = with_states(atoms) if (states and fragments is None) else atoms.copy()
    a.info = {"charge": 0.0, **{k: v for k, v in a.info.items() if k.startswith("charge_state")}}
    if fragments is not None:
        a.arrays["fragment_id"] = np.array([1 if i in AU else 0 for i in range(len(a))])
        a.info["fragment_charges"] = np.array(fragments, dtype=float)
    # ASE re-uses cached results when positions, numbers and cell are unchanged; it
    # does not compare info or custom arrays, so the fragment charges would be ignored.
    calc.reset()
    calc.model.last = None
    a.calc = calc
    energy = a.get_potential_energy()
    record = calc.model.last
    assert record is not None, "the model was not called: a cached result was returned"
    return energy, record


def main():
    args = sys.argv[1:]
    states = "--states" in args
    args = [a for a in args if a != "--states"]
    run = args[0]
    n_per = int(args[1]) if len(args) > 1 else 20
    calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
    calc.model = Capture(calc.model)
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    classes = {}
    for atoms in test:
        doping = "doped" if "Al" in atoms.get_chemical_symbols() else "undoped"
        key = (doping, ev.geometry_class(atoms))
        if len(classes.setdefault(key, [])) < n_per:
            classes[key].append(atoms)
    assert all(atoms.get_chemical_symbols()[i] == "Al" for atoms in classes[("doped", "non-wetting")]
               for i in DOPANT_SITES)

    mu_rms, t_rms, shells, ct = {}, [], {}, {}
    for key, frames in sorted(classes.items()):
        cum_trace, cum_dft, au_trace, au_dft, d_ct, dominant = [], [], [], [], [], []
        for atoms in frames:
            energy0, record = evaluate(calc, atoms, states=states)
            solve = record.solve_result
            q = solve.q.detach().numpy()
            mu = solve.mu.detach().numpy()
            for symbol, m in zip(atoms.get_chemical_symbols(), mu):
                mu_rms.setdefault(symbol, []).append(float(np.dot(m, m)))
            t_rms.append(float(np.sqrt(np.mean(np.square((q - solve.rho.detach().numpy()))))))
            dist = np.min([atoms.get_distances(s, range(len(atoms)), mic=True) for s in DOPANT_SITES], axis=0)
            hirsh = atoms.arrays["hirshfeld_charges"]
            cum_trace.append([q[dist < r].sum() for r in R_EDGES])
            cum_dft.append([hirsh[dist < r].sum() for r in R_EDGES])
            au_trace.append(q[AU].sum())
            au_dft.append(hirsh[AU].sum())
            dominant.append(int(round(float(q[AU].sum()))))
            e00, _ = evaluate(calc, atoms, fragments=[0.0, 0.0])
            e_ct, rec_ct = evaluate(calc, atoms, fragments=[1.0, -1.0])
            e_rev, _ = evaluate(calc, atoms, fragments=[-1.0, 1.0])
            d_ct.append((e_ct - e00, e00 - energy0, e_rev - e00))
        shells[key] = {"trace": np.mean(cum_trace, axis=0), "dft": np.mean(cum_dft, axis=0),
                       "au_trace": float(np.mean(au_trace)), "au_dft": float(np.mean(au_dft))}
        ct[key] = np.mean(d_ct, axis=0)
        tally = {c: dominant.count(c) for c in sorted(set(dominant))} if states else None
        print(f"{key}: n={len(frames)}  q(Au2) TRACE {shells[key]['au_trace']:+.3f}  Hirshfeld "
              f"{shells[key]['au_dft']:+.3f}   dE_CT = E(+1,-1) - E(0,0) = {ct[key][0]:+.3f} eV, "
              f"E(-1,+1) - E(0,0) = {ct[key][2]:+.3f} eV;  E(0,0) - E(free) = {ct[key][1] * 1e3:+.1f} meV"
              + (f";  dominant Au2 charge state counts {tally}" if states else ""), flush=True)

    print("\n1. Channels: RMS induced dipole |mu| (e A) per element: "
          + ", ".join(f"{s} {np.sqrt(np.mean(v)):.4f}" for s, v in sorted(mu_rms.items()))
          + f";  RMS charge moved by bond flow |q - rho| {np.mean(t_rms):.4f} e")
    print("\n2. Cumulative net charge around the dopant sites, doped minus undoped, e "
          "(R = distance to the nearest dopant site):")
    print("   R (A)      " + " ".join(f"{r:7.1f}" for r in R_EDGES))
    out = {"channels": {"mu_rms_eA": {s: float(np.sqrt(np.mean(v))) for s, v in mu_rms.items()},
                        "bond_flow_rms_e": float(np.mean(t_rms))}, "shells": {}, "charge_transfer_state": {}}
    for geometry in ("non-wetting", "wetting"):
        d, u = shells[("doped", geometry)], shells[("undoped", geometry)]
        for name in ("trace", "dft"):
            delta = d[name] - u[name]
            print(f"   {geometry:11s} {('TRACE' if name == 'trace' else 'DFT Hirshfeld'):13s}"
                  + " ".join(f"{x:+7.3f}" for x in delta))
            out["shells"][f"{geometry}/{name}"] = delta.tolist()
        dd = ct[("doped", geometry)][0] - ct[("undoped", geometry)][0]
        out["charge_transfer_state"][geometry] = {
            "dE_CT_doped_eV": float(ct[("doped", geometry)][0]),
            "dE_CT_undoped_eV": float(ct[("undoped", geometry)][0]),
            "doped_minus_undoped_eV": float(dd)}
    print("\n3. Integer transfer substrate -> Au2 in the model, dE_CT = E(+1,-1) - E(0,0):")
    for geometry, v in out["charge_transfer_state"].items():
        print(f"   {geometry:11s} doped {v['dE_CT_doped_eV']:+.3f} eV, undoped {v['dE_CT_undoped_eV']:+.3f} eV,"
              f" doped - undoped {v['doped_minus_undoped_eV']:+.3f} eV")
    out["r_edges_A"] = R_EDGES
    with open(os.path.join(run, "probe_aumgo_physics.json"), "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
