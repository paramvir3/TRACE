"""Build the entries of an s1 pair's state-pair slot beyond the neutral one (trace_redox SPEC 40.7; 2026-11-14).

For an applied pair with first-principles carriers (EA*, IP* in trace_redox/data/two_site.json), three functions of the
distance are fixed by three conditions on the ISOLATED pair, through the package's own fixed terms (no learned shift;
they do not depend on any weight, asserted here with two differently seeded modules) and the tabulated hop:

    lambda_ionic(R)  on (+1, -1), (-1, +1):   E[X2](R)  = E_gs(R)
    rho_minus(R)     on (-1, 0), (0, -1):     E[X2-](R) = E_gs(R) - EA*(R)
    rho_plus(R)      on (+1, 0), (0, +1):     E[X2+](R) = E_gs(R) + IP*(R)

(energies against two neutral atoms; the package's atomic EA and IP are the exact ones).  Each condition is one
equation in one unknown at each tabulated distance, solved by the secant method on the package's own energy.

    PYTHONPATH=. python tools/electrostatics/build_two_site_entries.py SYMBOL [--out PATH] [--check]

The entries are a property of the package's fixed terms under OPTIONS (the single model's option set, step 33) and
the electrostatic setting gaussian_a; they are written into the data file with both and with the table values they
were solved with (E_gs, E_T, t, EA*, IP*, rho at the knots), and a model whose options differ, or a table that has
changed since, is refused at run time.  The fixed terms hold no learned level shift and no learned coupling
(module.fixed_terms_only); the pair's hop acts on s moves only (SPEC 40.6 ix), so the cation's condition is the
two-level resonance of X+ X0 and X0 X+ and not a ground state reached through X2+ X-.
--check rebuilds and compares with the stored entries and their recorded settings; it exits non-zero on a difference.
"""
import argparse
import json
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
torch.set_default_dtype(torch.float64)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from trace_electrostatics.config import ElectrostaticConfig  # noqa: E402
from trace_electrostatics.heads import ElectrostaticHeads  # noqa: E402
from trace_redox import RedoxConfig, RedoxElectrostatics  # noqa: E402
from trace_redox.two_site import DATA_TWO_SITE, ENTRY_NAMES, ENTRY_OPTIONS, TwoSite  # noqa: E402
import trace_redox.two_site as two_site_file  # noqa: E402

Z = {"H": 1, "Ag": 47, "Au": 79}
R_MAX = 5.29
#: the single model's option set (experiments/benchmarks_4g, step 33) with SPEC 39, 29 and 40 on
OPTIONS = dict(anchor="formal", fragment_ensemble="diabatic", band_form="fermi", bond_partition=True, bond_pauli=True,
               bond_pauli_level="almo", bond_pauli_wall=True, carrier_band=True, carrier_induction=True,
               carrier_overlap="gordon_kim", carrier_penetration=True, coupling_bound="harrison", couplings=True,
               diabatic_selection="amplitude", diabatic_space="clusters", exclusion=False, frame_induction_damping=True,
               frame_polarizability_anchor=True, frame_polarizability_scope="ions", max_charge=3,
               pauli_table="first_principles", shift_gate=True, state_polarizability=True,
               state_polarizability_stiffness="onsite", frame_pair=True, spin_pairing=True, two_site=True)


def build_module(z, seed):
    torch.manual_seed(seed)
    heads = ElectrostaticHeads(r_max=R_MAX, hardness_margin=1.0)
    config = RedoxConfig(state_polarizability_elements=sorted({int(z), 8, 79}), **OPTIONS)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(bond_screening=(2.0, 2.8)), config)
    module.eval()
    module.fixed_terms_only = True
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(seed + 100))
    return module, h, config


def energy(module, h, z, R, q):
    """The package's energy of the isolated pair X2^q at R, against two neutral atoms (no backbone here)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, float(R)]])
    d = torch.cdist(pos, pos)
    src, dst = torch.nonzero((d < R_MAX) & (d > 0), as_tuple=True)
    ei = torch.stack([src, dst])
    vec = pos[ei[0]] - pos[ei[1]]
    with torch.no_grad():
        E = module(h, pos, ei, vec, vec.norm(dim=-1), data={"z": torch.tensor([z, z]), "total_charge": float(q)})[0]
    return float(E)


def secant(f, x0, x1, tol=1e-10, steps=40):
    f0, f1 = f(x0), f(x1)
    for _ in range(steps):
        if abs(f1) <= tol:
            return x1, f1
        if f1 == f0:
            break
        x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
        f1 = f(x1)
    return x1, f1


def solve(symbol, table, seed=0):
    """The pair's entries on ``table`` (a TwoSite object the default accessor returns while this runs)."""
    z = Z[symbol]
    pair = (z, z)
    module, h, config = build_module(z, seed)
    other, h2, _ = build_module(z, seed + 7)
    points, car = table.points[pair], table.carriers[pair]
    Rs = car["R"].tolist()
    e_gs = {round(float(r), 6): float(e) for r, e in zip(points["R"], points["E_gs"])}
    out = {name: [] for name in ENTRY_NAMES}
    record = {"E_zero": [], "residual": 0.0, "weights": 0.0}
    try:
        for k, R in enumerate(Rs):
            trial = [0.0, 0.0, 0.0]
            table._trial[pair] = tuple(trial)
            zero = [energy(module, h, z, R, q) for q in (0, -1, 1)]
            again = [energy(other, h2, z, R, q) for q in (0, -1, 1)]
            record["weights"] = max(record["weights"], max(abs(a - b) for a, b in zip(zero, again)))
            record["E_zero"].append(zero)
            targets = {2: (0, e_gs[round(R, 6)]), 0: (-1, e_gs[round(R, 6)] - float(car["EA"][k])), 1: (1, e_gs[round(R, 6)] + float(car["IP"][k]))}
            for slot in (2, 0, 1):                                    # lambda_ionic, rho_minus, rho_plus
                q, target = targets[slot]

                def f(x, slot=slot, q=q, target=target):
                    trial[slot] = float(x)
                    table._trial[pair] = tuple(trial)
                    return energy(module, h, z, R, q) - target

                start = f(0.0)
                x, res = secant(f, 0.0, start if slot != 2 else -2.0 * start)   # dE/drho = -1 (one-carrier), dE/dlambda in (0, 1)
                trial[slot] = float(x)
                record["residual"] = max(record["residual"], abs(res))
            for name, v in zip(ENTRY_NAMES, trial):
                out[name].append(float(v))
            print(f"  {symbol}-{symbol} R {R:.2f}: package without the entries E0 {zero[0]:+.4f} (E_gs {e_gs[round(R, 6)]:+.4f}), EA {zero[0] - zero[1]:+.4f} "
                  f"(EA* {float(car['EA'][k]):+.4f}), IP {zero[2] - zero[0]:+.4f} (IP* {float(car['IP'][k]):+.4f}) -> rho_minus {trial[0]:+.4f}, "
                  f"rho_plus {trial[1]:+.4f}, lambda_ionic {trial[2]:+.4f} eV", flush=True)
    finally:
        table._trial.pop(pair, None)
    atom = [abs(energy(module, h, z, 30.0, 0))]
    options = {k: getattr(config, k) for k in ENTRY_OPTIONS}
    options["state_polarizability_element"] = True
    options["gaussian_a"] = float(module.config.gaussian_a)
    knots = torch.tensor(Rs)
    solved_with = {"data_files": two_site_file.data_hashes(),
                   "E_gs": [e_gs[round(R, 6)] for R in Rs],
                   "E_T": [float(points["E_T"][int((points["R"] == R).nonzero()[0])]) for R in Rs],
                   "t": table.hop(z, z, knots).tolist(), "EA": car["EA"].tolist(), "IP": car["IP"].tolist(),
                   "rho": table.relaxation(z, z, knots).tolist()}
    block = {"R": Rs, **out, "options": options, "solved_with": solved_with,
             "built": {"kT": float(config.kT), "selection_amplitude_off": float(config.selection_amplitude_off),
                       "selection_amplitude_on": float(config.selection_amplitude_on), "coupling_cutoff": float(config.coupling_cutoff),
                       "largest_residual_eV": record["residual"], "largest_difference_between_two_seeded_modules_eV": record["weights"],
                       "two_far_neutral_atoms_eV": atom[0],
                       "package_energies_without_the_entries": {"R": Rs, "q": [0, -1, 1], "E": record["E_zero"]}}}
    return block, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--out", default=DATA_TWO_SITE)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with open(args.out) as fh:
        data = json.load(fh)
    stored = None
    if args.check:                                                       # what is in the file now; refused entries fail
        try:
            stored = TwoSite(args.out, allow_unbuilt=True)
        except ValueError as err:
            raise SystemExit(f"the stored entries are not usable: {err}")
    table = TwoSite(args.out, allow_unbuilt=True, ignore_entries=True)   # a private object the module is pointed at
    before = two_site_file._TWO_SITE.get(DATA_TWO_SITE)
    two_site_file._TWO_SITE[DATA_TWO_SITE] = table
    failed = False
    try:
        for symbol in args.symbols:
            pair = (Z[symbol], Z[symbol])
            key = f"{Z[symbol]}-{Z[symbol]}"
            table._applied[pair] = True                                  # the build needs the pair's hop and relaxation
            table.applied = [p for p in table.pairs if table._applied[p]]
            table.elements = sorted({z for p in table.applied for z in p})
            table.entries.pop(pair, None)                                # build from the bare pair
            block, record = solve(symbol, table)
            print(f"{symbol}-{symbol}: largest residual {record['residual']:.1e} eV; two differently seeded modules agree to {record['weights']:.1e} eV; "
                  f"two far neutral atoms {block['built']['two_far_neutral_atoms_eV']:.1e} eV")
            if record["weights"] > 1e-8 or record["residual"] > 1e-8:
                raise SystemExit(f"{symbol}-{symbol}: the package's fixed terms depend on the weights for this option set "
                                 "(or a condition was not met): nothing written")
            if args.check:
                old = stored.entries.get(pair) if stored is not None else None
                if old is None:
                    print(f"{symbol}-{symbol}: no stored entries to compare with")
                    failed = True
                    continue
                worst = max(abs(a - float(b)) for name in ENTRY_NAMES for a, b in zip(block[name], old[name].tolist()))
                same = (old["R"].tolist() == block["R"] and all(old["options"].get(k) == v for k, v in block["options"].items()))
                print(f"{symbol}-{symbol}: rebuilt against stored: largest difference {worst:.1e} eV; knots and recorded settings equal: {same}")
                failed = failed or worst > 1e-8 or not same
            else:
                data["pairs"][key]["entries"] = block
                data["pairs"][key]["applied"] = True
                data["pairs"][key]["why_not_applied"] = ""
    finally:
        if before is None:
            two_site_file._TWO_SITE.pop(DATA_TWO_SITE, None)
        else:
            two_site_file._TWO_SITE[DATA_TWO_SITE] = before
    if args.check:
        raise SystemExit(1 if failed else 0)
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1)
    two_site_file._TWO_SITE.pop(args.out, None)                          # the next reader sees the new file
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
