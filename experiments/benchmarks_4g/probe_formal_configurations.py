# Frozen probe (2026-10-09): pre-registered P5 (README, "Pre-registered probe P5").
# Usage (repo root): python experiments/benchmarks_4g/probe_formal_configurations.py CHECKPOINT [EA2_O] [AU_WIDTH_FACTOR]
"""Formal oxidation states selected by energy: do integer charges with compact sizes choose formal states
in ionic systems, neutral ones in covalent systems, and electrons on Au2 in Au2/MgO:Al?

    E(n) = sum_u E_Z(n_u) + 1/2 sum_{u != v} n_u n_v J^pair_uv   [ - 1/2 n^T A K^-1 A^T n   (screened) ]

E_Z: exact isolated-ion energies from NIST (successive IEs; -EA_1; O at -2: -EA_1 + EA2_O, default 8.75 eV).
J^pair: Gaussian Coulomb without self-energies (open: K_E erf(r/s)/r; periodic: Ewald pair part, own
images included).  Screened: the trained bound sector's linear response at these widths (b excluded).
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

from trace_electrostatics.constants import K_E  # noqa: E402
from trace_electrostatics.ewald import EwaldParams  # noqa: E402
from trace_electrostatics.kernels import gram_self_energy  # noqa: E402
from trace_electrostatics.lattice import free_charge_lattice_blocks  # noqa: E402
from trace_redox.constants import (  # noqa: E402
    ATOMIC_ELECTRON_AFFINITY_EV as EA,
    ATOMIC_IONIZATION_EV as IE1,
    ATOMIC_SUCCESSIVE_IONIZATION_EV as IES,
    SHANNON_CRYSTAL_RADIUS,
)
from trace_redox.diabatic import configuration_energies, search_configurations  # noqa: E402

EA2_O = float(sys.argv[2]) if len(sys.argv) > 2 else 8.75
AU_FACTOR = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
RANGE = {8: (-2, 1), 12: (-1, 2), 13: (-1, 3), 79: (-1, 2), 11: (-1, 1), 17: (-1, 1), 1: (-1, 1), 6: (-1, 4), 47: (-1, 1)}
FORMAL = {8: -2, 12: 2, 13: 3, 79: 0, 11: 1, 17: -1, 1: 0, 6: 0, 47: 0}
STATES = torch.arange(-2, 5, dtype=torch.float64)                   # global state range -2 .. +4


def isolated_ion(z: int, n: int) -> float:
    """E_Z(n) relative to the neutral atom, eV (inf outside the range)."""
    lo, hi = RANGE[z]
    if n < lo or n > hi:
        return math.inf
    if n > 0:
        return float(sum(IES[z][k] for k in range(n)))
    if n == 0:
        return 0.0
    if n == -1:
        return -float(EA[z])
    assert z == 8 and n == -2
    return -float(EA[z]) + EA2_O


def width(z: int) -> float:
    if z in SHANNON_CRYSTAL_RADIUS:
        return math.sqrt(0.4) * SHANNON_CRYSTAL_RADIUS[z] * (AU_FACTOR if z == 79 else 1.0)
    return math.sqrt(2.0) * K_E / (math.sqrt(math.pi) * (float(IE1[z]) - float(EA[z])))   # QEq width (H, C)


def ladders(z):
    return torch.tensor([[isolated_ion(int(x), int(s)) for s in STATES.tolist()] for x in z])


def pair_coulomb(pos, z, cell=None):
    sigma = torch.tensor([width(int(x)) for x in z])
    if cell is None:
        d = (pos.unsqueeze(1) - pos.unsqueeze(0)).norm(dim=-1)
        s = torch.sqrt(sigma.unsqueeze(1) ** 2 + sigma.unsqueeze(0) ** 2)
        J = K_E * torch.erf(d / s) / torch.where(d > 0, d, torch.ones_like(d))
        return J.fill_diagonal_(0.0), sigma
    # the real-space cutoff floored at the widest pair's accuracy radius s erfc^-1(10 accuracy)
    # (amended 2026-10-09: a fixed 10 A was refused for the Au width x 2 variant, s = 2.7 A)
    from scipy.special import erfcinv
    r_cut = max(10.0, 1.01 * math.sqrt(2.0) * float(sigma.max()) * float(erfcinv(1e-9)))
    ewald = EwaldParams.auto_tune(cell, r_cut, accuracy=1e-10)
    ones = torch.ones(len(z))
    return free_charge_lattice_blocks(pos, cell, sigma, sigma, sigma, ones, ewald)["j"], sigma


def lowest(e_hat, W, z, Q):
    """The lower minimum of the searches from all neutral and from the formal guess, and both values."""
    L = torch.tensor([RANGE[int(x)][0] for x in z])
    K = torch.tensor([RANGE[int(x)][1] for x in z])
    zero_b = torch.zeros(len(z))
    results = []
    for start in (None, torch.tensor([FORMAL[int(x)] for x in z])):
        configs, E = search_configurations(e_hat, STATES, zero_b, W, K, int(Q), window=0.3, cap=200000,
                                           start=start, min_charge=L)
        k = int(torch.argmin(E))
        results.append((float(E[k]), configs[k]))
    best = min(results, key=lambda r: r[0])
    neutral = torch.zeros(len(z), dtype=torch.long)
    E0 = float(configuration_energies(neutral.unsqueeze(0), e_hat, STATES, zero_b, W)[0]) if int(Q) == 0 else math.nan
    return best, [r[0] for r in results], E0


def summary(n, z):
    from collections import Counter
    return dict(sorted(Counter((int(a), int(b)) for a, b in zip(z.tolist(), n.tolist())).items()))


def screened_operator(checkpoint, atoms, sigma):
    """J_pair - A K^-1 A^T of the trained bound sector at these free widths (b excluded)."""
    import trace_redox.module as module_code
    from flashace.calculator import TransformersACECalculator
    from trace_redox.bound import eliminate_bound_sector
    calc = TransformersACECalculator(model_path=checkpoint, device="cpu")
    torch.set_num_threads(2)
    captured = {}
    original = module_code.eliminate_bound_sector

    def capture(assembly, *args, **kwargs):
        captured["assembly"] = assembly
        return original(assembly, *args, **kwargs)

    module_code.eliminate_bound_sector = capture
    try:
        a = atoms.copy()
        a.info = dict(atoms.info)
        a.calc = calc
        a.get_potential_energy()
    finally:
        module_code.eliminate_bound_sector = original
    assembly = captured["assembly"]
    with torch.no_grad():
        elim = eliminate_bound_sector(assembly, sigma, sigma_self=sigma)
        return elim.W.detach() - torch.diag(gram_self_energy(sigma))


def main():
    checkpoint = sys.argv[1]
    print(f"EA2(O) = {EA2_O} eV; Au width factor {AU_FACTOR}")
    # P5a: rocksalt MgO
    a = 4.212
    frac = torch.tensor([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]])
    cell = a * torch.eye(3)
    z = torch.tensor([12] * 4 + [8] * 4)
    J, _ = pair_coulomb(frac @ cell, z, cell)
    (E, n), starts, E0 = lowest(ladders(z), J, z, 0)
    formal = torch.tensor([2] * 4 + [-2] * 4, dtype=torch.float64)
    E_pair = 0.5 * float(formal @ J @ formal) / 4
    madelung = -1.7475645946331822 * 4 * K_E / (a / 2)
    closed = (float(IES[12][0] + IES[12][1]) - float(EA[8]) + EA2_O) + E_pair
    is_formal = bool((n == formal.long()).all())
    print(f"P5a MgO: lowest {summary(n, z)}; formal: {is_formal}; dE(formal - neutral) per MgO = {(E - E0) / 4:+.4f} eV, "
          f"closed form {closed:+.4f} (|diff| {abs((E - E0) / 4 - closed):.1e}); pair part / point Madelung = {E_pair / madelung:.4f}; "
          f"starts {[round(s - E0, 3) for s in starts]}")
    print(f"     P5a {'PASS' if is_formal and (E - E0) < 0 and abs((E - E0) / 4 - closed) <= 1e-9 else 'FAIL'}")

    frames = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    doped = [f for f in frames if f.info["system"].startswith("Al")][:6]
    undoped = [f for f in frames if not f.info["system"].startswith("Al")][:6]
    for label, group in (("P5b doped Au2/MgO:Al", doped), ("P5c undoped Au2/MgO", undoped)):
        brackets = ("bare", "screened") if label.startswith("P5b") else ("bare",)
        for bracket in brackets:
            hits = 0
            for k, atoms in enumerate(group):
                z = torch.tensor(atoms.get_atomic_numbers())
                pos = torch.tensor(atoms.positions)
                J, sigma = pair_coulomb(pos, z, torch.tensor(atoms.cell.array))
                W = J if bracket == "bare" else screened_operator(checkpoint, atoms, sigma)
                (E, n), starts, E0 = lowest(ladders(z), W, z, 0)
                au = n[z == 79]
                if label.startswith("P5b"):
                    ok = int(au.sum()) <= -1
                else:
                    ok = bool((n[z == 12] == 2).all() and (n[z == 8] == -2).all() and (au == 0).all())
                hits += ok
                print(f"  {label} [{bracket}] structure {k}: Al {n[z == 13].tolist()}, Au {au.tolist()}, "
                      f"Mg+ count {int((n[z == 12] == 1).sum())}, Mg2+ {int((n[z == 12] == 2).sum())}, "
                      f"O2- {int((n[z == 8] == -2).sum())}, O- {int((n[z == 8] == -1).sum())}; "
                      f"E - E(neutral) = {E - E0:+.2f} eV; starts {[round(s - E0, 2) for s in starts]}")
            print(f"     {label} [{bracket}]: {hits}/6 -> {'PASS' if hits >= 5 else 'FAIL'}")

    for label, path, criteria in (("P5d chain", "Carbon_chain_test", "chain"), ("P5e NaCl", "NaCl_test", "nacl")):
        frames = read(f"experiments/benchmarks_4g/data/{path}.xyz", ":")
        for formula in sorted({f.get_chemical_formula() for f in frames}):
            group = [f for f in frames if f.get_chemical_formula() == formula][:6]
            hits = 0
            for k, atoms in enumerate(group):
                z = torch.tensor(atoms.get_atomic_numbers())
                Q = int(round(float(atoms.info.get("charge", 0))))
                J, _ = pair_coulomb(torch.tensor(atoms.positions), z)
                (E, n), starts, E0 = lowest(ladders(z), J, z, Q)
                if criteria == "chain":
                    ok = bool((n == 0).all()) if Q == 0 else (int((n == 1).sum()) == 1 and int((n != 0).sum()) == 1)
                elif formula == "Cl8Na9":
                    ok = bool((n[z == 11] == 1).all() and (n[z == 17] == -1).all())
                else:
                    ok = bool((n[z == 11] == 1).all() and int((n[z == 17] == -1).sum()) == 7 and int((n[z == 17] == 0).sum()) == 1)
                hits += ok
                print(f"  {label} {formula} (Q {Q}) structure {k}: {summary(n, z)}; starts {[round(s, 2) for s in starts]}")
            print(f"     {label} {formula}: {hits}/6 -> {'PASS' if hits >= 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
