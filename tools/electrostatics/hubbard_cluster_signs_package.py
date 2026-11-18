"""The package's own configuration Hamiltonian (trace_redox.diabatic.diabatic_hamiltonian: Jordan-Wigner signs of
spin-less local orbitals, SPEC 29's sqrt 2) against the exact lowest-spin level of s-orbital Hubbard clusters."""
import itertools, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "hubbard_cluster_signs.py")).read()
ns = {}
exec(src[:src.index("t = 1.0\n")], ns)                      # the exact solver and the two reference rules only
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from trace_redox.diabatic import diabatic_hamiltonian
torch.set_default_dtype(torch.float64)
t = 1.0
shapes = {"dimer": (2, [(0, 1)]), "chain of 3": (3, [(0, 1), (1, 2)]), "triangle": (3, [(0, 1), (1, 2), (0, 2)]),
          "square": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]), "tetrahedron": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])}


def code_model(bonds, n, N, U, order=None, spin=True):
    """Lowest eigenvalue of the package's Hamiltonian: units = sites with charges +1, 0, -1 (0, 1, 2 s electrons)."""
    order = list(range(n)) if order is None else order      # the units' order (the Jordan-Wigner string depends on it)
    pos = {site: k for k, site in enumerate(order)}
    cfgs = [c for c in itertools.product((1, 0, -1), repeat=n) if sum(1 - q for q in c) == N]
    configs = torch.tensor(cfgs, dtype=torch.long)
    E = torch.tensor([U * sum(1 for q in c if q == -1) for c in cfgs], dtype=torch.float64)
    T = torch.zeros(n, n)
    for (i, j) in bonds:
        T[pos[i], pos[j]] = T[pos[j], pos[i]] = -t
    K = torch.ones(n, dtype=torch.long)
    H = diabatic_hamiltonian(E, configs, K, T, electrons=torch.full((n,), 79, dtype=torch.long) if spin else None)
    assert torch.allclose(H, H.T)
    return float(torch.linalg.eigvalsh(H)[0])


print(f"{'cluster':12s} {'N':>2s} {'U/t':>4s} {'low spin':>9s} {'package':>9s} {'err':>8s} | {'other unit orders (err)':s}")
worst_exact = 0.0
for name, (n, bonds) in shapes.items():
    for N in range(1, 2 * n):
        for U in (1.0, 4.0):
            e, g, same = ns["exact"](bonds, n, N, t, U)
            c = code_model(bonds, n, N, U)
            others = sorted({round(code_model(bonds, n, N, U, order=list(o)) - e, 4) for o in itertools.permutations(range(n))})
            print(f"{name:12s} {N:2d} {U:4.1f} {e:+9.4f} {c:+9.4f} {c - e:+8.4f} | {others}")
