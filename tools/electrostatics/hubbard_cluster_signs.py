"""Exact diagonalisation of small s-orbital Hubbard clusters against the charge-configuration model (one state per
occupation pattern, couplings t or sqrt(2) t), with a uniform sign and with the particle-hole sign."""
import itertools, math
import numpy as np

def spectrum(bonds, n_sites, N, t, U, twice_sz):
    """Lowest eigenvalue per total spin in the N-electron, S_z = 0 (or 1/2) sector; H = -t sum c+c + U sum n_up n_dn."""
    orb = 2 * n_sites                                     # orbital index 2 i + spin
    states = [s for s in range(1 << orb) if bin(s).count("1") == N
              and sum(((s >> (2 * i)) & 1) - ((s >> (2 * i + 1)) & 1) for i in range(n_sites)) == twice_sz]
    if not states:
        return np.zeros(0)
    index = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    def hop(s, a, b):                                    # c+_a c_b |s>, with the fermion sign
        if not (s >> b) & 1 or (s >> a) & 1:
            return None, 0
        sign = (-1) ** bin(s & ((1 << b) - 1)).count("1")
        s2 = s ^ (1 << b)
        sign *= (-1) ** bin(s2 & ((1 << a) - 1)).count("1")
        return s2 | (1 << a), sign
    for s in states:
        k = index[s]
        H[k, k] = U * sum(((s >> (2 * i)) & 1) & ((s >> (2 * i + 1)) & 1) for i in range(n_sites))
        for (i, j) in bonds:
            for sp in (0, 1):
                for a, b in ((2 * i + sp, 2 * j + sp), (2 * j + sp, 2 * i + sp)):
                    s2, sign = hop(s, a, b)
                    if s2 is not None:
                        H[index[s2], k] += -t * sign
    return np.linalg.eigvalsh(H)


def exact(bonds, n_sites, N, t, U):
    """(lowest level of the lowest total spin, overall ground level, twice the ground state's spin)."""
    low = spectrum(bonds, n_sites, N, t, U, N % 2)
    high = list(spectrum(bonds, n_sites, N, t, U, N % 2 + 2))
    own = []
    for e in low:                                        # remove the higher-spin multiplets' S_z = Smin components
        hit = next((k for k, x in enumerate(high) if abs(x - e) < 1e-9), None)
        if hit is None:
            own.append(e)
        else:
            high.pop(hit)
    ground_is_low_spin = abs(own[0] - low[0]) < 1e-9
    return float(own[0]), float(low[0]), ground_is_low_spin

def config_model(bonds, n_sites, N, t, U, rule):
    cfgs = [c for c in itertools.product((0, 1, 2), repeat=n_sites) if sum(c) == N]
    index = {c: k for k, c in enumerate(cfgs)}
    H = np.zeros((len(cfgs), len(cfgs)))
    for c in cfgs:
        k = index[c]
        H[k, k] = U * sum(1 for n in c if n == 2)
        for (i, j) in bonds:
            for a, b in ((i, j), (j, i)):               # an electron moves from b to a
                if c[b] == 0 or c[a] == 2:
                    continue
                d = list(c); d[b] -= 1; d[a] += 1
                pairs = (c[b] % 2) == (c[a] % 2)          # SPEC 29: the move pairs or unpairs two frontier electrons
                amp = t * (math.sqrt(2.0) if pairs else 1.0)
                sign = -1.0
                if rule == "particle-hole" and c[b] + c[a] == 3:    # (2, 1) -> (1, 2): the bond is more than half filled
                    sign = +1.0
                H[index[tuple(d)], k] += sign * amp
    return float(np.linalg.eigvalsh(H)[0]), len(cfgs)

t = 1.0
shapes = {"dimer": (2, [(0, 1)]), "chain of 3": (3, [(0, 1), (1, 2)]), "triangle": (3, [(0, 1), (1, 2), (0, 2)]),
          "square": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]), "tetrahedron": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])}
print(f"{'cluster':12s} {'N':>2s} {'U/t':>4s} {'low spin':>9s} {'ground':>9s} {'uniform':>9s} {'err':>8s} {'p-h sign':>9s} {'err':>8s}  (err against the lowest state of the lowest spin)")
for name, (n, bonds) in shapes.items():
    for N in range(1, 2 * n):
        for U in (1.0, 4.0):
            e, g, same = exact(bonds, n, N, t, U)
            u, m = config_model(bonds, n, N, t, U, "uniform")
            p, _ = config_model(bonds, n, N, t, U, "particle-hole")
            print(f"{name:12s} {N:2d} {U:4.1f} {e:+9.4f} {g:+9.4f}{' ' if same else '*'} {u:+9.4f} {u - e:+8.4f} {p:+9.4f} {p - e:+8.4f}")
print("* the ground state has a higher spin than the lowest one")
