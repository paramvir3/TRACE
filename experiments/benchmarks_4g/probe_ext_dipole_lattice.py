"""Tin-foil Ewald sum of a lattice of parallel unit point dipoles: Lambda_ab (1/A^3), energy per dipole = K_E/2 p.Lambda.p."""
import itertools, math, sys
import numpy as np
from scipy.special import erfc

def Lam(cell, eta=0.35, rc=40.0, gc=4.0):
    cell = np.asarray(cell, float); V = abs(np.linalg.det(cell)); rec = 2 * np.pi * np.linalg.inv(cell).T
    nmax = [int(math.ceil(rc / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    out = np.zeros((3, 3))
    for n in itertools.product(*[range(-m, m + 1) for m in nmax]):
        if n == (0, 0, 0): continue
        R = np.array(n) @ cell; r = np.linalg.norm(R)
        if r > rc: continue
        B = erfc(eta * r) / r**3 + 2 * eta / math.sqrt(math.pi) * math.exp(-(eta * r) ** 2) / r**2
        C = 3 * erfc(eta * r) / r**5 + 2 * eta / math.sqrt(math.pi) * (2 * eta**2 + 3 / r**2) * math.exp(-(eta * r) ** 2) / r**2
        out += B * np.eye(3) - C * np.outer(R, R)
    gmax = [int(math.ceil(gc / np.linalg.norm(rec[i]))) + 1 for i in range(3)]
    for n in itertools.product(*[range(-m, m + 1) for m in gmax]):
        if n == (0, 0, 0): continue
        G = np.array(n) @ rec; g2 = G @ G
        if g2 > gc**2: continue
        out += 4 * np.pi / V * np.outer(G, G) / g2 * math.exp(-g2 / (4 * eta**2))
    return out - 4 * eta**3 / (3 * math.sqrt(math.pi)) * np.eye(3)

if __name__ == "__main__":
    a, c = 9.0474, 26.4589
    # convergence check in the Ewald parameter (an artifact check): two splittings must agree
    l1, l2 = Lam(np.diag([a, a, c]), eta=0.35), Lam(np.diag([a, a, c]), eta=0.45, gc=5.0)
    print("eta check, max difference %.2e 1/A^3" % np.abs(l1 - l2).max())
    base = l1
    print("cell   Lxx        Lzz        dLxx       dLzz   (1e-3 / A^3; d = against 1x1)")
    for mx, my in ((1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (4, 1), (1, 4), (2, 2), (3, 3), (4, 4), (8, 8)):
        L = Lam(np.diag([mx * a, my * a, c]))
        print(f"{mx}x{my}  {1e3 * L[0, 0]:+9.4f}  {1e3 * L[2, 2]:+9.4f}  {1e3 * (L[0, 0] - base[0, 0]):+9.4f}  {1e3 * (L[2, 2] - base[2, 2]):+9.4f}")
