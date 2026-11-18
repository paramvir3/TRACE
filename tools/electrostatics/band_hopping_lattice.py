"""Post-hoc (labelled so; not a pre-registered check): the two-site minimal-basis numbers of band_hopping.py placed
in the fcc lattice (2026-10-22).

A two-site generalized eigenproblem H = [[e, h], [h, e]], S = [[1, s], [s, 1]] has E+- = (e -+ h)/(1 -+ s) and
tau_2site = (E+ - E-)/2 = t / (1 - s^2) with t = e s - h the coupling invariant under a shift of the energy zero.
The same orbitals on the fcc cation lattice with nearest-neighbour h, s only (non-orthogonal tight binding,
ESTABLISHED) give E(k) = [e + 12 h f(k)] / [1 + 12 s f(k)], f(k) = (1/12) sum_nn exp(i k.R) = 1 - k^2 a^2 / 12 + ...,
so near Gamma E(k) = E(Gamma) + t a^2 k^2 / (1 + 12 s)^2: the orthogonal model's equivalent hopping is
tau_eff = t / (1 + 12 s)^2, m* = hbar^2 / (2 tau_eff a^2), and the band bottom lies 12 t / (1 + 12 s) below e.
"""
import json
import sys

HBAR2_OVER_ME = 7.619964
A = 2 * 2.1325

r = json.load(open(sys.argv[1]))
print(f"{'case':16s} {'tau_2site':>9s} {'s':>7s} {'t':>7s} {'tau_eff':>8s} {'m*':>6s} {'bottom-e':>9s}")
for key in ("shared_lattice", "all_lattice", "shared_vacuum", "all_vacuum"):
    if key not in r:
        continue
    v = r[key]
    tau2, s = v["tau_projected"], v["S_AB_projected"]
    t = tau2 * (1 - s * s)
    teff = t / (1 + 12 * s) ** 2
    print(f"{key:16s} {tau2:9.4f} {s:7.4f} {t:7.4f} {teff:8.4f} {HBAR2_OVER_ME / (2 * teff * A * A):6.2f} "
          f"{-12 * t / (1 + 12 * s):+9.3f}")
print(f"PBE: m* 0.322-0.330, i.e. tau_eff 0.635-0.651 eV")
