"""The isolated gold pair through the package's fixed terms beyond the last tabulated distance (SPEC 40.7, limit d;
2026-11-15).  A probe: nothing trained, nothing written outside its output.

    PYTHONPATH=../.. python probe_two_site_beyond_points.py [step=0.01]

Au2, Au2- and Au2+ from 2.8 to 4.6 A: the energy against the separated limit (two atoms, or the ion and the atom, with
the package's exact atomic EA and IP) and -dE/dR by central differences on the grid; the largest force beyond 3.0 A
and where it is.  The reference option set of tests/redox/test_redox_two_site.py.
"""
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
sys.path.insert(0, "../../tests/redox")
import test_redox_two_site as TT  # noqa: E402

OPTS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
step = float(OPTS.get("step", 0.01))
grid = np.round(np.arange(2.8, 4.6 + 1e-9, step), 6)
limit = {0.0: 0.0, -1.0: -2.30861, 1.0: 9.225554}
print(f"grid 2.8-4.6 A, step {step} A; energies in eV against the separated limit, forces -dE/dR in eV/A")
for q, name in ((0.0, "Au2"), (-1.0, "Au2-"), (1.0, "Au2+")):
    E = np.array([TT._dimer(float(R), charge=q, fixed=True, **TT.ON)[0] for R in grid]) - limit[q]
    F = -np.gradient(E, grid)
    beyond = grid > 3.0 + 1e-9
    k = int(np.argmax(np.abs(np.where(beyond, F, 0.0))))
    at = {r: float(E[int(np.argmin(np.abs(grid - r)))]) for r in (3.0, 3.3, 3.6, 4.0, 4.5)}
    print(f"{name:5s}: E at 3.0 / 3.3 / 3.6 / 4.0 / 4.5 A: " + " / ".join(f"{at[r]:+.3f}" for r in (3.0, 3.3, 3.6, 4.0, 4.5))
          + f"; force at 3.0 A {F[int(np.argmin(np.abs(grid - 3.0)))]:+.2f}; largest |force| beyond 3.0 A {abs(F[k]):.2f} at {grid[k]:.2f} A", flush=True)
