"""C4 stage 1 (2026-11-10): trace_redox.fermi on the saved band arrays of the Au2/MgO cells (R18's arrays).

    python probe_ext_fermi_module.py ARRAYS.npz [kT]

Per cell: the polynomial route (the one a derivative can be taken through) against an eigendecomposition in numpy,
F_b(N) for N = 0 .. 4 per copy; the first derivative against a fourth-order difference of the energy and the second
against a difference of the first, along random symmetric directions of h; timings.  Loads no model."""
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "tests/redox")
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from test_redox_fermi_band import _reference_energy, _stencil, _symmetric
from trace_redox.fermi import band_free_energy, fermi_operator

data = np.load(sys.argv[1])
kT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
torch.set_num_threads(4)
print(f"kT {kT} eV")
for tag in ("doped_wetting_1x1", "doped_wetting_2x1", "doped_wetting_2x2", "doped_non-wetting_2x2", "undoped_wetting_1x1", "undoped_wetting_2x2"):
    h0 = torch.as_tensor(np.diag(data[tag + "/eps"]) - data[tag + "/tau"]); V0 = torch.as_tensor(data[tag + "/V"])
    B = int(h0.shape[0]); copies = int(tag[-3]) * int(tag[-1])
    fillings = list(range(0, 4 * copies + 1))
    reference = np.array([_reference_energy(h0.numpy(), V0.numpy(), N, kT) for N in fillings])
    t0 = time.time(); h = h0.clone().requires_grad_(True)
    F, rho = band_free_energy(h, V0, fillings, kT); t1 = time.time()
    with torch.no_grad():
        Fe, _ = band_free_energy(h0, V0, fillings, kT)
    order = fermi_operator(h0, [1], kT).order
    print(f"== {tag}: {B} sites, order {order}; max|F_b - reference| over N = 0..{fillings[-1]}: polynomial route "
          f"{np.abs(F.detach().numpy() - reference).max():.1e} eV, eigenvector route {np.abs(Fe.numpy() - reference).max():.1e} eV; "
          f"max|sum rho - N| {float((rho.sum(-1) - torch.tensor(fillings, dtype=torch.float64)).abs().max()):.1e}  ({t1 - t0:.1f} s for {len(fillings)} fillings)")
    N = 3 * copies                                                  # the doped cell's excess electrons
    X, Y = _symmetric(B, 31) / B ** 0.5, _symmetric(B, 32) / B ** 0.5

    def slope(s):
        hh = (h0 + s * Y).requires_grad_(True)
        f, _ = band_free_energy(hh, V0, [N], kT)
        (g,) = torch.autograd.grad(f[0], hh, create_graph=True)
        return (g * X).sum(), hh

    t0 = time.time(); g, hh = slope(0.0); t1 = time.time()
    (gh,) = torch.autograd.grad(g, hh); t2 = time.time()
    first = _stencil(lambda s: float(band_free_energy(h0 + s * X, V0, [N], kT)[0][0]), 1e-3)
    second = _stencil(lambda s: float(slope(s)[0]), 1e-3)
    print(f"   N = {N}: dF {float(g):+.10f} (differences {float(g) - first:+.1e});  d2F {float((gh * Y).sum()):+.10f} "
          f"(differences {float((gh * Y).sum()) - second:+.1e});  energy + gradient {t1 - t0:.1f} s, second backward {t2 - t1:.1f} s", flush=True)
