"""fp64 vs fp32 throughput of the three kernels the electrostatic solve is made of.

1. dense matvec K v (n x n), the inner loop of the dense CG;
2. Gaussian Coulomb kernel assembly erf(r/s)/r over n^2 pairs, plus its backward
   (what forces cost on the dense route);
3. 3-D real FFT round trip of a PME mesh.

Prints median wall time per call (CUDA-synchronised) and the fp64/fp32 ratio.
Also reports nvidia-smi utilisation at start, because a shared card makes the
absolute numbers indicative only.
"""
import statistics
import subprocess
import time

import torch

dev = torch.device("cuda")
util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, GPU busy at start: {util}")


def bench(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def row(label, f64, f32):
    print(f"{label:38s} fp64 {f64 * 1e3:9.3f} ms   fp32 {f32 * 1e3:9.3f} ms   ratio {f64 / f32:6.1f}x", flush=True)


for n in (1000, 4000, 8000):
    out = {}
    for dtype in (torch.float64, torch.float32):
        K = torch.randn(n, n, device=dev, dtype=dtype)
        v = torch.randn(n, device=dev, dtype=dtype)
        out[dtype] = bench(lambda: K @ v)
    row(f"matvec n={n}", out[torch.float64], out[torch.float32])

for n in (1000, 3000):
    out = {}
    for dtype in (torch.float64, torch.float32):
        pos = (torch.rand(n, 3, device=dev, dtype=dtype) * 20.0).requires_grad_(True)
        sigma = torch.full((n,), 0.8, device=dev, dtype=dtype)

        def kernel():
            r = torch.cdist(pos, pos) + torch.eye(n, device=dev, dtype=dtype)
            s = torch.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
            J = torch.erf(r / s) / r
            e = J.sum()
            torch.autograd.grad(e, pos)

        out[dtype] = bench(kernel, repeats=10)
    row(f"erf kernel + backward n={n}", out[torch.float64], out[torch.float32])

for m in (48, 84, 128):
    out = {}
    for dtype in (torch.float64, torch.float32):
        grid = torch.randn(m, m, m, device=dev, dtype=dtype)
        out[dtype] = bench(lambda: torch.fft.irfftn(torch.fft.rfftn(grid), s=(m, m, m)))
    row(f"rfft/irfft mesh {m}^3", out[torch.float64], out[torch.float32])
