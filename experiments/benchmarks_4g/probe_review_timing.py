"""Review check: wall time of one energy + force evaluation on a 111-atom Au2/MgO cell (4 threads, float64, CPU), the
local model against the full model, and where the full model's time goes (cProfile, cumulative)."""
import cProfile, pstats, sys, time, warnings, io
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
at = references()[("doped", "non-wetting")]
def run(calc, n=3):
    ts = []
    for k in range(n):
        a = at.copy(); a.info = {"charge": 0.0}; a.positions += 1e-4 * np.random.RandomState(k).randn(*a.positions.shape)
        a.calc = calc; calc.reset(); t = time.perf_counter(); a.get_forces(); ts.append(time.perf_counter() - t)
    return ts
for label, path in (("local", sys.argv[1]), ("full", sys.argv[2])):
    calc = TransformersACECalculator(model_path=path, device="cpu"); torch.set_num_threads(4)
    ts = run(calc)
    print(f"{label}: energy+forces, {len(at)} atoms, 4 threads: {['%.2f' % t for t in ts]} s (first call includes warm-up)", flush=True)
    if label == "full":
        pr = cProfile.Profile(); pr.enable(); run(calc, 2); pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(45)
        keep = [l for l in s.getvalue().splitlines() if ("trace_" in l or "flashace" in l or "torch" in l) and "calculator.py" not in l]
        print("\n".join(l[:190] for l in keep[:34]))
