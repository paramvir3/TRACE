"""S2 (README, pre-registered 2026-11-11 19:02 IST): GFN1-xTB (DFTB+ 25.1, tblite) energy and forces on S1's cell.

The undoped wetting reference (110 atoms), displaced by the seeded 0.01 A Gaussian of probe_speed_profile.py (the
same draws: index 0 is S1's warm-up geometry, 1-5 its timed ones).  Each run is one DFTB+ process in its own
directory under runs/speed_2026-11-11/dftb/; wall time from Python, DFTB+'s own timer from its output.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_speed_dftb.py kpts=1|3 threads=4 [restart]
    (DFTB_BINARY=/path/to/dftb+ if dftb+ is not on PATH)
"""
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
from ase.io import write

_skip = __import__("torch").set_num_threads; __import__("torch").set_num_threads = lambda n: None
from eval_aumgo import references                                   # noqa: E402  (its thread call is irrelevant here)
__import__("torch").set_num_threads = _skip

DFTB = os.environ.get("DFTB_BINARY") or shutil.which("dftb+")   # the DFTB+ 25.1 executable (with tblite); $DFTB_BINARY or PATH
if not DFTB:
    sys.exit("dftb+ not found: set DFTB_BINARY to the DFTB+ executable")
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
kpts, threads, restart = int(opt.get("kpts", 1)), opt.get("threads", "4"), "restart" in sys.argv[1:]
root = f"experiments/benchmarks_4g/runs/speed_2026-11-11/dftb/k{kpts}_t{threads}{'_restart' if restart else ''}"
os.makedirs(root, exist_ok=True)
base = references()[("undoped", "wetting")].copy(); base.set_constraint()
rng = np.random.default_rng(20261111)
HSD = """Geometry = GenFormat {{
  <<< "geo.gen"
}}
Hamiltonian = xTB {{
  Method = "GFN1-xTB"
  Charge = 0.0
  SCCTolerance = 1.0e-5
  MaxSCCIterations = 300
  ReadInitialCharges = {read}
  Filling = Fermi {{
    Temperature [Kelvin] = 300.0
  }}
  KPointsAndWeights = SupercellFolding {{
    {k} 0 0
    0 {k} 0
    0 0 1
    0.0 0.0 0.0
  }}
}}
Analysis = {{
  PrintForces = Yes
}}
Options = {{
  TimingVerbosity = 1
  WriteChargesAsText = No
}}
ParserOptions = {{
  ParserVersion = 14
}}
"""
rows = []
print(f"DFTB+ GFN1-xTB; k {kpts} x {kpts} x 1; OMP_NUM_THREADS {threads}; {'restart from the previous charges' if restart else 'SCC from scratch'}")
for i in range(6):
    a = base.copy(); a.positions = a.positions + rng.normal(0.0, 0.01, a.positions.shape)
    d = f"{root}/g{i}"; os.makedirs(d, exist_ok=True)
    write(f"{d}/geo.gen", a, format="gen")
    read = "No"
    if restart and i > 0:
        shutil.copy(f"{root}/g{i - 1}/charges.bin", f"{d}/charges.bin"); read = "Yes"
    open(f"{d}/dftb_in.hsd", "w").write(HSD.format(read=read, k=kpts))
    env = dict(os.environ, OMP_NUM_THREADS=threads)
    t0 = time.perf_counter()
    p = subprocess.run([DFTB], cwd=d, env=env, stdout=open(f"{d}/stdout.txt", "w"), stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    out = open(f"{d}/stdout.txt").read(); det = open(f"{d}/detailed.out").read() if os.path.exists(f"{d}/detailed.out") else ""
    conv = "SCC converged" in det and "NOT converged" not in det and p.returncode == 0
    n_scc = len(re.findall(r"^\s+\d+\s+[-+0-9.E]+\s+[-+0-9.E]+\s+[-+0-9.E]+\s*$", out, flags=re.M))
    tot = re.findall(r"^Total\s+=\s+.*$", out, flags=re.M)
    e = re.search(r"Total energy:\s+([-0-9.E+]+) H", det)
    rows.append((wall, conv, n_scc))
    print(f"   g{i}: wall {wall:7.3f} s; exit {p.returncode}; converged {conv}; SCC iterations {n_scc}; "
          f"E {e.group(1) if e else None} H; timer: {tot[-1].strip()[:110] if tot else None}", flush=True)
good = [w for (w, c, n), j in zip(rows, range(6)) if c and j >= 1]
if good:
    print(f"== median wall over g1-g5 (converged only, {len(good)} of 5): {statistics.median(good):.3f} s [{min(good):.3f}, {max(good):.3f}]")
print("done " + time.strftime("%Y-%m-%d %H:%M IST"))
