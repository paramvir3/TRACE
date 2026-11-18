"""Frozen-weights probe (2026-11-07): the ensemble's dynamic charge transfer in bulk MgO, i.e. the Born-charge
contribution of the integer configurations beyond the frame, Z*_ens(O) = d P_ens / d r_O with P_ens = sum_n p_n d_n,
p_n the configuration populations of the diabatic ensemble (Boltzmann over the eigenstates of H, projected) and d_n the
configuration's dipole relative to the frame (a hole on O_i: +e r_i; an electron on Mg_j: -e r_j; a band electron: -e
times the cell centroid, position independent).  Measured with the learned couplings and with the Harrison anchor
(SPEC 31) switched on post hoc, on the 64-atom cell at the PBE lattice constant, by central differences of one O along z.
Usage: python experiments/benchmarks_phonons/probe_ensemble_born.py MODEL.pt [--a 4.25566] [--delta 0.01]"""
import argparse, sys, warnings
sys.path.insert(0, "."); warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
from ase.build import bulk
from flashace.calculator import TransformersACECalculator
from trace_redox.constants import formal_frame


def ensemble_dipole(mod, numbers, positions):
    last = mod._ensemble_last; H = last["H"].detach(); cfgs = last["configs"]; kT = float(last["kT"]); n = len(numbers)
    f = formal_frame(torch.as_tensor(numbers)).to(cfgs.dtype)
    w, V = torch.linalg.eigh(H); boltz = torch.softmax(-(w - w.min()) / kT, dim=0)
    p = (V ** 2 * boltz).sum(dim=1)                                   # configuration populations
    idx = int(((cfgs[:, :n] == f).all(1) & (cfgs[:, n:] == 0).all(1)).nonzero()[0])
    dq = (cfgs[:, :n] - f).to(torch.float64)                          # [M, n] charge change per atom vs the frame
    pos = torch.as_tensor(positions, dtype=torch.float64)
    d = dq @ pos                                                      # [M, 3]: + e r for a hole (dq = +1), - e r for an electron
    P = (p.unsqueeze(1) * d).sum(0)
    q = f + (p.unsqueeze(1) * dq).sum(0)                              # ensemble-averaged integer charge per atom
    return P, float(p[idx]), int(cfgs.shape[0]), q, float((H - torch.diag(torch.diagonal(H))).abs().max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--a", type=float, default=4.255656884758435); ap.add_argument("--delta", type=float, default=0.01)
    args = ap.parse_args()
    calc = TransformersACECalculator(model_path=args.model, device="cpu"); mod = calc.model.electrostatics
    cell = bulk("MgO", "rocksalt", a=args.a, cubic=True).repeat((2, 2, 2)); cell.info = {"charge": 0.0}
    iO = int(np.nonzero(cell.numbers == 8)[0][0])
    print(f"=== {args.model.split('/')[-2]}, {len(cell)} atoms, a = {args.a:.5f}, O atom {iO} displaced by ±{args.delta} A along z")
    for setting in ("learned", "harrison-anchor"):
        if setting == "harrison-anchor":
            if not hasattr(mod.coupling, "anchor"):
                print("  (checkpoint predates SPEC 31: no anchor attribute; setting it)")
            mod.coupling.anchor = "harrison"
        rows = {}
        for sgn in (+1.0, 0.0, -1.0):
            a = cell.copy(); a.info = {"charge": 0.0}; a.positions[iO, 2] += sgn * args.delta; a.calc = calc; calc.reset()
            try:
                E = a.get_potential_energy()
            except Exception as err:
                print(f"  {setting}: forward refused at {sgn:+.0f}: {type(err).__name__}: {str(err)[:120]}"); rows = None; break
            P, p_frame, M, q, cmax = ensemble_dipole(mod, a.numbers, a.positions)
            rows[sgn] = (P, p_frame, M, q, cmax, E)
        if rows is None:
            mod.coupling.anchor = None; continue
        P0, pf, M, q, cmax, E0 = rows[0.0]
        Zz = (rows[1.0][0] - rows[-1.0][0]) / (2 * args.delta)
        print(f"  {setting}: {M} configurations, frame population {pf:.6f}, largest coupling {cmax:.3f} eV, ensemble charge "
              f"q(O) {float(q[cell.numbers == 8].mean()):+.5f}, q(Mg) {float(q[cell.numbers == 12].mean()):+.5f} e; "
              f"Z*_ens(O) = dP_ens/dr_O,z = {np.round(Zz.numpy(), 5).tolist()} e (zz component is the dynamic transfer; the frame's own -2 is not included)")
        mod.coupling.anchor = None


if __name__ == "__main__":
    main()
