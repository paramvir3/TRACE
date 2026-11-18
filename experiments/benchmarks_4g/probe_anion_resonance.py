# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_anion_resonance.py CKPT [D]
"""Why the model's Au2- parallel polarisability is what it is: the two-state resonance.

For the isolated Au2- (one extra electron, configurations Au-Au0 and Au0Au-), alpha_CT = K_E mu_eff^2 / (4 |tau|) for a
symmetric pair, with mu_eff the change of the TOTAL dipole (free charges plus the bound moments they induce) between
the two configurations and tau their coupling.  The probe reads, under the full first-principles stack (SPEC 22, 23,
25, 26, 28), the configurations, their energies, the coupling, the bare and the screened dipole difference, and
compares the two-state alpha_CT with the finite-field alpha_par - alpha_perp of the same model."""
import dataclasses, sys, warnings
import numpy as np, torch
warnings.simplefilter("ignore"); sys.path.insert(0, "."); torch.set_default_dtype(torch.float64)
from ase import Atoms
from flashace.calculator import TransformersACECalculator
import trace_redox.module as TM
K_E = 14.399645
ck = sys.argv[1]; d = float(sys.argv[2]) if len(sys.argv) > 2 else 2.47
calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
mod = calc.model.electrostatics; mod.exclusion = None
mod.redox = dataclasses.replace(mod.redox, bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim",
                                exclusion=False, carrier_induction=True, state_polarizability=True)
store = {}
orig_elim, orig_sol = TM.eliminate_bound_sector, TM.RedoxElectrostatics._diabatic_solution
def elim_wrap(*a, **k):
    e = orig_elim(*a, **k); store["elim"] = e; return e
def sol_wrap(self, e_hat, states, b, W, L, K, Q, T=None, **kw):
    store.update(T=None if T is None else T.detach(), states=states)
    out = orig_sol(self, e_hat, states, b, W, L, K, Q, T, **kw)
    return out
TM.eliminate_bound_sector = elim_wrap; TM.RedoxElectrostatics._diabatic_solution = sol_wrap
import trace_redox.diabatic as D
orig_h = TM.switched_hamiltonian
def h_wrap(E, configs, K, T, a_off, a_on, kT):
    H = orig_h(E, configs, K, T, a_off, a_on, kT); store.update(H=H.detach(), configs=configs.detach(), E=E.detach()); return H
TM.switched_hamiltonian = h_wrap
orig_d = TM.diabatic_hamiltonian
def d_wrap(E, configs, K, T):
    H = orig_d(E, configs, K, T); store.update(H=H.detach(), configs=configs.detach(), E=E.detach()); return H
TM.diabatic_hamiltonian = d_wrap
def energy(field, q=-1.0):
    a = Atoms("Au2", positions=[[0, 0, 0], [0, 0, d]]); a.info = {"charge": q, "external_field": list(field)}
    a.calc = calc; calc.reset(); return a.get_potential_energy()
e0 = energy([0, 0, 0])
H, cfg, E = store["H"], store["configs"], store["E"]
print(f"Au2- at {d} A: {cfg.shape[0]} configurations")
for k in range(cfg.shape[0]):
    print(f"  config {cfg[k].tolist()}  E - E_min {float(E[k] - E.min()):+.4f} eV")
off = (H - torch.diag(torch.diagonal(H))).abs()
print(f"  largest coupling |H_ij| {float(off.max()):.4f} eV")
# dipole of each configuration: free charges q_i z_i plus the bound moments they induce
elim = store["elim"]; pos = torch.tensor([[0, 0, 0], [0, 0, d]], dtype=torch.float64)
mus = []
for k in range(cfg.shape[0]):
    q = cfg[k].to(torch.float64)[:2]
    y = elim.bound_moments(q)
    t, mu_hat, _ = elim.split_moments(y)
    qb = elim.assembly.charges(t, torch.zeros(2, dtype=torch.float64))
    mu_b = elim.assembly.dipoles(mu_hat)
    mus.append(float(((q + qb) * pos[:, 2]).sum() + mu_b[:, 2].sum()))
pair = [int(k) for k in torch.argsort(E)[:2]]
dmu_bare = float(d)                                   # one electron moved across the bond
dmu = abs(mus[pair[0]] - mus[pair[1]])
tau = float(off[pair[0], pair[1]])
print(f"  dipole difference between the two lowest: bare {dmu_bare:.3f}, screened (free + bound) {dmu:.3f} e A; coupling {tau:.4f} eV")
print(f"  two-state alpha_CT = K_E mu_eff^2 / (4 tau) = {K_E * dmu ** 2 / (4 * tau):.2f} A^3 (bare mu: {K_E * dmu_bare ** 2 / (4 * tau):.2f})")
h = 0.01
ap = -(energy([0, 0, h]) - 2 * e0 + energy([0, 0, -h])) / h ** 2 * K_E
an = -(energy([h, 0, 0]) - 2 * e0 + energy([-h, 0, 0])) / h ** 2 * K_E
print(f"  finite field: alpha_par {ap:.2f}, alpha_perp {an:.2f}, difference {ap - an:.2f} A^3")
