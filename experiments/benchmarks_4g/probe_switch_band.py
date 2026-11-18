# Diagnostic probe (2026-10-19; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_switch_band.py CKPT [...]
"""How much of a structure's free energy rides on SPEC 21's switch band: the coupled pairs with a_off < e < a_on, and
F(switched) - F(band pairs fully on), on the four T5 reference geometries."""
import sys, warnings
import torch
warnings.simplefilter("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
torch.set_default_dtype(torch.float64)
import trace_redox.module as TM
from trace_redox.diabatic import amplitude_switch, canonical_free_energy, configuration_importance
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
store = {}
orig = TM.switched_hamiltonian
def wrap(E, configs, K, T, a_off, a_on, kT, electrons=None):
    H = orig(E, configs, K, T, a_off, a_on, kT, electrons=electrons)
    _, e, (rows, cols, donor, acceptor, sign) = configuration_importance(E.detach(), configs, K, T.detach(), kT)
    sigma = amplitude_switch(e, a_off, a_on)
    band = (e > a_off) & (e < a_on)
    H_on = torch.diag(E.detach()).index_put((rows, cols), sign.to(E.dtype) * torch.where(band, torch.ones_like(sigma), sigma) * T.detach()[donor, acceptor], accumulate=True)
    store.update(n_pairs=int(rows.numel()) // 2, n_band=int(band.sum()) // 2, dF=float(canonical_free_energy(H.detach(), kT) - canonical_free_energy(H_on, kT)),
                 e_band=sorted(set(round(float(x), 3) for x in e[band].tolist()))[:6], n_cfg=int(configs.shape[0]))
    return H
TM.switched_hamiltonian = wrap
refs = references()
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    for key, a0 in refs.items():
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset(); store.clear(); a.get_potential_energy()
        print(f"{ck.split('/')[-2][-12:]} {key[0]:8s} {key[1]:12s}: {store.get('n_cfg')} configs, {store.get('n_pairs')} coupled pairs, "
              f"{store.get('n_band')} in the band (e {store.get('e_band')}); F(switched) - F(band on) {1e3 * store.get('dF', 0.0):+.1f} meV", flush=True)
