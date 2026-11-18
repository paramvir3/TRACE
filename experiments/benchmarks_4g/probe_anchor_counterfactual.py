"""Frozen-weight counterfactual of SPEC 31 (Harrison-anchored couplings) on a trained model (2026-10-27; a probe).

Usage (repo root): python experiments/benchmarks_4g/probe_anchor_counterfactual.py MODEL.pt [...]

Every SPEC 31 pair (all but the frame's cation-anion pairs) takes Harrison's universal value (factor 1: the anchored
head at its initialisation), or only the element pairs of a subset; every other pair keeps its trained coupling under
SPEC 27's bound.  (A first version anchored only the pairs with an open-shell frame, the Au pairs: measured, it let
seed 42's opened O -> Au channel delocalise its hole through the 28 eV O-O couplings.)  The four T5 references
(unrelaxed) give wet_u, wet_d, T5 as trained and as anchored, with the configuration counts (the SPEC 21 closure's
cap is 20000 here).  Direction and size only: the other weights were trained with the learned couplings.
"""
import dataclasses
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.heads import TransferCoupling, anchored_pairs, harrison_bound  # noqa: E402
from eval_aumgo import references  # noqa: E402

DFT = (928.9, -2.7, -931.6)


SUBSETS = {"all": None, "Au-O+O-O": {(79, 8), (8, 8)}, "Au-O": {(79, 8)}, "O-O": {(8, 8)}, "Au-Au": {(79, 79)}}


def anchored_values(trained, anchor_head, pairs=None):
    """Anchored pairs (optionally only the element pairs listed) take the anchored head's value; the rest keep the
    trained coupling under SPEC 27's bound."""
    def values(invariants, edge_index, edge_len, species=None):
        raw = trained.raw_values(invariants, edge_index, edge_len)
        a = anchor_head.edge_values(invariants, edge_index, edge_len, species=species)
        src, dst = edge_index[0], edge_index[1]
        anchored = anchored_pairs(species, src, dst)
        if pairs is not None:
            zs, zd = species[src], species[dst]
            keep = torch.zeros_like(anchored)
            for a_, b_ in pairs:
                keep = keep | ((zs == a_) & (zd == b_)) | ((zs == b_) & (zd == a_))
            anchored = anchored & keep
        return torch.where(anchored, a, harrison_bound(raw, edge_len))
    return values


def energies(calc, refs):
    mod = calc.model.electrostatics
    out = {}
    for key, at in refs.items():
        a = at.copy()
        a.info = dict(at.info)
        a.calc = calc
        calc.reset()
        out[key] = (float(a.get_potential_energy()), int(mod._ensemble_last["configs"].shape[0]))
    return out


def main():
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        name = os.path.basename(os.path.dirname(model))
        rows = {}
        head = TransferCoupling(mod.coupling.pair[0].in_features // 2, r_cut=mod.coupling.r_cut,
                                n_radial=mod.coupling.n_radial, anchor="harrison")
        with torch.no_grad():
            head.pair[-1].weight.zero_()
            head.pair[-1].bias.zero_()
        trained_values = mod.coupling.edge_values
        for label in ["as trained"] + [f"anchored {k} (factor 1)" for k in SUBSETS]:
            if label == "as trained":
                mod.coupling.edge_values = trained_values
            else:
                mod.coupling.edge_values = anchored_values(mod.coupling, head, SUBSETS[label.split()[1]])
            E = energies(calc, refs)
            wu = E[("undoped", "wetting")][0] - E[("undoped", "non-wetting")][0]
            wd = E[("doped", "wetting")][0] - E[("doped", "non-wetting")][0]
            rows[label] = (wu, wd)
            n = {k[0][0] + k[1][0]: v[1] for k, v in E.items()}
            print(f"=== {name} {label}: wet_u {wu * 1e3:+.1f}, wet_d {wd * 1e3:+.1f}, T5 {(wd - wu) * 1e3:+.1f} meV "
                  f"(DFT {DFT[0]:+.1f} / {DFT[1]:+.1f} / {DFT[2]:+.1f}); configurations {n}", flush=True)


if __name__ == "__main__":
    main()
