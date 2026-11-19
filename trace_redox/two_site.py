"""The two-site sector of a pair of s1 frame atoms from first principles (trace_redox SPEC 40; 2026-11-13, reviewed
2026-11-14, completed by SPEC 40.7 the same day).  Built and tested, off by default, nothing trained with it.

NEW-TO-TRACE: the covalent (no-hop) level and the hop of a pair of half-filled s frames are the pair's own
first-principles functions, so that a bond is (the frame pair's interaction, SPEC 39) - (the covalent relaxation) +
(the ensemble's hopping energy), and the learned coupling is no longer free to change sign or to make a wall.
ESTABLISHED: the two-electron, two-orbital Hubbard / valence-bond mapping (as for H2): with the ground-state energy
E_gs and the vertical sigma sigma* excitations E_T, E_S of the pair,

    E_cov(R) = E_gs + E_T          the covalent configuration (no hop), by the identity (W - U) / 2 = E_T
    U(R)     = E_S - E_T           the ionic gap,   W = sqrt(U^2 + 16 t^2)
    t(R)     = sqrt(E_T E_S) / 2   the hop

Why (measured 2026-11-12 / 13, experiments/benchmarks_4g/README.md).  In every trained model the charge module's only
Au-Au repulsion was the learned coupling shrinking under compression; it passes through zero (2.06 A in step 31,
2.13-2.16 A in the single model) and the dimer collapses (4G test forces).  First principles has one sign and a
monotone fall of t with R for Au2, Ag2 and H2, and no universal form.  SPEC 39's frozen frame term lies 3.5 / 2.2 /
1.7 / 1.0 eV above E_cov at 2.0 / 2.3 / 2.47 / 2.8 A (Au2).

The terms, for the APPLIED pairs of trace_redox/data/two_site.json (tools/electrostatics/pair_no_hop_curve.py,
pair_one_carrier_curve.py, collect_two_site.py, build_two_site_entries.py; Au-Au; Ag-Ag and H-H are tabulated and
not applied, the file says why):

  1. The covalent relaxation, one entry of the state-pair slot that SPEC 23 / 25 leave at zero:
         O_ab(f_a, f_b; R) = -s_rho(R) rho_ab(R),      rho_ab = Phi_ab(R) - E_cov,ab(R) > 0
     (Phi the SPEC 39 term), so that the pair's neutral configuration sits at E_cov.  ln rho is interpolated by a
     monotone cubic through the points and continued linearly beyond them (an exponential); s_rho switches it off
     over [RHO_ON, RHO_CUT].
  2. The hop:  tau_ab(R) = -t_ab(R) w(R),  ln t through the points and linear beyond them, w the coupling window; it
     replaces the learned coupling of the pair.  With RedoxConfig.spin_pairing (SPEC 29) the neutral pair's
     covalent-ionic element is sqrt(2) t and a one-carrier resonance t.  (First principles: the anion's own hop is
     this t within 10 % for Au, the cation's is 13-28 % larger; one hop per pair is kept, see 3.)
  3. SPEC 40.7, the pair's other three levels, each fixed by one condition on the ISOLATED pair through the package's
     own fixed terms (no learned shift; they depend on no weight, tested) and the hop of 2:
         O_ab(-1, 0) = O_ab(0, -1) = -s rho_minus(R)     so that  E[X2-] = E_gs - EA*
         O_ab(+1, 0) = O_ab(0, +1) = -s rho_plus(R)      so that  E[X2+] = E_gs + IP*
         O_ab(+1, -1) = O_ab(-1, +1) = +s lambda_ionic(R) so that  E[X2]  = E_gs
     (EA*, IP* the first-principles values on the exact atomic values).  Solved by build_two_site_entries.py and
     stored with the option set they were solved under; a model with other fixed-term options is refused.

Limits, numbered as in SPEC 40.6 (read them there; the ones a user of this file must know):
  (iii) Terms 1 and 2 ALONE correct one of the pair's four levels: gas-phase Au2 then has its electron affinity
        0.8-1.7 eV too low and its ionisation energy 1.2-1.7 eV too high (two_site.json, entries.built.
        package_energies_without_the_entries; 1.0-1.2 eV while d moves still carried the s-s hop), and no one-body learned
        shift can repair E(a+b-) + E(a0b0) - E(a+b0) - E(a-b0).  Term 3 answers it for Au-Au (the isolated pair's EA
        and IP are then first principles'); a pair without built entries cannot be applied (the file is refused).
        What remains: the conditions are the isolated pair's, transferred to a surface (limit i).
  (i) gas-phase pair functions on adsorbed and embedded pairs;  (ii) rho pair-additive, not measured;  (iv)
  homonuclear pairs only;  (v) mixed levels of theory in rho;  (vi) continuations beyond the points (the hop and rho
  as exponentials; term 3's entries end 1 A after their last point, and below the first continue on a line);
  (vii) the configuration Hamiltonian's signs are not exact for two or more carriers on three or more sites, and at
  half filling it over-binds;  (ix) cured: the hop is resolved by charge step (diabatic.StepCouplings, s moves only);
  (x) the mapping is applied to TDA roots;  (xi) the curves are C^1 at the points.
Refusals (SPEC 40.4, 40.7): another selection than 'amplitude' (in RedoxConfig); a coupling window that starts inside
a pair's tabulated distances; a model whose fixed-term options are not the ones the pair's entries were built with;
an s1 atom of an applied pair meeting its own periodic image within RHO_CUT; a covalent level above the frozen term;
a malformed table.
"""
from __future__ import annotations

import json
import math
import os

import torch
from torch import Tensor

DATA_TWO_SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "two_site.json")
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
#: the data files the isolated pair's fixed terms read (SPEC 23, 25, 28, 39): built entries record their content hashes
#: and are refused when one has changed (the second review of SPEC 40.7: alpha(Au+) x 1.5 moved the conditions by
#: 0.016-0.023 eV with no refusal)
FIXED_TERM_DATA = {name: os.path.join(_DATA, name) for name in
                   ("ion_densities.npz", "pauli_fp.npz", "ion_polarizabilities.json", "frame_pair.npz")}


def data_hashes() -> dict:
    """{file name: sha256 of its content} for FIXED_TERM_DATA."""
    import hashlib
    out = {}
    for name, path in FIXED_TERM_DATA.items():
        with open(path, "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out
#: SPEC 40.7: the pair slot's entries beyond the neutral one: -rho_minus on (-1, 0) and (0, -1), -rho_plus on (+1, 0) and
#: (0, +1), +lambda_ionic on (+1, -1) and (-1, +1)
ENTRY_NAMES = ("rho_minus", "rho_plus", "lambda_ionic")
#: every entry is switched off (C^2 quintic) over this length beyond its last knot (A)
ENTRY_TAIL = 1.0
#: the options of RedoxConfig that enter the fixed terms of an isolated s1 pair: a model must agree with the option set
#: the entries were built with (state_polarizability_elements is compared for the pair's own element only)
ENTRY_OPTIONS = ("anchor", "fragment_ensemble", "carrier_penetration", "carrier_overlap", "pauli_table", "carrier_induction",
                 "state_polarizability", "state_polarizability_stiffness", "frame_induction_damping",
                 "frame_polarizability_anchor", "frame_polarizability_scope", "bond_partition", "exclusion", "frame_pair",
                 "spin_pairing", "max_charge")
#: the covalent relaxation is switched off (C^2) between these distances (A): rho(Au2) is 0.65 eV at 3.0 A and falls by
#: e in 0.46 A beyond its last point (Ag2: 0.62 A), so it is 0.008 eV (Ag2: 0.018 eV) where the switch starts
RHO_ON, RHO_CUT = 5.0, 6.0


def _pchip_slopes(x: Tensor, y: Tensor) -> Tensor:
    """Fritsch-Carlson slopes with the shape-preserving end slopes (the builder's rule, tools/.../build_frame_pair.py)."""
    n = int(x.numel())
    m = torch.zeros_like(y)
    if n < 2:
        return m
    h, d = x[1:] - x[:-1], (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    if n == 2:
        return m + d[0]
    for k in range(1, n - 1):
        if float(d[k - 1] * d[k]) > 0.0:
            w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def end(h0, h1, d0, d1):
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if float(s * d0) <= 0.0:
            return torch.zeros_like(s)
        if float(d0 * d1) <= 0.0 and abs(float(s)) > 3.0 * abs(float(d0)):
            return 3.0 * d0
        return s
    m[0], m[-1] = end(h[0], h[1], d[0], d[1]), end(h[-1], h[-2], d[-1], d[-2])
    return m


def _log_curve(R: Tensor, x: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """exp of the C^1 cubic Hermite through (x, y) with slopes m, continued linearly (in the exponent) beyond the ends."""
    x, y, m = (v.to(R.dtype).to(R.device) for v in (x, y, m))
    k = (torch.searchsorted(x, R.detach().contiguous(), right=True) - 1).clamp(0, int(x.numel()) - 2)
    h = x[k + 1] - x[k]
    u = ((R - x[k]) / h).clamp(0.0, 1.0)
    inside = ((2 * u ** 3 - 3 * u ** 2 + 1) * y[k] + (u ** 3 - 2 * u ** 2 + u) * h * m[k]
              + (-2 * u ** 3 + 3 * u ** 2) * y[k + 1] + (u ** 3 - u ** 2) * h * m[k + 1])
    value = torch.where(R < x[0], y[0] + m[0] * (R - x[0]), inside)
    value = torch.where(R > x[-1], y[-1] + m[-1] * (R - x[-1]), value)
    return torch.exp(value)


def _switch(R: Tensor, a: float, b: float) -> Tensor:
    x = ((R - a) / (b - a)).clamp(0.0, 1.0)
    return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


class TwoSite:
    def __init__(self, path: str = DATA_TWO_SITE, allow_unbuilt: bool = False, ignore_entries: bool = False) -> None:
        """``allow_unbuilt``: accept an applied pair without built entries; ``ignore_entries``: do not read built
        entries at all.  Both are for tools/electrostatics/build_two_site_entries.py, which builds them."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it with tools/electrostatics/collect_two_site.py")
        with open(path) as fh:
            data = json.load(fh)
        self.method = data["method"]
        self.points, self.flagged, self._t, self._rho, self._applied = {}, {}, {}, {}, {}
        self.carriers, self.entries, self._trial = {}, {}, {}
        names = ("R", "E_gs", "E_T", "E_S", "E_cov", "U", "t", "weight_T", "weight_S")
        for key, p in data["pairs"].items():
            za, zb = (int(v) for v in key.split("-"))
            pair = (min(za, zb), max(za, zb))
            bad = lambda why: ValueError(f"two-site table {path}: pair {key}: {why}")
            if za != zb:
                raise bad("the mapping E_cov, U, t of SPEC 40.1 is the symmetric one; a pair of two different atoms needs "
                          "the three-level mapping (SPEC 40.6 iv), which is not built")
            self.points[pair] = {name: torch.tensor([float("nan") if v is None else v for v in p[name]], dtype=torch.float64)
                                 for name in names}
            d = self.points[pair]
            R = d["R"]
            if len({int(v.numel()) for v in d.values()}) != 1 or int(R.numel()) < 2:
                raise bad("columns of unequal length, or fewer than two points")
            if not bool((R[1:] > R[:-1]).all()):
                raise bad("the distances must increase strictly")
            if not bool(torch.isfinite(d["E_cov"]).all()):
                raise bad("E_cov must be given at every point")
            ok = torch.isfinite(d["t"])
            if int(ok.sum()) < 2:
                raise bad("fewer than two points with a hop")
            if not bool((d["t"][ok] > 0).all()):
                raise bad("the hop must be positive (it is interpolated in ln t)")
            x, y = R[ok], torch.log(d["t"][ok])
            self._t[pair] = (x, y, _pchip_slopes(x, y))
            self.flagged[pair] = list(p.get("flagged", []))
            self._applied[pair] = bool(p.get("applied", True))
            col = lambda block, name: torch.tensor([float(v) for v in p[block][name]], dtype=torch.float64)
            if "carriers" in p:                              # first-principles EA*, IP* of the pair (SPEC 40.7)
                self.carriers[pair] = {"R": col("carriers", "R"), "EA": col("carriers", "EA"), "IP": col("carriers", "IP"),
                                       "method": p["carriers"].get("method", "")}
                if not bool((self.carriers[pair]["R"][1:] > self.carriers[pair]["R"][:-1]).all()):
                    raise bad("the carriers' distances must increase strictly")
            if "entries" in p and not ignore_entries:       # built against the package's fixed terms (build_two_site_entries.py)
                e = {name: col("entries", name) for name in ("R",) + ENTRY_NAMES}
                if not bool((e["R"][1:] > e["R"][:-1]).all()) or int(e["R"].numel()) < 2:
                    raise bad("the entries' distances must increase strictly, two at least")
                e["options"] = dict(p["entries"]["options"])
                e["slopes"] = {name: _pchip_slopes(e["R"], e[name]) for name in ENTRY_NAMES}
                solved = p["entries"].get("solved_with")
                if solved is None or "gaussian_a" not in e["options"]:
                    raise bad("built entries without the record of what they were solved with ('solved_with' and the "
                              "electrostatic 'gaussian_a'): no consistency check could run on them; rebuild them with "
                              "tools/electrostatics/build_two_site_entries.py")
                if True:                                     # the table the conditions were solved with must be this one
                    def at(column, r):                       # a column's value at a distance of the entries (nan if absent)
                        hit = (R == r).nonzero().flatten()
                        return float(d[column][int(hit[0])]) if int(hit.numel()) else float("nan")

                    def carrier(name, r):
                        if pair not in self.carriers:
                            return float("nan")
                        c = self.carriers[pair]
                        hit = (c["R"] == r).nonzero().flatten()
                        return float(c[name][int(hit[0])]) if int(hit.numel()) else float("nan")

                    knots = e["R"].tolist()
                    here = {"E_gs": [at("E_gs", r) for r in knots], "E_T": [at("E_T", r) for r in knots],
                            "t": self.hop(za, zb, e["R"]).tolist(),
                            "EA": [carrier("EA", r) for r in knots], "IP": [carrier("IP", r) for r in knots]}
                    for name, values in here.items():
                        old = [float(v) for v in solved[name]]
                        same = len(old) == len(values) and all(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9
                                                               for a, b in zip(old, values))
                        if not same:
                            raise bad(f"the entries were solved with another table ({name} differs or is missing at one "
                                      "of their distances): rebuild them with tools/electrostatics/build_two_site_entries.py")
                    recorded = solved.get("data_files")
                    if recorded is None:
                        raise bad("built entries without the record of the data files their fixed terms read: rebuild them "
                                  "with tools/electrostatics/build_two_site_entries.py")
                    now = data_hashes()
                    changed = sorted(k for k in now if recorded.get(k) != now[k])
                    if changed:
                        raise bad(f"the data file(s) {changed} behind the pair's fixed terms are not the ones the entries "
                                  "were solved with: rebuild them with tools/electrostatics/build_two_site_entries.py")
                    e["rho"] = torch.tensor([float(v) for v in solved["rho"]], dtype=torch.float64)
                self.entries[pair] = e
        self.pairs = sorted(self.points)
        unbuilt = [p for p in self.pairs if self._applied[p] and p not in self.entries]
        if unbuilt and not allow_unbuilt:
            raise ValueError(f"two-site table {path}: the pairs {unbuilt} are marked applied and have no built entries "
                             "(SPEC 40.7): with the neutral entry alone a pair's electron affinity and ionisation energy "
                             "are about 1 eV off; build them with tools/electrostatics/build_two_site_entries.py")
        #: the pairs the module uses (the file's "applied"; see tools/electrostatics/collect_two_site.py for the rule)
        self.applied = [p for p in self.pairs if self._applied[p]]
        self.elements = sorted({z for p in self.applied for z in p})

    def has(self, Za: int, Zb: int) -> bool:
        return (min(int(Za), int(Zb)), max(int(Za), int(Zb))) in self.points

    def is_applied(self, Za: int, Zb: int) -> bool:
        """Whether the module's two hooks act on this element pair (tabulated AND applied)."""
        return (min(int(Za), int(Zb)), max(int(Za), int(Zb))) in self.applied

    def hop(self, Za: int, Zb: int, R: Tensor) -> Tensor:
        """t_ab(R) > 0 (eV): through the first-principles points in ln t, an exponential beyond them (no window)."""
        return _log_curve(R, *self._t[(min(int(Za), int(Zb)), max(int(Za), int(Zb)))])

    def _relaxation_knots(self, pair):
        """(R_k, ln rho_k, slopes): rho = Phi_frozen - E_cov at the first-principles distances, with SPEC 39's table.
        Kept per table object: if the process's frame-pair table is replaced, the knots are computed again."""
        from .frame_pair import frame_pair
        table = frame_pair()
        kept = self._rho.get(pair)
        if kept is None or kept[0] is not table:
            R = self.points[pair]["R"]
            es, pauli = table.pair_parts(pair[0], pair[1], R)
            rho = es + pauli - self.points[pair]["E_cov"]
            if not bool((rho > 0).all()):
                raise ValueError(f"the covalent level of the pair {pair} lies above SPEC 39's frozen term at some point: "
                                 f"rho = {rho.tolist()}")
            y = torch.log(rho)
            kept = (table, (R, y, _pchip_slopes(R, y)))
            self._rho[pair] = kept
        return kept[1]

    def relaxation(self, Za: int, Zb: int, R: Tensor) -> Tensor:
        """rho_ab(R) > 0 (eV) with its switch: what the pair's frame configuration lies below SPEC 39's frozen term."""
        pair = (min(int(Za), int(Zb)), max(int(Za), int(Zb)))
        return _log_curve(R, *self._relaxation_knots(pair)) * _switch(R, RHO_ON, RHO_CUT)

    def entry_values(self, Za: int, Zb: int, R: Tensor):
        """``(rho_minus, rho_plus, lambda_ionic)(R)``; None when the pair has no entries.  Each is the C^1
        shape-preserving cubic through its knots (in the value: lambda_ionic changes sign).  Below the first knot it
        continues with its end slope.  Beyond the last knot it continues on its line and is switched off over
        ENTRY_TAIL: the entries complete a resonance whose hop the coupling's window closes over the same ångström
        (SPEC 40.4 requires the window to start at the last knot or beyond), and nothing is tabulated further out."""
        pair = (min(int(Za), int(Zb)), max(int(Za), int(Zb)))
        if pair in self._trial:                                  # the builder's solve: constants, no interpolation
            return tuple(torch.full_like(R, float(v)) for v in self._trial[pair])
        e = self.entries.get(pair)
        if e is None:
            return None
        x = e["R"].to(R.dtype).to(R.device)
        k = (torch.searchsorted(x, R.detach().contiguous(), right=True) - 1).clamp(0, int(x.numel()) - 2)
        h = x[k + 1] - x[k]
        u = ((R - x[k]) / h).clamp(0.0, 1.0)
        out = []
        for name in ENTRY_NAMES:
            y, m = e[name].to(R.dtype).to(R.device), e["slopes"][name].to(R.dtype).to(R.device)
            inside = ((2 * u ** 3 - 3 * u ** 2 + 1) * y[k] + (u ** 3 - 2 * u ** 2 + u) * h * m[k]
                      + (-2 * u ** 3 + 3 * u ** 2) * y[k + 1] + (u ** 3 - u ** 2) * h * m[k + 1])
            value = torch.where(R < x[0], y[0] + m[0] * (R - x[0]), inside)
            # beyond the last knot: its line, switched off (C^2 quintic) over ENTRY_TAIL, so that every entry ends where
            # the coupling's window closes the hop it completes (no first-principles point supports a longer tail)
            tail = (y[-1] + m[-1] * (R - x[-1]).clamp_min(0.0)) * _switch(R, float(x[-1]), float(x[-1]) + ENTRY_TAIL)
            out.append(torch.where(R > x[-1], tail, value))
        return tuple(out)

    def check_options(self, pair, config, gaussian_a=None) -> None:
        """Refuse a model whose fixed terms are not the ones the pair's entries were built against: the options of
        ENTRY_OPTIONS, the pair's element in the state-polarizability list, the Gaussian exponent factor of the
        electrostatic settings (``gaussian_a``; it sets the dipoles' widths) and the frame table rho was taken from."""
        e = self.entries.get(pair)
        if e is None or config is None:
            return
        built = e["options"]
        differ = {k: (built.get(k), getattr(config, k, None)) for k in ENTRY_OPTIONS if built.get(k) != getattr(config, k, None)}
        listed = getattr(config, "state_polarizability_elements", None)
        here = listed is None or all(int(z) in {int(v) for v in listed} for z in pair)      # None: every element
        if bool(built.get("state_polarizability_element", True)) != bool(here):
            differ["state_polarizability_elements"] = (built.get("state_polarizability_element"), listed)
        if gaussian_a is None:
            raise ValueError(f"two_site: the entries of the pair {pair} need the model's electrostatic gaussian_a to be "
                             "checked against (SPEC 40.7)")
        if abs(float(built["gaussian_a"]) - float(gaussian_a)) > 1e-12:
            differ["gaussian_a"] = (built["gaussian_a"], float(gaussian_a))
        stored = e["rho"]                                    # rho of (40.2) as it was when the entries were solved
        if True:
            now = self.relaxation(pair[0], pair[1], e["R"])
            if float((now - stored).abs().max()) > 1e-8:
                differ["frame table (rho at the knots)"] = (stored.tolist(), now.tolist())
        if differ:
            raise ValueError(f"two_site: the entries of the pair {pair} were built against other fixed terms than this "
                             f"model's (option: built, model): {differ}; rebuild them with "
                             "tools/electrostatics/build_two_site_entries.py for this option set (SPEC 40.7)")

    # ---------------------------------------------------------------------------------------------- for the module

    def replace_hops(self, T: Tensor, species: Tensor, edge_index: Tensor, edge_len: Tensor, r_cut: float) -> Tensor:
        """``T`` [n, n] with the entries of every tabulated element pair replaced by -t(R) w(R), summed over the pair's
        edges (periodic images add, as in the learned head); every other entry is the head's."""
        from .heads import ANCHOR_SWITCH_WIDTH, window_switch
        z = torch.as_tensor(species).to(torch.long).to(T.device)
        src, dst = edge_index[0], edge_index[1]
        n = int(T.shape[0])
        mask = torch.zeros(n, n, dtype=torch.bool, device=T.device)
        table = torch.zeros_like(T)
        for (a, b) in self.applied:
            on_pair = ((z.unsqueeze(1) == a) & (z.unsqueeze(0) == b)) | ((z.unsqueeze(1) == b) & (z.unsqueeze(0) == a))
            on_pair = on_pair & ~torch.eye(n, dtype=torch.bool, device=T.device)
            if not bool(on_pair.any()):
                continue
            last = float(self._t[(a, b)][0][-1])
            if (a, b) in self.entries:                       # the entries were solved with the whole hop at their knots
                last = max(last, float(self.entries[(a, b)]["R"][-1]))
            if float(r_cut) - ANCHOR_SWITCH_WIDTH < last - 1e-12:
                raise ValueError(f"two_site: the coupling's window starts at {float(r_cut) - ANCHOR_SWITCH_WIDTH:.3f} A "
                                 f"(coupling_cutoff {float(r_cut)} A), inside the tabulated distances of the pair "
                                 f"{(a, b)} (to {last} A): it would cut the first-principles hop at its own data, and the "
                                 "pair's entries were solved with the whole hop there (SPEC 40.4, 40.7)")
            mask = mask | on_pair
            sel = (((z[src] == a) & (z[dst] == b)) | ((z[src] == b) & (z[dst] == a))) & (src != dst) & (edge_len < r_cut)
            if bool(sel.any()):
                Rs = edge_len[sel].to(T.dtype)
                value = -self.hop(a, b, Rs) * window_switch(Rs, r_cut, ANCHOR_SWITCH_WIDTH)
                table = table.index_put((src[sel], dst[sel]), value, accumulate=True)
        mask = mask & ~torch.eye(n, dtype=torch.bool, device=T.device)
        return torch.where(mask, 0.5 * (table + table.transpose(0, 1)), T)

    def resolve_hops(self, T: Tensor, species: Tensor, edge_index: Tensor, edge_len: Tensor, r_cut: float):
        """``(T', steps)``: :meth:`replace_hops`, and the record that makes the table's hop the coupling of the pair's
        s moves only (``trace_redox.diabatic.StepCouplings``: every other move of an applied pair, a d electron
        leaving an Au+ for example, keeps the learned coupling ``T`` handed in).  ``steps`` is None when the structure
        holds no applied pair."""
        from .diabatic import StepCouplings
        z = torch.as_tensor(species).to(torch.long).to(T.device)
        n = int(T.shape[0])
        mask = torch.zeros(n, n, dtype=torch.bool, device=T.device)
        for (a, b) in self.applied:
            mask = mask | ((z.unsqueeze(1) == a) & (z.unsqueeze(0) == b)) | ((z.unsqueeze(1) == b) & (z.unsqueeze(0) == a))
        mask = mask & ~torch.eye(n, dtype=torch.bool, device=T.device)
        new = self.replace_hops(T, species, edge_index, edge_len, r_cut)
        return new, (StepCouplings(mask=mask, other=T) if bool(mask.any()) else None)

    def relaxation_pairs(self, species: Tensor, states: Tensor, frames: Tensor, pos: Tensor, cell, pbc, config=None,
                         gaussian_a=None):
        """``(i, j, Xi [P, S, S])`` for the state-pair slot of every applied pair within RHO_CUT; None when there is none.
        Always -rho(R) at the entry (0, 0) (SPEC 40.2).  Where the pair has built entries (SPEC 40.7): -rho_minus at
        (-1, 0) and (0, -1), -rho_plus at (+1, 0) and (0, +1), +lambda_ionic at (+1, -1) and (-1, +1); ``config`` (the
        model's RedoxConfig) is then checked against the option set the entries were built with."""
        from .penetration import pairs_within, self_images_within
        z = torch.as_tensor(species).to(torch.long).to(pos.device)
        fr = torch.as_tensor(frames).to(torch.long).to(pos.device)
        listed = torch.zeros_like(z, dtype=torch.bool)
        for e in self.elements:
            listed = listed | (z == e)
        atoms = listed.nonzero().flatten()
        if int(atoms.numel()) == 0:
            return None
        if bool((fr[atoms] != 0).any()):
            raise ValueError("two_site: the tabulated pairs are pairs of NEUTRAL s1 frames; an atom of "
                             f"{sorted(set(z[atoms][fr[atoms] != 0].tolist()))} has another frame here (SPEC 40.4)")
        # own images are lattice vectors: one atom decides for all (also when the cell holds a single s1 atom)
        if int(self_images_within(pos[atoms[:1]], cell, pbc, RHO_CUT)[0].numel()) > 0:
            raise ValueError(f"two_site: an s1 atom meets its own periodic image within {RHO_CUT} A; the hop to an own "
                             "image does not exist in the cell's ensemble, so the relaxation of that pair would stand "
                             "alone, and the same crystal would depend on the cell chosen (SPEC 40.4)")
        if int(atoms.numel()) < 2:
            return None
        a_loc, b_loc, R = pairs_within(pos[atoms], cell, pbc, RHO_CUT)[:3]     # pairs among the s1 atoms only
        if int(a_loc.numel()) == 0:
            return None
        i, j = atoms[a_loc], atoms[b_loc]
        st = torch.round(states).to(torch.long).to(pos.device)
        S = int(st.numel())
        on = lambda n: (st == n).to(pos.dtype)                              # [S]: the row of charge n (zeros if absent)
        outer = lambda a, b: on(a).view(S, 1) * on(b).view(1, S)
        neutral = outer(0, 0)
        minus, plus = outer(-1, 0) + outer(0, -1), outer(1, 0) + outer(0, 1)
        ionic = outer(1, -1) + outer(-1, 1)
        keep = torch.zeros_like(i, dtype=torch.bool)
        Xi = torch.zeros(int(i.numel()), S, S, dtype=pos.dtype, device=pos.device)
        for (a, b) in self.applied:
            sel = ((z[i] == a) & (z[j] == b)) | ((z[i] == b) & (z[j] == a))
            if not bool(sel.any()):
                continue
            keep = keep | sel
            rows = sel.nonzero().flatten()
            block = -self.relaxation(a, b, R[sel]).view(-1, 1, 1) * neutral
            extra = self.entry_values(a, b, R[sel])
            if extra is not None:
                if (a, b) not in self._trial:
                    if config is None:
                        raise ValueError(f"two_site: the pair {(a, b)} has built entries, which are only valid for the fixed "
                                         "terms they were solved against: pass the model's settings (config=RedoxConfig, "
                                         "gaussian_a) so that they can be checked (SPEC 40.7)")
                    self.check_options((a, b), config, gaussian_a)
                rho_minus, rho_plus, lam = extra
                block = (block - rho_minus.view(-1, 1, 1) * minus - rho_plus.view(-1, 1, 1) * plus
                         + lam.view(-1, 1, 1) * ionic)
            Xi = Xi.index_put((rows,), block)
        if not bool(keep.any()):
            return None
        return i[keep], j[keep], Xi[keep]


_TWO_SITE: dict = {}


def two_site(path: str = DATA_TWO_SITE) -> TwoSite:
    if path not in _TWO_SITE:
        _TWO_SITE[path] = TwoSite(path)
    return _TWO_SITE[path]
