# TRACE-Electrostatics: benchmark results

This page summarises every scored benchmark of TRACE-Electrostatics and TRACE-Redox: what was measured, what the
pre-registered criteria said, and what failed. It adds no new measurement. Every number is copied from a score file or
from one of the two lab notebooks, and each table names its source.

- **Au₂/MgO(001), the 4G-HDNNP "Table II" test** (section 1): every trained Au₂/MgO step, both seeds where both
  were trained.
- **The two featured models**, step 34b and step 31 (section 2), with all their secondary scores and their known
  defects.
- **Bulk MgO phonons against PBE** (section 3), criteria P0–P6.
- **The 4G cluster sets** Ag₃±, NaCl and the carbon chain (section 4), including PFN's training sizes.
- **What failed, and what was never measured** (section 5).

No trained model passes both the Table II criteria and the phonon criteria (section 2.3).

## How to read the sources

| abbreviation | file |
|---|---|
| `R4G:n` | line n of [`experiments/benchmarks_4g/README.md`](../../experiments/benchmarks_4g/README.md), the 4G lab notebook |
| `RPH:n` | line n of [`experiments/benchmarks_phonons/README.md`](../../experiments/benchmarks_phonons/README.md), the phonon lab notebook |
| `SPEC:n` | line n of [`trace_redox/SPEC.md`](../../trace_redox/SPEC.md) |
| `runs/<run>/scores/<file>` | a score file under `experiments/benchmarks_4g/` |
| `PH runs/<dir>/mgo_phonons.log` | a phonon log under `experiments/benchmarks_phonons/` |

- **The notebooks are dated and are kept as written.** Where they were corrected later, the correction sits next to
  the original text. A line number points to the entry; read the entry for its context.
- **Score files are not in git.** `runs/` is gitignored. The checkpoints released as GitHub Release assets are listed
  with their sha256 in [`MODELS.md`](MODELS.md). For those models,
  [`experiments/benchmarks_4g/score_model.sh`](../../experiments/benchmarks_4g/score_model.sh) rewrites the score
  files ([`REPRODUCE.md`](REPRODUCE.md)). For every other step the checkpoint is not released, and the numbers below
  are as recorded.
- **"Registered verdict"** is the verdict the notebook wrote against criteria fixed before the run. It is quoted, not
  re-judged. A step with no scored entry in the notebook says so.
- **"Seed"** is the training seed (42 or 43). The model is the best checkpoint by validation loss unless a row says
  otherwise.

---

## 1. Au₂/MgO(001): Table II of the 4G-HDNNP paper

### 1.1 The test and the references

Two Au₂ geometries on a 110-atom MgO(001) slab: the dimer lying flat ("wetting") and standing upright
("non-wetting"). The slab is either undoped or doped with three Al donors more than 10 Å from the gold. The test
quantities, in meV per cell:

- **wet_u** = E(wetting) − E(non-wetting), undoped slab;
- **wet_d** = the same, doped slab;
- **T5** = wet_d − wet_u, the doping-induced change. Only non-local charge transfer can produce it, because the nearest
  Al is 10.71 Å from the Au `R4G:87`.

`eval_aumgo.py` relaxes the Au positions with the substrate fixed, on the mean substrate of each doping
`R4G:82`. It also prints the test-set
RMSE on 200 undoped and 200 doped structures.

| | wet_u | wet_d | T5 (% of DFT) | relaxed non-wetting Au–O, doped / undoped (Å) |
|---|---|---|---|---|
| DFT (PBE, FHI-aims "light") | **+928.9** | **−2.7** | **−931.6** | 2.332 / 2.190 |
| 4G-HDNNP (Ko et al., Table II) | +975.4 | −41.0 | −1016.4 (109.1 %) | 2.342 / 2.177 |
| 2G-HDNNP | +375.0 | +375.0 | 0 | 2.256 / 2.256 |

Source: the reference line that `eval_aumgo.py` prints into every `runs/<run>/scores/T5_wetting.txt`, and
`R4G:92`.

**Caveats of the comparison.**
- **Training data.** 4G-HDNNP trained on about 4,500 Au₂/MgO structures. PFN's NequIP runs used 4,000. Every TRACE
  model here trained on **300** (76 validation, 400 test) `R4G:279`.
  Steps 33 and 34b add 146 bulk-MgO PBE force cells from another data set (section 2.1).
- **4G's force error.** The 4G pooled force RMSE of 0.067 eV/Å on Au₂/MgO is a reference line digitised from a figure
  in PFN's blog. It is not a number from the 4G paper `R4G:283`. The
  4G paper's test RMSEs are in its supplementary information, which was not read
  `R4G:258`.
- **4G's energy error.** The notebook quotes a 4G-HDNNP test energy RMSE of 0.219 meV/atom on this set
  `R4G:4747`. **No source for that number is recorded**, and it is
  not used as a comparator here.
- **One cell, one k-point.** Table II is scored on the 1×1 benchmark cell. The reference DFT sampled 3 × 3 × 1
  k-points `R4G:6460`. Table II on a 2×2 supercell was
  never scored for any step (section 5).
- **Charges are not comparable in size.** TRACE-Redox moves an integer electron onto Au₂ (T2 ≈ −1.00 e); Hirshfeld
  moves −0.266 e. The two are different partitions `R4G:4761`.
- **Seeds.** From step 9 onward most steps ran seeds 42 and 43. The two seeds often differ by more than the gap to
  DFT (rows 9, 13, 21, 25 in table 1.2). Earlier steps have one seed.

### 1.2 Every trained Au₂/MgO step

Wetting energies in meV per cell; test E RMSE in meV/atom; test F RMSE in eV/Å; "u / d" = undoped / doped. T5 is
wet_d − wet_u. Source of the numbers: `runs/<run>/scores/T5_wetting.txt` (the T5 protocol: Au relaxed), except the
rows marked *json*, which come from `runs/<run>/eval_aumgo.json` (same script, run before the score folder existed).
The run names are in table 1.3.

| step | change (one per step unless stated) | seed | wet_u | wet_d | T5 (% DFT) | E u / d | F u / d | registered verdict |
|---|---|---|---|---|---|---|---|---|
| DFT | | | +928.9 | −2.7 | −931.6 | | | |
| 4G-HDNNP | | | +975.4 | −41.0 | −1016.4 (109.1 %) | | | |
| R1 local *json* | TRACE without electrostatics, 25 epochs | 42 | +3.7 | +2.8 | −0.9 (0.1 %) | 5.349 / 1.285 | 0.1314 / 0.1709 | fails: no doping dependence `R4G:105` |
| R1 elec *json* | + SQE electrostatics, 25 epochs | 42 | +192.7 | +195.0 | +2.3 (−0.2 %) | 6.442 / 1.545 | 0.1151 / 0.1219 | fails, as above |
| elecq *json* | elec + Hirshfeld loss, 25 epochs | 42 | +166.7 | +166.5 | −0.2 (0.0 %) | 4.390 / 2.027 | 0.1150 / 0.1314 | T2 shift has the wrong sign `R4G:950` |
| elec *json* | elec, 40 epochs (transfer baseline) | 42 | +254.7 | +260.0 | +5.3 (−0.6 %) | 3.965 / 2.281 | 0.1104 / 0.1049 | comparator; no transfer `R4G:758` |
| elecCS *json* | elec with declared charge states | 42 | +367.8 | +168.6 | −199.2 (21.4 %) | 3.634 / 1.723 | 0.1053 / 0.0985 | no notebook entry |
| redox *json* | TRACE-Redox, energies and forces | 42 | +175.4 | +247.7 | +72.3 (−7.8 %) | 3.537 / 3.001 | 0.0851 / 0.0897 | FAIL T1, T2, T4, T5, P1 `R4G:866` |
| redoxq *json* | TRACE-Redox + Hirshfeld loss | 42 | +87.5 | +176.2 | +88.7 (−9.5 %) | 3.917 / 1.537 | 0.0829 / 0.0906 | FAIL T2, T4, T5, P1 `R4G:914` |
| 1 | physical bond hardness (two-level law) | 42 | +88.9 | +100.7 | +11.8 (−1.3 %) | 3.764 / 1.190 | 0.0882 / 0.0943 | PASS (S1 ε∞ ≤ 30, S2 accuracy); T2–T5 were reported, not criteria, and fail `R4G:1434` |
| 2 | formal oxidation states | 42 | +617.1 | −28.2 | −645.3 (69.3 %) | 1.434 / 0.606 | 0.0602 / 0.0816 | passes numerically; ruled an over-screening artifact (transfer 0/6 at ε∞ = 3) `R4G:1602` |
| 3 | + training electronic temperature | 42 | +823.6 | −147.5 | −971.1 (104.2 %) | 0.752 / 0.666 | 0.0631 / 0.0753 | same artifact (ε∞ 6.50) `R4G:1606` |
| 4 | + measured ε∞(MgO) prior | 42 | +510.7 | +79.6 | −431.1 (46.3 %) | 2.170 / 1.159 | 0.0749 / 0.0847 | F1, F5 FAIL `R4G:1609` |
| 5 | + one-electron couplings | 42 | +482.2 | −23.2 | −505.4 (54.3 %) | 1.944 / 0.678 | 0.0590 / 0.0754 | PASS (E1, F1–F5) `R4G:1650` |
| E-weight | step 5 + energy weight 1000, 10-epoch fine-tune | 42 | +582.7 | +46.5 | −536.2 (57.6 %) | 1.485 / 0.941 | 0.0573 / 0.0768 | H1, H3 FAIL `R4G:1873` |
| 6 | + Au atom polarisability prior | 42 | +592.8 | +55.6 | −537.2 (57.7 %) | 1.938 / 1.376 | 0.1256 / 0.0931 | F1 FAIL `R4G:1745` |
| 7 | core + carrier free charges | 42 | +306.3 | +226.3 | −80.0 (8.6 %) | 2.962 / 2.393 | 0.1117 / 0.0955 | FAIL `R4G:1906` |
| 8 | free-charge widths by frame | 42 | +548.2 | +61.5 | −486.7 (52.2 %) | 2.480 / 1.148 | 0.0836 / 0.0748 | F1, F5′ FAIL `R4G:1958` |
| 9 | exclusion by closed shells (SPEC 18) | 42 | +874.9 | −6.3 | −881.2 (94.6 %) | 0.936 / 0.976 | 0.0944 / 0.0690 | PASS `R4G:2045` |
| 9 | (seed replicate) | 43 | +854.3 | −313.9 | −1168.2 (125.4 %) | 0.931 / 1.317 | 0.0854 / 0.0865 | S1, S3 FAIL `R4G:2185` |
| 10 | relaxed exclusion | 42 | +1278.1 | −509.9 | −1788.0 (191.9 %) | 2.031 / 2.047 | 0.1003 / 0.0841 | FAIL `R4G:2128` |
| 11 | Pauli exclusion between carrier units | 42 | +747.5 | −18.5 | −766.0 (82.2 %) | 1.435 / 1.168 | 0.0899 / 0.0700 | P2, P3 FAIL `R4G:2314` |
| 11 | | 43 | +875.3 | −323.3 | −1198.6 (128.7 %) | 1.206 / 1.488 | 0.0907 / 0.0858 | P1, P2, X1 FAIL `R4G:2291` |
| 12 | band unit for the donor electrons (SPEC 20) | 42 | +918.6 | −41.5 | −960.1 (103.1 %) | 0.847 / 0.703 | 0.1052 / 0.0764 | B5 FAIL (forces, by about 1 %) `R4G:2368` |
| 12 | | 43 | +636.1 | −23.7 | −659.8 (70.8 %) | 2.294 / 1.793 | 0.1140 / 0.0743 | B2, B5 FAIL `R4G:2400` |
| 13 | amplitude selection (SPEC 21) | 42 | +506.3 | −37.8 | −544.1 (58.4 %) | 1.842 / 0.431 | 0.0929 / 0.0705 | A3, A4, A5 FAIL `R4G:2522` |
| 13 | | 43 | +1288.7 | −80.0 | −1368.7 (146.9 %) | 2.177 / 1.149 | 0.0735 / 0.0818 | as above |
| 13S | step 13 + spin pairing (SPEC 29) | 42 | +501.8 | −92.6 | −594.4 (63.8 %) | 2.171 / 1.218 | 0.0905 / 0.0727 | no scored entry in the notebook |
| 13S | | 43 | −28.3 | +35.2 | +63.5 (−6.8 %) | 2.834 / 1.270 | 0.0826 / 0.0941 | no scored entry |
| 14 | one sector per bond (SPEC 22) | 42 | +249.6 | −473.2 | −722.8 (77.6 %) | 3.170 / 1.990 | 0.1632 / 0.0844 | no scored entry |
| 14 | | 43 | +323.1 | −347.1 | −670.2 (71.9 %) | 2.968 / 1.822 | 0.0877 / 0.0899 | no scored entry |
| 16 | carrier induction (SPEC 26) | 43 | +575.9 | −222.7 | −798.6 (85.7 %) | 1.630 / 0.878 | 0.1425 / 0.0946 | no scored entry; seed 42 refused in epoch 4 `R4G:4562` |
| 17 | state-dependent polarisabilities (SPEC 28) | 42 | +480.8 | −362.7 | −843.5 (90.5 %) | 2.378 / 2.301 | 0.1547 / 0.0929 | S1 PASS; S4, B3 FAIL `R4G:4561` |
| 17 | | 43 | +660.9 | −243.2 | −904.1 (97.0 %) | 1.287 / 1.341 | 0.0985 / 0.0900 | as above |
| 18 | + charged-dimer field prior | 42 | −76.3 | +22.0 | +98.3 (−10.6 %) | 4.457 / 2.138 | 0.1868 / 0.2069 | G1, G3, G5 FAIL `R4G:4461` |
| 18 | | 43 | −310.4 | −315.0 | −4.6 (0.5 %) | 5.433 / 2.512 | 0.1736 / 0.2247 | as above |
| 19 | band hopping anchored to PBE's band mass | 42 | +312.2 | −391.6 | −703.8 (75.5 %) | 2.601 / 1.647 | 0.0666 / 0.0867 | Q2, Q4 FAIL; Q5 fails on s42 `R4G:4422` |
| 19 | | 43 | +346.8 | −264.4 | −611.2 (65.6 %) | 2.735 / 1.165 | 0.1263 / 0.0910 | Q2, Q4 FAIL; Q3 fails on s43 |
| 20 | Pauli table from first principles (SPEC 25) | 42 | +536.3 | −453.1 | −989.4 (106.2 %) | 2.493 / 2.728 | 0.0689 / 0.0886 | R1, R4 PASS; R2, R3 FAIL `R4G:4339` |
| 20 | | 43 | +753.8 | −222.1 | −975.9 (104.8 %) | 0.905 / 1.272 | 0.0726 / 0.1091 | as above |
| 21 | Harrison-anchored couplings (SPEC 31) | 42 | — | — | — (relaxation refused) | 2.457 / 8.102 | 0.1713 / 0.1791 | FAIL `R4G:4170` |
| 21 | | 43 | −1718.5 | +204.4 | +1922.9 (−206.4 %) | 2.297 / 1.291 | 0.1068 / 0.1041 | FAIL |
| 22 | downfolded O 2p ↔ Au 6s bond (SPEC 32), warm from step 20 | 42 | +685.9 | −386.0 | −1071.9 (115.1 %) | 0.823 / 1.548 | 0.0764 / 0.0838 | FAIL (E1 and E5 needed on both seeds) `R4G:3626` |
| 22 | | 43 | +220.7 | −420.0 | −640.7 (68.8 %) | 4.354 / 1.785 | 0.0976 / 0.1063 | as above |
| 23 | + the coupling's measured radial decay | 42 | +742.2 | −386.3 | −1128.5 (121.1 %) | 0.763 / 1.563 | 0.0751 / 0.0841 | FAIL (E4 both; E5 on s43) `R4G:3852` |
| 23 | | 43 | +371.5 | −282.0 | −653.5 (70.1 %) | 4.403 / 1.181 | 0.0896 / 0.1054 | as above |
| 24 | + the frame pair's Pauli wall | 42 | −102.5 | −349.9 | −247.4 (26.6 %) | 0.795 / 1.475 | 0.0850 / 0.0843 | FAIL on both seeds `R4G:3829` |
| 24 | | 43 | +323.9 | −315.3 | −639.2 (68.6 %) | 4.585 / 1.353 | 0.0896 / 0.1065 | as above |
| 25 | bond-oriented Pauli repulsion (SPEC 33), cold start from step 20's options | 42 | +1461.0 | +1162.1 | −298.9 (32.1 %) | 2.027 / 5.922 | 0.1380 / 0.3281 | no scored entry in the notebook |
| 25 | | 43 | +1067.4 | −79.3 | −1146.7 (123.1 %) | 0.848 / 0.585 | 0.0744 / 0.1135 | no scored entry |
| 26 | + SPEC 33's wall (cold start) | 42 | +612.5 † | −422.0 † | −1034.5 (111 %) † | 2.30 / 2.65 | 0.062 / 0.085 | B3 FAIL on both seeds, so FAIL (PASS = B1–B3 and C4) `R4G:4725` |
| 26 | | 43 | +810.6 | −198.2 | −1008.8 (108.3 %) | 1.399 / 2.274 | 0.0768 / 0.1056 | as above |
| 28b | SPEC 35 frame-polarizability anchor + SPEC 34 damping, on step 20 | 42 | +174.4 | +296.5 | +122.1 (−13.1 %) | 5.464 / 3.933 | 0.1627 / 0.1250 | fails its 4G criteria `RPH:299` |
| 28b | | 43 | +718.3 | +103.7 | −614.6 (66.0 %) | 1.475 / 2.010 | 0.1554 / 0.1110 | as above |
| 29 | step 26 s43 + energy weight 1000, 10-epoch warm start | 43 | +906.8 | −98.1 | −1004.9 (107.9 %) | 0.424 / 0.665 | 0.0778 / 0.1049 | FAIL (W1) `R4G:4816` |
| 29 control | the same with energy weight 10 | 43 | +822.9 | −190.2 | −1013.1 (108.7 %) | 1.952 / 1.598 | 0.0762 / 0.1039 | control of step 29 |
| **31** | step 26 + energy weight 1000, 20-epoch warm start | 42 | +851.0 | −177.4 | −1028.4 (110.4 %) | 0.590 / 0.937 | 0.0614 / 0.0867 | FAIL: s42 fails Y1, Y3 `R4G:4895` |
| **31** | | 43 | **+920.4** | −75.7 | −996.1 (106.9 %) | **0.469 / 0.621** | 0.0776 / 0.1045 | s43 passes Y1–Y5 |
| 32a | step 26 recipe + coupling bound (SPEC 27) | 42 | +606.7 | −406.3 | −1013.0 (108.7 %) | 2.336 / 2.580 | 0.0600 / 0.0866 | stage of step 32 `R4G:7227` |
| 32b | step 31 recipe, bound throughout | 42 | +828.1 | −186.8 | −1014.9 (108.9 %) | 0.635 / 0.866 | 0.0600 / 0.0877 | FAIL (Z1, Z4; Z3 unscored, seed 43 dropped) `R4G:7247` |
| 33 | single model, every term on, + bulk PBE cells | 42 | +768.4 | −9.2 | −777.6 (83.5 %) | 1.339 / 0.877 | 0.1930 / 0.1043 | FAIL (S2, S3, S5 on both seeds) `R4G:8042` |
| 33 | | 43 | +789.9 | +92.7 | −697.2 (74.8 %) | 1.396 / 0.880 | 0.2211 / 0.1563 | as above |
| **34b** | step 33 + SPEC 39, 29, 40/40.7, cold start, band factor 0.55 | 42 | +610.4 | +92.3 | −518.1 (55.6 %) | 1.353 / 0.989 | 0.0845 / 0.1242 | FAIL (S2, S5, S8 on both seeds) `R4G:9223` |
| **34b** | | 43 | +729.7 | −27.7 | −757.4 (81.3 %) | 1.142 / 0.671 | 0.0616 / 0.1112 | as above |

† Step 26 seed 42: the T5-protocol relaxation crashed (`cholesky_inverse` singular in the induction block at a relaxed
geometry). The three marked numbers are at the reference geometries, and E / F are the notebook's two-decimal values,
all from `R4G:4725`. `runs/<run>/scores/T5_wetting.txt` for this run holds only
the traceback.

Notes on the table.
- **Steps 22–24** also have Table II at the unrelaxed reference geometries in the notebook. Their T5 differs from the T5
  protocol's by as much as 1079.1 meV (step 24 s42: −1326.5 at the references, −247.4 relaxed; `R4G:3835`).
- **Step 21 seed 42** refused the T5 relaxation (`ConfigurationSearchError`, more than 512 configurations); its E / F
  are from the same score file.
- **Step 12 seed 42 is not a featured model.** Its T5 (−960.1, 103.1 %, 28.5 meV from DFT) is the closest to DFT of
  any row in the table; its seed-43 replicate gave 70.8 %.

### 1.3 Run names

All under `experiments/benchmarks_4g/runs/`. `F` = `AuMgO_formal_kT_eps_cpl_alpha_`, `P` = `F` +
`exband_amp_part_pen_ind_spol_pfp_`, `_sNN` = seed.

| step | run |
|---|---|
| R1 local / R1 elec / elecq / elec / elecCS | `AuMgO_base_s42`, `AuMgO_elec_s42`, `AuMgO_elecq_s42`, `AuMgO_elec_r5_s42`, `AuMgO_elecCS_r5_s42` |
| redox / redoxq | `AuMgO_redox_r5_s42`, `AuMgO_redoxq_r5_s42_nanexact` |
| 1–5 | `AuMgO_redoxq_bondlaw_r5_s42`, `AuMgO_formal_bondlaw_r5_s42`, `AuMgO_formal_kT_bondlaw_r5_s42`, `AuMgO_formal_kT_eps_bondlaw_r5_s42`, `AuMgO_formal_kT_eps_cpl_bondlaw_r5_s42` |
| E-weight | `AuMgO_step5_Ew1000_ft` |
| 6–13 | `F` + `bondlaw`, `cc`, `fw`, `ex`, `exr`, `expair`, `exband`, `exband_amp` (each `_bondlaw_r5_sNN`) |
| 13S, 14, 16, 17, 18, 19 | `F` + `exband_amp_spin`, `exband_amp_part`, `exband_amp_part_pen_ind`, `exband_amp_part_pen_ind_spol`, `…_spol_fpriorq`, `…_spol_bprior` |
| 20–26 | `P` + (nothing), `anc`, `dfct`, `dfct_dec`, `dfct_dec_wall`, `bop`, `bopw` |
| 28b, 29, 29 control, 31 | `P` + `fpa_fid`, `bopw_ew1000`, `bopw_ew10`, `bopw_e1kw` |
| 32a, 32b | `P` + `bopw_cbh`, `bopw_cbh_e1kw` |
| 33 (stage A, B) | `AuMgO_single_all_bondlaw_r5_sNN`, `AuMgO_single_all_e1kw_bondlaw_r5_sNN` |
| 34b (stage A, B) | `AuMgO_twosite_bf055_bondlaw_r5_sNN`, `AuMgO_twosite_bf055_e1kw_bondlaw_r5_sNN` |

### 1.4 Steps with no Table II number

| step | what happened | source |
|---|---|---|
| 13F | the ε∞ prior refused its structure (5 configurations in the window); stopped at epoch 7 (s42) and 15 (s43); unscored | `R4G:2552` |
| 13FC | not run; its slot went to step 15 | `R4G:3184` |
| 15 | carrier penetration (SPEC 23): refused at the second forward call on both seeds | `R4G:4562` |
| 16 s42 | refused at epoch 4 | as above |
| 27 | built and pre-registered, never launched | `R4G:4737` |
| 28 | withdrawn before launch, replaced by 28a and 28b | `R4G:4686` |
| 28a | cancelled in the queue, never trained | `RPH:290` |
| 30 | the energy weight on a cold start: cancelled at epoch 21 / 20 of 40, forces 3–6× behind step 26; unscored | `R4G:4856` |
| 32 s43 | dropped after a GPU out-of-memory error | `R4G:7257` |
| 34 | stage A stopped in epoch 2 on both seeds (`ConfigurationSearchError`, 529 > 512 configurations on bulk MgO); relaunched as 34b with one change, `band_initial_factor: 0.55` | `R4G:8893` |
| 33 and 34b stage A | trained, used as warm starts for stage B; Table II not scored on stage A | `runs/<run>/scores/` holds phonons only |

**Mid-training reads, not verdicts.** Three Mac copies of step 33's stage A (seed 43) were read at epochs 14, 25 and
39: T5 −563.5, −557.6 and −427.4 meV, undoped test F 0.1793, 0.1596 and 0.1696 eV/Å
(`runs/AuMgO_single_all_mac_s43_epoch{14,25,39}/scores/T5_wetting.txt`;
`R4G:7761`).

**Frozen copies are not trained models.** `runs/AuMgO_frozen_step20_fid_s42` is step 20's weights with SPEC 34's flag
switched on and no training. Its undoped test F RMSE is 87.95 eV/Å (`scores/T5_wetting.txt`). It is excluded from
table 1.2.

---

## 2. The two featured models

Both are released as checkpoints ([`MODELS.md`](MODELS.md)). Neither passed its own pre-registration on both seeds.

| | DFT | 4G-HDNNP | step 31 s42 | step 31 s43 | step 34b s42 | step 34b s43 |
|---|---|---|---|---|---|---|
| wet_u (meV) | +928.9 | +975.4 | +851.0 | **+920.4** | +610.4 | +729.7 |
| wet_d (meV) | −2.7 | −41.0 | −177.4 | −75.7 | +92.3 | −27.7 |
| T5 (meV, % of DFT) | −931.6 | −1016.4 (109.1 %) | −1028.4 (110.4 %) | −996.1 (106.9 %) | −518.1 (55.6 %) | −757.4 (81.3 %) |
| test F u / d (eV/Å) | | | 0.0614 / 0.0867 | 0.0776 / 0.1045 | 0.0845 / 0.1242 | 0.0616 / 0.1112 |
| test E u / d (meV/atom) | | | 0.590 / 0.937 | 0.469 / 0.621 | 1.353 / 0.989 | 1.142 / 0.671 |
| Au–Au bond force at 2.089 Å, test 167 (eV/Å) | +15.25 | | not measured | +6.18 ‡ | +14.25 | +15.82 |
| MgO phonons, 2×2×2 / 4×4×4 (of 7) | | | — / 3 | — / 2 | 6 / 7 | 6 / 7 |
| Z*(Mg), PBE 1.992 | | | 2.402 | 2.534 | 1.950 | 1.961 |
| LO (cm⁻¹), PBE 681.3 | | | 807.9 | 851.5 | 669.3 | 667.4 |

‡ From a different probe than step 34b's S7 (section 2.2). Sources: the sections below.

### 2.1 Step 34b

**What it is.** The single model of step 33 (every term of TRACE-Redox on), with SPEC 39 (the frame pair's own
interaction), SPEC 29 (spin pairing) and SPEC 40 / 40.7 (the two-site sector of two s¹ frames, Au–Au only), from a cold
start, with the band unit's hopping factor started at 0.55 (SPEC 20).
- **Training data:** 436 structures = the 300 Au₂/MgO training slabs + 136 bulk-MgO PBE force cells from Togo's
  phono3py set (76 + 10 validation), with a per-structure force weight of 122.4 on the PBE cells; an ε∞ = 3.235 prior
  cell (PBE) at weight 0.05; the Au atom α prior at weight 0.3. The phonon reference's own 2 × 512-atom fc2 cells are
  held out.
- **Recipe:** stage A 40 epochs from scratch (energy weight 10, lr 1e-3 → 5e-5); stage B 20 epochs from stage A's best
  checkpoint (energy weight 1000, lr 1e-4 → 5e-6). Best checkpoints: stage A epoch 39 (both seeds); stage B epoch 15
  (s42) and 18 (s43). About 20 min per epoch in stage B, both seeds sharing one GPU.
- Run names: `AuMgO_twosite_bf055_bondlaw_r5_sNN` (A) and `AuMgO_twosite_bf055_e1kw_bondlaw_r5_sNN` (B). Config:
  [`experiments/benchmarks_4g/configs/`](../../experiments/benchmarks_4g/configs/).

**Pre-registered criteria and scores** (PASS = S1–S5, S7 and S8 on both seeds;
`R4G:9223`):

| criterion | target | s42 | s43 |
|---|---|---|---|
| S1 | both stages finish | pass | pass |
| S2 | wet_u +928.9 ± 100; wet_d −2.7 ± 150; T5 within ±15 % of −931.6 | **FAIL**: +610.4; +92.3 (pass); −518.1 (55.6 %) | **FAIL**: +729.7; −27.7 (pass); −757.4 (81.3 %) |
| S3 | test F ≤ 0.10 undoped, ≤ 0.13 doped (eV/Å) | pass: 0.0845 / 0.1242 | pass: 0.0616 / 0.1112 |
| S4 | phonons on 2×2×2: P0, P1, P2, P4 pass and ≥ 6 / 7 | pass: 6 / 7 (P5 29.5) | pass: 6 / 7 (P5 29.9) |
| S5 | crystal factor g in 0.85–0.95; prior cell's ε∞ within 0.05 of 3.235 | **FAIL**: g 0.9515, ε∞ 3.333 | **FAIL**: g 0.9396, ε∞ 3.300 |
| S7 | Au–Au bond force on test 167 repulsive and monotone over 2.00–2.40 Å, within ±30 % of +15.25 eV/Å at 2.089 Å | pass: +14.25 (−7 %) | pass: +15.82 (+4 %) |
| S8 | undoped test F without the five dimers below 2.18 Å ≤ 0.060 eV/Å | **FAIL**: 0.0844 | **FAIL**: 0.0614 |

S5's ε∞ is read at the training's last evaluation, as the scorer does; at the best checkpoints it is 3.3337 (s42) and
3.3001 (s43) `R4G:9258`.

**Secondary scores** (`runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_sNN/scores/`):

| quantity | s42 | s43 | reference | file |
|---|---|---|---|---|
| S7 scan, bond force at 2.00 / 2.05 / 2.15 / 2.20 / 2.30 / 2.40 Å (eV/Å) | +22.47 / +17.37 / +10.57 / +7.99 / +4.49 / +2.22 | +23.83 / +19.02 / +11.75 / +8.82 / +4.85 / +2.37 | | `S7_S8.txt` |
| share of the undoped squared force error carried by the five dimers below 2.18 Å | 2.7 % | 3.1 % | step 33: 92.2 % / 94.2 % | `S7_S8.txt`; `R4G:9246` |
| test class residual, flat − upright, undoped (meV) | −273.5 ± 8.1 | −95.3 ± 15.3 | 0 | `H1_class_residuals.txt` |
| test class residual, flat − upright, doped (meV) | +165.4 ± 7.9 | −46.1 ± 10.6 | 0 | `H1_class_residuals.txt` |
| T2, q(Au₂) shift doped − undoped (e) | −1.0000 ± 0.0000 | −0.9966 ± 0.0019 | Hirshfeld −0.2656 ± 0.0083 | `T1_T3_transfer.txt` |
| per-structure Pearson r(q_Au₂, Hirshfeld) | +0.848 | +0.849 | | `T1_T3_transfer.txt` |
| relaxed non-wetting Au–O, doped / undoped (Å) | 2.297 / 2.195 | 2.430 / 2.165 | DFT 2.332 / 2.190 | `T5_wetting.txt` |
| gas-phase Au₂ EA error, model − reference, 2.30–2.80 Å (eV), as trained | +0.197 to +0.431 | −0.034 to −0.001 | | `S6_au2_ea_ip.txt` |
| gas-phase Au₂ IP error, model − reference, 2.30–2.80 Å (eV), as trained | +0.562 to +0.593 | +0.070 to +0.081 | | `S6_au2_ea_ip.txt` |
| the same with the learned parts off (fixed terms only, both seeds): EA / IP error (eV) | −0.159 to −0.135 / −0.040 to +0.012 | same | | `S6_au2_ea_ip.txt` |
| largest \|τ\| / B on the 8-atom bulk MgO prior cell | 0.012 | 0.008 | ≤ 1 | `Z1_Z2_couplings.txt` |
| lowest state below the reference, four benchmark cells (eV) | −0.81 to −1.51 | −0.95 to −2.53 | | `Z1_Z2_couplings.txt` |

The EA / IP columns of `S6_au2_ea_ip.txt` labelled "error" are model minus reference; the notebook first misread them
as values and corrected this `R4G:9280`.

**Phonons** (section 3): 6 / 7 on 2×2×2 and 7 / 7 on 4×4×4 for stage B on both seeds.

**Known defects of step 34b.**
- **Table II is worse than step 33's and step 31's:** wet_u +610.4 / +729.7 against DFT +928.9 (S2). The error is a
  class offset: the flat dimer is too stable against the upright one on the undoped slab (−273.5 / −95.3 meV).
- **S8:** the undoped forces without the five compressed dimers got worse than step 33's (0.0844 / 0.0614 against
  0.0544 / 0.0540 eV/Å). Not attributed; the candidates (the cold start, SPEC 39's change of the frozen Au–O
  interaction, seed spread) are untested `R4G:9253`.
- **S5:** ε∞ of the prior cell ends 0.065–0.098 above PBE's 3.235.
- **The Au–Au pair is switched off between 3.0 and 4.0 Å.** SPEC 40.7's entries end at 3.0 Å; beyond it the hop's
  window and the entries' switch remove the isolated pair's binding, from −1.65 eV at 3.0 Å to −0.20 eV at 3.6 Å for
  Au₂, with a force of up to 3.09 eV/Å at 3.32 Å. One 4G structure lies beyond 3.0 Å (3.056 Å)
  `SPEC:4474`.
- **Three or more s¹ atoms over-bind at half filling.** On exact small clusters, the model's neutral sector of three or
  more s¹ atoms lies below the exact ground state of any spin (chain −0.11 / −0.23 t, triangle −0.61 / −0.73 t, square
  −0.50 / −0.48 t, tetrahedron −1.08 / −1.19 t, at U = t / 4t) `SPEC:4356`.
  Only Au–Au carries SPEC 40.7's entries; Au₂ on MgO has two s¹ atoms, so the benchmark does not test this.
- **T2:** the model moves an integer electron (−1.00 e); Hirshfeld moves −0.27 e. Different partitions; not evidence
  either way.
- **Not measured:** Table II on the 2×2 supercell.

### 2.2 Step 31

**What it is.** Step 26 (TRACE-Redox through SPEC 33's bond-oriented Pauli repulsion and its wall), warm-started from
step 26's final weights and trained 20 more epochs with the energy weight raised from 10 to 1000 (lr 1e-4 → 5e-6).
- **Training data:** the 300 Au₂/MgO slabs only, the ε∞ = 2.95 (experiment) prior cell at a = 4.2650 Å, and the Au α
  prior. No bulk force cells.
- **Machine:** CPU (a Mac), both seeds together. Best checkpoints: epoch 20 (s42), 17 (s43).
- Run names: `P` + `bopw_e1kw_bondlaw_r5_sNN` (table 1.3); its parent step 26 is `P` + `bopw_bondlaw_r5_sNN`.

**Pre-registered criteria and scores** (PASS = Y1–Y5 on both seeds;
`R4G:4895`):

| | Y1 training-set undoped class offset (≤ 30 meV) | Y2 wet_u | Y3 wet_d | Y4 T5 | Y5 test F u / d (eV/Å) |
|---|---|---|---|---|---|
| s42 | −73.3 **FAIL** | +851.0 | −177.4 **FAIL** (by 25) | −1028.4 (110.4 %) | 0.0614 / 0.0867 |
| s43 | +7.4 | +920.4 | −75.7 | −996.1 (106.9 %) | 0.0776 / 0.1045 |

**Secondary scores** (`runs/P…bopw_e1kw_bondlaw_r5_sNN/scores/` unless stated):

| quantity | s42 | s43 | reference | file |
|---|---|---|---|---|
| test E u / d (meV/atom) | 0.590 / 0.937 | 0.469 / 0.621 | | `T5_wetting.txt` |
| training-set class offset, flat − upright, undoped / doped (meV) | −73.3 ± 7.1 / −124.1 ± 6.9 | +7.4 ± 6.7 / −65.2 ± 6.4 | 0 | `H1_class_residuals_train.txt` |
| Y6, doped − undoped training offset (meV) | −50.8 | −72.6 | predicted −30 to −120 | `R4G:4911` |
| test class residual, undoped / doped (meV) | −64.5 ± 5.5 / −142.5 ± 7.4 | +15.3 ± 6.3 / −69.9 ± 8.1 | 0 | `H1_class_residuals.txt` |
| T2, q(Au₂) shift (e) | −1.0000 ± 0.0000 | −0.9988 ± 0.0003 | Hirshfeld −0.2656 ± 0.0083 | `T1_T3_transfer.txt` |
| per-structure r with Hirshfeld | +0.848 | +0.849 | | `T1_T3_transfer.txt` |
| T6, ε∞ of bulk MgO | 3.284 | 2.934 | experiment 2.95; window 2.5–3.5 | `T6_eps.txt` |
| relaxed non-wetting Au–O, doped / undoped (Å) | 2.301 / 2.171 | 2.306 / 2.168 | DFT 2.332 / 2.190 | `T5_wetting.txt` |
| S7 (the Au–Au bond-force criterion of steps 34 / 34b) | not scored | not scored | | |
| S8 (the undoped force without the compressed dimers) | not scored | not scored | | |

Against 4G-HDNNP, step 31 s43's wet_u (8.5 meV from DFT, against 4G's 46.5) and T5 (106.9 % against 109.1 %) are
closer to DFT; its wet_d is not (73.0 meV from DFT, against 4G's 38.3) `R4G:4907`.
4G trained on about 15 times as many Au₂/MgO structures (section 1.1).

**Phonons** (`score_phonons.py --p5-supercell 4`; `PH runs/step31_P…bopw_e1kw_bondlaw_r5_sNN{,_sc4}/mgo_phonons.log`;
`RPH:1141`):

| | Z*(Mg) | ε∞ | TO (cm⁻¹) | LO (cm⁻¹) | P5 no NAC, 4×4×4 (2×2×2) | P6 NAC, 2×2×2 (4×4×4) | passes |
|---|---|---|---|---|---|---|---|
| PBE | 1.992 | 3.235 | 373.3 | 681.3 | | | |
| step 31 s42 | 2.402 (+21 %) | 3.025 | 384.3 (+2.9 %) | 807.9 | 22.0 (29.6) | 28.9 (28.9) | 3 / 7 (P0, P2, P3) |
| step 31 s43 | 2.534 (+27 %) | 3.002 | 398.7 (+6.8 %) | 851.5 | 47.8 (45.5) | 53.9 (53.8) | 2 / 7 (P0, P2) |

The LO error follows from the Z* error: the Born–Huang relation with each seed's own Z*, ε∞ and TO predicts LO 807.8
and 851.6 cm⁻¹, against 807.9 and 851.5 scored `RPH:1162`. The logs give the
static dielectric constant ε₀ = ε∞ (LO/TO)² as 13.37 (s42) and 13.69 (s43), against the experimental 9.8.

**Known defects of step 31.**
- **Seed 42's bulk MgO has a spurious state 203 eV below its reference configuration.** On the 8-atom ε∞ prior cell,
  s42's learned O–O hopping (up to +17.72 eV at 3.015 Å, 6.5 times Harrison's ceiling B(r); 576 O–O edges above B in
  every benchmark cell) produces a state 203 eV below the reference configuration. Hole configurations never enter the
  neutral training ensembles, so no data constrains these couplings
  `R4G:6949`, `R4G:7115`.
  The coupling does not enter the slab benchmark cells' own Hamiltonians. Seed 43 is clean.
- **Seed 43's Au–Au wall is too soft and collapses.** On test structure 167 (Au–Au 2.089 Å) the total bond force is
  +6.18 eV/Å against DFT's +15.25 `R4G:7389`. Under compression the dimer falls
  into one Au⁺Au⁻ configuration between 2.10 and 2.06 Å; the energy has a maximum at 2.07 Å and the bond force turns to
  −14 to −23 eV/Å below it. The shortest test structure (2.089 Å) lies 0.02 Å outside the collapse
  `R4G:7433`. This is from `probe_auau_wall_terms.py`, not from step
  34b's S7 scorer; S7 was never run on step 31.
- **Phonons fail:** Z*(Mg) 2.40 / 2.53 against PBE's 1.99, LO 808 / 852 cm⁻¹ against 681, 3 / 7 and 2 / 7. Hypothesis,
  not tested: formal ±2 ions with too little electronic screening, and no bulk force cells in training.
- **Legacy, non-extensive forms of the equations.** Step 31 trained SPEC 28's `'block'` stiffness (the current default
  is `'onsite'`), the `'global'` configuration space (SPEC 36: `'clusters'`) and the `'phi1'` band (SPEC 38:
  `'fermi'`) `R4G:7405`. The `'block'` form scales an atom's interaction with its
  own periodic images `R4G:5190`. The checkpoint loader restores
  `'block'`; a retraining from its configuration file needs `state_polarizability_stiffness: block` written explicitly
  `R4G:7060`.
- **Not measured:** S7, S8, Table II on the 2×2 supercell, and whether today's code reproduces step 31's stored
  numbers (section 5).

### 2.3 Which model for what

- **Table II:** step 31 seed 43 is closest to DFT (wet_u within 8.5 meV, T5 106.9 %). It fails the phonons (2 / 7).
- **Phonons and the Au–Au wall:** step 34b passes 6 / 7 (2×2×2) and 7 / 7 (4×4×4) on both seeds, and S7 on both. Its
  Table II reaches 55.6 % and 81.3 % of DFT's T5.
- No trained model does both `RPH:1174`.

---

## 3. Bulk MgO phonons against PBE

### 3.1 Reference and protocol

- **Reference:** A. Togo's phono3py data set for rocksalt MgO (NIMS MDR, CC BY 4.0): VASP PBE, a = 4.25566 Å, fc2 from
  0.03 Å displacements on the 4×4×4 conventional supercell (512 atoms), Z* = ±1.9923, ε∞ = 3.235; TO 373.3, LO
  681.3 cm⁻¹; bands recomputed with phonopy 2.17.2 along Γ–X–K–Γ–L (204 points) `RPH:37`.
  The derived band files ship in `experiments/benchmarks_phonons/reference/` with attribution; the raw data does not.
- **Model protocol** (`score_phonons.py`): the model at PBE's lattice constant; Z* and ε∞ from the bound sector on the
  64-atom cell; 0.01 Å finite displacements on the **2×2×2 conventional supercell (64 atoms)**; NAC from the model's own
  Z* and ε∞ `RPH:51`.
- **Experiment** at Γ: TO 401, LO 718 cm⁻¹, ε∞ 2.95. The criteria are against PBE.

**Criteria** (pre-registered 2026-11-06 `RPH:58`):

| | criterion |
|---|---|
| P0 | the charge ensemble stays in the formal frame on every displaced supercell (population ≥ 1 − 10⁻⁴) |
| P1 | Z*(Mg) within 1.90–2.10 |
| P2 | ε∞ within 2.9–3.5 |
| P3 | TO within 3 % of 373.3 cm⁻¹ |
| P4 | LO within 660–700 cm⁻¹ |
| P5 | path RMS without NAC ≤ 15 cm⁻¹ |
| P6 | path RMS with NAC ≤ 20 cm⁻¹ |

### 3.2 Results

TO, LO in cm⁻¹; P5, P6 in cm⁻¹. "2×2×2" is the registered protocol of every step up to 34b; "4×4×4" uses the model's
force constants on PBE's own 512-atom supercell and is recorded alongside (section 3.3).

| model | Z*(Mg) | ε∞ | TO | LO | P5, 2×2×2 / 4×4×4 | P6, 2×2×2 / 4×4×4 | passes, 2×2×2 / 4×4×4 | source |
|---|---|---|---|---|---|---|---|---|
| PBE | 1.992 | 3.235 | 373.3 | 681.3 | | | | |
| step 20 s42 (at a = 4.265) | 2.43 | 3.06 | 389.8 | 817 | — / — | 30.3 / — | P2 only; P0 fails at 4.2557 | `RPH:62` |
| step 20 s43 | 2.54 | 3.05 | 407.0 | 850 | 40.4 / — | 48.5 / — | 2 / 7 (P0, P2) | as above |
| step 28b s42 (best, epoch 35) | 1.932 | 3.469 | 337.8 | 631.6 | 36.1 / — | 20.7 / — | 3 / 7 | `RPH:247` |
| step 28b s43 | 1.934 | 3.468 | 339.5 | 633.0 | 33.1 / — | 16.5 / — | 4 / 7 | as above |
| step 31 s42 | 2.402 | 3.025 | 384.3 | 807.9 | 29.6 / 22.0 | 28.9 / 28.9 | — / 3 | section 2.2 |
| step 31 s43 | 2.534 | 3.002 | 398.7 | 851.5 | 45.5 / 47.8 | 53.8 / 53.9 | — / 2 | section 2.2 |
| step 33 A s42 | 1.948 | 3.293 | 361.0 | 659.8 | 32.6 / — | 9.0 / — | 4 / 7 | `runs/AuMgO_single_all_bondlaw_r5_s42/scores/P_phonons_pbe.txt` |
| step 33 A s43 | 1.942 | 3.309 | 370.2 | 662.4 | 30.7 / — | 6.0 / — | 6 / 7 | `…_s43/scores/P_phonons_pbe.txt` |
| step 33 B s42 | 1.948 | 3.291 | 363.4 | 661.2 | 32.1 / — | 8.4 / — | 6 / 7 | `runs/AuMgO_single_all_e1kw_bondlaw_r5_s42/scores/P_phonons_pbe.txt` |
| step 33 B s43 | 1.942 | 3.308 | 371.5 | 663.2 | 30.4 / — | 5.6 / — | 6 / 7 | `…_s43/scores/P_phonons_pbe.txt` |
| step 34b A s42 | 1.950 | 3.334 | 384.7 | 670.7 | 29.3 / 8.4 | 7.5 / 8.6 | 5 / 7, 6 / 7 | `runs/AuMgO_twosite_bf055_bondlaw_r5_s42/scores/P_phonons_pbe{,_sc4}.txt` |
| step 34b A s43 | 1.960 | 3.303 | 366.2 | 664.9 | 30.6 / 9.8 | 8.1 / 10.1 | 6 / 7, 7 / 7 | `…_s43/scores/P_phonons_pbe{,_sc4}.txt` |
| **step 34b B s42** | 1.950 | 3.334 | 382.1 | 669.3 | 29.5 / **8.1** | 7.1 / 8.5 | 6 / 7, **7 / 7** | `runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s42/scores/P_phonons_pbe{,_sc4}.txt` |
| **step 34b B s43** | 1.961 | 3.300 | 370.3 | 667.4 | 29.9 / **8.6** | 6.8 / 8.8 | 6 / 7, **7 / 7** | `…_s43/scores/P_phonons_pbe{,_sc4}.txt` |

- **Failing criteria.** Step 20: P1, P3, P4, P5, P6 (and P0 on s42 at a = 4.2557). Step 28b: P3, P4, P5 (P6 also on
  s42 best). Step 31: P1, P4, P5, P6 (and P3 on s43). Step 33 B and step 34b B: P5 on 2×2×2 only. Step 33 A s42: P3,
  P4, P5. Step 34b A s42: P3 by 0.05 percentage points (+3.05 %, limit 3 %).
- **Step 31 was scored only with the 4×4×4 protocol** (`--p5-supercell 4`), after step 34b. Its "2×2×2" P5 and P6
  values are the records printed beside it.
- **Step 33 was never scored on 4×4×4.**
- Step 34b B, both seeds: no imaginary mode (minimum frequency −0.00 cm⁻¹ on both supercells), and C1 0.005 (s42) and
  0.001 (s43) cm⁻¹ (`P_phonons_sc2_vs_sc4.txt`).

### 3.3 Why P5 depends on the supercell

P5 read on 2×2×2 against PBE's 4×4×4 bands compares two different interpolations. A finite-displacement supercell gives
the dynamical matrix exactly only at the wave vectors commensurate with it; elsewhere the band is a Fourier
interpolation of the folded force constants. Without NAC, the top optical branch near Γ ramps from TO (at Γ, where a
periodic supercell has no macroscopic field) to an LO-like value at the first commensurate q ≠ 0: t = 1/4 of Γ–X on
4×4×4 and t = 1/2 on 2×2×2. PBE's interpolation peaks at 686.4 cm⁻¹ at t = 0.32; the model on 2×2×2 peaks near
t = 0.54 `RPH:1002`.

Step 34b stage B, path points without NAC (`runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_sNN/scores/P_phonons_sc2_vs_sc4.txt`):

| path points (no NAC) | n | max \|2×2×2 − 4×4×4\| (cm⁻¹), s42 / s43 | RMS vs PBE, 2×2×2, s42 / s43 | RMS vs PBE, 4×4×4, s42 / s43 |
|---|---|---|---|---|
| exact on both supercells (Γ ×3, middle of Γ–X, X ×2, L) | 7 | 0.005 / 0.001 | 8.2 / 9.1 | 8.2 / 9.1 |
| exact on 4×4×4 only | 3 | 73.9 / 84.2 | 28.6 / 28.9 | 9.2 / 8.9 |
| interpolated on both | 194 | 152.7 / 161.3 | 30.0 / 30.4 | 8.1 / 8.6 |

The same breakdown for stage A (s42 epoch 30 / s43 epoch 39) is in `RPH:1016`.

- **Where both supercells are exact, the model agrees with itself to 0.005 cm⁻¹** and its error against PBE is the
  same on both. The 2×2×2 excess of P5 is entirely in the interpolation.
- **The hypothesis "P5 > 15 on 2×2×2 is the supercell mismatch" was pre-registered and held**: on the pre-registered
  checkpoint (step 34b stage A, s42 epoch 21) P5 went from 30.4 on 2×2×2 to 11.5 on 4×4×4
  `RPH:968`. For steps before 34b this is a hypothesis:
  they were not rerun (except step 31, scored later on 4×4×4, which still fails P5 at 22.0 and 47.8).
- **What remains is the model's own error.** At X, commensurate with both supercells, the top branch is 511.2 (s42)
  and 514.9 (s43) cm⁻¹ against PBE's 528.2; the largest single deviation on 4×4×4 is 29.8 (s42) and 26.8 (s43) cm⁻¹,
  at interpolated points (`P_phonons_sc2_vs_sc4.txt`).
- **P6 also moves with the supercell.** On step 34b's checkpoints P6 on 4×4×4 is 1.1–2.2 cm⁻¹ higher than on 2×2×2,
  so P6 read on 2×2×2 is optimistic by up to 2.2 cm⁻¹ `RPH:1092`.
  For earlier models the change is unmeasured.
- **One protocol difference remains:** the model uses 0.01 Å displacements, PBE used 0.03 Å. Its effect on P5 is
  unmeasured.
- **For every step registered after 34b,** P5 is read on 4×4×4 (`score_phonons.py --p5-supercell 4`), valid only if
  C1 holds (both supercells agree within 1 cm⁻¹ at the 7 common exact points); P1–P4 and P6 stay on 2×2×2
  `RPH:1043`. Steps up to 34b keep the
  protocol they were registered with.

The phonon notebook also holds a PBEsol line (bulk-only models against phonondb mp-1265). It is a separate validation
and is not summarised here.

---

## 4. The 4G cluster sets: Ag₃±, NaCl, carbon chain

**Read these as separate models.** Each system is its own model, trained on its own data, **one seed (42)**, with an
older equation stack than sections 1–3: the round-1 and round-5 "TRACE + elec" arms are the mean-field SQE
electrostatics (no TRACE-Redox); the "step 9" rows use step 9's TRACE-Redox stack of October 2026; the carbon-chain
"free carriers" row adds SPEC 19's covalent units. None uses the terms of steps 10–34b.

**PFN's numbers are validation RMSEs digitised from a figure in PFN's blog** (56 px per decade). Ours are on held-out
test sets never used for training or early stopping `R4G:260`.
The 4G-HDNNP pooled numbers in these tables are the reference lines of the same figure.

### 4.1 First round: 1,000 training structures, 40 epochs

Test RMSE, E in meV/atom / F in eV/Å, 1,000 test structures `R4G:193`:

| state | TRACE local | TRACE + elec |
|---|---|---|
| Ag₃⁺ | 751.3 / 0.463 | 3.55 / 0.0084 |
| Ag₃⁻ | 639.9 / 0.592 | 11.28 / 0.0234 |
| C₁₀H₂ | 7.10 / 0.145 | 2.19 / 0.055 |
| C₁₀H₃⁺ | 4.87 / 0.216 | 1.48 / 0.096 |
| Na₈Cl₈⁺ | 1.68 / 0.047 | 1.43 / 0.024 |
| Na₉Cl₈⁺ | 2.15 / 0.068 | 0.97 / 0.027 |

Pooled force RMSE (meV/Å) against the published models `R4G:275`:

| | TRACE + elec | TRACE local | 4G-HDNNP | NequIP | NequIP+E_ele | NequIP+QEq |
|---|---|---|---|---|---|---|
| training structures | 1,000 (Au₂/MgO 300) | 1,000 (300) | ≈ 9,000 (4,500) | 8,000 (4,000) | 8,000 (4,000) | 8,000 (4,000) |
| C₁₀H₂ / C₁₀H₃⁺ | 79 | 185 | 76 | 50 | 316 | 60 |
| NaCl | 25 | 59 | 33 | 10 | 16 | 12 |
| Ag₃± | 18 | 537 | 31 | 97 | 25 | 20 (18.5–22.8) |
| Au₂/MgO | 119 | 152 | 67 | 9 | 186 | 84 |

At the DFT minima (4G's Table I metric), TRACE + elec's energy errors were 4–11 times 4G's on three of four structures
`R4G:321`. The NaCl shift between the two charge states' scan minima:
DFT 0.107 Å, 4G 0.109 Å, TRACE + elec 0.055 Å, local 0 `R4G:309`.

### 4.2 PFN's training sizes (round 5), 300 epochs

Same model and seed, PFN's split sizes, the rest of each set held out as test
`R4G:336`. Pooled test F RMSE in eV/Å.

| system (train / valid / test) | TRACE local | TRACE + elec | + Hirshfeld loss | best published (digitised) | 4G-HDNNP line |
|---|---|---|---|---|---|
| Ag₃± (8000 / 1000 / 2013) | 1.892 | **0.0159** | 0.0157 | NequIP+QEq 0.0204 (0.0185–0.0228) | 0.031 |
| NaCl (4000 / 500 / 500) | 0.0553 | **0.0068** | 0.0079 | NequIP 0.0104 | 0.0329 |
| C₁₀H₂ / C₁₀H₃⁺ (8000 / 1000 / 1019) | 0.0805 | **0.0356** | 0.0370 | NequIP 0.0495 (0.0458–0.0518) | 0.0757 |

Per state and at the minima:
- **Ag₃±:** E 0.572 / 3.416 meV/atom, F 0.0022 / 0.0215 eV/Å (Ag₃⁺ / Ag₃⁻). At the DFT minima 0.975 / 0.047 meV/atom
  (Ag₃⁻ / Ag₃⁺), against 4G's 1.166 / 0.320; relaxation RMSD 0.007 / 0.000 Å (4G 0.006 / 0.002). Both minimum
  structures are in the training set `R4G:364`.
- **NaCl:** E 0.739 / 1.588 meV/atom, F 0.0057 / 0.0077 eV/Å (Na₈Cl₈⁺ / Na₉Cl₈⁺). Scan minima 3.367 / 3.471 Å against
  DFT 3.366 / 3.473: shift 0.104 Å against DFT 0.107 and 4G 0.109 `R4G:375`.
- **Carbon chain:** E 0.413 / 0.990 meV/atom, F 0.0145 / 0.0473 eV/Å (C₁₀H₂ / C₁₀H₃⁺). At the minima |ΔE| 0.036 /
  0.190 meV/atom against 4G's 0.463 / 0.176 (both in our training set). C₁₀H₃⁺'s force RMSE is 3.3 times C₁₀H₂'s
  `R4G:407`.
- The Hirshfeld loss lowers the NaCl energy error 3–4 times and raises its force error by 16 %.

### 4.3 Step 9's TRACE-Redox stack on the clusters

Pre-registered U1–U6 `R4G:442`; PFN sizes, 300 epochs:

| id | criterion | measured | verdict |
|---|---|---|---|
| U1 | NaCl pooled F < NequIP's 0.0104 eV/Å | 0.0064 | PASS |
| U2 | NaCl shift within 0.010 Å of 0.107; minima within 0.01 Å | 0.102 Å; 3.369 / 3.471 Å | PASS |
| U3 | Ag₃ pooled F < NequIP+QEq's 0.0204 eV/Å | 0.0160 | PASS |
| U4 | carbon-chain pooled F < NequIP's 0.0495 eV/Å | refused with step 9's stack; 0.0329 with SPEC 19's covalent units | PASS (covalent units) |
| U5 | no charge state worse than round 5 by > 10 % | Na₈Cl₈⁺ F +19 %, Ag₃⁺ F +23 % | **FAIL** |
| U6 | ε∞(NaCl) 2.0–2.7; Ag atom α within 10 % | not measured | open |

Sources: `R4G:494`, `R4G:641`,
`R4G:680`. Step 9's stack improves the NaCl energies 4–10 times (0.18 / 0.16
against 0.74 / 1.59 meV/atom); Ag₃⁺ at its minimum is worse than round 5 (0.303 against 0.047 meV/atom). The
carbon-chain run with covalent units passes U5 (E +6.3 % / −3.4 %, F 0 / −8.0 %).

Never run: the Ag₃± model with the overlap stack of steps 14–16 (pre-registered, V1–V4;
`R4G:619`).

---

## 5. What failed, and what was never measured

### 5.1 Failed

- **No Au₂/MgO step trained on both seeds passed its pre-registration on both.** Passes on one seed: step 9 s42 (T5
  94.6 %; its replicate s43 failed at 125.4 %), step 22 s42 (primary criteria E1 and E5), step 23 s42 and step 31 s43.
  Steps 1 and 5, trained on one seed, passed (step 1's criteria did not include T5; step 5's T5 was 54.3 %).
- **No model passes both Table II and the phonons** (section 2.3).
- **The seed spread of T5 is large:** 287 meV (step 9), 825 meV (step 13), 848 meV (step 25), 431 meV (step 22),
  80 meV (step 33), 239 meV (step 34b), 32 meV (step 31) (table 1.2, |T5(s42) − T5(s43)|).
- **The wet_d error** (the doped flat dimer too stable) follows the doped class offset of the training data across 36
  runs (Pearson r = +0.90); the wet_u error does not `R4G:4739`.
- **T2, the doped − undoped shift of q(Au₂)** (Hirshfeld −0.266 e): −0.69 to −1.00 e in every formal-charge step whose
  T2 is quoted in this page or the notebook (steps 2–6, 8, 9, 31, 32b, 33, 34b), +0.012 e in step 7, −0.02 to −0.05 e
  in the bound-sector redox arms (redox, redoxq, step 1), and the wrong sign (+0.012 e) in the SQE arms. The integer
  model and Hirshfeld are different partitions (section 1.1).
- **Refusals and crashes in training or scoring:** steps 13F, 15, 16 s42, 34 (configuration caps); step 21 s42 and
  step 26 s42 (T5 relaxation); step 32 s43 (out of GPU memory); step 9's stack on the carbon chain (configuration cap).
- **Cluster sets:** U5 failed on two cation force entries (section 4.3).

### 5.2 Never measured

- **Table II on the 2×2 supercell**, for every step (recorded as NOT DONE for steps 33 and 34b:
  `R4G:8067`, `R4G:9291`).
- **Step 31:** S7 and S8 (the Au–Au wall and the undoped force without the compressed dimers, as scored for 34b).
- **Step 33:** phonons on 4×4×4.
- **Step 32:** T6 ε∞ and the relaxation-failure count `R4G:7268`.
- **Verdicts of steps 13S, 14, 16 (s43) and 25:** trained and scored by the scripts, but the notebook has no scored
  entry against their criteria.
- **Reproduction by today's code** of the stored numbers of steps 26 and 31 and of the cluster runs. Step 31's
  seed-43 Table II was reproduced on the code of 2026-11-12 `R4G:7063`;
  no full re-score on the release code was run.
- **Whether PBE has the flat dimer as a minimum against the tilt** that steps 23–24 found `R4G:3845`.
- **The 4G-HDNNP test RMSEs** (the paper's supplementary information was not read), and the source of the 0.219
  meV/atom quoted in the notebook.
- **Phonons:** the effect of 0.01 against 0.03 Å displacements on P5; the 2×2×2 → 4×4×4 change of P6 for any model
  before step 34b; P5 on 4×4×4 for steps before 34b except step 31.
- **Cluster sets:** U6 (ε∞ of NaCl, the Ag atom α); a second seed of any cluster model; the Ag₃± overlap-stack run.
- **Environment:** the scores were computed on CPU (Python 3.8.16, torch 2.2.0, e3nn 0.4.4). Steps 28b, 32, 33 and
  34b and the round-5 carbon-chain arms were trained on a GPU host; for step 32 the notebook records torch 2.11.0+cu128
  there `R4G:7093`. The pinned `requirements.txt` set has not been used for any
  reported run. Retraining is not bitwise reproducible (GPU summation order, transient non-finite steps recomputed by the NaN
  guard); the released checkpoints are the reproduction path for these numbers ([`REPRODUCE.md`](REPRODUCE.md)).
