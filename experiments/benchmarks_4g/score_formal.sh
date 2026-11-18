#!/bin/zsh
# Score a FORMAL run: T1-T6, P1 (score_run.sh), the transfer margin as trained and at eps_inf = 3
# (s = 2 / (eps_model - 1), eps_model from this checkpoint's T6), the covalency probe, and the diagnostics
# (per-atom Au charges, Au polarizability, electron-site and wetting decompositions, class residuals).
# Usage (any directory): score_formal.sh RUN_NAME
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
$HERE/score_run.sh $1
cd $HERE/../..
R=experiments/benchmarks_4g/runs/$1
EPS=$(grep -o "eps_inf diag \[[0-9.]*" $R/scores/T6_eps.txt | head -1 | grep -o "[0-9.]*$")
S=$(python -c "print(round(2.0 / ($EPS - 1.0), 6))")
echo "eps_inf(model) = $EPS -> s = $S" > $R/scores/M_transfer_margin.txt
python experiments/benchmarks_4g/probe_transfer_margin.py $R/model.pt $R/model.pt:$S >> $R/scores/M_transfer_margin.txt 2>&1
python experiments/benchmarks_4g/probe_couplings.py $R/model.pt > $R/scores/C_couplings.txt 2>&1
python experiments/benchmarks_4g/probe_au_pattern.py $R/model.pt > $R/scores/A_au_pattern.txt 2>&1
python experiments/benchmarks_4g/probe_au_polarizability.py $R/model.pt > $R/scores/A_au_polarizability.txt 2>&1
python experiments/benchmarks_4g/probe_site_decomposition.py $R/model.pt > $R/scores/S_site_decomposition.txt 2>&1
python experiments/benchmarks_4g/probe_class_residuals.py $R/model.pt > $R/scores/H1_class_residuals.txt 2>&1
(cd experiments/benchmarks_4g && python probe_wetting_decomposition.py runs/$1 > runs/$1/scores/W_wetting_decomposition.txt 2>&1)
echo "scored (formal) $1 at $(date)"
