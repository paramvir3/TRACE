#!/bin/zsh
# Step 19 (2026-10-22, branch band-prior-2026-10-22; pre-registered in README): step 17 + band_initial_factor 0.55.
# Launched once the full test suites pass; scored as step 17, plus score_extra.sh and the model's own spectrum.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S19=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bprior_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bprior_bondlaw_r5_s43)
P=()
for r in $S19; do nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 19 ended $(date)"
for r in $S19; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE|ConfigurationSearchError" runs/$r/log.txt | head -3
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 python probe_window_deficit.py runs/$r > runs/$r/scores/R_window_deficit.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$r/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$r/scores/D_dimer_ct.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
    nice -n 5 ./score_extra.sh $r
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_spectrum.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S_spectrum.txt 2>&1)
  fi
done
echo "step 19 scored $(date)"
