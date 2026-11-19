#!/bin/zsh
# Watcher (2026-10-18): launch step 17 (pre-registered; step 16 + state_polarizability, SPEC 28) when step 16 has been
# trained and scored, then score it as for steps 15 and 16.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S17=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s43)
while ! grep -q "step 16 scored" runs/watch_step16.log 2>/dev/null; do sleep 300; done
echo "step 16 scored $(date)"
P=()
for r in $S17; do nohup python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 17 ended $(date)"
for r in $S17; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 python probe_window_deficit.py runs/$r > runs/$r/scores/R_window_deficit.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$r/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$r/scores/D_dimer_ct.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
  fi
done
echo "step 17 scored $(date)"
