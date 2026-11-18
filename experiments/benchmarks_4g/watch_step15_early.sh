#!/bin/zsh
# Watcher (2026-10-19 13:10): launch step 15 (pre-registered; unchanged) in the slot freed by step 13S, i.e. when
# step 13S has been scored, instead of after step 14 (13FC was disarmed: confounded by the state-independent alpha_b,
# README).  Scoring as in watch_step15.sh; writes "step 15 scored" to runs/watch_step15.log, which step 16 waits for.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S15=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s43)
while ! grep -q "step 13S scored" runs/watch_step13s.log 2>/dev/null; do sleep 300; done
echo "step 13S scored; launching step 15 in its slot $(date)"
P=()
for r in $S15; do nohup python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 15 ended $(date)"
for r in $S15; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 python probe_window_deficit.py runs/$r > runs/$r/scores/R_window_deficit.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$r/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$r/scores/D_dimer_ct.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_alpha_scorecard.py experiments/benchmarks_4g/data/Au2_charges_alpha_field_prior.xyz experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/F_alpha_scorecard.txt 2>&1)
  fi
done
echo "step 15 scored $(date)"
