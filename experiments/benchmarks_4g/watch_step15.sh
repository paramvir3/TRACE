#!/bin/zsh
# Watcher (2026-10-18): launch step 15 (pre-registered) when both step 14 runs have finished (Training finished or
# stopped), then score it as score_formal.sh + the SPEC 21 window test + the isolated Au2 probe + eps via bound_state.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S14=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s43)
S15=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s43)
running() { pgrep -f "run_train_nanguard.py runs/$1/config" > /dev/null; }
# wait until both step-14 logs exist and neither run is alive
while [ ! -s runs/$S14[1]/log.txt ] || [ ! -s runs/$S14[2]/log.txt ] || running $S14[1] || running $S14[2]; do sleep 300; done
echo "step 14 ended $(date)"
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
  fi
done
echo "step 15 scored $(date)"
