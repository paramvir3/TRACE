#!/bin/zsh
# Watcher (2026-10-18): when step 13 (its training PIDs, given as arguments: seed 42 resumed after the epoch-8 NaN, and seed 43) ends, launch step 14 (pre-registered) and score step 13;
# when step 14 ends, score it.  Scoring = score_formal.sh + the SPEC 21 window test + the isolated Au2 probe.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S13=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_bondlaw_r5_s43)
S14=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s43)
score() {
  nice -n 5 ./score_formal.sh $1
  nice -n 5 python probe_window_deficit.py runs/$1 > runs/$1/scores/R_window_deficit.txt 2>&1
  (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$1/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$1/scores/D_dimer_ct.txt 2>&1)
  (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$1/model.pt > experiments/benchmarks_4g/runs/$1/scores/T6_eps_bound.txt 2>&1)
}
for p in "$@"; do while ps -p $p > /dev/null; do sleep 300; done; done   # the PIDs given as arguments (none: start at once)
echo "step 13 ended $(date)"; for r in $S13; do grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2; done
P=()
for r in $S14; do nohup python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
for r in $S13; do grep -q "Training finished" runs/$r/log.txt && score $r & done
wait $P[1] $P[2] 2>/dev/null; while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 14 ended $(date)"; for r in $S14; do grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2; done
for r in $S14; do grep -q "Training finished" runs/$r/log.txt && score $r & done
wait
echo "all scored $(date)"
