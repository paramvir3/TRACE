#!/bin/zsh
# Launch step 13FC (pre-registered; step 13 + the field prior on Au2 0, +1, -1) when step 13S has ended and
# been scored (watch_step13f_score.sh), then score it.
set -u
cd "$(dirname "$0")"
S=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fpriorq_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fpriorq_bondlaw_r5_s43)
while ! grep -q "step 13S scored" runs/watch_step13s.log 2>/dev/null; do sleep 300; done
echo "launching step 13FC $(date)"
P=()
for r in $S; do nohup python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 13FC ended $(date)"
for r in $S; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_alpha.py Au 2.30,2.47,2.60,2.80,3.10 experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/F_dimer_alpha.txt 2>&1)
  fi
done
echo "step 13FC scored $(date)"
