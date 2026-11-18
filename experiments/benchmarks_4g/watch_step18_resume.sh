#!/bin/zsh
# Step 18 was suspended 2026-10-22 14:31 (decision of that day; memory: 16 GB over-committed, heavy swap, four NaN-guard
# transients on s42 that morning) so that step 19 and the wetting decomposition run at full speed.  This resumes it from
# its best checkpoints (s42 epoch 7, s43 epoch 8; resume_run.py, tag suspend1) once step 19 is scored, then scores it
# exactly as watch_step18.sh would have.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S18=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s43)
while ! grep -q "step 19 scored" runs/watch_step19.log 2>/dev/null; do sleep 300; done
echo "step 19 scored; resuming step 18 $(date)"
P=()
for r in $S18; do
  cmd=$(python resume_run.py runs/$r suspend1) || { echo "resume_run failed for $r"; exit 1; }
  echo "$cmd"
  eval "$cmd"
  P+=($!); echo "resumed $r pid $!"
done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 18 ended $(date)"
for r in $S18; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 python probe_window_deficit.py runs/$r > runs/$r/scores/R_window_deficit.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_alpha_scorecard.py experiments/benchmarks_4g/data/Au2_charges_alpha_field_prior.xyz experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/F_alpha_scorecard.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_explicit_gap.py Au 2.3,2.5,2.8,3.4 experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/G_explicit_gap.txt 2>&1)
    nice -n 5 python probe_gauge_split.py runs/$r/model.pt > runs/$r/scores/G_gauge_split.txt 2>&1
  fi
done
echo "step 18 scored $(date)"
