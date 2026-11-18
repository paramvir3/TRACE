#!/bin/zsh
# 2026-10-24 13:10, the decision of that day: memory (16 GB, 8.3 GB swap) starved the 34-ion wetting run, so step 18 s42 was
# suspended a second time (at epoch 32; best checkpoint epoch 31); s43 was in its final epoch and finishes on its own.
# This waits for s43 to end and for the 34-ion run to exit (their PIDs given as arguments 1 and 2), resumes s42 (resume_run.py, tag suspend2), scores
# both as watch_step18.sh would, then launches the doped wetting decomposition (memory again).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S42=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s42
S43=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s43
S43_PID=${1:-}; RUN34_PID=${2:-}   # the running s43 training and the 34-ion run (empty: do not wait)
[ -n "$S43_PID" ] && while kill -0 $S43_PID 2>/dev/null; do sleep 120; done
echo "step 18 s43 ended $(date)"
[ -n "$RUN34_PID" ] && while kill -0 $RUN34_PID 2>/dev/null; do sleep 120; done
echo "34-ion run ended; resuming step 18 s42 $(date)"
cmd=$(python resume_run.py runs/$S42 suspend2) || { echo "resume_run failed"; exit 1; }
echo "$cmd"; eval "$cmd"; P=$!; echo "resumed $S42 pid $P"
while kill -0 $P 2>/dev/null; do sleep 300; done
echo "step 18 ended $(date)"
for r in $S42 $S43; do
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
cd "$HERE/../.."
# needs PySCF in the active Python environment; first-principles outputs go to the cache $TRACE_FP_CACHE (default <repo>/fp_cache)
FP=${TRACE_FP_CACHE:-fp_cache}; mkdir -p $FP/wetting_eda
OMP_NUM_THREADS=4 nice -n 5 python tools/electrostatics/wetting_eda.py --doped $FP/wetting_eda 3.8 > $FP/wetting_eda/doped_r3.8.log 2>&1
echo "doped wetting decomposition exit $? $(date)"
