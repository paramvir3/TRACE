#!/bin/zsh
# Score one AuMgO run on the pre-registered tests (T1-T6, P1).  Usage (any directory): score_run.sh RUN_NAME
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
R=runs/$1
OUT=$R/scores
mkdir -p $OUT
EVAL_THREADS=2 python eval_aumgo_transfer.py $R > $OUT/T1_T3_transfer.txt 2>&1 &
(cd ../.. && python experiments/benchmarks_4g/probe_free_charges.py experiments/benchmarks_4g/$R/model.pt > experiments/benchmarks_4g/$OUT/T4_free_charges.txt 2>&1) &
python eval_aumgo.py $R > $OUT/T5_wetting.txt 2>&1 &
EPS_PROBE=probe_eps_mgo.py; grep -q "bond_partition: true" $R/config.yaml && EPS_PROBE=probe_eps_bound.py   # SPEC 22 models: the module's own bound_state
(cd ../.. && python experiments/benchmarks_4g/$EPS_PROBE experiments/benchmarks_4g/$R/model.pt > experiments/benchmarks_4g/$OUT/T6_eps.txt 2>&1 && python experiments/benchmarks_4g/probe_one_curve.py experiments/benchmarks_4g/$R/model.pt > experiments/benchmarks_4g/$OUT/P1_one_curve.txt 2>&1) &
wait
echo "scored $1 at $(date)"
