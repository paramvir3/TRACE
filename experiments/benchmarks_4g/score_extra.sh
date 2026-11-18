#!/bin/zsh
# The 2026-10-19 probes for a trained formal AuMgO run (after score_formal.sh): the exact T5 split into the terms of
# the configuration energy (G_t5_terms); the T5 split into E_s / E_q with the
# Au shifts and p(Au+ Au-); the isolated Au2 ionic gap and tau, learned and zeroed shifts, with and without the
# overlap stack; the complete Au2 configuration ladder; the class offsets on the training set; T5 under the later
# steps' settings on this run's frozen weights.  Usage (any directory): score_extra.sh RUN_NAME
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HERE/runs/$1
[ -f $R/model.pt ] || { echo "no model for $1"; exit 1; }
cd $HERE
nice -n 5 python probe_gauge_split.py $R/model.pt > $R/scores/G_gauge_split.txt 2>&1
nice -n 5 python probe_t5_stack.py $R/model.pt > $R/scores/G_t5_stack.txt 2>&1
nice -n 5 python probe_t5_terms.py $R/model.pt > $R/scores/G_t5_terms.txt 2>&1
cd $HERE/../..
nice -n 5 python experiments/benchmarks_4g/probe_explicit_gap.py Au 2.3,2.5,2.8,3.4 $R/model.pt > $R/scores/G_explicit_gap.txt 2>&1
nice -n 5 python experiments/benchmarks_4g/probe_dimer_ladder.py Au Au 2.47 $R/model.pt > $R/scores/G_dimer_ladder.txt 2>&1
CLASS_SPLIT=train nice -n 5 python experiments/benchmarks_4g/probe_class_residuals.py $R/model.pt > $R/scores/H1_class_residuals_train.txt 2>&1
grep -l "Traceback" $R/scores/G_*.txt $R/scores/H1_class_residuals_train.txt 2>/dev/null | sed 's/^/FAILED: /'
echo "extra scores $1 at $(date)"
