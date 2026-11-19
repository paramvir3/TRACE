#!/bin/zsh
# Score the carbon chain with covalent units (pre-registered U4, U5, U7) when its resumed run (PID given as the argument) ends.
cd "$(dirname "$0")"
R=runs/protocol/Carbon_chain_step9cu_s42
for p in "$@"; do while ps -p $p > /dev/null; do sleep 300; done; done   # the PIDs given as arguments (none: start at once)
echo "chain ended $(date)"; grep -E "Training finished|Traceback|FIRST NON-FINITE" $R/log.txt | cut -c1-200
if grep -q "Training finished" $R/log.txt; then
  mkdir -p $R/scores
  EVAL_DATA_DIR=data_pfn nice -n 5 python eval_4g.py Carbon_chain $R runs/protocol/Carbon_chain_elec_s42 > $R/scores/eval_4g.txt 2>&1
  nice -n 5 python compare_protocol.py Carbon_chain $R runs/protocol/Carbon_chain_elec_s42 > $R/scores/compare_protocol.txt 2>&1
  echo "scored $(date)"
fi
