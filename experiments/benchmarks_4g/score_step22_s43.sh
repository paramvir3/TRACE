#!/bin/zsh
# Step 22 seed 43 after its resume (2026-10-31: crashed at epoch 31 with an autograd-engine error, resumed from epoch 26):
# the same scoring as watch_step22.sh for this run only, then step 23 (both seeds) via watch_step23.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
r=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_dfct_bondlaw_r5_s43
[ -n "${1:-}" ] && while ps -p $1 > /dev/null; do sleep 300; done   # the resumed training's PID, given as the argument
echo "step 22 s43 ended $(date)"
grep -E "Training finished|Traceback|EXCEPTION|FIRST NON-FINITE|ConfigurationSearchError" runs/$r/log.txt | head -3
if grep -q "Training finished" runs/$r/log.txt; then
  sed -n '/if grep -q "Training finished" runs\/\$r\/log.txt; then/,/^  fi/p' watch_step22.sh | sed '1d;$d' > runs/score_block_s43.sh
  source runs/score_block_s43.sh
  rm -f runs/score_block_s43.sh
  echo "step 22 s43 scored $(date)"
  ./watch_step23.sh > runs/watch_step23.log 2>&1
else
  echo "step 22 s43 did not finish: step 23 NOT launched"
fi
