#!/bin/zsh
# Step 30 (pre-registered in README, 2026-11-08 21:55 IST): step 26 + energy_weight 1000, cold start, both seeds,
# concurrently on the Mac; scored as step 26 (score_formal.sh, score_extra.sh); read with read_step30.py.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
B=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1k_bondlaw_r5
RUNS=(${B}_s42 ${B}_s43)
python check_run_config.py runs/$RUNS[1] runs/$RUNS[2] || exit 1
P=()
for r in $RUNS; do nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "$(date): launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "$(date): step 30 trainings ended"
for r in $RUNS; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE|ConfigurationSearchError" runs/$r/log.txt | head -3
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 ./score_extra.sh $r
    echo "$(date): scored $r"
  fi
done
python read_step30.py
echo "$(date): step 30 scored"
