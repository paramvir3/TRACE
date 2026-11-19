#!/bin/zsh
# Step 29, the energy-weight test (pre-registered in README, 2026-11-08 17:02 IST): test (energy_weight 1000) and its
# control (10), both warm from step 26 s43, 10 epochs, concurrently on the Mac; then the scores W1-W4 need:
# eval_aumgo.py (E/F per subset, Table II by the T5 protocol) and the class residuals on the test and training sets.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
B=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw
RUNS=(${B}_ew1000_bondlaw_r5_s43 ${B}_ew10_bondlaw_r5_s43)
python check_run_config.py runs/$RUNS[1] runs/$RUNS[2] || exit 1
P=()
for r in $RUNS; do nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "$(date): launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "$(date): step 29 trainings ended"
for r in $RUNS; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE|ConfigurationSearchError" runs/$r/log.txt | head -3
  if grep -q "Training finished" runs/$r/log.txt; then
    mkdir -p runs/$r/scores
    nice -n 5 python eval_aumgo.py runs/$r > runs/$r/scores/T5_wetting.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_class_residuals.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/H1_class_residuals.txt 2>&1)
    (cd ../.. && CLASS_SPLIT=train nice -n 5 python experiments/benchmarks_4g/probe_class_residuals.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/H1_class_residuals_train.txt 2>&1)
    echo "$(date): scored $r"
  fi
done
python read_step29.py
echo "$(date): step 29 scored"
