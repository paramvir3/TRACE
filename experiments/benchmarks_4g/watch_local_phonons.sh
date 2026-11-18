#!/bin/zsh
# TRACE without electrostatics on the PBEsol cells (pre-registered in benchmarks_phonons/README.md, 2026-11-09 16:04):
# train both seeds concurrently, then phonons by benchmarks_phonons/phonons_model.py (no NAC; DFT's Z*/eps applied as a reference),
# path RMS against PBEsol (score_band_dir.py), and the model-vs-DFT pair plots.
set -u
H=$(cd "$(dirname "$0")/../.." && pwd); B=$H/experiments/benchmarks_4g; P=$H/experiments/benchmarks_phonons
cd $B
python check_run_config.py runs/MgO_PBEsol_bulkonly_local_r5_s42 runs/MgO_PBEsol_bulkonly_local_r5_s43 || exit 1
echo '{"Z_Mg": 1.9791, "eps": 3.2417}' > $P/runs/dft_pbesol_born.json
PIDS=()
for s in 42 43; do nohup nice -n 10 python run_train_nanguard.py runs/MgO_PBEsol_bulkonly_local_r5_s$s/config.yaml > runs/MgO_PBEsol_bulkonly_local_r5_s$s/log.txt 2>&1 & PIDS+=($!); echo "$(date): launched local s$s pid $!"; done
while ps -p $PIDS[1] > /dev/null || ps -p $PIDS[2] > /dev/null; do sleep 120; done
echo "$(date): trainings ended"
for s in 42 43; do
  R=MgO_PBEsol_bulkonly_local_r5_s$s; O=$P/runs/${R}_pbesol
  grep -E "Training finished|Traceback" runs/$R/log.txt | head -2
  mkdir -p $O/own $O/dft_born
  nice -n 5 python $P/phonons_model.py runs/$R --a 4.21726 --out $O/own > $O/own/phonons.log 2>&1
  cp $O/own/band_nonac.yaml $O/own/band_nac.yaml        # no Born charge: the NAC term is zero, the NAC band is the no-NAC band
  nice -n 5 python $P/phonons_model.py runs/$R --a 4.21726 --out $O/dft_born --born-from $P/runs/dft_pbesol_born.json > $O/dft_born/phonons.log 2>&1
  (cd $P && python score_band_dir.py $O/own; python score_band_dir.py $O/dft_born) | tee $O/scores.txt
  grep -E "^\[t=.*\] +[0-9]+ \|" runs/$R/log.txt | tail -1 | awk -F'|' '{print "final epoch val F RMSE (eV/A):", $13}' | tee -a $O/scores.txt
  (cd $P && python plot_pbesol_pair.py $O/own "local (no electrostatics) s$s" $P/${R}_vs_pbesol.png)
done
echo "$(date): local phonons scored"
