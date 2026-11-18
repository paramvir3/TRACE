#!/bin/bash
# Au2 on MgO(001), periodic (110 atoms): local TRACE vs TRACE + electrostatics (new-run defaults:
# solver auto -> matrix-free 3-D Ewald, SPEC 23 contact law, no seed loss). 300/76/400, 25 epochs,
# seed 42, cutoff 10 Bohr = 5.29 A as in the paper. Starts after the cluster runs (run_4g.sh).
# Run as: ./run_4g.sh > run_4g_driver.log & ./run_aumgo.sh > run_aumgo_driver.log   (it waits for ALL_DONE there)
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
until grep -q ALL_DONE run_4g_driver.log 2>/dev/null; do sleep 60; done
pids=()
for arm in base elec; do
  name=AuMgO_${arm}_s42; extra=""; [ "$arm" = elec ] && extra=elec
  $PY make_configs.py AuMgO $name 25 5.29 5 $extra seed=42 > /dev/null
  $PY run_train.py runs/$name/config.yaml > runs/$name/log.txt 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo AUMGO_DONE
