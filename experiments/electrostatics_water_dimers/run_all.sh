#!/bin/bash
# Two waves of four concurrent CPU trainings (2 threads each): seed 42 then seed 7.
# Usage: bash run_all.sh   (from this directory; logs in runs/<name>/log.txt)
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for seed in 42 7; do
  pids=()
  for r in 3 6; do
    for arm in base elec; do
      name=r${r}_${arm}_s${seed}
      extra=""; [ "$arm" = elec ] && extra=elec
      $PY make_configs.py $name 80 ${r}.0 2 $extra seed=$seed > /dev/null
      $PY run_train.py runs/$name/config.yaml > runs/$name/log.txt 2>&1 &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait $p; done
done
echo ALL_DONE
