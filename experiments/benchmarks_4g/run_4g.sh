#!/bin/bash
# 4G-HDNNP benchmarks (Ko et al., Nat. Commun. 2021): local TRACE vs TRACE + electrostatics
# (new-run defaults: solver auto, SPEC 23 contact law), 1000/250/1000, 40 epochs, seed 42.
# Cutoffs as in the paper: 4.23 A carbon chain, 10 Bohr = 5.29 A otherwise.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
pids=()
for spec in "Ag_cluster 5.29" "Carbon_chain 4.23" "NaCl 5.29"; do
  set -- $spec; data=$1; rmax=$2
  for arm in base elec; do
    name=${data}_${arm}_s42; extra=""; [ "$arm" = elec ] && extra=elec
    $PY make_configs.py $data $name 40 $rmax 2 $extra seed=42 > /dev/null
    $PY run_train.py runs/$name/config.yaml > runs/$name/log.txt 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait $p; done
echo ALL_DONE
