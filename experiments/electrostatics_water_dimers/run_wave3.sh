#!/bin/bash
# Wave 3 (seed 42), after the PBE0 reference is complete (data_qm/{train,valid,test}.xyz):
#   r6_elec_mu       energies + forces + dipoles          -> is the (untrained) alpha right?
#   r6_elec_mualpha  energies + forces + dipoles + alpha   (upper reference for alpha)
#   r6_elec_ct       energies + forces, charge-transfer contact law (SPEC 23): is alpha
#                    size-extensive WITHOUT electrical data?
#   r6_elec_ct_mu    contact law + dipoles
#   r3_elec_disp     + many-body dispersion from the same moment block (SPEC 13)
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for f in data_qm/train.xyz data_qm/valid.xyz data_qm/test.xyz; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
pids=()
launch() { name=$1; shift; $PY make_configs.py $name 80 "$@" > /dev/null
           $PY run_train.py runs/$name/config.yaml > runs/$name/log.txt 2>&1 & pids+=($!); }
launch r6_elec_mu_s42      6.0 2 elec seed=42 data_dir=data_qm elec.dipole_weight=10
launch r6_elec_mualpha_s42 6.0 2 elec seed=42 data_dir=data_qm elec.dipole_weight=10 elec.polarizability_weight=1
launch r6_elec_ct_s7       6.0 2 elec seed=7 elec.heads.charge_transfer_decay=3.0
launch r6_elec_ct_mu_s42   6.0 2 elec seed=42 data_dir=data_qm elec.heads.charge_transfer_decay=3.0 elec.dipole_weight=10
launch r3_elec_disp_s42    3.0 2 elec seed=42 elec.heads.dispersion=true elec.config.dispersion=true elec.config.dispersion_range=2.0
for p in "${pids[@]}"; do wait $p; done
echo WAVE3_DONE
