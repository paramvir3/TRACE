#!/bin/zsh
# Bulk-augmented step (2026-11-06, branch phonon-benchmark-2026-11-06; pre-registered in experiments/benchmarks_phonons/README.md): step 20's stack trained on the 4G slabs plus 136 displaced PBE MgO cells (forces only).
# Scored as step 20 (T1-T6 and the probes) plus the phonon benchmark (score_phonons.py).
# 2026-11-07 02:15: relaunched as warm starts from step 20 (resume_from warm_from_step20_sXX.pt, epoch 0, weights only): the cold start exploded the configuration search on the bulk cells (ConfigurationSearchError, > 512 configurations within 2.5 eV at random init) in epoch 1 on both seeds and on the GPU copy.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S25=(AuMgO_bulkaug_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bondlaw_r5_s43)   # s43 here; s42 trains on the GPU host and is scored by hand when its model.pt is copied back
P=()
for r in $S25; do nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while true; do alive=0; for p in $P; do ps -p $p > /dev/null && alive=1; done; [ $alive -eq 0 ] && break; sleep 300; done
echo "bulk-augmented step ended $(date)"
for r in $S25; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE|ConfigurationSearchError" runs/$r/log.txt | head -3
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$r/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$r/scores/D_dimer_ct.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
    nice -n 5 ./score_extra.sh $r
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_spectrum.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S_spectrum.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_hole_shift.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S_hole_shift.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_t5_first_order.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S5_first_order.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_hole_terms.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S6_hole_terms.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_otop_channels.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/S_otop_channels.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_downfold_counterfactual.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/D6_fold.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_mgtop_curve.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/W1_mgtop_curve.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_tilt_path.py --scan ${TRACE_FP_CACHE:-fp_cache}/tilt_path/path_step24_s42.json experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T_tilt_path.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_tilt_path.py --scan ${TRACE_FP_CACHE:-fp_cache}/tilt_path/rotation_path.json experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T_rotation.txt 2>&1)
  fi
done
echo "bulk-augmented step scored $(date)"
# the 28-ion ALMO split was paused (SIGSTOP) for the training: resume it (exact PID, given as ALMO_PID)
if [ -n "${ALMO_PID:-}" ] && kill -0 $ALMO_PID 2>/dev/null; then kill -CONT $ALMO_PID; echo "ALMO $ALMO_PID resumed $(date)"; fi
for r in $S25; do
  if grep -q "Training finished" runs/$r/log.txt; then
    (cd ../.. && nice -n 5 python experiments/benchmarks_phonons/score_phonons.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/P_phonons.txt 2>&1)
  fi
done
echo "bulk-augmented step phonons scored $(date)"
