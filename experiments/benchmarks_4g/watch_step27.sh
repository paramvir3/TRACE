#!/bin/zsh
# Step 27 (2026-11-06, branch bond-pauli-wall-level-2026-11-05; pre-registered in README before steps 25/26 were scored): step 26 + bond_pauli_states (SPEC 33 per charge state), cold start.
# Scored as step 26.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
S27=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bops_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bops_bondlaw_r5_s43)
P=()
for r in $S27; do nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 & P+=($!); echo "launched $r pid $!"; done
while ps -p $P[1] > /dev/null || ps -p $P[2] > /dev/null; do sleep 300; done
echo "step 27 ended $(date)"
for r in $S27; do
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
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_upright_approach.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T_upright_approach.txt 2>&1)
  fi
done
echo "step 27 scored $(date)"
# the 28-ion ALMO split was paused (SIGSTOP) for the training: resume it (exact PID, given as ALMO_PID)
if [ -n "${ALMO_PID:-}" ] && kill -0 $ALMO_PID 2>/dev/null; then kill -CONT $ALMO_PID; echo "ALMO $ALMO_PID resumed $(date)"; fi
