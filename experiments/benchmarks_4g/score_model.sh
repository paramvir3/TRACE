#!/usr/bin/env bash
# Score one trained run locally (CPU; no remote host), writing every output under RUN_DIR/scores/.  The recipe is the
# one that scored the released Au2/MgO runs (steps 33 and 34b), plus the 512-atom phonon score recorded for step 34b:
#
#   P_phonons_pbe.txt             MgO phonons against Togo's PBE set, P0-P6 on the 2x2x2 conventional supercell (64 atoms;
#                                 the pre-registered protocol of steps up to 34b); band yaml in scores/phonons_sc2/
#   P_phonons_pbe_sc4.txt         the same with the force constants on 4x4x4 (512 atoms, PBE's own fc2 supercell);
#                                 band yaml in scores/phonons_sc4/ (recorded for step 34b, not a criterion there)
#   P_phonons_sc2_vs_sc4.txt      the consistency check C1 of the two supercells (compare_sc2_sc4.py)
#   Z1_Z2_couplings.txt           the coupling sizes (probe_coupling_sizes.py)
#   T5_wetting.txt                Table II of the 4G paper by the T5 protocol and the test E / F RMSE (eval_aumgo.py;
#                                 also writes RUN_DIR/eval_aumgo.json); reads the raw 4G set in $TRACE_4G_RAW
#   T1_T3_transfer.txt            the transfer set (eval_aumgo_transfer.py; also RUN_DIR/eval_aumgo_transfer.json)
#   H1_class_residuals.txt        the class residuals on the test set, and *_train.txt on the training set
#   S7_auau_scan.txt              the Au-Au bond-force scan of test structure 167 (probe_auau_two_site.py)
#   S8_force_outliers.{txt,npz}   the per-structure force breakdown of the test set (probe_force_outliers.py)
#   S6_au2_ea_ip.txt              the gas-phase Au2 electron affinity and ionisation energy (probe_au2_ea_ip.py)
#   S7_S8.txt                     S7, S8 and S5 as scored for step 34b (score_step34_s7_s8.py).  S5 also reads the
#                                 training log RUN_DIR/log.txt; for a released checkpoint without its log, S5 prints
#                                 NOT SCORED (the crystal factor g is in the checkpoint, and the phonon scorer's eps_inf
#                                 is P2 of P_phonons_pbe.txt)
#
# For a cluster run of the PFN-size protocol (runs/protocol/{Ag_cluster,NaCl,Carbon_chain}_*), it writes instead
#   eval_4g.txt                   per-state and pooled RMSE on the run's own test set (EVAL_DATA_DIR=data_pfn eval_4g.py;
#                                 also RUN_DIR/eval_4g.json); compare several runs with compare_protocol.py
#
# Usage (any directory):
#   experiments/benchmarks_4g/score_model.sh RUN_DIR [--stage a|b]
#     RUN_DIR holds model.pt (a released checkpoint: put <run_name>.pt there as model.pt; docs/electrostatics/MODELS.md).
#     --stage a scores only the phonons (both supercells) and the couplings, as was done for the stage-A checkpoints;
#     the default (b) scores everything.
#   PYTHON selects the interpreter (default python3).  EVAL_THREADS (default 2) sets the threads of the transfer, S7
#   and S8 probes.  Each score runs on its own; a failing one is reported and the others continue.
# Needs: experiments/benchmarks_4g/prepare_all.sh run first (data/AuMgO_*, the priors and $TRACE_4G_RAW), phonopy for
# the phonon scores, and the trace_redox/data tables of the release (read at run time, not stored in the checkpoint).
# Cost on the machine of record (Mac CPU, torch 2.2.0): the four step-34b checkpoints took about 3 h together for the
# set without the 512-atom phonons; the 512-atom phonon score takes about 62 s and 6.6 GB resident memory per model.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
B4G="$REPO/experiments/benchmarks_4g"
PH="$REPO/experiments/benchmarks_phonons"
PY="${PYTHON:-python3}"
THREADS="${EVAL_THREADS:-2}"

[ $# -ge 1 ] || { echo "usage: $0 RUN_DIR [--stage a|b]" >&2; exit 2; }
RUN="$(cd "$1" 2>/dev/null && pwd)" || { echo "no directory $1" >&2; exit 2; }
STAGE=b
if [ $# -ge 3 ] && [ "$2" = "--stage" ]; then STAGE="$3"; fi
case "$STAGE" in a|b) ;; *) echo "--stage must be a or b" >&2; exit 2 ;; esac
MODEL="$RUN/model.pt"
[ -f "$MODEL" ] || { echo "no $MODEL" >&2; exit 2; }
NAME="$(basename "$RUN")"
OUT="$RUN/scores"
mkdir -p "$OUT"
FAILED=()

# step NAME OUTFILE CMD...: run CMD with stdout and stderr to OUTFILE; record a non-zero exit
step() {
  local label="$1" file="$2"; shift 2
  echo "$(date '+%Y-%m-%d %H:%M:%S')  $label -> ${file#$RUN/}"
  if ! "$@" > "$file" 2>&1; then
    FAILED+=("$label"); echo "    failed (exit status non-zero; see $file)"
  fi
}

in_repo() { (cd "$REPO" && "$@"); }

export PYTHONPATH="$REPO:$B4G${PYTHONPATH:+:$PYTHONPATH}"
cd "$B4G"

case "$NAME" in
  Ag_cluster_*|NaCl_*|Carbon_chain_*)
    case "$NAME" in Ag_cluster_*) DATASET=Ag_cluster ;; NaCl_*) DATASET=NaCl ;; *) DATASET=Carbon_chain ;; esac
    step "eval_4g ($DATASET, data_pfn test set)" "$OUT/eval_4g.txt" env EVAL_DATA_DIR=data_pfn "$PY" eval_4g.py "$DATASET" "$RUN"
    ;;
  *)
    step "phonons, 2x2x2 (64 atoms)" "$OUT/P_phonons_pbe.txt" \
      "$PY" "$PH/score_phonons.py" "$MODEL" --out "$OUT/phonons_sc2"
    step "phonons, 4x4x4 (512 atoms)" "$OUT/P_phonons_pbe_sc4.txt" \
      "$PY" "$PH/score_phonons.py" "$MODEL" --supercell 4 --out "$OUT/phonons_sc4"
    step "phonons, 2x2x2 against 4x4x4 (C1)" "$OUT/P_phonons_sc2_vs_sc4.txt" \
      "$PY" "$PH/compare_sc2_sc4.py" "$OUT/phonons_sc2" "$OUT/phonons_sc4"
    # probe_coupling_sizes.py and probe_class_residuals.py read data paths relative to the repository root
    step "coupling sizes" "$OUT/Z1_Z2_couplings.txt" in_repo "$PY" "$B4G/probe_coupling_sizes.py" "$MODEL"
    if [ "$STAGE" = b ]; then
      step "Table II (T5) and test RMSE" "$OUT/T5_wetting.txt" "$PY" eval_aumgo.py "$RUN"
      step "transfer set (T1-T3)" "$OUT/T1_T3_transfer.txt" env EVAL_THREADS="$THREADS" "$PY" eval_aumgo_transfer.py "$RUN"
      step "class residuals, test" "$OUT/H1_class_residuals.txt" in_repo "$PY" "$B4G/probe_class_residuals.py" "$MODEL"
      step "class residuals, train" "$OUT/H1_class_residuals_train.txt" in_repo env CLASS_SPLIT=train "$PY" "$B4G/probe_class_residuals.py" "$MODEL"
      step "S7 Au-Au scan" "$OUT/S7_auau_scan.txt" "$PY" probe_auau_two_site.py "$RUN" "threads=$THREADS" h=0.0005
      step "S8 force outliers" "$OUT/S8_force_outliers.txt" \
        "$PY" probe_force_outliers.py "$RUN" data/AuMgO_test.xyz "threads=$THREADS" "out=$OUT/S8_force_outliers.npz"
      step "S6 Au2 EA / IP" "$OUT/S6_au2_ea_ip.txt" "$PY" probe_au2_ea_ip.py "$RUN" 2.30,2.47,2.60,2.80
      step "S7, S8, S5 summary" "$OUT/S7_S8.txt" "$PY" score_step34_s7_s8.py "$RUN"
    fi
    ;;
esac

echo
grep -h -E "passed;|^S[578]:|PASS|FAIL" "$OUT"/P_phonons_pbe*.txt "$OUT"/S7_S8.txt 2>/dev/null | grep -E "passed;|^S[578]:" | cut -c1-160
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "$NAME: ${#FAILED[@]} score(s) failed: ${FAILED[*]}"; exit 1
fi
echo "$NAME scored; outputs in $OUT"
