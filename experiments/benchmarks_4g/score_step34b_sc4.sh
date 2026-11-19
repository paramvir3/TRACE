#!/bin/zsh
# Step 34b's final phonon scoring on PBE's own force-constant supercell (decision of 2026-11-16: report both supercells
# in the final scoring; README, "Step 34b's final scoring reports both supercells").  Not a criterion: S4 is scored as
# pre-registered on the 2x2x2 conventional supercell (64 atoms) by the final scoring (score_model.sh): its output
# runs/<run>/scores/P_phonons_pbe.txt is read here, and score_phonons.py's 2x2x2 band yaml sits by default under
# experiments/benchmarks_phonons/runs/<run>/.  This
# adds, for the same four checkpoints, P0-P6 with the force constants on 4x4x4 (512 atoms, Togo's fc2 supercell), the
# 2x2x2 / 4x4x4 consistency check C1 (Gamma, X, L within 1 cm^-1) and the figures.  Run it after the final scoring;
# given a PID, it first waits for that process (e.g. the scoring) to exit.  Each model.pt's md5 is checked before and
# after.  A run without a 2x2x2 score is skipped.
#     zsh score_step34b_sc4.sh [PID]
# The paths below can be overridden from the environment (used only to test this script on scratch copies).
set -u
H=$(cd "$(dirname "$0")/../.." && pwd); B=$H/experiments/benchmarks_4g; PH=$H/experiments/benchmarks_phonons
RB=${SC4_RUNS:-$B/runs}            # runs/<run>/model.pt and runs/<run>/scores/P_phonons_pbe.txt (the final scoring's)
PR=${SC4_PHONONS:-$PH/runs}        # <run>/ holds the final scoring's 2x2x2 band yaml; <run>_sc4/ gets the 4x4x4 one
FIG=${SC4_FIGURES:-$PH}
LOG=${SC4_LOG:-$B/runs/step34b_sc4.log}
md5of() { if command -v md5 > /dev/null; then md5 -q $1; else md5sum $1 | cut -d' ' -f1; fi; }
TIMER=(/usr/bin/time -l); [ "$(uname)" = Linux ] && TIMER=(/usr/bin/time -v)   # peak memory: BSD -l, GNU -v
STEMS=(${=SC4_STEMS:-AuMgO_twosite_bf055_e1kw_bondlaw_r5 AuMgO_twosite_bf055_bondlaw_r5})   # stage B first, as the final scoring
W=${1:-}
if [ -n "$W" ]; then
  echo "$(date): waiting for PID $W to end" >> $LOG
  while kill -0 $W 2>/dev/null; do sleep 300; done
fi
typeset -A ep
for stem in $STEMS; do
  for s in 42 43; do
    r=${stem}_s$s; m=$RB/$r/model.pt; s2=$RB/$r/scores/P_phonons_pbe.txt; s4=$RB/$r/scores/P_phonons_pbe_sc4.txt
    grep -q "passed" $s2 2>/dev/null || { echo "$(date): $r has no 2x2x2 score; not scored on 4x4x4" >> $LOG; continue; }
    md0=$(md5of $m)
    ep[$r]=$(python -c "import torch; print(torch.load('$m', map_location='cpu', weights_only=False)['epoch'])")
    (cd $H && $TIMER nice -n 5 python experiments/benchmarks_phonons/score_phonons.py $m --supercell 4 --out $PR/${r}_sc4 \
        > $s4 2> $RB/$r/scores/P_phonons_pbe_sc4_time.txt)
    python $PH/compare_sc2_sc4.py $PR/$r $PR/${r}_sc4 > $RB/$r/scores/P_phonons_sc2_vs_sc4.txt 2>&1
    [ "$(md5of $m)" = "$md0" ] && same=same || same=CHANGED
    { echo "--- $(date '+%Y-%m-%d %H:%M') $r, best checkpoint epoch ${ep[$r]} (model.pt md5 $md0, $same after scoring)"
      echo "  2x2x2 (64 atoms, pre-registered):"; grep -E "PASS|FAIL|passed" $s2 | sed 's/; log and band yaml.*//' | cut -c1-110 | sed 's/^/  /'
      echo "  4x4x4 (512 atoms, PBE's supercell; recorded, not a criterion):"; grep -E "PASS|FAIL|passed" $s4 | sed 's/; log and band yaml.*//' | cut -c1-110 | sed 's/^/  /'
      grep -E "C1|RMS model" $RB/$r/scores/P_phonons_sc2_vs_sc4.txt | cut -c1-200
      grep -i -E "maximum resident" $RB/$r/scores/P_phonons_pbe_sc4_time.txt; } >> $LOG
  done
  st=B; [ $stem = AuMgO_twosite_bf055_bondlaw_r5 ] && st=A
  e42=${ep[${stem}_s42]:-}; e43=${ep[${stem}_s43]:-}
  [ -n "$e42" ] && [ -n "$e43" ] || continue
  f1=$FIG/step34b_${st}_final_s42ep${e42}_s43ep${e43}_nonac_sc2_vs_sc4.png; f2=$FIG/step34b_${st}_final_s42ep${e42}_s43ep${e43}_nac_sc4_vs_pbe.png
  (cd $PH && python plot_nonac_sc2_sc4.py $f1 \
     "Step 34b stage $st, final (s42 epoch $e42, s43 epoch $e43): MgO phonons without NAC vs PBE without NAC"$'\n'"force constants on 64 atoms (top) and on 512 atoms, PBE's own supercell (bottom)" \
     $PR/${stem}_s42 $PR/${stem}_s43 > /dev/null 2>> $LOG) \
  && (cd $PH && python plot_vs_pbe.py $f2 \
     "Step 34b stage $st, final (s42 epoch $e42, s43 epoch $e43): MgO phonons with NAC vs PBE, force constants on 512 atoms" \
     "TRACE step 34b (NAC, 512 atoms)" $PR/${stem}_s42_sc4/band_nac.yaml $PR/${stem}_s43_sc4/band_nac.yaml > /dev/null 2>> $LOG) \
  && echo "    figures: $f1 and $f2" >> $LOG
done
echo "$(date): step 34b's 512-atom phonon scoring done; read $LOG" >> $LOG
