#!/bin/zsh
# 2026-10-25: finishes step 18's pre-registered scoring (as watch_step18.sh), split off the watcher so that the doped
# wetting decomposition can run beside it: waits for s42's running window-deficit probe (its PID given as the argument), runs s42's remaining probes,
# then scores s43 in full.
set -u
cd "$(dirname "$0")"
S42=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s42
S43=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s43
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 120; done; done   # the PID given as the argument (none: start at once)
extras() {
  local r=$1
  (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_eps_bound.py experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/T6_eps_bound.txt 2>&1)
  (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_alpha_scorecard.py experiments/benchmarks_4g/data/Au2_charges_alpha_field_prior.xyz experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/F_alpha_scorecard.txt 2>&1)
  (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_explicit_gap.py Au 2.3,2.5,2.8,3.4 experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/G_explicit_gap.txt 2>&1)
  nice -n 5 python probe_gauge_split.py runs/$r/model.pt > runs/$r/scores/G_gauge_split.txt 2>&1
}
extras $S42
echo "s42 scored in full $(date)"
nice -n 5 ./score_formal.sh $S43
nice -n 5 python probe_window_deficit.py runs/$S43 > runs/$S43/scores/R_window_deficit.txt 2>&1
extras $S43
echo "step 18 scored $(date)"
