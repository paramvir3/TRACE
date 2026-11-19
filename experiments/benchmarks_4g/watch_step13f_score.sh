#!/bin/zsh
# Score step 13F when its resumed runs (PIDs given as arguments; resumed after the 22:03 reboot) end.
set -u
cd "$(dirname "$0")"
S13F=(AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fprior_bondlaw_r5_s42 AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fprior_bondlaw_r5_s43)
for p in "$@"; do while ps -p $p > /dev/null; do sleep 300; done; done   # the PIDs given as arguments (none: start at once)
echo "step 13F ended $(date)"
for r in $S13F; do
  grep -E "Training finished|Traceback|FIRST NON-FINITE" runs/$r/log.txt | head -2
  if grep -q "Training finished" runs/$r/log.txt; then
    nice -n 5 ./score_formal.sh $r
    nice -n 5 python probe_window_deficit.py runs/$r > runs/$r/scores/R_window_deficit.txt 2>&1
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_alpha.py Au 2.30,2.47,2.60,2.80,3.10 experiments/benchmarks_4g/runs/$r/model.pt > experiments/benchmarks_4g/runs/$r/scores/F_dimer_alpha.txt 2>&1)
    (cd ../.. && nice -n 5 python experiments/benchmarks_4g/probe_dimer_ct.py experiments/benchmarks_4g/runs/$r/model.pt Au 2.47 > experiments/benchmarks_4g/runs/$r/scores/D_dimer_ct.txt 2>&1)
  fi
done
echo "step 13F scored $(date)"
