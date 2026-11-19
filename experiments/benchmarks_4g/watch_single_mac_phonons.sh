#!/bin/zsh
# The single model's early look on the Mac (decision of 2026-11-12: run it on the Mac and compute the phonons first):
# trains stage A of step 33, seed 43, on the Mac's CPU, and scores the phonons against PBE DFT at every new best
# checkpoint (a copy, so the training's file is never read while it is written).  Not the pre-registered run: that
# is the two-seed GPU run on the GPU host; this is a mid-training read, updated every epoch.
set -u
H=$(cd "$(dirname "$0")/../.." && pwd); B=$H/experiments/benchmarks_4g; cd $B
r=AuMgO_single_all_mac_s43
nohup nice -n 5 python run_train_nanguard.py runs/$r/config.yaml > runs/$r/log.txt 2>&1 &
P=$!; echo "$(date): launched $r pid $P"
last=""
while kill -0 $P 2>/dev/null; do
  sleep 120
  [ -f runs/$r/model.pt ] || continue
  now=$(stat -f %m runs/$r/model.pt)
  [ "$now" = "$last" ] && continue
  last=$now; sleep 5
  mkdir -p runs/${r}_mid && cp runs/$r/model.pt runs/${r}_mid/model.pt
  ep=$(python -c "import torch, math; c = torch.load('runs/${r}_mid/model.pt', map_location='cpu', weights_only=False); print(c['epoch'], round(math.exp(float(c['model_state_dict']['electrostatics.log_alpha_scale'])), 4))")
  (cd $H && nice -n 5 python experiments/benchmarks_phonons/score_phonons.py experiments/benchmarks_4g/runs/${r}_mid/model.pt > experiments/benchmarks_4g/runs/${r}_mid/P_phonons_pbe.txt 2>&1)
  echo "--- $(date '+%H:%M') checkpoint epoch and crystal factor: $ep"
  grep "PASS\|FAIL\|passed" runs/${r}_mid/P_phonons_pbe.txt | sed 's/ (population.*): population/: population/' | cut -c1-110
  grep -E "^\[t=.*\] +[0-9]+ \|" runs/$r/log.txt | tail -1 | awk -F'|' '{print "    training: epoch", $1, "| E", $3, "| force RMSE", $5}' | sed 's/\[t=[^]]*\]//'
done
echo "$(date): $r exited ($(grep -c 'Training finished' runs/$r/log.txt) finished, $(grep -c Traceback runs/$r/log.txt) tracebacks)"
