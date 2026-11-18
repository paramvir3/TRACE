#!/bin/zsh
# Watcher (2026-10-19): when LOG contains PATTERN, run score_extra.sh for each RUN whose training finished.
# Usage: watch_extra.sh LOG PATTERN RUN [RUN ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd $HERE
LOG=$1; PATTERN=$2; shift 2
while ! grep -q "$PATTERN" $LOG 2>/dev/null; do sleep 300; done
echo "$PATTERN seen $(date)"
for r in "$@"; do
  if grep -q "Training finished" runs/$r/log.txt 2>/dev/null; then ./score_extra.sh $r; else echo "skip $r (not finished)"; fi
done
echo "extra scored: $PATTERN $(date)"
