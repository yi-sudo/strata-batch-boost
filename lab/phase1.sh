#!/bin/bash
# 階段 1：實驗版不改程式碼，p1 / p4 基準，對照正式版 (p1 180 / p4@4 268)
. "$(dirname "$0")/env.sh"
OUT=$LAB/results/phase1; mkdir -p $OUT
trap '$LAB/lab-stop.sh' EXIT
for P in 1 4; do
  $LAB/lab-start.sh $P || exit 1
  $PY $LAB/bench_concurrency.py --url $URL --clients 1,2,4 --runs 2 --max-tokens 1024 \
      --label lab-p$P --json-out $OUT/p$P.json
  cp $LAB/engine.log $OUT/engine-p$P.log 2>/dev/null
done
