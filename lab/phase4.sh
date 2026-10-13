#!/bin/bash
# 階段 4：並行時 attention 一次處理所有行（STRATA_BATCH_ATTN_ROWS=1）vs 上游逐行
. "$(dirname "$0")/env.sh"
OUT=$LAB/results/phase4; mkdir -p $OUT
trap '$LAB/lab-stop.sh' EXIT
run() {  # $1 label, rest: env
  local L=$1; shift
  env "$@" $LAB/lab-start.sh 4 || exit 1
  local start=$(wc -l < $LAB/engine.log)
  $PY $LAB/parity.py $URL 4 $OUT/parity-$L.json
  $PY $LAB/bench_concurrency.py --url $URL --clients 1,2,3,4 --runs 2 --max-tokens 1024 \
      --label $L --json-out $OUT/$L.json
  tail -n +$start $LAB/engine.log > $OUT/engine-$L.log
}
run base
run attnrows STRATA_BATCH_ATTN_ROWS=1
echo "== parity base vs attnrows"; $PY $LAB/compare_parity.py $OUT/parity-base.json $OUT/parity-attnrows.json
