#!/bin/bash
# 階段 6：p8，experts 全在 VRAM（降低 reserve），上游 vs 兩個合併修改
. "$(dirname "$0")/env.sh"
OUT=$LAB/results/phase6; mkdir -p $OUT
trap '$LAB/lab-stop.sh' EXIT
run() {  # $1 label, rest: env
  local L=$1; shift
  env LAB_RESERVE=${RES:-6144} "$@" $LAB/lab-start.sh 8 || exit 1
  local start=$(wc -l < $LAB/engine.log)
  grep -E "expert cache [0-9]+ slots" $LAB/engine.log | tail -1 | tee $OUT/cache-$L.txt
  $PY $LAB/bench_concurrency.py --url $URL --clients 1,4,6,8 --runs 2 --max-tokens 1024 \
      --label $L --json-out $OUT/$L.json
  tail -n +$start $LAB/engine.log > $OUT/engine-$L.log
}
run p8-upstream
run p8-attn+gdn STRATA_BATCH_ATTN_ROWS=1 STRATA_BATCH_GDN_GROUPS=1
