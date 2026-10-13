#!/bin/bash
# 階段 2：拆解每一輪的時間。STRATA_DECODE_TIMING（單請求路徑）+ STRATA_VERIFY_PROFILE（各步驟 GPU 時間）
# 注意：profile 會關掉部分 kernel 融合，絕對時間會變大，看比例與「每多一行增加多少」
. "$(dirname "$0")/env.sh"
OUT=$LAB/results/phase2; mkdir -p $OUT
trap '$LAB/lab-stop.sh' EXIT
prof() {  # $1 label, $2 extra env
  local L=$1; shift
  env "$@" $LAB/lab-start.sh 4 || exit 1
  : > $LAB/engine.log.mark; local start=$(wc -l < $LAB/engine.log)
  for n in 1 2 3 4; do
    $PY $LAB/bench_concurrency.py --url $URL --clients $n --runs 1 --max-tokens 512 --label $L
  done
  tail -n +$start $LAB/engine.log > $OUT/engine-$L.log
}
prof timing STRATA_DECODE_TIMING=1
prof profile STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1
