#!/bin/bash
# 階段 3a：batch-mtp 每個 slot 多猜幾個（STRATA_BATCH_MTP_D），不改 kVerifyMaxT
. "$(dirname "$0")/env.sh"
OUT=$LAB/results/phase3a; mkdir -p $OUT
trap '$LAB/lab-stop.sh' EXIT
run() {  # $1 label, $2 parallel, rest: engine args ; env vars set by caller
  local L=$1 P=$2; shift 2
  $LAB/lab-start.sh $P "$@" || exit 1
  $PY $LAB/parity.py $URL 2 $OUT/parity-$L.json
  $PY $LAB/bench_concurrency.py --url $URL --clients ${CLIENTS:-1,2,3,4} --runs 2 --max-tokens 1024 \
      --label $L --json-out $OUT/$L.json
  grep -E "strata batch:" $LAB/engine.log | tail -6 > $OUT/batchstats-$L.txt
  cp $LAB/engine.log $OUT/engine-$L.log
}
run p4 4
STRATA_BATCH_MTP_D=1 run p4mtp-d1 4 --batch-mtp
STRATA_BATCH_MTP_D=3 run p4mtp-d3 4 --batch-mtp
for L in p4mtp-d1 p4mtp-d3; do echo "== parity p4 vs $L"; $PY $LAB/compare_parity.py $OUT/parity-p4.json $OUT/parity-$L.json; done
