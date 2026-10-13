#!/bin/bash
# --batch 4 VRAM vs --max-context: restart Strata through a vllm_my-style dashboard at each setting, record the GPU
# memory and a short 1- and 4-client run, then restore p4 / 262144.
DASH=${DASH:-http://127.0.0.1:3567}; API=${API:-http://127.0.0.1:8081}; PY=${PY:-python3}
MODEL_KEY=${MODEL_KEY:-Qwen3.8-Flash-Next-IQ3_S-Strata}; HERE=$(cd "$(dirname "$0")" && pwd)
start() {  # $1 max-context, $2 parallel
  curl -s -X POST $DASH/api/stop -m 120 >/dev/null
  for i in $(seq 1 60); do pgrep -x strata >/dev/null || break; sleep 2; done
  curl -s -X POST $DASH/api/start -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_KEY\",\"params\":{\"vram-reserve-mib\":20480,\"max-context\":$1,\"kv\":\"int8\",\"strata-parallel\":$2}}" >/dev/null
  for i in $(seq 1 180); do
    st=$(curl -s $DASH/api/status | $PY -c 'import json,sys;print(json.load(sys.stdin)["status"])' 2>/dev/null)
    [ "$st" = ready ] && return 0; [ "$st" = error ] && return 1; sleep 5; done; return 1
}
for cfg in "131072 4" "65536 4" "262144 1"; do
  set -- $cfg
  start $1 $2 || { echo "ctx=$1 p=$2 FAILED"; continue; }
  sleep 5
  eng=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | grep "engine/strata," | awk -F, '{print $2}')
  tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  m=$(curl -s $API/metrics | $PY -c 'import json,sys;e=json.load(sys.stdin)["engine"];print("experts",e["expert_slots"],"arena",e["arena_mib"],"free",e["vram_free_mib"])')
  echo "ctx=$1 p=$2  strata=$eng  gpu_total=$tot  $m"
  $PY "$HERE/bench_concurrency.py" --url $API --clients 1,4 --runs 1 --max-tokens 512 --label c$1p$2 2>&1 | grep clients
done
start 262144 4 && echo "restored p4 262144"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
