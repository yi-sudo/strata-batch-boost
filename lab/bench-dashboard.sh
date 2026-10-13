#!/bin/bash
# Strata 單請求 vs 多並發 tokens/s 比較。
# 透過 vllm_my 風格的控制台 ($DASH, /api/start /api/stop /api/status) 依序用不同設定重啟 Strata，每種設定各跑 1/2/4/8 個同時請求。
#
# 用法：
#   lab/bench-dashboard.sh                # 跑全部設定
#   lab/bench-dashboard.sh p1 p4          # 只跑指定設定
#   CLIENTS=1,4 RUNS=1 MAX_TOKENS=512 lab/bench-dashboard.sh p4
#   NO_RESTORE=1 ...                                  # 跑完不切回 p1
#
# 設定：
#   p1     strata-parallel 1（現狀：一次一個，其他排隊）
#   p4     strata-parallel 4
#   p4mtp  strata-parallel 4 + --batch-mtp
#   p8     strata-parallel 8
#   p8mtp  strata-parallel 8 + --batch-mtp
# 結果：~/strata-test/bench-results/<時間>/summary.txt 與各設定 JSON
set -u
DASH=${DASH:-http://127.0.0.1:3567}
API=${API:-http://127.0.0.1:8081}
MODEL_KEY=${MODEL_KEY:-"Qwen3.8-Flash-Next-IQ3_S-Strata"}
CLIENTS=${CLIENTS:-1,2,4,8}
RUNS=${RUNS:-2}
MAX_TOKENS=${MAX_TOKENS:-1024}
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/bench-results/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

declare -A CFG=(
  [p1]='"strata-parallel":1'
  [p4]='"strata-parallel":4'
  [p4mtp]='"strata-parallel":4,"batch-mtp":true'
  [p8]='"strata-parallel":8'
  [p8mtp]='"strata-parallel":8,"batch-mtp":true'
)
ORDER=(p1 p4 p4mtp p8 p8mtp)
[ $# -gt 0 ] && ORDER=("$@")
BASE='"vram-reserve-mib":20480,"max-context":262144,"kv":"int8"'

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/summary.txt"; }

# 其他程式在用 GPU 會讓數字偏低，先提醒
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v strata)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
log "GPU 使用率 ${util}%；其他 GPU 程序："
echo "${others:-  (無)}" | tee -a "$OUT/summary.txt"
[ "${util:-0}" -gt 30 ] && log "⚠️  GPU 目前很忙，結果會偏低（Strata 閒置時 GPU 應接近 0%）"

start_cfg() {  # $1 = 設定名
  local extra=${CFG[$1]}
  curl -s -X POST $DASH/api/stop >/dev/null
  for i in $(seq 1 60); do pgrep -f "engine/strata " >/dev/null || break; sleep 2; done
  curl -s -X POST $DASH/api/start -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_KEY\",\"params\":{${BASE},${extra}}}" >/dev/null
  for i in $(seq 1 180); do   # 最多等 15 分鐘
    st=$(curl -s $DASH/api/status | $PY -c 'import json,sys;d=json.load(sys.stdin);print(d["status"])' 2>/dev/null)
    [ "$st" = ready ] && return 0
    [ "$st" = error ] && return 1
    sleep 5
  done
  return 1
}

for name in "${ORDER[@]}"; do
  [ -z "${CFG[$name]+x}" ] && { log "未知設定 $name，略過"; continue; }
  log "=== $name (${CFG[$name]}) 啟動中 ==="
  if ! start_cfg "$name"; then log "❌ $name 啟動失敗，看 ~/vllm_my/logs/"; continue; fi
  slots=$(curl -s $API/metrics | $PY -c 'import json,sys;e=json.load(sys.stdin)["engine"];print(e.get("expert_slots"),"slots /", e.get("vram_free_mib"),"MiB free")' 2>/dev/null)
  log "ready：experts $slots"
  $PY "$HERE/bench_concurrency.py" --url $API --clients "$CLIENTS" --runs "$RUNS" \
      --max-tokens "$MAX_TOKENS" --label "$name" --json-out "$OUT/$name.json" 2>&1 | tee -a "$OUT/summary.txt"
done

# 彙整成一張表
$PY - "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    rows += json.load(open(f))
if not rows: sys.exit()
base = next((r["per_req_tok_s"] for r in rows if r["label"] == "p1" and r["clients"] == 1), None)
print("\n設定    同時  總吞吐tok/s  單請求tok/s  TTFT平均  TTFT最久  錯誤")
for r in rows:
    rel = f" ({r['aggregate_tok_s']/base:.2f}x)" if base else ""
    print(f"{r['label']:<7} {r['clients']:>4}  {r['aggregate_tok_s']:8.1f}{rel:<9} {r['per_req_tok_s']:8.1f}   "
          f"{r['ttft_mean_s']:7.2f}s  {r['ttft_max_s']:7.2f}s  {r['errors']}")
EOF

if [ -z "${NO_RESTORE:-}" ]; then
  log "切回原設定 p1 ..."
  start_cfg p1 && log "已恢復 strata-parallel 1" || log "❌ 恢復失敗，請到 3567 手動啟動"
fi
log "結果在 $OUT/summary.txt"
