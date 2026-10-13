#!/bin/bash
# Start the lab Strata (serve/server.py + engine built from this checkout) on $LAB_PORT.
# Usage: lab-start.sh <parallel> [extra engine args...]     e.g. lab-start.sh 4 --batch-mtp
# STRATA_* environment variables reach the engine.  LAB_RESERVE replaces --vram-reserve-mib.
set -u
. "$(dirname "$0")/env.sh"
PAR=${1:-1}; shift || true
if [ -n "$DASH" ]; then   # stop the production instance first (only when it runs: its stop waits for a free GPU)
  st=$(curl -sf -m 10 $DASH/api/status) || { echo "dashboard $DASH not answering: abort"; exit 1; }
  echo "$st" | grep -q '"running":true' && curl -s -m 120 -X POST $DASH/api/stop >/dev/null
fi
"$LAB/lab-kill.sh"
# the lab needs ~60 GB: abort when something else still holds the GPU
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -gt 8000 ] && { echo "GPU still has ${used} MiB used: abort"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; }
sed -e "s|@ROOT@|$ROOT|g; s|@DATA@|$STRATA_DATA|g; s|@PROFILE@|$PROFILE|g; s|@LAB@|$LAB|g; s|@CUDA@|$CUDA_HOME|g" \
    -e "s|@PORT@|$LAB_PORT|g; s|@VISION@|$VISION_EXE|g" "$LAB/lab-config.template.json" > "$LAB/run.json.in"
"$PY" - "$LAB" "$PAR" "$@" <<'PYEOF'
import json, os, sys
lab, par, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
c = json.load(open(f"{lab}/run.json.in"))
if par > 1: c["parallel"] = par
c["args"] += extra
if os.environ.get("LAB_RESERVE"):
    a = c["args"]; a[a.index("--vram-reserve-mib") + 1] = os.environ["LAB_RESERVE"]
json.dump(c, open(f"{lab}/run.json", "w"), indent=1)
PYEOF
cd "$ROOT"
# own session / process group: its PGID goes to lab.pgid, lab-kill ends the whole group (server + engine + vision).
# Never `pkill -f` a path here: it also matches the shell running this script.
setsid bash -c 'echo $$ > "$1/lab.pgid"; exec "$0" serve/server.py --engine strata --config "$1/run.json" --port "$2"' \
  "$PY" "$LAB" "$LAB_PORT" > "$LAB/server.out" 2>&1 < /dev/null &
sleep 1
for i in $(seq 1 180); do
  curl -s $URL/health 2>/dev/null | grep -q '"loaded": true' && { echo "lab ready (parallel=$PAR $*)"; exit 0; }
  kill -0 "$(cat "$LAB/lab.pgid" 2>/dev/null)" 2>/dev/null || { echo "lab server died"; tail -20 "$LAB/server.out"; exit 1; }
  sleep 5
done
echo "lab start timeout"; exit 1
