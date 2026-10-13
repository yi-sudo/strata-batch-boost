#!/bin/bash
# Stop the lab; with $DASH set, start the production instance again with $DASH_START_JSON.
. "$(dirname "$0")/env.sh"
"$LAB/lab-kill.sh"
[ -z "$DASH" ] && exit 0
curl -s -X POST $DASH/api/start -H 'Content-Type: application/json' -d "$DASH_START_JSON" >/dev/null
for i in $(seq 1 120); do
  st=$(curl -s $DASH/api/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])' 2>/dev/null)
  [ "$st" = ready ] && { echo "production restored"; exit 0; }
  sleep 5
done
echo "production restore NOT confirmed — check $DASH"; exit 1
