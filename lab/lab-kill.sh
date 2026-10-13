#!/bin/bash
# Stop the lab's process group (PGID from lab.pgid); never touches anything else.
. "$(dirname "$0")/env.sh"
F=$LAB/lab.pgid
[ -f "$F" ] || exit 0
G=$(cat "$F")
kill -TERM -- -"$G" 2>/dev/null
for i in $(seq 1 30); do pgrep -g "$G" >/dev/null || break; sleep 1; done
kill -KILL -- -"$G" 2>/dev/null
rm -f "$F"
