#!/bin/bash
# Apply the --batch kernel patch to a Strata git clone.
# Usage: ./apply.sh /path/to/Strata [--experimental]
#   --experimental  also apply the multi-draft batch MTP patch (no speed gain, for research only)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
S=${1:?usage: ./apply.sh /path/to/Strata [--experimental]}
[ -e "$S/.git" ] || { echo "$S is not a git clone of Strata"; exit 1; }
cd "$S"
if git apply --reverse --check "$HERE"/patches/*/batch-row-kernels.patch 2>/dev/null; then
  echo "already applied"; exit 0
fi
# newest patch first: patches/latest (made against the newest upstream tested), then older versions
for p in "$HERE"/patches/latest/batch-row-kernels.patch $(ls -r "$HERE"/patches/v*/batch-row-kernels.patch); do
  [ -f "$p" ] || continue
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p"
    echo "applied $(basename "$(dirname "$p")")/$(basename "$p")"
    if [ "${2:-}" = "--experimental" ]; then
      git apply "$HERE/patches/experimental/batch-mtp-multi-draft.patch" && echo "applied experimental/batch-mtp-multi-draft.patch"
    fi
    cat <<'MSG'

Next:
  1. rebuild the engine from source:   ./setup.sh --build --no-start
  2. turn it on: add to your strata-<model>.json (setup keeps this block):
       "env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
  3. start with parallel requests, e.g. "parallel": 4 in the same file (or --batch 4)

Strata's ./update.sh runs "git pull": run "git checkout -- ." first, update, then apply this patch again.
MSG
    exit 0
  fi
done
echo "no patch here applies to this Strata version ($(git log -1 --format='%h %cd' --date=short)); open an issue"
exit 1
