# Shared settings for the lab scripts.  Override any of them in the environment.
LAB=${LAB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"   # untracked machine settings
ROOT=${ROOT:-$HOME/Strata}                            # the patched Strata checkout (engine/strata built from it)
PY=${PY:-$ROOT/.venv/bin/python}                      # python with serve/server.py's requirements
STRATA_DATA=${STRATA_DATA:-$HOME/Strata-data}         # packs/, models/, mtp/ from setup.py
PROFILE=${PROFILE:-$ROOT/data/expert-profile.bin}
VISION_EXE=${VISION_EXE:-$ROOT/engine/strata-vision}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
LAB_PORT=${LAB_PORT:-8095}
URL=http://127.0.0.1:$LAB_PORT
# Optional: a vllm_my-style dashboard that runs a production Strata on the same GPU.  When set, lab-start stops it
# and lab-stop starts it again with $DASH_START_JSON.  Leave empty when the GPU is free.
DASH=${DASH:-}
DASH_START_JSON=${DASH_START_JSON:-}
