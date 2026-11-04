#!/bin/bash
# Activate Holosoma's isaacgym conda env for RPL. Source it, don't execute it:
#   source scripts/source_isaacgym_setup.sh
# Uses thirdparty/holosoma unless HOLOSOMA_LOCAL_DIR points at another checkout.
# ENV_NAME overrides the conda env name (Holosoma's default otherwise).

_rpl_script_dir() {
  local source="${BASH_SOURCE[0]:-${(%):-%x}}"
  cd -P "$( dirname "$source" )" >/dev/null && pwd
}
RPL_ROOT_DIR="$(dirname "$(_rpl_script_dir)")"
unset -f _rpl_script_dir
HOLOSOMA_DIR="${HOLOSOMA_LOCAL_DIR:-$RPL_ROOT_DIR/thirdparty/holosoma}"

if [ ! -f "$HOLOSOMA_DIR/scripts/source_isaacgym_setup.sh" ]; then
    echo "ERROR: holosoma not found at $HOLOSOMA_DIR. Run: git submodule update --init --recursive" >&2
    return 1
fi
echo "[rpl] Using holosoma from $HOLOSOMA_DIR"
if [ -n "${ENV_NAME:-}" ]; then export CONDA_ENV_NAME="$ENV_NAME"; fi
# Holosoma appends to this variable; launchers may enable nounset in a fresh shell.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$HOLOSOMA_DIR/scripts/source_isaacgym_setup.sh"
