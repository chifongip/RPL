#!/bin/bash
# One-machine X2 training. Flat/slopes default to one GPU; other stages use four.
# num-envs is global. The depth student retains 8192 total environments on four GPUs.
# TASK=flat|expert|stones|stairs|slopes|distill; CLI overrides follow the preset.
set -euo pipefail
case "${TASK:-expert}" in
    flat) preset=x2-29dof-rpl-flat ;;
    expert) preset=x2-29dof-rpl-expert ;;
    stones|stairs|slopes) preset="x2-29dof-rpl-expert-${TASK}" ;;
    distill) preset=x2-29dof-rpl-distill ;;
    *) echo "ERROR: TASK must be flat, expert, stones, stairs, slopes or distill" >&2; exit 1 ;;
esac
if [[ "${TASK:-expert}" == flat || "${TASK:-expert}" == slopes ]]; then
    export NGPUS="${NGPUS:-1}"
else
    export NGPUS="${NGPUS:-4}"
fi
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
rpl_launcher_init
[[ "$SIM" == isaacgym ]] || { rpl_error "X2 runtime is validated with IsaacGym"; exit 1; }
rpl_activate_sim
rpl_training_command
exec "${TRAIN_COMMAND[@]}" -m rpl_training.train_agent "exp:$preset" \
    --training.headless True "${LOGGER_ARGS[@]}" "$@"
