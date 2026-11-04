#!/bin/bash
# Front-camera-only DAgger student; defaults to four GPUs and 4096 total environments.
# Uses the shipped height-map teachers. Start a fresh run, not a dual-camera student resume.
# Same SIM, LOGGER, WANDB_* and extra CLI arguments as run_distill.sh.
set -euo pipefail
export NGPUS="${NGPUS:-4}"
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
rpl_launcher_init
rpl_activate_sim
rpl_training_command
exec "${TRAIN_COMMAND[@]}" -m rpl_training.train_agent \
    exp:g1-29dof-rpl-distill-front "${SIM_ARGS[@]}" \
    --training.headless True \
    "${LOGGER_ARGS[@]}" \
    "$@"
