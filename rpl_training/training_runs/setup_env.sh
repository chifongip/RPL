#!/bin/bash
# Execute to bootstrap RPL; source to load the shared launcher helpers.
#   SIM=isaacgym (default, FALCON's simulator) | isaacsim
#   ENV_NAME overrides the conda env name; HOLOSOMA_LOCAL_DIR another holosoma checkout.

rpl_error() {
    echo "ERROR: $*" >&2
    return 1
}

rpl_launcher_init() {
    RPL_TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    RPL_REPO_ROOT="$(cd "$RPL_TRAINING_DIR/../.." && pwd)"
    SIM="${SIM:-isaacgym}"
    case "$SIM" in
        isaacgym | isaacsim) ;;
        *) rpl_error "SIM must be isaacgym or isaacsim (got '$SIM')"; return 1 ;;
    esac
    # The presets default to IsaacGym with FALCON's PhysX settings (isaacgym_falcon).
    SIM_ARGS=()
    [[ "$SIM" == isaacgym ]] || SIM_ARGS=("simulator:$SIM")
    NGPUS="${NGPUS:-1}"
    [[ "$NGPUS" =~ ^[1-9][0-9]*$ ]] || { rpl_error "NGPUS must be a positive integer (got '$NGPUS')"; return 1; }
    LOGGER="${LOGGER:-wandb}"
    case "$LOGGER" in
        disabled) LOGGER_ARGS=(logger:disabled) ;;
        wandb)
            LOGGER_ARGS=(logger:wandb)
            [[ -z "${WANDB_PROJECT:-}" ]] || LOGGER_ARGS+=(--logger.project "$WANDB_PROJECT")
            [[ -z "${WANDB_ENTITY:-}" ]] || LOGGER_ARGS+=(--logger.entity "$WANDB_ENTITY")
            [[ -z "${RUN_NAME:-}" ]] || LOGGER_ARGS+=(--logger.name "$RUN_NAME")
            ;;
        *) rpl_error "LOGGER must be disabled or wandb (got '$LOGGER')"; return 1 ;;
    esac
}

rpl_activate_sim() {
    cd "$RPL_REPO_ROOT"
    # shellcheck disable=SC1090
    source "$RPL_REPO_ROOT/scripts/source_${SIM}_setup.sh" || return 1
    if [[ "$SIM" == isaacsim ]]; then export OMNI_KIT_ACCEPT_EULA=YES; fi
}

rpl_training_command() {
    if [[ "$SIM" == isaacgym ]]; then
        # Use the cache and compiler concurrency verified on the training server.
        export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions/rpl_hsgym_front}"
        export MAX_JOBS="${MAX_JOBS:-2}"
    fi
    # --training.num-envs is the TOTAL across GPUs; holosoma splits it per rank.
    if [[ "$NGPUS" -gt 1 ]]; then
        if [[ "$SIM" == isaacgym ]]; then
            # Build/load the shared extension before torchrun workers contend for its lock.
            echo "[rpl] Preparing gymtorch before starting $NGPUS GPU workers..."
            python -c 'from isaacgym import gymapi, gymtorch; print("[rpl] gymtorch ready", flush=True)' || return 1
        fi
        TRAIN_COMMAND=(torchrun --standalone --nnodes=1 "--nproc_per_node=$NGPUS")
    else
        TRAIN_COMMAND=(python)
    fi
}

rpl_setup_environment() {
    rpl_launcher_init || return 1
    local holosoma_dir="${HOLOSOMA_LOCAL_DIR:-$RPL_REPO_ROOT/thirdparty/holosoma}"
    cd "$RPL_REPO_ROOT"
    [[ -n "${HOLOSOMA_LOCAL_DIR:-}" ]] || git submodule update --init --recursive
    [[ -f "$holosoma_dir/scripts/setup_${SIM}.sh" ]] || { rpl_error "Holosoma setup not found in $holosoma_dir"; return 1; }
    if [[ -n "${ENV_NAME:-}" ]]; then export CONDA_ENV_NAME="$ENV_NAME"; fi
    # Holosoma's installer builds the conda env and installs holosoma (editable) from $holosoma_dir.
    bash "$holosoma_dir/scripts/setup_${SIM}.sh"
    rpl_activate_sim
    pip install -e "$RPL_REPO_ROOT"
    echo "[setup] Done. Run rpl_training/training_runs/run_expert.sh or run_distill.sh."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    rpl_setup_environment "$@"
fi
