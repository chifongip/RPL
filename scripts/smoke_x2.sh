#!/bin/bash
# Single-GPU integration checks. All weights produced here are smoke artifacts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NGPUS=1 LOGGER=disabled
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/rpl-x2-torch}"
export MAX_JOBS="${MAX_JOBS:-2}"
source rpl_training/training_runs/setup_env.sh
rpl_launcher_init
rpl_activate_sim
TRAIN_PY="$(command -v python)"
SIM_PY="${CONDA_ROOT}/envs/${SIM_ENV:-rplmujoco}/bin/python"
POLICY_PY="${CONDA_ROOT}/envs/${POLICY_ENV:-rplinference}/bin/python"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
REPORT_DIR="$RPL_REPO_ROOT/logs/x2-runtime-smoke/$STAMP"
mkdir -p "$REPORT_DIR"
export REPORT_DIR STAMP
echo "X2 smoke artifacts: $REPORT_DIR"
"$TRAIN_PY" -m rpl_training.training_runs.probe_x2_asset > "$REPORT_DIR/asset-probe.log" 2>&1
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$TRAIN_PY" -m pytest rpl_training/tests -q -p no:cacheprovider > "$REPORT_DIR/training-tests.log" 2>&1
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$SIM_PY" -m pytest rpl_sim2sim/tests/test_sim.py rpl_sim2sim/tests/test_x2_runtime.py rpl_training/tests/test_x2_parameters.py -q -p no:cacheprovider > "$REPORT_DIR/sim-tests.log" 2>&1
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$POLICY_PY" -m pytest rpl_sim2sim/tests/test_policy.py rpl_sim2sim/tests/test_x2_policy.py -q -p no:cacheprovider > "$REPORT_DIR/policy-tests.log" 2>&1

run_training() {
    local stage=$1
    shift
    echo "Training smoke: $stage"
    TASK="$stage" bash rpl_training/training_runs/run_x2.sh \
        --training.num-envs "${SMOKE_ENVS:-32}" --training.project x2-runtime-smoke \
        --training.name "$stage-$STAMP" --algo.config.num-learning-iterations "${ITERS:-3}" \
        --algo.config.save-interval 100 --terrain.terrain-term.num-rows 2 \
        --terrain.terrain-term.num-cols 20 --terrain.terrain-term.border-size 2 \
        "$@" > "$REPORT_DIR/$stage.log" 2>&1
}
checkpoint() {
    "$TRAIN_PY" - "$1" "$STAMP" <<'PY'
import sys
from pathlib import Path
files = sorted(Path('logs/x2-runtime-smoke').glob(f'*-{sys.argv[1]}-{sys.argv[2]}-locomotion/model_*.pt'))
assert files, f'No {sys.argv[1]} smoke checkpoint was produced'
print(files[-1].resolve())
PY
}
run_training flat
FLAT_CKPT="$(checkpoint flat)"
run_training expert
EXPERT_CKPT="$(checkpoint expert)"
"$TRAIN_PY" - "$EXPERT_CKPT" "$REPORT_DIR" <<'PY'
import sys
from pathlib import Path
for terrain in ['slopes', 'stairs_up', 'stairs_down', 'stepping_stones_columns']:
    target = Path(sys.argv[2]) / 'smoke-teachers' / terrain / 'smoke' / 'model_00000.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(sys.argv[1]))
PY
run_training distill --algo.config.ckpt-teacher "$REPORT_DIR/smoke-teachers"
DEPTH_CKPT="$(checkpoint distill)"
for model in "$FLAT_CKPT" "$EXPERT_CKPT" "$DEPTH_CKPT"; do
    "$TRAIN_PY" -m rpl_training.training_runs.check_x2_export "$model" >> "$REPORT_DIR/export-checks.log" 2>&1
done
"$TRAIN_PY" scripts/smoke_x2_transport.py --flat "$FLAT_CKPT" --depth "$DEPTH_CKPT" \
    --report-dir "$REPORT_DIR" --conda-root "$CONDA_ROOT"
echo "X2 smoke checks passed. Logs: $REPORT_DIR"
