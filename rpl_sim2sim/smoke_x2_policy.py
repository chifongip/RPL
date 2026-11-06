"""Bounded X2 ONNX/DDS/shared-depth smoke consumer; run alongside run_sim.

The startup support band should remain enabled for short, untrained smoke
checkpoints. This checks integration and finite commands, not locomotion quality.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from holosoma_inference.inputs.api.commands import StateCommand

from rpl_sim2sim.inference_presets import x2_29dof_rpl_depth, x2_29dof_rpl_flat
from rpl_sim2sim.policy import RplDepthPolicy


class _NoInput:
    def start(self):
        pass

    def zero(self):
        pass

    def poll_velocity(self):
        return None

    def poll_commands(self):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=["flat", "depth"], required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--shm-name", default="x2_runtime_smoke")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.domain_id != 0:
        raise ValueError("The current C++ Unitree sim2sim transport uses DDS domain 0")
    preset = x2_29dof_rpl_flat if args.kind == "flat" else x2_29dof_rpl_depth
    config = replace(
        preset,
        task=replace(
            preset.task,
            model_path=str(args.model),
            domain_id=args.domain_id,
            velocity_input="injected",
            state_input="injected",
            depth_shm=replace(preset.task.depth_shm, name=args.shm_name, heartbeat=False),
        ),
    )
    policy = object.__new__(RplDepthPolicy)
    policy._injected_velocity_input = _NoInput()
    policy._injected_command_provider = _NoInput()
    policy.__init__(config)
    deadline = time.monotonic() + 10
    while np.linalg.norm(policy.interface.get_low_state()[0, 3:7]) < 0.5:
        if time.monotonic() > deadline:
            raise RuntimeError("No valid robot state received over DDS")
        time.sleep(0.05)
    policy._dispatch_command(StateCommand.INIT)
    policy.policy_action()
    assert policy.get_ready_state
    policy._dispatch_command(StateCommand.START)
    maximum = 0.0
    for _ in range(args.steps):
        policy.policy_action()
        assert np.isfinite(policy.last_policy_action).all()
        maximum = max(maximum, float(np.abs(policy.last_policy_action).max()))
        time.sleep(0.02)
    state = policy.interface.get_low_state()
    assert np.isfinite(state).all()
    report = dict(
        kind=args.kind,
        steps=args.steps,
        max_absolute_raw_action=maximum,
        state_shape=list(state.shape),
        arm_ik_enabled=policy.enable_arm_ik,
        observation_shapes={k: list(v.shape) for k, v in policy.prepare_obs_for_rl(state).items()},
    )
    policy._dispatch_command(StateCommand.STOP)
    policy.policy_action()
    assert not policy.use_policy_action
    report["init_start_stop_passed"] = True
    if policy.depth_sensor is not None:
        image = policy.depth_sensor.get_latest()
        assert np.isfinite(image).all()
        # Training preserves antialiased-bicubic overshoot beyond [-0.5, 0.5].
        assert image.min() >= -1.0 and image.max() <= 1.0
        report["depth_shape"] = list(image.shape)
        report["depth_range"] = [float(image.min()), float(image.max())]
        policy.depth_sensor.stop()
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
