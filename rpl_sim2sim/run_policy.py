"""Run the RPL / FALCON-Terrains depth student against ``python -m rpl_sim2sim.run_sim``.

Usage::

    python -m rpl_sim2sim.run_policy                                    # shipped student, Xbox pad
    python -m rpl_sim2sim.run_policy --task.model-path /path/model_XXXXX.onnx   # your export
    python -m rpl_sim2sim.run_policy --task.velocity-input keyboard --task.state-input keyboard
    python -m rpl_sim2sim.run_policy --no-arm-ik                        # Xbox pad, fixed arm reference

Xbox pad (FALCON's mapping, read by the sim and relayed as the wireless controller): ``A+X`` init
pose, ``start`` start, ``R2`` stand/walk, left stick vx, right stick yaw, ``L2`` zero velocity,
``B+Y`` stop, ``B+up/down`` base height, ``Y`` or ``select`` + d-pad waist pitch/yaw, ``R1`` / ``X`` +
d-pad hand targets (arm IK).
Keyboard (policy terminal): ``i`` init pose, ``]`` start, ``=`` stand/walk, ``w/s/a/d`` velocity,
``q/e`` yaw, ``z`` zero velocity, ``o`` stop. Any ``holosoma_inference`` argument can be added.
"""

from __future__ import annotations

import sys

import rpl_sim2sim.inference_presets  # noqa: F401  (registers the presets)

FALCON_JOYSTICK_GUIDE = """\
FALCON Xbox controls (sim2sim: the pad is plugged into the run_sim host):
  A+X            - init pose (ramp to the stand pose)
  start          - start the policy
  B+Y            - stop the policy (hold current pose)
  R2             - toggle stand / walk
  L2             - zero velocity
  left stick     - forward/backward velocity (lateral is ignored, as in FALCON)
  right stick    - yaw rate
  B+up / B+down  - base height +/- 0.1
  Y+up / Y+down (or select+up/down)   - waist pitch -/+ 0.05
  select+left / select+right          - waist yaw -/+ 0.1
  Y+left/right, A+left/right, A+Y     - kp level -/+ 0.1, -/+ 0.01, reset
  A+up / A+down, L1+R1                - fixed forward speed +/- 0.1, toggle it
  R1+up / R1+down                     - hand targets forward/back 0.05 (arm IK)
  R1+left / R1+right                  - hands apart/together 0.02
  X+up / X+down                       - hands up/down 0.05
MuJoCo window: 7/8 raise/lower the gantry, 9 releases it."""


def main() -> None:
    import holosoma_inference.run_policy as holosoma_run_policy
    from loguru import logger

    from rpl_sim2sim.policy import RplDepthPolicy

    if "--no-arm-ik" in sys.argv:
        sys.argv[:] = [arg for arg in sys.argv if arg != "--no-arm-ik"]
        RplDepthPolicy.enable_arm_ik = False
        logger.info("ArmIK disabled; joystick locomotion uses the default arm reference pose.")

    holosoma_guide = holosoma_run_policy._print_control_guide

    def _print_control_guide(policy_class, use_joystick: bool, dual_mode: bool = False):
        if use_joystick and issubclass(policy_class, RplDepthPolicy):
            for line in FALCON_JOYSTICK_GUIDE.splitlines():
                if not policy_class.enable_arm_ik and "(arm IK)" in line:
                    continue
                logger.info(line)
        else:
            holosoma_guide(policy_class, use_joystick, dual_mode=dual_mode)

    holosoma_run_policy._print_control_guide = _print_control_guide
    if not any(a.startswith("inference:") for a in sys.argv[1:]):
        sys.argv.insert(1, "inference:g1-29dof-rpl-depth")
    holosoma_run_policy.main()


if __name__ == "__main__":
    main()
