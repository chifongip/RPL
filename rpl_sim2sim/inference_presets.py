"""``holosoma_inference`` presets for the RPL / FALCON-Terrains depth student.

``inference:g1-29dof-rpl-depth`` pairs with ``python -m rpl_sim2sim.run_sim`` (FALCON's sim2sim
environment). Robot gains, stand pose, action scale and observation layout follow FALCON's
``loco_manip_cam`` / ``g1_29dof_depth`` sim2sim configs, which are also what the distillation
preset trains against.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from holosoma_inference.config.config_types.camera import CameraConfig, CameraPose, CameraProps
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.camera import CAMERA_REGISTRY
from holosoma_inference.config.config_values.inference import INFERENCE_REGISTRY
from holosoma_inference.config.config_values.observation import OBSERVATION_REGISTRY
from holosoma_inference.config.config_values.robot import ROBOT_REGISTRY, g1_29dof
from holosoma_inference.config.config_values.task import TASK_REGISTRY, locomotion

from rpl_sim2sim.policy import RplObservationConfig
from rpl_training.x2 import DOF_NAMES, HOME, KD, KP, MODEL

MODELS_DIR = Path(__file__).resolve().parent / "models"
# DEFAULT_MODEL = MODELS_DIR / "rpl_original_model_19999.onnx"
DEFAULT_MODEL = MODELS_DIR / "rpl_holosoma_model_19999.onnx"

# FALCON ``g1_29dof.yaml``: PD gains, stand pose (legs bent, arms at zero), motor order.
# fmt: off
_KP = (100, 100, 100, 150, 40, 40) * 2 + (200, 40, 40) + (40,) * 14
_KD = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2 + (5.0, 5.0, 5.0) + (1.0,) * 14
_STAND = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0,) * 17
# fmt: on

g1_29dof_rpl = replace(
    g1_29dof,
    default_dof_angles=_STAND,
    default_motor_angles=_STAND,
    motor_kp=tuple(float(k) for k in _KP),
    motor_kd=_KD,
)

# 115-dim frame x 5 (actor), 2 x 27 x 48 depth x 2 frames pushed at 10 Hz, command velocity.
rpl_depth_observation = RplObservationConfig(
    obs_dict={
        "actor_obs": [
            "actions",
            "base_ang_vel",
            "command_ang_vel",
            "command_base_height",
            "command_lin_vel",
            "command_stand",
            "command_waist_dofs",
            "dof_pos",
            "dof_vel",
            "projected_gravity",
            "ref_upper_dof_pos",
        ],
        "perception_obs": ["cam_depth"],
        "command_lin_vel_obs": ["command_lin_vel"],
    },
    obs_dims={
        "actions": 29,
        "base_ang_vel": 3,
        "command_ang_vel": 1,
        "command_base_height": 1,
        "command_lin_vel": 2,
        "command_stand": 1,
        "command_waist_dofs": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "projected_gravity": 3,
        "ref_upper_dof_pos": 14,
        "cam_depth": 2 * 27 * 48,
    },
    obs_scales={
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "command_ang_vel": 1.0,
        "command_base_height": 2.0,
        "command_lin_vel": 1.0,
        "command_stand": 1.0,
        "command_waist_dofs": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "projected_gravity": 1.0,
        "ref_upper_dof_pos": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={"actor_obs": 5, "perception_obs": 2, "command_lin_vel_obs": 1},
    obs_intervals={"actor_obs": 1, "perception_obs": 5, "command_lin_vel_obs": 1},
)

rpl_depth_task = replace(
    locomotion,
    model_path=str(DEFAULT_MODEL),
    policy_type="rpl_depth",
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=False,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    interface="lo",
    # FALCON's ``task.USE_JOYSTICK=true``: the Xbox pad the sim publishes as the wireless controller.
    # ``--task.velocity-input keyboard --task.state-input keyboard`` drives it from the terminal.
    velocity_input="interface",
    state_input="interface",
)

# Front then back, as published by the sim's ``depth-shm-falcon`` plugin. Only the camera count and
# resized size are read by the policy; poses/props document the rig (FALCON sim2sim cameras).
rpl_dual_depth = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="torso_link", camera_offset=(0.15, 0.06, 0.02), camera_rotation=(0.0, 75.0, 0.0)
        ),
        "cam_back_depth": CameraPose(
            parent_link="torso_link", camera_offset=(-0.13, -0.06, 0.02), camera_rotation=(0.0, 75.0, 180.0)
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=48,
        resized_height=27,
        horizontal_fov=101.41,
        vertical_fov=69.0,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
    ),
)

g1_29dof_rpl_depth = InferenceConfig(
    robot=g1_29dof_rpl,
    observation=rpl_depth_observation,
    task=rpl_depth_task,
    camera=rpl_dual_depth,
)

# Use with the front-only sim config and an explicitly supplied single-camera export.
rpl_front_depth = replace(rpl_dual_depth, poses={"cam_front_depth": rpl_dual_depth.poses["cam_front_depth"]})
rpl_front_depth_observation = replace(
    rpl_depth_observation, obs_dims={**rpl_depth_observation.obs_dims, "cam_depth": 27 * 48}
)
g1_29dof_rpl_depth_front = replace(
    g1_29dof_rpl_depth,
    camera=rpl_front_depth,
    observation=rpl_front_depth_observation,
    task=replace(rpl_depth_task, model_path=""),
)

# Unitree HG messages are used only as a local sim2sim transport. The robot
# family selects the SDK transport implementation, not X2 kinematics/control.
x2_29dof_rpl = replace(
    g1_29dof_rpl,
    robot_type="x2_29dof",
    robot="g1",
    dof_names=tuple(DOF_NAMES),
    dof_names_lower_body=tuple(MODEL["lower_dof_names"]),
    dof_names_upper_body=tuple(MODEL["upper_dof_names"]),
    default_dof_angles=tuple(HOME),
    default_motor_angles=tuple(HOME),
    motor_kp=tuple(KP),
    motor_kd=tuple(KD),
    motor2joint=tuple(range(29)),
    joint2motor=tuple(range(29)),
    default_per_joint_action_scale=(0.25,) * 29,
    left_hand_link_name="left_wrist_roll_link",
    right_hand_link_name="right_wrist_roll_link",
)
x2_dual_depth = replace(
    rpl_dual_depth,
    poses={
        "cam_front_depth": replace(rpl_dual_depth.poses["cam_front_depth"], camera_offset=(0.135, 0.06, 0.02)),
        "cam_back_depth": replace(rpl_dual_depth.poses["cam_back_depth"], camera_offset=(-0.125, -0.06, 0.02)),
    },
)
x2_29dof_rpl_depth = replace(
    g1_29dof_rpl_depth,
    robot=x2_29dof_rpl,
    camera=x2_dual_depth,
    task=replace(rpl_depth_task, model_path="", desired_base_height=0.68),
)
x2_flat_observation = replace(
    rpl_depth_observation,
    obs_dict={"actor_obs": rpl_depth_observation.obs_dict["actor_obs"]},
    history_length_dict={"actor_obs": 5},
    obs_intervals={"actor_obs": 1},
)
x2_29dof_rpl_flat = replace(x2_29dof_rpl_depth, camera=None, observation=x2_flat_observation)


def register() -> None:
    """Add the presets to ``holosoma_inference``'s registries (idempotent)."""
    ROBOT_REGISTRY.add("g1-29dof-rpl", g1_29dof_rpl)
    ROBOT_REGISTRY.add("x2-29dof-rpl", x2_29dof_rpl)
    OBSERVATION_REGISTRY.add("rpl-depth", rpl_depth_observation)
    TASK_REGISTRY.add("rpl-depth", rpl_depth_task)
    CAMERA_REGISTRY.add("rpl-dual-depth", rpl_dual_depth)
    CAMERA_REGISTRY.add("rpl-front-depth", rpl_front_depth)
    CAMERA_REGISTRY.add("x2-dual-depth", x2_dual_depth)
    INFERENCE_REGISTRY.add("g1-29dof-rpl-depth", g1_29dof_rpl_depth)
    INFERENCE_REGISTRY.add("g1-29dof-rpl-depth-front", g1_29dof_rpl_depth_front)
    INFERENCE_REGISTRY.add("x2-29dof-rpl-depth", x2_29dof_rpl_depth)
    INFERENCE_REGISTRY.add("x2-29dof-rpl-flat", x2_29dof_rpl_flat)


register()

__all__ = ["DEFAULT_MODEL", "g1_29dof_rpl", "g1_29dof_rpl_depth", "rpl_depth_observation", "rpl_depth_task"]
