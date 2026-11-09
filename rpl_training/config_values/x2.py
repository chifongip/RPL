"""X2 presets: retain G1 RPL tasks/learning settings; replace robot-specific data."""

from copy import deepcopy
from dataclasses import replace

from rpl_training.config_values import experiment as g1
from rpl_training.config_values.camera import warp_dual_depth_cameras_falcon
from rpl_training.config_values.robot import g1_29dof_falcon
from rpl_training.paths import DATA_DIR
from rpl_training.x2 import ASSET_DIR, BASELINE, DOF_NAMES, HOME, KD, KP, MODEL, POSITION_LIMITS

x2_29dof = replace(
    g1_29dof_falcon,
    num_bodies=MODEL["expected_num_bodies"],
    body_names=MODEL["expected_collapsed_body_names"],
    dof_names=DOF_NAMES,
    lower_dof_names=MODEL["lower_dof_names"],
    upper_dof_names=MODEL["upper_dof_names"],
    upper_left_arm_dof_names=DOF_NAMES[15:22],
    upper_right_arm_dof_names=DOF_NAMES[22:],
    arm_dof_names=DOF_NAMES[15:],
    left_arm_dof_names=DOF_NAMES[15:22],
    right_arm_dof_names=DOF_NAMES[22:],
    waist_dof_names=DOF_NAMES[12:15],
    dof_pos_lower_limit_list=[limits[0] for limits in POSITION_LIMITS],
    dof_pos_upper_limit_list=[limits[1] for limits in POSITION_LIMITS],
    dof_vel_limit_list=[j["velocity"] for j in MODEL["dofs"]],
    dof_effort_limit_list=[j["effort"] for j in MODEL["dofs"]],
    dof_armature_list=BASELINE["armature_kg_m2"],
    dof_joint_friction_list=BASELINE["frictionloss_Nm"],
    symmetry_joint_names={j["name"]: j["mirror_joint"] for j in MODEL["dofs"]},
    flip_sign_joint_names=[j["name"] for j in MODEL["dofs"] if j["mirror_sign"] < 0],
    init_state=replace(
        g1_29dof_falcon.init_state, pos=MODEL["home_root_position"], default_joint_angles=dict(zip(DOF_NAMES, HOME))
    ),
    control=replace(g1_29dof_falcon.control, stiffness=dict(zip(DOF_NAMES, KP)), damping=dict(zip(DOF_NAMES, KD))),
    asset=replace(
        g1_29dof_falcon.asset,
        asset_root=str(ASSET_DIR),
        urdf_file="x2_ultra_29dof.urdf",
        xml_file="x2_ultra_29dof.xml",
        usd_file=None,
        robot_type="x2_29dof",
        mesh_root=str(ASSET_DIR / "meshes"),
        replace_cylinder_with_capsule=False,
    ),
    feet_front_offset_in_body=BASELINE["four_point_foot_edge_offsets_m"]["front"],
    feet_back_offset_in_body=BASELINE["four_point_foot_edge_offsets_m"]["back"],
    feet_left_offset_in_body=BASELINE["four_point_foot_edge_offsets_m"]["left"],
    feet_right_offset_in_body=BASELINE["four_point_foot_edge_offsets_m"]["right"],
    lidar_height_offset=0.68,
    motion=replace(
        g1.g1_29dof_falcon_hm.robot.motion,
        motion_file=str(DATA_DIR / "motions/x2/falcon_accad_all"),
    ),
)

_camera_term = warp_dual_depth_cameras_falcon.terms["warp_depth"]
x2_dual_depth = replace(
    warp_dual_depth_cameras_falcon,
    terms={
        "warp_depth": replace(
            _camera_term,
            params={
                **deepcopy(_camera_term.params),
                "dynamic_meshes": {
                    "enable": True,
                    "meshes_root": str(ASSET_DIR / "camera_meshes"),
                    "body_meshes": {f"{name}.stl": name for name in MODEL["expected_collapsed_body_names"]},
                },
            },
        ),
    },
)


def _x2(base, name, *, flat=False, distill=False):
    command = deepcopy(base.command)
    command.setup_terms["locomotion_command"].params["default_base_height"] = 0.64
    # Retain G1's force task; X2's hand component extends along local -Z.
    command.params["ee_force_config"] = {
        **command.params.get("ee_force_config", {}),
        "enabled": not flat,
        "left_hand_link": "left_wrist_roll_link",
        "right_hand_link": "right_wrist_roll_link",
        "left_extend_parent_link": "left_wrist_roll_link",
        "right_extend_parent_link": "right_wrist_roll_link",
        "extend_offset": (0.0, 0.0, -0.15),
        "apply_force_pos_ratio_range": (0.2, 1.0),
    }
    sim = replace(
        base.simulator,
        config=replace(base.simulator.config, sim=replace(base.simulator.config.sim, max_episode_length_s=10.0)),
    )
    algorithm = base.algo
    if distill:
        # Keep the G1 teacher layout/architecture, but never silently load G1 weights.
        algorithm = replace(
            base.algo,
            _target_="rpl_training.x2_dagger.X2DAggerMoE",
            config=replace(base.algo.config, ckpt_teacher=str(DATA_DIR / "models/x2")),
        )
    # X2's last wrist joint is roll; G1's wrist-yaw fallback is proximal on X2.
    hand_links = ("left_wrist_roll_link", "right_wrist_roll_link")
    reward_params = {
        "penalty_ee_lin_acc": {"hand_link_names": hand_links},
        "penalty_ee_ang_acc": {"hand_link_names": hand_links},
        # The ankle-roll frame is 73 mm above the contact sole. Sampling at
        # local Z=0 incorrectly fails the inherited 50 mm foothold tolerance.
        "penalty_foothold": {
            "foothold_sample_z_offsets": tuple(
                MODEL["feet"][f"{side}_ankle_roll_link"]["sole_z"] for side in ("left", "right")
            ),
        },
    }
    reward = replace(
        base.reward,
        terms={
            name: replace(term, params={**term.params, **reward_params[name]})
            if name in reward_params
            else term
            for name, term in base.reward.terms.items()
        },
    )
    return replace(
        base,
        robot=replace(
            x2_29dof,
            motion=replace(base.robot.motion, motion_file=x2_29dof.motion.motion_file),
        ),
        reward=reward,
        command=command,
        simulator=sim,
        algo=algorithm,
        camera=x2_dual_depth if distill else base.camera,
        training=replace(
            base.training,
            name=name,
            project="rpl-x2-" + ("distill" if distill else "experts"),
        ),
    )


x2_29dof_rpl_flat = _x2(g1.g1_29dof_falcon_hm, "x2_29dof_rpl_flat", flat=True)
x2_29dof_rpl_expert = _x2(g1.g1_29dof_rpl_expert, "x2_29dof_rpl_expert")
x2_29dof_rpl_expert_stones = _x2(g1.g1_29dof_rpl_expert_stones, "x2_29dof_rpl_expert_stones")
x2_29dof_rpl_expert_stairs = _x2(g1.g1_29dof_rpl_expert_stairs, "x2_29dof_rpl_expert_stairs")
x2_29dof_rpl_expert_slopes = _x2(g1.g1_29dof_rpl_expert_slopes, "x2_29dof_rpl_expert_slopes")
x2_29dof_rpl_distill = _x2(g1.g1_29dof_rpl_distill, "x2_29dof_rpl_distill", distill=True)

EXPERIMENTS = {name: value for name, value in list(globals().items()) if name.startswith("x2_29dof_rpl_")}
