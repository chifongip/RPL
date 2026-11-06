"""X2 robot symmetry with the unchanged G1 PPO and augmentation implementation."""

from types import SimpleNamespace

import torch
from holosoma.agents.modules.augmentation_utils import SymmetryUtils

from rpl_training.config_values.x2 import x2_29dof_rpl_distill, x2_29dof_rpl_expert, x2_29dof_rpl_flat
from rpl_training.x2 import DOF_NAMES, HOME, MODEL, POSITION_LIMITS


def symmetry_env():
    cfg = x2_29dof_rpl_expert
    dims = dict(
        base_lin_vel=3,
        base_ang_vel=3,
        projected_gravity=3,
        command_lin_vel=2,
        command_ang_vel=1,
        command_stand=1,
        command_waist_dofs=3,
        command_base_height=1,
        ref_upper_dof_pos=14,
        dof_pos=29,
        dof_vel=29,
        actions=29,
        ee_apply_force=6,
        height_measurements=160,
    )
    env = SimpleNamespace(
        device="cpu",
        robot_config=cfg.robot,
        upper_dof_indices=torch.arange(15, 29),
        history_length={k: group.history_length for k, group in cfg.observation.groups.items()},
        observation_manager=SimpleNamespace(
            cfg=cfg.observation, _compute_term=lambda group, term, config: torch.zeros(1, dims[term])
        ),
        terrain_manager=SimpleNamespace(
            terrain_term=SimpleNamespace(num_height_scan_points_x=16, num_height_scan_points_y=10)
        ),
    )
    return env


def test_symmetry_presets_and_vendor_limits():
    for cfg in (x2_29dof_rpl_flat, x2_29dof_rpl_expert):
        assert cfg.algo.config.use_symmetry
        assert cfg.algo.config.symmetry_actor_coef == 1
        assert cfg.algo.config.symmetry_critic_coef == 0
    assert not getattr(x2_29dof_rpl_distill.algo.config, "use_symmetry", False)
    robot = x2_29dof_rpl_expert.robot
    yaw = DOF_NAMES.index("waist_yaw_joint")
    assert robot.dof_pos_lower_limit_list[yaw] == -2.2078
    assert robot.dof_pos_upper_limit_list[yaw] == 2.2078
    assert MODEL["dofs"][yaw]["lower"] == -3.43
    for joint, limits in zip(MODEL["dofs"], POSITION_LIMITS):
        expected = [-2.2078, 2.2078] if joint["name"] == "waist_yaw_joint" else [joint["lower"], joint["upper"]]
        assert limits == expected
        partner = POSITION_LIMITS[DOF_NAMES.index(joint["mirror_joint"])]
        reflected = sorted(joint["mirror_sign"] * bound for bound in partner)
        assert limits == reflected


def test_mirroring_twice_preserves_all_expert_observations():
    env = symmetry_env()
    symmetry = SymmetryUtils(env)
    for group, size in symmetry.observation_dims.items():
        observation = torch.randn(4, size)
        original = observation.clone()
        mirrored = symmetry.mirror_xz_plane(observation, env, [group])
        torch.testing.assert_close(symmetry.mirror_xz_plane(mirrored, env, [group]), original)
        torch.testing.assert_close(observation, original)
    for key, size in [("lower_body", 15), ("upper_body", 14), (None, 29)]:
        action = torch.randn(4, size)
        torch.testing.assert_close(
            symmetry.mirror_action_xz_plane(symmetry.mirror_action_xz_plane(action, body_key=key), body_key=key), action
        )
    home = torch.tensor(HOME)
    torch.testing.assert_close(symmetry.mirror_action_xz_plane(home), home)


def test_each_joint_reflects_to_its_named_partner_with_axis_correct_sign():
    symmetry = SymmetryUtils(symmetry_env())
    partner_indices, signs = [], []
    for joint in MODEL["dofs"]:
        name = joint["name"]
        partner = name.replace("left_", "right_") if name.startswith("left_") else name.replace("right_", "left_")
        partner_indices.append(DOF_NAMES.index(partner))
        # Angular axes are axial vectors: sagittal reflection reverses X/Z,
        # while preserving Y. Derive the expected sign from the vendor axis.
        axis = joint["axis"]
        signs.append(-1.0 if axis[0] or axis[2] else 1.0)
    basis = torch.eye(29)
    expected = basis[:, partner_indices] * torch.tensor(signs)
    torch.testing.assert_close(symmetry.mirror_obs_dof_pos(basis), expected)
    torch.testing.assert_close(symmetry.mirror_obs_dof_vel(basis), expected)
    torch.testing.assert_close(symmetry.mirror_action_xz_plane(basis), expected)
    split = torch.cat(
        (
            symmetry.mirror_action_xz_plane(basis[:, :15], body_key="lower_body"),
            symmetry.mirror_action_xz_plane(basis[:, 15:], body_key="upper_body"),
        ),
        dim=1,
    )
    torch.testing.assert_close(split, expected)
    # Command order is semantic yaw/roll/pitch, independent of joint order.
    torch.testing.assert_close(
        symmetry.mirror_obs_command_waist_dofs(torch.tensor([[0.1, 0.2, 0.3]])), torch.tensor([[-0.1, -0.2, 0.3]])
    )


def test_standard_soft_limit_penalty_is_symmetric():
    from rpl_training.terms.reward import limits_lower_body_dof_pos

    cfg = x2_29dof_rpl_expert
    yaw = DOF_NAMES.index("waist_yaw_joint")
    hard = torch.tensor(POSITION_LIMITS)
    mid = hard.mean(dim=1)
    half = (hard[:, 1] - hard[:, 0]) * cfg.robot.soft_dof_pos_limit / 2
    soft = torch.stack((mid - half, mid + half), dim=1)
    torch.testing.assert_close(soft[yaw], torch.tensor([-2.09741, 2.09741]))
    positions = torch.tensor(HOME).repeat(2, 1)
    positions[:, yaw] = torch.tensor([-2.15, 2.15])
    env = SimpleNamespace(
        device="cpu", robot_config=cfg.robot, dof_pos_limits=soft, simulator=SimpleNamespace(dof_pos=positions)
    )
    penalty = limits_lower_body_dof_pos(env)
    torch.testing.assert_close(penalty[0], penalty[1])
