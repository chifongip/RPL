"""X2 motion/force task integration using production commands and force hooks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from holosoma.utils.safe_torch_import import torch

from rpl_training.config_values import experiment as g1
from rpl_training.config_values import x2
from rpl_training.envs.loco_manip_force_manager import LeggedRobotLocoManipForceManager
from rpl_training.envs.loco_manip_manager import LeggedRobotLocoManipManager
from rpl_training.terms.command import UpperBodyRefPoseCommand
from rpl_training.terms.termination import motion_ends
from rpl_training.x2 import DOF_NAMES, HOME


@pytest.mark.parametrize("stage", ["flat", "expert", "expert_stones", "expert_stairs", "expert_slopes", "distill"])
def test_x2_task_matches_corresponding_g1(stage):
    old = g1.g1_29dof_falcon_hm if stage == "flat" else getattr(g1, "g1_29dof_rpl_" + stage)
    new = getattr(x2, "x2_29dof_rpl_" + stage)
    assert replace(new.robot.motion, motion_file=old.robot.motion.motion_file) == old.robot.motion
    assert new.robot.motion.standardize_motion_length
    assert new.robot.motion.standardize_motion_length_value == 10.0
    assert new.robot.motion.motion_file.endswith("motions/x2/falcon_accad_all")
    assert new.curriculum == old.curriculum
    assert new.env_class == old.env_class
    assert new.termination == old.termination
    assert new.observation == old.observation
    assert new.command.reset_terms == old.command.reset_terms
    assert new.command.step_terms == old.command.step_terms
    for name, term in new.command.setup_terms.items():
        if name == "locomotion_command":
            assert term.params["default_base_height"] == 0.64
            assert replace(term, params={**term.params, "default_base_height": 0.75}) == old.command.setup_terms[name]
        else:
            assert term == old.command.setup_terms[name]
    forces = LeggedRobotLocoManipForceManager._resolve_force_config(new)
    original = LeggedRobotLocoManipForceManager._resolve_force_config(old)
    geometry_keys = {
        "enabled",
        "left_hand_link",
        "right_hand_link",
        "left_extend_parent_link",
        "right_extend_parent_link",
        "extend_offset",
    }
    assert {k: v for k, v in forces.items() if k not in geometry_keys} == {
        k: v for k, v in original.items() if k not in geometry_keys
    }
    assert forces["enabled"] == (stage != "flat")


def test_real_x2_motion_reference_playback_freeze_reset_and_end():
    cfg = x2.x2_29dof_rpl_expert
    n = 3
    env = SimpleNamespace(
        robot_config=cfg.robot,
        num_envs=n,
        device="cpu",
        dt=0.02,
        is_evaluating=True,
        max_episode_length_s=10.0,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        default_dof_pos=torch.tensor(HOME).repeat(n, 1),
        simulator=SimpleNamespace(dof_pos=torch.tensor(HOME).repeat(n, 1), dof_vel=torch.zeros(n, 29)),
    )
    term = UpperBodyRefPoseCommand(cfg.command.setup_terms["upper_body_ref_pose"], env)
    term.setup()
    term.reset(None)
    assert env.motion_reference_enabled and env.num_motions == 252
    torch.testing.assert_close(env.motion_len, torch.full((n,), 10.0))
    assert term.upper_dof_indices.tolist() == list(range(15, 29))
    initial = env.ref_upper_dof_pos.clone()
    env.episode_length_buf[:] = 100
    term.step()
    assert env.ref_upper_dof_pos.shape == (n, 14)
    assert not torch.allclose(initial, env.ref_upper_dof_pos)
    # Independent frame interpolation at normalized playback time, including
    # the inherited one-control-step lookahead. Stored velocities stay unscaled.
    for i, path in enumerate(term.motion_lib._motion_data_paths[:n]):
        with np.load(path) as data:
            assert data["dof_names"].tolist() == DOF_NAMES
            frame = 2.02 / 10.0 * (len(data["joint_pos"]) - 1)
            lo = int(frame)
            expected = (1 - frame + lo) * data["joint_pos"][lo, 7:] + (frame - lo) * data["joint_pos"][lo + 1, 7:]
            np.testing.assert_allclose(env.ref_upper_dof_pos[i], expected[15:], atol=2e-5)
    env.is_evaluating = False
    term.fix_upper_body[:] = True
    term.fix_upper_body_motion_times[:] = 1.25
    term.step()
    frozen = env.ref_upper_dof_pos.clone()
    env.episode_length_buf[:] = 150
    term.step()
    torch.testing.assert_close(env.ref_upper_dof_pos, frozen)
    env._pending_episode_lengths = torch.full((n,), 50, dtype=torch.long)
    expected = term.motion_lib.get_motion_state(env.motion_ids, torch.ones(n))
    term._reset_upper_dofs_to_motion(torch.arange(n))
    torch.testing.assert_close(env.simulator.dof_pos[:, 15:], expected["dof_pos"][:, 15:])
    torch.testing.assert_close(env.simulator.dof_vel[:, 15:], expected["dof_vel"][:, 15:])
    torch.testing.assert_close(env.simulator.dof_pos[:, :15], torch.tensor(HOME[:15]).repeat(n, 1))
    env.episode_length_buf[:] = torch.tensor([499, 500, 501])
    assert motion_ends(env).tolist() == [False, False, True]


def test_x2_nonzero_forces_reach_correct_bodies_and_observations(monkeypatch):
    cfg = x2.x2_29dof_rpl_expert
    mgr = object.__new__(LeggedRobotLocoManipForceManager)
    mgr.robot_config = cfg.robot
    mgr.dof_names = DOF_NAMES
    mgr.num_envs, mgr.num_bodies, mgr.device = 2, cfg.robot.num_bodies, "cpu"
    mgr.force_config = mgr._resolve_force_config(cfg)
    mgr.command_manager = SimpleNamespace(commands=torch.zeros(2, 9))
    mgr.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    positions = torch.arange(2 * mgr.num_bodies * 3, dtype=torch.float).reshape(2, mgr.num_bodies, 3)
    rotations = mgr.base_quat[:, None, :].repeat(1, mgr.num_bodies, 1)
    applied = []
    mgr.simulator = SimpleNamespace(
        find_rigid_body_indice=cfg.robot.body_names.index,
        jacobian=torch.ones(2, mgr.num_bodies, 6, 35) * 0.1,
        _rigid_body_pos=positions,
        _rigid_body_rot=rotations,
        apply_rigid_body_force_at_pos_tensor=lambda force, pos: applied.append((force.clone(), pos.clone())),
    )
    monkeypatch.setattr(LeggedRobotLocoManipManager, "_setup_robot_body_indices", lambda self: None)
    monkeypatch.setattr(LeggedRobotLocoManipManager, "_pre_compute_observations_callback", lambda self, ids: None)
    mgr._setup_robot_body_indices()
    assert mgr.left_arm_dof_indices.tolist() == list(range(15, 22))
    assert mgr.right_arm_dof_indices.tolist() == list(range(22, 29))
    mgr._init_force_settings()
    mgr.apply_force_scale[:] = 1.0
    mgr.left_zero_force.zero_()
    mgr.right_zero_force.zero_()
    mgr.left_ee_apply_force_phase.zero_()
    mgr.right_ee_apply_force_phase.zero_()
    mgr.force_xyz_scale[:] = torch.tensor([0.0, 0.0, 1.0])
    mgr._pre_compute_observations_callback()
    mgr._calculate_ee_forces()
    mgr.action_manager = None
    mgr._apply_force_in_physics_step()
    force, pos = applied[0]
    assert torch.isfinite(force).all() and torch.isfinite(pos).all()
    hands = [cfg.robot.body_names.index(f"{side}_wrist_roll_link") for side in ("left", "right")]
    assert torch.all(force[:, hands, 2] < 0)
    torch.testing.assert_close(pos[:, hands], positions[:, hands])
    torch.testing.assert_close(mgr.left_ee_apply_force, force[:, hands[0]])
    torch.testing.assert_close(mgr.right_ee_apply_force, force[:, hands[1]])
    other = [i for i in range(mgr.num_bodies) if i not in hands]
    assert torch.count_nonzero(force[:, other]) == 0
