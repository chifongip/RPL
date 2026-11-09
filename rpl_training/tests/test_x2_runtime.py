"""X2 port contracts and retained G1 learning settings (training environment)."""

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("holosoma")

from rpl_training.config_values import experiment as g1
from rpl_training.config_values import x2
from rpl_training.x2 import DOF_NAMES, HOME


def test_native_x2_order_and_body_part_configuration():
    legs = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    waist = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
    arms = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        )
    ]
    expected = legs + waist + arms
    robot = x2.x2_29dof
    assert DOF_NAMES == robot.dof_names == expected
    assert robot.lower_dof_names == legs + waist
    assert robot.upper_dof_names == robot.arm_dof_names == arms
    assert robot.left_arm_dof_names == robot.upper_left_arm_dof_names == arms[:7]
    assert robot.right_arm_dof_names == robot.upper_right_arm_dof_names == arms[7:]
    assert robot.waist_dof_names == waist
    assert [robot.waist_yaw_dof_name, robot.waist_pitch_dof_name, robot.waist_roll_dof_name] == waist
    assert list(robot.init_state.default_joint_angles) == expected
    assert [g1.g1_29dof_rpl_expert.robot.dof_names.index(name) for name in expected] == (
        list(range(13)) + [14, 13, 15, 16, 17, 18, 21, 20, 19, 22, 23, 24, 25, 28, 27, 26]
    )


def test_pd_action_channels_and_gains_follow_native_joint_names():
    from types import SimpleNamespace

    import torch
    from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

    from rpl_training.x2 import EFFORT, KD, KP

    cfg = x2.x2_29dof_rpl_expert
    env = SimpleNamespace(
        robot_config=cfg.robot,
        dof_names=DOF_NAMES,
        num_envs=29,
        num_dof=29,
        device="cpu",
        default_dof_pos=torch.tensor(HOME).repeat(29, 1),
        torque_limits=torch.tensor(EFFORT),
        simulator=SimpleNamespace(dof_pos=torch.tensor(HOME).repeat(29, 1), dof_vel=torch.zeros(29, 29)),
        log_dict={},
    )
    term = JointPositionActionTerm(cfg.action.terms["joint_control"], env)
    torch.testing.assert_close(term.p_gains, torch.tensor(KP))
    torch.testing.assert_close(term.d_gains, torch.tensor(KD))
    # One active action per environment: every waist and wrist channel must
    # affect only its named joint through the actual action-term constructor.
    actions = torch.eye(29) * 0.01
    term.process_actions(actions)
    torch.testing.assert_close(term._compute_torques(term._actions_after_delay), torch.diag(torch.tensor(KP) * 0.0025))
    env.simulator.dof_vel[:] = torch.eye(29) * 0.1
    torch.testing.assert_close(term._compute_torques(torch.zeros(29, 29)), torch.diag(-torch.tensor(KD) * 0.1))


@pytest.mark.parametrize("suffix", ["expert", "expert_stones", "expert_stairs", "expert_slopes", "distill"])
def test_g1_algorithm_and_batch_settings_retained(suffix):
    old = getattr(g1, "g1_29dof_rpl_" + suffix)
    new = getattr(x2, "x2_29dof_rpl_" + suffix)
    for key in ["num_learning_iterations", "num_steps_per_env", "num_mini_batches", "actor_learning_rate"]:
        if hasattr(old.algo.config, key):
            assert getattr(new.algo.config, key) == getattr(old.algo.config, key)
    assert new.algo.config.module_dict == old.algo.config.module_dict
    assert new.env_class == old.env_class
    assert new.action == old.action
    if suffix != "distill":
        assert new.algo == old.algo
    assert new.training.num_envs == old.training.num_envs
    assert (
        new.training.num_envs * new.algo.config.num_steps_per_env
        == old.training.num_envs * old.algo.config.num_steps_per_env
    )
    assert replace(new.reward, terms=old.reward.terms) == old.reward
    for name, term in new.reward.terms.items():
        if name in ("penalty_ee_lin_acc", "penalty_ee_ang_acc"):
            assert term.params["hand_link_names"] == ("left_wrist_roll_link", "right_wrist_roll_link")
            assert replace(term, params=old.reward.terms[name].params) == old.reward.terms[name]
        elif name == "penalty_foothold":
            assert term.params["foothold_sample_z_offsets"] == pytest.approx((-0.073, -0.073))
            assert replace(term, params=old.reward.terms[name].params) == old.reward.terms[name]
        else:
            assert term == old.reward.terms[name]
    assert new.terrain == old.terrain
    assert new.randomization == old.randomization


@pytest.mark.parametrize(
    "task, gpus", [("flat", 1), ("expert", 4), ("stones", 4), ("stairs", 4), ("slopes", 1), ("distill", 4)]
)
def test_x2_launcher_default_gpu_allocation(tmp_path, task, gpus):
    import os
    import subprocess

    launcher = Path(__file__).resolve().parents[1] / "training_runs/run_x2.sh"
    (tmp_path / "run_x2.sh").write_text(launcher.read_text())
    # Exercise the actual launcher without activating conda or starting a job.
    # The fake backend prints the selected GPU count and forwarded CLI args.
    (tmp_path / "setup_env.sh").write_text(
        "rpl_launcher_init() { SIM=isaacgym; LOGGER_ARGS=(); }\n"
        "rpl_activate_sim() { :; }\n"
        'rpl_training_command() { TRAIN_COMMAND=(/usr/bin/printf "%s\\n" "$NGPUS"); }\n'
    )
    env = {**os.environ, "TASK": task}
    env.pop("NGPUS", None)
    result = subprocess.run(
        ["bash", str(tmp_path / "run_x2.sh"), "--training.num-envs", "32"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert args[0] == str(gpus)
    assert args[-2:] == ["--training.num-envs", "32"]
    env["NGPUS"] = "2"
    overridden = subprocess.run(
        ["bash", str(tmp_path / "run_x2.sh")], env=env, capture_output=True, text=True, check=True
    )
    assert overridden.stdout.splitlines()[0] == "2"


def test_x2_runtime_constants_cover_all_joints():
    robot = x2.x2_29dof
    assert robot.dof_names == DOF_NAMES
    assert list(robot.init_state.default_joint_angles.values()) == HOME
    assert set(robot.control.stiffness) == set(DOF_NAMES)
    assert Path(robot.motion.motion_file).is_dir()
    assert len(robot.body_names) == robot.num_bodies == 30
    assert robot.asset.robot_type == "x2_29dof"
    assert robot.asset.replace_cylinder_with_capsule is False
    term = x2.x2_dual_depth.terms["warp_depth"]
    original = g1.g1_29dof_rpl_distill.camera.terms["warp_depth"]
    for key in ("poses", "props", "noise", "pose_randomization", "fov_randomization"):
        assert term.params[key] == original.params[key]
    assert all(
        (Path(term.params["dynamic_meshes"]["meshes_root"]) / f).is_file()
        for f in term.params["dynamic_meshes"]["body_meshes"]
    )


@pytest.mark.parametrize("name", ["penalty_ee_lin_acc", "penalty_ee_ang_acc"])
def test_x2_end_effector_rewards_use_final_wrist_links(name):
    from types import SimpleNamespace

    import torch

    from rpl_training.terms import reward

    names = x2.x2_29dof.body_names
    velocity = torch.zeros(1, len(names), 3)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        body_names=names,
        episode_length_buf=torch.tensor([2]),
        simulator=SimpleNamespace(
            find_rigid_body_indice=names.index, _rigid_body_vel=velocity, _rigid_body_ang_vel=velocity
        ),
    )
    term = getattr(reward, name)
    params = x2.x2_29dof_rpl_expert.reward.terms[name].params
    assert term(env, **params).item() == 0
    velocity[:, names.index("left_wrist_roll_link"), 0] = 1
    velocity[:, names.index("right_wrist_roll_link"), 0] = 2
    velocity[:, names.index("left_wrist_yaw_link"), 0] = 100
    assert term(env, **params).item() == 3
    # Legacy G1 selection still follows its existing final wrist-yaw links.
    env.body_names = g1.g1_29dof_rpl_expert.robot.body_names
    env.simulator.find_rigid_body_indice = env.body_names.index
    assert reward._ee_hand_indices(env) == tuple(
        env.body_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")
    )


def test_g1_teacher_rejected():
    from rpl_training.x2_dagger import X2DAggerMoE

    fake = object.__new__(X2DAggerMoE)
    root = Path(__file__).resolve().parents[1] / "data/models"
    with pytest.raises(ValueError, match="compatible X2"):
        fake._resolve_per_terrain_ckpt(str(root), "slopes", {})


def test_depth_resize_matches_training_torch():
    import numpy as np
    import torch

    from rpl_sim2sim.depth_processing import resize_bicubic_antialias

    image = np.random.default_rng(3).uniform(0.1, 2.0, (135, 240)).astype(np.float32)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None, None], size=(27, 48), mode="bicubic", align_corners=False, antialias=True
    )[0, 0].numpy()
    np.testing.assert_allclose(resize_bicubic_antialias(image, 27, 48), expected, atol=2e-6)


@pytest.mark.parametrize("motion_enabled, expected_level", [(True, 2), (False, 1)])
def test_fixed_reference_terrain_curriculum_uses_episode_horizon(motion_enabled, expected_level):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from rpl_training.terms.curriculum import MotionTimeTerrainCurriculum

    terrain = SimpleNamespace(
        env_origins=torch.zeros(1, 3), terrain_levels=torch.tensor([2]), max_terrain_level=10, set_terrain_levels=Mock()
    )
    manager = SimpleNamespace(
        get_state=lambda _: terrain,
        cfg=SimpleNamespace(
            terrain_term=SimpleNamespace(
                mesh_type="trimesh", terrain_length=8, terrain_move_up_ratio=0.3125, terrain_move_down_ratio=0.5
            )
        ),
    )
    env = SimpleNamespace(
        device="cpu",
        init_done=True,
        terrain_manager=manager,
        max_episode_length_s=10,
        motion_reference_enabled=motion_enabled,
        motion_len=torch.zeros(1),
        motion_start_times=torch.zeros(1),
        simulator=SimpleNamespace(robot_root_states=torch.zeros(1, 13)),
        time_out_buf=torch.tensor([False]),
        command_manager=SimpleNamespace(commands=torch.tensor([[1.0, 0.0, 0.0]])),
    )
    term = MotionTimeTerrainCurriculum(SimpleNamespace(params={}), env)
    term.reset(torch.tensor([0]))
    assert terrain.set_terrain_levels.call_args[0][1].item() == expected_level
