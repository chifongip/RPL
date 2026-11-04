"""Policy-side checks (need holosoma_inference + onnxruntime): ONNX contract and history layout."""

from __future__ import annotations

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("holosoma_inference")
ort = pytest.importorskip("onnxruntime")

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd  # noqa: E402

from rpl_sim2sim.inference_presets import MODELS_DIR, g1_29dof_rpl_depth, rpl_depth_observation  # noqa: E402
from rpl_sim2sim.policy import FALCON_JOYSTICK_COMMANDS, FalconCommand, RplDepthPolicy  # noqa: E402


def _group_dim(group: str) -> int:
    obs = rpl_depth_observation
    frame = sum(obs.obs_dims[t] for t in obs.obs_dict[group])
    return frame * obs.history_length_dict[group]


@pytest.mark.parametrize("model", sorted(MODELS_DIR.glob("*.onnx")), ids=lambda p: p.name)
def test_shipped_students_match_the_observation_layout(model) -> None:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape[-1] for i in session.get_inputs()}
    lower = _group_dim("actor_obs") + _group_dim("perception_obs") + _group_dim("command_lin_vel_obs")
    assert shapes == {"actor_obs_lower_body": lower, "actor_obs_upper_body": _group_dim("actor_obs")}
    assert (lower, _group_dim("actor_obs")) == (5761, 575)
    (out,) = session.get_outputs()
    assert out.shape[-1] == 29
    feed = {name: np.zeros((1, dim), dtype=np.float32) for name, dim in shapes.items()}
    assert np.isfinite(session.run(None, feed)[0]).all()


def test_depth_shape_matches_the_sim_block() -> None:
    cam = g1_29dof_rpl_depth.camera
    assert (cam.num_cameras, cam.props.resized_height, cam.props.resized_width) == (2, 27, 48)
    assert rpl_depth_observation.obs_dims["cam_depth"] == 2 * 27 * 48


def test_front_depth_shape_matches_the_sim_block() -> None:
    from rpl_sim2sim.inference_presets import g1_29dof_rpl_depth_front
    from rpl_sim2sim.run_sim import PACKAGE_DIR, load_config

    cfg = g1_29dof_rpl_depth_front
    sim = load_config(PACKAGE_DIR / "config/loco_manip_cam_front.yaml")
    assert list(cfg.camera.poses) == list(sim["ROBOT_CFG"]["cameras"]["poses"]) == ["cam_front_depth"]
    assert cfg.camera.num_cameras == 1
    assert cfg.observation.obs_dims["cam_depth"] == 27 * 48
    assert cfg.observation.history_length_dict["perception_obs"] == 2
    assert cfg.task.model_path == ""


def _bare_policy(history: dict[str, int], intervals: dict[str, int]) -> RplDepthPolicy:
    policy = object.__new__(RplDepthPolicy)
    policy.obs_terms_sorted = {group: ["b", "a"] if group == "g" else ["x"] for group in history}
    policy.history_length_dict = history
    policy.obs_intervals = intervals
    policy.obs_interval_counter = 0
    policy.group_history = {group: deque(maxlen=n) for group, n in history.items()}
    return policy


def test_history_is_time_major_with_zero_padding() -> None:
    policy = _bare_policy({"g": 3}, {"g": 1})
    outs = []
    for t in range(1, 4):
        outs.append(policy._update_obs_history({"g": {"a": np.full((1, 1), 10 * t), "b": np.full((1, 2), t)}})["g"])
    # terms in obs_terms_sorted order (b, a), frames oldest first, zeros before the first push
    np.testing.assert_array_equal(outs[0], [[0, 0, 0, 0, 0, 0, 1, 1, 10]])
    np.testing.assert_array_equal(outs[2], [[1, 1, 10, 2, 2, 20, 3, 3, 30]])


def test_interval_holds_frames_between_pushes() -> None:
    policy = _bare_policy({"d": 2}, {"d": 5})
    seen = [policy._update_obs_history({"d": {"x": np.full((1, 1), t)}})["d"].ravel().tolist() for t in range(11)]
    assert seen[0] == [0, 0] and seen[4] == [0, 0]
    assert seen[5] == [0, 5] and seen[9] == [0, 5]
    assert seen[10] == [5, 10]


def _velocity_policy(joystick=False):
    policy = object.__new__(RplDepthPolicy)
    policy._falcon_joystick = joystick
    policy.fixed_x_speed = 0.0
    policy.fixed_x_speed_enabled = False
    policy._last_velocity_cmd = VelCmd((0.0, 0.0), 0.0)
    policy.stand_command = np.zeros((1, 1), dtype=int)
    policy.lin_vel_command = np.zeros((1, 2))
    policy.ang_vel_command = np.zeros((1, 1))
    policy.base_height_command = np.array([[0.60]])
    policy.waist_dofs_command = np.zeros((1, 3))
    policy.desired_base_height = 0.75
    policy._velocity_input = Mock()
    policy.logger = Mock()
    return policy


@pytest.mark.parametrize("joystick", [False, True])
@pytest.mark.parametrize(
    "linear,yaw,moving",
    [
        ((0.0, 0.0), 0.0, False),
        ((0.1, 0.0), 0.1, False),
        ((0.11, 0.0), 0.0, True),
        ((-0.11, 0.0), 0.0, True),
        ((0.0, 0.0), 0.11, True),
        ((0.0, 0.0), -0.11, True),
    ],
)
def test_mode_follows_velocity_and_returns_to_stance(joystick, linear, yaw, moving):
    policy = _velocity_policy(joystick)
    policy._apply_velocity(VelCmd(linear, yaw))
    assert bool(policy.stand_command[0, 0]) == moving
    np.testing.assert_allclose(policy.lin_vel_command[0], linear)
    assert policy.ang_vel_command[0, 0] == yaw
    assert policy.base_height_command[0, 0] == (0.75 if moving else 0.60)
    policy._apply_velocity(VelCmd((0.0, 0.0), 0.0))
    assert policy.stand_command[0, 0] == 0


def test_joystick_ignores_lateral_but_keyboard_uses_xy_norm():
    for joystick in (False, True):
        policy = _velocity_policy(joystick)
        policy._apply_velocity(VelCmd((0.08, 0.08), 0.0))
        assert bool(policy.stand_command[0, 0]) == (not joystick)
        assert policy.lin_vel_command[0, 1] == (0.0 if joystick else 0.08)


def test_fixed_speed_updates_mode_immediately_and_restores_stick_when_disabled():
    policy = _velocity_policy(joystick=True)
    policy._apply_velocity(VelCmd((0.0, 0.0), 0.0))
    policy._dispatch_command(FalconCommand.FIXED_X_UP)
    policy._dispatch_command(FalconCommand.FIXED_X_UP)
    assert policy.stand_command[0, 0] == 0  # Override still disabled.
    policy._dispatch_command(FalconCommand.FIXED_X_TOGGLE)
    assert policy.stand_command[0, 0] == 1
    assert policy.lin_vel_command[0, 0] == pytest.approx(0.2)
    policy._dispatch_command(FalconCommand.FIXED_X_TOGGLE)
    assert policy.stand_command[0, 0] == 0
    assert policy.lin_vel_command[0, 0] == 0.0


@pytest.mark.parametrize("cmd", [StateCommand.ZERO_VELOCITY, StateCommand.STAND])
def test_explicit_zero_cancels_fixed_speed_and_enters_stance(cmd):
    policy = _velocity_policy(joystick=True)
    policy.fixed_x_speed = 0.5
    policy.fixed_x_speed_enabled = True
    policy._apply_velocity(VelCmd((0.2, 0.0), 0.3))
    policy._dispatch_command(cmd)
    policy._velocity_input.zero.assert_called_once()
    assert not policy.fixed_x_speed_enabled
    assert policy.stand_command[0, 0] == 0
    np.testing.assert_array_equal(policy.lin_vel_command, 0.0)
    np.testing.assert_array_equal(policy.ang_vel_command, 0.0)
    policy._dispatch_command(FalconCommand.BASE_HEIGHT_DOWN)
    assert policy.stand_command[0, 0] == 0


def test_manual_mode_commands_do_not_override_automatic_mode():
    policy = _velocity_policy()
    assert StateCommand.STAND_TOGGLE not in FALCON_JOYSTICK_COMMANDS.values()
    for velocity in (0.0, 0.5):
        policy._apply_velocity(VelCmd((velocity, 0.0), 0.0))
        for cmd in (StateCommand.STAND_TOGGLE, StateCommand.WALK):
            policy._dispatch_command(cmd)
            assert policy.stand_command[0, 0] == int(velocity > 0.1)


def test_height_resets_only_when_entering_walk():
    policy = _velocity_policy()
    policy._apply_velocity(VelCmd((0.5, 0.0), 0.0))
    assert policy.base_height_command[0, 0] == 0.75
    policy.base_height_command[0, 0] = 0.70
    policy._apply_velocity(VelCmd((0.6, 0.0), 0.0))
    assert policy.base_height_command[0, 0] == 0.70
