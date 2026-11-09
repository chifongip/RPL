"""X2-specific inference contracts in the policy environment."""

import numpy as np
import pytest

pytest.importorskip("holosoma_inference")
pytest.importorskip("onnxruntime")

from rpl_sim2sim.inference_presets import DEFAULT_MODEL, x2_29dof_rpl_depth, x2_29dof_rpl_flat
from rpl_sim2sim.policy import RplDepthPolicy


def test_g1_export_rejected_for_x2():
    policy = object.__new__(RplDepthPolicy)
    policy.config = x2_29dof_rpl_depth
    with pytest.raises(ValueError, match="X2 export"):
        policy.setup_policy(str(DEFAULT_MODEL))


def _flat_policy_and_state():
    policy = object.__new__(RplDepthPolicy)
    policy.config = x2_29dof_rpl_flat
    policy._init_robot_config(policy.config.robot)
    policy._init_obs_config()
    policy.last_policy_action = np.zeros((1, 29))
    policy.lin_vel_command = np.zeros((1, 2))
    policy.ang_vel_command = np.zeros((1, 1))
    policy.stand_command = np.zeros((1, 1))
    policy.base_height_command = np.array([[0.64]])
    policy.desired_base_height = 0.64
    policy.waist_dofs_command = np.zeros((1, 3))
    policy.ref_upper_dof_pos = np.array([policy.config.robot.default_dof_angles[15:]])
    policy.depth_sensor = None
    state = np.zeros((1, 71))
    state[0, 3] = 1
    state[0, 7:36] = policy.config.robot.default_dof_angles
    return policy, state


def test_flat_height_observation_matches_training_delta():
    policy, state = _flat_policy_and_state()
    obs = policy.get_current_obs_buffer_dict(state)
    np.testing.assert_allclose(obs["command_base_height"], 0)
    np.testing.assert_allclose(obs["dof_pos"], 0)
    assert {k: v.shape for k, v in policy.prepare_obs_for_rl(state).items()} == {
        "actor_obs_lower_body": (1, 575),
        "actor_obs_upper_body": (1, 575),
    }


def test_joint_observations_keep_x2_order_and_commands_keep_semantic_order():
    from rpl_training.x2 import DOF_NAMES

    policy, state = _flat_policy_and_state()
    assert list(policy.dof_names) == DOF_NAMES
    assert policy.lower_dof_indices == list(range(15))
    assert policy.upper_dof_indices == list(range(15, 29))
    delta = np.arange(1, 30)[None, :] * 0.01
    state[:, 7:36] += delta
    state[:, 42:71] = delta * 2
    policy.last_policy_action = delta * 3
    policy.ref_upper_dof_pos = delta[:, 15:] * 4
    policy.waist_dofs_command[:] = [0.11, 0.22, 0.33]  # yaw / roll / pitch
    obs = policy.get_current_obs_buffer_dict(state)
    np.testing.assert_allclose(obs["dof_pos"], delta)
    np.testing.assert_allclose(obs["dof_vel"], delta * 2)
    np.testing.assert_allclose(obs["actions"], delta * 3)
    np.testing.assert_allclose(obs["ref_upper_dof_pos"], delta[:, 15:] * 4)
    np.testing.assert_allclose(obs["command_waist_dofs"], [[0.11, 0.22, 0.33]])
    inputs = policy.prepare_obs_for_rl(state)
    frame = inputs["actor_obs_lower_body"][:, -115:]
    offset = 0
    for name in sorted(policy.config.observation.obs_dict["actor_obs"]):
        width = policy.config.observation.obs_dims[name]
        expected = obs[name] * policy.config.observation.obs_scales[name]
        np.testing.assert_allclose(frame[:, offset : offset + width], expected, atol=1e-7)
        offset += width
    assert offset == 115
    np.testing.assert_array_equal(inputs["actor_obs_lower_body"], inputs["actor_obs_upper_body"])


def test_x2_inference_nominal_height_matches_training():
    from rpl_sim2sim.inference_presets import x2_29dof_rpl_depth, x2_29dof_rpl_flat

    assert x2_29dof_rpl_depth.task.desired_base_height == 0.64
    assert x2_29dof_rpl_flat.task.desired_base_height == 0.64
