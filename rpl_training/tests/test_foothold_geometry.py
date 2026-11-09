"""Foothold support geometry through the real sampler and reward backends."""

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from types import SimpleNamespace

import pytest
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz
from holosoma.utils.safe_torch_import import torch

from rpl_training.config_values import experiment as g1
from rpl_training.config_values import x2
from rpl_training.envs.loco_manip_manager import LeggedRobotLocoManipManager
from rpl_training.terms.reward import penalty_foothold


def _sampling_env(cfg):
    env = object.__new__(LeggedRobotLocoManipManager)
    env.device = "cpu"
    env.robot_config = cfg.robot
    env.reward_manager = SimpleNamespace(cfg=cfg.reward)
    env._init_foothold_sampling()
    return env


def test_x2_sample_planes_match_packaged_contact_geometry():
    cfg = x2.x2_29dof_rpl_expert_slopes
    env = _sampling_env(cfg)
    asset = cfg.robot.asset
    root = ET.parse(str(x2.ASSET_DIR / asset.urdf_file)).getroot()
    for index, side in enumerate(("left", "right")):
        link = root.find(f"link[@name='{side}_ankle_roll_link']")
        bottoms = []
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is not None:
                z = float(collision.find("origin").get("xyz").split()[2])
                bottoms.append(z - float(sphere.get("radius")))
        assert min(bottoms) == pytest.approx(-0.073)
        torch.testing.assert_close(
            env.foot_samples_local[index, :, 2], torch.full((env.num_foot_samples,), min(bottoms))
        )


def test_g1_sampling_is_unchanged():
    env = _sampling_env(g1.g1_29dof_rpl_expert_slopes)
    samples = env.foot_samples_local.clone()
    LeggedRobotLocomotionManager._init_foothold_sampling(env)
    torch.testing.assert_close(samples, env.foot_samples_local, rtol=0, atol=0)


@pytest.mark.parametrize("falcon_heightfield", [False, True])
@pytest.mark.parametrize("roll,pitch,yaw", [(0.0, 0.0, 0.0), (0.2, 0.35, 0.4), (-0.2, -0.35, -0.4)])
def test_sole_support_on_flat_and_rotated_slopes(falcon_heightfield, roll, pitch, yaw):
    env = _sampling_env(x2.x2_29dof_rpl_expert_slopes)
    env.num_envs = 3
    env.feet_indices = [0, 1]
    quat = quat_from_euler_xyz(*(torch.tensor([angle]) for angle in (roll, pitch, yaw)))[0]
    # Rotate a flat support plane together with the feet, including yaw and roll.
    normal = quat_apply(quat[None], torch.tensor([[0.0, 0.0, 1.0]]), w_last=True)[0]

    def ground_height(points):
        return -(normal[0] * points[..., 0] + normal[1] * points[..., 1]) / normal[2]

    terrain = SimpleNamespace(
        _height_samples=torch.zeros(1),
        _falcon_heightfield_heights=lambda points, _: ground_height(points),
    )
    env.terrain_manager = SimpleNamespace(get_state=lambda _: terrain)
    env._get_terrain_heights_at_points_world = ground_height
    offsets = env.foot_samples_local[:, 0, 2]
    origins_local = torch.tensor([[0.0, 0.14, -offsets[0]], [0.0, -0.14, -offsets[1]]])
    origins = quat_apply(quat.expand(2, -1), origins_local, w_last=True)
    positions = origins.repeat(3, 1, 1)
    # Env 0: both soles supported. Env 1: right sole 6 cm clear despite contact
    # elsewhere on that foot. Env 2: the same clearance during a non-contact swing.
    positions[1:, 1, 2] += 0.06
    contacts = torch.zeros(3, 2, 3)
    contacts[:, :, 2] = 10.0
    contacts[2, 1, 2] = 0.0
    env.simulator = SimpleNamespace(
        _rigid_body_pos=positions,
        _rigid_body_rot=quat.repeat(3, 2, 1),
        contact_forces=contacts,
    )
    torch.testing.assert_close(
        penalty_foothold(env, falcon_heightfield=falcon_heightfield), torch.tensor([0.0, 1.0, 0.0])
    )
    # Reproduce the original geometry bug without weakening the 5 cm tolerance.
    if roll == pitch == yaw == 0:
        env.foot_samples_local[:, :, 2] = 0.0
        assert penalty_foothold(env, falcon_heightfield=falcon_heightfield)[0].item() == 2.0


@pytest.mark.parametrize("offsets", [(0.0,), (0.0, 0.0, 0.0), (math.nan, 0.0), (0.0, math.inf)])
def test_invalid_sample_planes_rejected(offsets):
    cfg = deepcopy(x2.x2_29dof_rpl_expert_slopes)
    cfg.reward.terms["penalty_foothold"].params["foothold_sample_z_offsets"] = offsets
    with pytest.raises(ValueError, match="two finite left/right sole coordinates"):
        _sampling_env(cfg)
