"""Register the RPL presets with Holosoma's config registries.

Importing this module makes ``exp:g1-29dof-rpl-expert`` etc. selectable in
``python -m rpl_training.train_agent`` without installing the package. When the package is
installed, the ``holosoma.config.*`` entry points in ``pyproject.toml`` register the same
objects for ``python -m holosoma.train_agent``.
"""

from __future__ import annotations

from holosoma.config_values.camera import CAMERA_MANAGER_REGISTRY
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.config_values.terrain import TERRAIN_REGISTRY

from rpl_training.config_values import camera, experiment, robot, simulator, terrain

EXPERIMENTS = {
    # Per-terrain RL experts (FALCON ``+exp=falcon_hm_multi_terrain algo=ppo_ma_hm_cnn``).
    "g1_29dof_rpl_expert": experiment.g1_29dof_rpl_expert,
    # FALCON's final expert stages per terrain.
    "g1_29dof_rpl_expert_stones": experiment.g1_29dof_rpl_expert_stones,
    "g1_29dof_rpl_expert_stairs": experiment.g1_29dof_rpl_expert_stairs,
    "g1_29dof_rpl_expert_slopes": experiment.g1_29dof_rpl_expert_slopes,
    # Multi-depth DAgger student (FALCON ``+exp=falcon_hm_dagger_moe algo=dagger_ma_moe_hm_cnn_depth_vtr``).
    "g1_29dof_rpl_distill": experiment.g1_29dof_rpl_distill,
    "g1_29dof_rpl_distill_front": experiment.g1_29dof_rpl_distill_front,
}


def register() -> None:
    """Add the presets to the registries (idempotent)."""
    for name, cfg in EXPERIMENTS.items():
        EXPERIMENT_REGISTRY.add(name, cfg)
    CAMERA_MANAGER_REGISTRY.add("warp_dual_depth_cameras_falcon", camera.warp_dual_depth_cameras_falcon)
    CAMERA_MANAGER_REGISTRY.add("warp_front_depth_camera_falcon", camera.warp_front_depth_camera_falcon)
    TERRAIN_REGISTRY.add("terrain_locomotion_multi", terrain.terrain_locomotion_multi)
    TERRAIN_REGISTRY.add("terrain_locomotion_multi_falcon", terrain.terrain_locomotion_multi_falcon)
    TERRAIN_REGISTRY.add("terrain_locomotion_multi_falcon_expert", terrain.terrain_locomotion_multi_falcon_expert)
    SIMULATOR_REGISTRY.add("isaacgym_falcon", simulator.isaacgym_falcon)
    ROBOT_REGISTRY.add("g1_29dof_falcon", robot.g1_29dof_falcon)


register()
