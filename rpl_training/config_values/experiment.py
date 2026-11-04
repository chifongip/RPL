"""RPL / FALCON-Terrains experiment presets (G1 29-DoF loco-manipulation).

``g1_29dof_rpl_expert`` trains the per-terrain RL experts and
``g1_29dof_rpl_distill`` distills them into the multi-depth
student. ``g1_29dof_falcon_hm`` / ``g1_29dof_falcon_hm_force`` are their bases.
"""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_types.motion import MotionLibConfig
from holosoma.config_values import action
from holosoma.config_values.terrain import terrain_locomotion_plane

from rpl_training.config_values import algo, robot, simulator, terrain
from rpl_training.config_values.camera import warp_dual_depth_cameras_falcon, warp_front_depth_camera_falcon
from rpl_training.config_values.command import (
    g1_29dof_falcon_hm_command,
    g1_29dof_rpl_command_distill,
    g1_29dof_rpl_command_expert,
)
from rpl_training.config_values.curriculum import (
    g1_29dof_falcon_hm_curriculum,
    g1_29dof_falcon_hm_force_curriculum,
    g1_29dof_rpl_curriculum_distill,
    g1_29dof_rpl_curriculum_expert,
)
from rpl_training.config_values.observation import (
    g1_29dof_falcon_hm_history_wolinvel,
    g1_29dof_rpl_depth_distill,
    g1_29dof_rpl_history_wolinvel,
    g1_29dof_rpl_history_wolinvel_hm_noise,
)
from rpl_training.config_values.randomization import (
    g1_29dof_falcon_hm_randomization,
    g1_29dof_rpl_randomization_distill,
    g1_29dof_rpl_randomization_expert,
    g1_29dof_rpl_randomization_expert_slopes,
    g1_29dof_rpl_randomization_expert_stairs,
    g1_29dof_rpl_randomization_expert_stones,
)
from rpl_training.config_values.reward import (
    g1_29dof_falcon_hm_reward,
    g1_29dof_rpl_reward_expert,
    g1_29dof_rpl_reward_expert_slopes,
    g1_29dof_rpl_reward_expert_stairs,
    g1_29dof_rpl_reward_expert_stones,
)
from rpl_training.config_values.termination import g1_29dof_falcon_hm_termination
from rpl_training.paths import DATA_DIR

g1_29dof_falcon_hm = ExperimentConfig(
    env_class="rpl_training.envs.loco_manip_manager.LeggedRobotLocoManipManager",
    training=TrainingConfig(
        project="falcon-holosoma-g1-hm",
        name="g1_29dof_falcon_hm_flat",
        num_envs=4096,
        seed=1,
    ),
    algo=replace(
        algo.ppo_ma,
        config=replace(
            algo.ppo_ma.config,
            num_learning_iterations=50000,
            # Mirror FAR-FALCON's ``falcon_hm_multi_terrain.yaml`` symmetry_cfg:
            # data augmentation + actor mirror loss enabled for both bodies
            # (SymmetryUtils handles per-body slicing via body_key).
            use_symmetry=True,
            symmetry_actor_coef=1.0,
            symmetry_critic_coef=0.0,
        ),
    ),
    simulator=simulator.isaacgym,
    robot=replace(
        robot.g1_29dof_falcon,
        motion=MotionLibConfig(
            motion_file=str(DATA_DIR / "motions/g1/falcon_accad_all"),
            standardize_motion_length=True,
            standardize_motion_length_value=10.0,
        ),
    ),
    terrain=terrain_locomotion_plane,
    observation=g1_29dof_falcon_hm_history_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_falcon_hm_termination,
    randomization=g1_29dof_falcon_hm_randomization,
    command=g1_29dof_falcon_hm_command,
    curriculum=g1_29dof_falcon_hm_curriculum,
    reward=g1_29dof_falcon_hm_reward,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={
            "Episode/rew_tracking_ang_vel": [0.7, "inf"],
            "Episode/rew_tracking_lin_vel": [0.55, "inf"],
            "Episode/rew_tracking_upper_body_dofs": [0.5, "inf"],
        },
    ),
)

g1_29dof_falcon_hm_force = replace(
    g1_29dof_falcon_hm,
    env_class="rpl_training.envs.loco_manip_force_manager.LeggedRobotLocoManipForceManager",
    curriculum=g1_29dof_falcon_hm_force_curriculum,
)


# Reproduces FAR-FALCON-Terrains' depth distillation
# (``+exp=falcon_hm_dagger_moe algo=dagger_ma_moe_hm_cnn_depth_vtr``) with FALCON's own MoE
# teacher checkpoints (``moe_cnn/20251105``, migrated to
# ``data/models`` under the Holosoma terrain names). Every
# ``*_falcon`` / ``*_falcon_distill`` preset below mirrors the resolved FALCON config:
#   * obs: 5-frame state history with ``command_stand`` and absolute base height (the
#     FALCON teacher's 115-dim frame), depth history pushed every 5 steps, noisy heightmap;
#   * camera: 2-frame depth delay, sensor noise, per-env pose / FOV randomization,
#     240x135 ray cast + bicubic resize to 48x27;
#   * command: per-terrain sampled stand flag with the tapping mode, no yaw on stones,
#     upper-body freeze / waist zero-fix, FALCON force settings;
#   * curriculum: FALCON step sizes, penalty / DR-scale curricula off;
#   * terrain: FALCON mix (stones .35, rough slopes .15, stairs .25/.25) and heightmap lookup;
#   * randomization: base COM, upper-body-only control delay, pelvis added mass, friction;
#   * simulator: FALCON PhysX solver / contact settings;
#   * robot: FALCON armature + upper-body motion clips;
#   * algo: channel vision tokens, independent per-camera CNN init.
# FALCON's armature values (``g1_29dof_rev_1_0.yaml``) and its PHC-retargeted ACCAD motion set
# (``g1_29dof/v1/accad_all.pkl``), converted by ``scripts/convert_falcon_motion_pkl.py``, so the
# teacher's ``ref_upper_dof_pos`` inputs come from the clips it was trained on.
_FALCON_ARMATURE = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + [0.003597] * 14

g1_29dof_rpl_distill = replace(
    g1_29dof_falcon_hm_force,
    robot=replace(
        g1_29dof_falcon_hm_force.robot,
        dof_armature_list=_FALCON_ARMATURE,
        motion=replace(
            g1_29dof_falcon_hm_force.robot.motion,
            motion_file=str(DATA_DIR / "motions/g1/falcon_accad_all"),
        ),
    ),
    training=TrainingConfig(
        project="falcon-holosoma-g1-hm-distill",
        name="g1_29dof_rpl_distill",
        # Total across GPUs (train_agent divides by world size); FALCON's
        # ``num_envs=1024`` is per GPU, i.e. 8192 on its 8-GPU runs.
        num_envs=8192,
        seed=1,
    ),
    algo=replace(
        algo.dagger_ma_moe_cnn_depth_vtr_falcon,
        config=replace(
            algo.dagger_ma_moe_cnn_depth_vtr_falcon.config,
            num_learning_iterations=20000,
            num_scheduler_iterations=10000,
            num_mini_batches=4,
            ckpt_teacher=str(DATA_DIR / "models"),
        ),
    ),
    simulator=simulator.isaacgym_falcon,
    terrain=terrain.terrain_locomotion_multi_falcon,
    camera=warp_dual_depth_cameras_falcon,
    observation=g1_29dof_rpl_depth_distill,
    randomization=g1_29dof_rpl_randomization_distill,
    command=g1_29dof_rpl_command_distill,
    curriculum=g1_29dof_rpl_curriculum_distill,
)


# Front-only student, sized for four GPUs with 1024 environments per GPU.
# Teachers still consume height maps; only the student's visual input changes.
_distill_modules = g1_29dof_rpl_distill.algo.config.module_dict
g1_29dof_rpl_distill_front = replace(
    g1_29dof_rpl_distill,
    training=replace(g1_29dof_rpl_distill.training, name="g1_29dof_rpl_distill_front", num_envs=4096),
    camera=warp_front_depth_camera_falcon,
    algo=replace(
        g1_29dof_rpl_distill.algo,
        config=replace(
            g1_29dof_rpl_distill.algo.config,
            module_dict=replace(
                _distill_modules,
                actor_lower_body=replace(
                    _distill_modules.actor_lower_body,
                    layer_config=replace(
                        _distill_modules.actor_lower_body.layer_config,
                        num_cameras=1,
                        # Direction-based front/back selection is unnecessary with one camera.
                        use_vision_mask=False,
                    ),
                ),
            ),
        ),
    ),
)


# Reproduces FAR-FALCON-Terrains' RL experts (``+exp=falcon_hm_multi_terrain
# algo=ppo_ma_hm_cnn``) with FALCON's task: stand flag sampled per terrain with the
# tapping mode and ``command_stand`` in the actor obs (not the velocity-derived stance
# variant), FALCON rewards, absolute ``command_base_height``, history kept across resets,
# heightmap / base / feet heights read from the raw heightfield, directional
# ``slope_up`` / ``slope_down`` terrains, FALCON PhysX settings, armature, ACCAD upper-body
# clips, upper-body freeze / waist zero-fix, upper-body-only control delay, pelvis added
# mass, and the penalty / DR-scale curricula at FALCON's degree, plus the settings all three
# of FALCON's final expert stages share (IMU noise, lower-body torque weight, hand-force
# budget). The per-terrain experts below add the rest of their run args. ``num_envs`` is the
# total across GPUs (FALCON's ``num_envs=4096`` on 4 GPUs is ``--training.num-envs=16384``).
g1_29dof_rpl_expert = replace(
    g1_29dof_falcon_hm_force,
    robot=g1_29dof_rpl_distill.robot,
    training=TrainingConfig(
        project="falcon-holosoma-g1-hm-experts",
        name="g1_29dof_rpl_expert",
        num_envs=16384,
        seed=1,
    ),
    algo=replace(
        algo.ppo_ma_cnn,
        config=replace(
            algo.ppo_ma_cnn.config,
            num_learning_iterations=50000,
            use_symmetry=True,
            symmetry_actor_coef=1.0,
            symmetry_critic_coef=0.0,
        ),
    ),
    simulator=simulator.isaacgym_falcon,
    terrain=terrain.terrain_locomotion_multi_falcon_expert,
    observation=g1_29dof_rpl_history_wolinvel,
    randomization=g1_29dof_rpl_randomization_expert,
    command=g1_29dof_rpl_command_expert,
    curriculum=g1_29dof_rpl_curriculum_expert,
    reward=g1_29dof_rpl_reward_expert,
)


def _falcon_expert(name: str, **overrides) -> ExperimentConfig:
    training = replace(g1_29dof_rpl_expert.training, name=name, **overrides.pop("training", {}))
    return replace(g1_29dof_rpl_expert, training=training, **overrides)


# FALCON's final expert stages (wandb run args; the curricula and the stairs lower-body
# control delay are left at the base preset's from-scratch settings).
# far-wandb/falcon-29-terrains-experts-stones/lrlo8qpt, 4096 x 4 GPUs.
g1_29dof_rpl_expert_stones = _falcon_expert(
    "g1_29dof_rpl_expert_stones",
    terrain=terrain.terrain_locomotion_falcon_expert_stones,
    observation=g1_29dof_rpl_history_wolinvel_hm_noise,
    randomization=g1_29dof_rpl_randomization_expert_stones,
    reward=g1_29dof_rpl_reward_expert_stones,
)
# far-wandb/falcon-29-terrains-experts/h5d52hm0, 4096 x 4 GPUs.
g1_29dof_rpl_expert_stairs = _falcon_expert(
    "g1_29dof_rpl_expert_stairs",
    terrain=terrain.terrain_locomotion_falcon_expert_stairs,
    randomization=g1_29dof_rpl_randomization_expert_stairs,
    reward=g1_29dof_rpl_reward_expert_stairs,
)
# far-wandb/falcon-29-terrains-experts/b04ael3p, 4096 envs on 1 GPU (~28k fps vs ~112k).
g1_29dof_rpl_expert_slopes = _falcon_expert(
    "g1_29dof_rpl_expert_slopes",
    training={"num_envs": 4096},
    terrain=terrain.terrain_locomotion_falcon_expert_slopes,
    observation=g1_29dof_rpl_history_wolinvel_hm_noise,
    randomization=g1_29dof_rpl_randomization_expert_slopes,
    reward=g1_29dof_rpl_reward_expert_slopes,
)
