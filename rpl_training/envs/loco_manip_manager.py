"""Loco-manipulation environment (RPL HM-style).

Use with decoupled multi-agent PPO: per-body rewards are exposed via
``extras[\"rewards_ma\"]`` when the reward config sets ``multi_agent_body_keys``.
"""

from __future__ import annotations

import numpy as np
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.utils.safe_torch_import import torch
from loguru import logger


class LeggedRobotLocoManipManager(LeggedRobotLocomotionManager):
    """Same runtime as :class:`LeggedRobotLocomotionManager`; distinct task name for logging."""

    def __init__(self, tyro_config, *, device):
        self._next_task_requests = 0
        super().__init__(tyro_config, device=device)
        self._align_max_episode_length_to_motion()

    def _align_max_episode_length_to_motion(self):
        """Match ``max_episode_length_s`` to the standardised motion duration.

        When motions are length-standardised **and** the ``motion_end``
        termination term is active, episodes effectively last
        ``standardize_motion_length_value`` seconds.  The logged per-term
        rewards are normalised by ``max_episode_length_s``, so it must
        reflect the true episode duration to keep metrics comparable with
        RPL (which performs the same override).
        """
        motion_cfg = getattr(self.robot_config, "motion", None)
        if motion_cfg is None or not motion_cfg.standardize_motion_length:
            return

        has_motion_end_term = (
            self.termination_manager is not None and "motion_end" in self.termination_manager.cfg.terms
        )
        if not has_motion_end_term:
            return

        self.max_episode_length_s = motion_cfg.standardize_motion_length_value
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        logger.info(
            f"Aligned max_episode_length_s to standardised motion length: "
            f"{self.max_episode_length_s}s ({int(self.max_episode_length)} steps)"
        )

    def _init_foothold_sampling(self) -> None:
        """Place footprint samples on configured sole planes in the foot frames.

        ``foothold_sample_z_offsets`` contains left/right local Z coordinates
        in metres. Leaving it unset preserves the inherited FALCON sampling.
        Both foothold reward backends transform this grid with the foot pose,
        so the sole offset also follows foot pitch and roll on slopes.
        """
        super()._init_foothold_sampling()
        offsets = self._foothold_params_from_reward_cfg().get("foothold_sample_z_offsets")
        if offsets is None:
            return
        offsets = torch.as_tensor(offsets, device=self.device, dtype=self.foot_samples_local.dtype)
        if offsets.shape != (2,) or not torch.isfinite(offsets).all():
            raise ValueError("foothold_sample_z_offsets must contain two finite left/right sole coordinates in metres")
        self.foot_samples_local[:, :, 2] = offsets[:, None]

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Like :meth:`~holosoma.envs.base_task.BaseTask.reset_envs_idx` but runs curriculum before robot respawn.

        Terrain and motion-time terrain curricula need end-of-episode root pose and commands, and they
        update ``env_origins`` that :meth:`_reset_root_states` must apply on spawn. The default base
        order runs curriculum after robot reset, which breaks those signals for loco-manip.
        """
        for env_id in env_ids:
            if hasattr(self.simulator, "on_episode_end"):
                self.simulator.on_episode_end(env_id.item())

        self.observation_manager.reset(env_ids)

        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self._pending_episode_update_mask[env_ids] = True

        if self.curriculum_manager is not None:
            self.curriculum_manager.reset(env_ids)

        self._reset_envs_idx_impl(env_ids, target_states, target_buf)

        if self.randomization_manager is not None:
            self.randomization_manager.reset(env_ids)

        if self.action_manager is not None:
            self.action_manager.reset(env_ids)

        if self.command_manager is not None:
            self.command_manager.reset(env_ids)
        self._post_command_reset_callback(env_ids)

        if self.termination_manager is not None:
            self.termination_manager.reset(env_ids)

        self.reset_manager.reset_scene(env_ids)

        for env_id in env_ids:
            if hasattr(self.simulator, "on_episode_start"):
                self.simulator.on_episode_start(env_id.item())

    def _post_command_reset_callback(self, env_ids):
        """Hook run right after the command reset (default no-op)."""
        return

    def _get_task_name(self) -> str:
        return "loco_manip"

    def set_is_evaluating(self, command=None):
        """Eval hook: reset commands only (does not call :meth:`LeggedRobotLocomotionManager.set_is_evaluating`)."""
        logger.info("Setting Env is evaluating")
        super(LeggedRobotLocomotionManager, self).set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)

    def _update_log_dict(self):
        super()._update_log_dict()
        if hasattr(self, "command_height_scale"):
            self.log_dict["command_height_scale"] = self.command_height_scale.mean().detach().cpu()
        if hasattr(self, "waist_command_scale"):
            self.log_dict["waist_command_scale"] = self.waist_command_scale.mean().detach().cpu()
        if hasattr(self, "upper_body_motion_scale"):
            self.log_dict["upper_body_motion_scale"] = self.upper_body_motion_scale.mean().detach().cpu()

    def next_task(self):
        """Advance to the next block of motions during evaluation."""
        motion_state = self.command_manager.get_state("upper_body_ref_pose")
        motion_lib = getattr(motion_state, "motion_lib", None)
        if motion_lib is None:
            return

        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        print(f"Loaded {self.motion_start_idx} motions")
        print(f"Num motions: {self.num_motions}")
        self.reset_all()

    def request_next_task(self):
        """Thread-safe signal from keyboard listener to switch task on sim thread."""
        self._next_task_requests += 1

    def _setup_simulator_next_task(self):
        """Process queued task-switch requests on the simulation thread."""
        if self._next_task_requests <= 0:
            return
        self._next_task_requests -= 1
        self.next_task()
