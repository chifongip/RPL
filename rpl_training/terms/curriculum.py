"""Curriculum hooks for loco-manipulation tasks."""

from __future__ import annotations

from typing import Any

import torch
from holosoma.managers.curriculum.base import CurriculumTermBase


class CommandScaleCurriculum(CurriculumTermBase):
    """RPL-style per-env command scale curriculum for loco-manip.

    Updates ``env.command_height_scale`` (for command index 8) and
    ``env.waist_command_scale`` (for command indices 5:8) based on finished
    episode lengths, then re-samples locomotion commands for updated envs.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled = bool(params.get("enabled", True))
        self.command_height_enabled = bool(params.get("command_height_scale_curriculum", True))
        self.waist_enabled = bool(params.get("waist_command_scale_curriculum", True))
        self.upper_body_enabled = bool(params.get("upper_body_motion_scale_curriculum", True))

        self.command_height_initial_scale = float(params.get("command_height_scale_initial_scale", 0.1))
        self.command_height_up_threshold = float(params.get("command_height_scale_up_threshold", 210.0))
        self.command_height_down_threshold = float(params.get("command_height_scale_down_threshold", 200.0))
        self.command_height_up = float(params.get("command_height_scale_up", 0.02))
        self.command_height_down = float(params.get("command_height_scale_down", 0.02))
        self.command_height_min = float(params.get("command_height_scale_min", 0.0))
        self.command_height_max = float(params.get("command_height_scale_max", 1.0))

        self.waist_initial_scale = float(params.get("waist_command_scale_initial_scale", 0.1))
        self.waist_up_threshold = float(params.get("waist_command_scale_up_threshold", 210.0))
        self.waist_down_threshold = float(params.get("waist_command_scale_down_threshold", 200.0))
        self.waist_up = float(params.get("waist_command_scale_up", 0.02))
        self.waist_down = float(params.get("waist_command_scale_down", 0.02))
        self.waist_min = float(params.get("waist_command_scale_min", 0.0))
        self.waist_max = float(params.get("waist_command_scale_max", 1.0))

        self.upper_body_initial_scale = float(params.get("upper_body_motion_scale_initial_scale", 0.1))
        self.upper_body_up_threshold = float(params.get("upper_body_motion_scale_up_threshold", 210.0))
        self.upper_body_down_threshold = float(params.get("upper_body_motion_scale_down_threshold", 200.0))
        self.upper_body_up = float(params.get("upper_body_motion_scale_up", 0.02))
        self.upper_body_down = float(params.get("upper_body_motion_scale_down", 0.02))
        self.upper_body_min = float(params.get("upper_body_motion_scale_min", 0.0))
        self.upper_body_max = float(params.get("upper_body_motion_scale_max", 1.0))

        self.resample_after_update = bool(params.get("resample_after_update", True))

    def setup(self) -> None:
        env = self.env
        device = env.device
        n = env.num_envs

        if not hasattr(env, "command_height_scale"):
            env.command_height_scale = torch.ones(n, 1, dtype=torch.float32, device=device)
        if not hasattr(env, "waist_command_scale"):
            env.waist_command_scale = torch.ones(n, 1, dtype=torch.float32, device=device)
        if not hasattr(env, "upper_body_motion_scale"):
            env.upper_body_motion_scale = torch.ones(n, 1, dtype=torch.float32, device=device)

        if self.enabled and self.command_height_enabled:
            env.command_height_scale[:] = self.command_height_initial_scale
        else:
            env.command_height_scale[:] = 1.0

        if self.enabled and self.waist_enabled:
            env.waist_command_scale[:] = self.waist_initial_scale
        else:
            env.waist_command_scale[:] = 1.0

        if self.enabled and self.upper_body_enabled:
            env.upper_body_motion_scale[:] = self.upper_body_initial_scale
        else:
            env.upper_body_motion_scale[:] = 1.0

        env.use_command_height_scale_curriculum = bool(self.enabled and self.command_height_enabled)
        env.use_waist_command_scale_curriculum = bool(self.enabled and self.waist_enabled)
        env.use_upper_body_motion_scale_curriculum = bool(self.enabled and self.upper_body_enabled)
        self._update_log_dict()

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        pending_lengths = getattr(self.env, "_pending_episode_lengths", None)
        pending_mask = getattr(self.env, "_pending_episode_update_mask", None)
        if pending_lengths is None or pending_mask is None:
            return

        valid_mask = pending_mask.index_select(0, idx)
        if not torch.any(valid_mask):
            return
        active_ids = idx[valid_mask]
        episode_lengths = pending_lengths.index_select(0, active_ids).to(dtype=torch.float32)

        if self.command_height_enabled:
            self._update_scale_(
                self.env.command_height_scale,
                active_ids,
                episode_lengths,
                up_threshold=self.command_height_up_threshold,
                down_threshold=self.command_height_down_threshold,
                up_step=self.command_height_up,
                down_step=self.command_height_down,
                min_scale=self.command_height_min,
                max_scale=self.command_height_max,
            )

        if self.waist_enabled:
            self._update_scale_(
                self.env.waist_command_scale,
                active_ids,
                episode_lengths,
                up_threshold=self.waist_up_threshold,
                down_threshold=self.waist_down_threshold,
                up_step=self.waist_up,
                down_step=self.waist_down,
                min_scale=self.waist_min,
                max_scale=self.waist_max,
            )

        if self.upper_body_enabled:
            self._update_scale_(
                self.env.upper_body_motion_scale,
                active_ids,
                episode_lengths,
                up_threshold=self.upper_body_up_threshold,
                down_threshold=self.upper_body_down_threshold,
                up_step=self.upper_body_up,
                down_step=self.upper_body_down,
                min_scale=self.upper_body_min,
                max_scale=self.upper_body_max,
            )

        if self.resample_after_update and not self.env.is_evaluating:
            cmd_term = getattr(self.env, "command_manager", None)
            if cmd_term is not None:
                loco_cmd = cmd_term.get_state("locomotion_command")
                if loco_cmd is not None and hasattr(loco_cmd, "_resample"):
                    loco_cmd._resample(active_ids)

        self._update_log_dict()

    def step(self) -> None:
        if not self.enabled:
            return
        # Keep scales bounded even if they are adjusted externally.
        self.env.command_height_scale.clamp_(self.command_height_min, self.command_height_max)
        self.env.waist_command_scale.clamp_(self.waist_min, self.waist_max)
        self.env.upper_body_motion_scale.clamp_(self.upper_body_min, self.upper_body_max)

    def _update_scale_(
        self,
        scale_buf: torch.Tensor,
        env_ids: torch.Tensor,
        episode_lengths: torch.Tensor,
        *,
        up_threshold: float,
        down_threshold: float,
        up_step: float,
        down_step: float,
        min_scale: float,
        max_scale: float,
    ) -> None:
        curr = scale_buf[env_ids].squeeze(-1)
        curr = torch.where(episode_lengths > up_threshold, curr + up_step, curr)
        curr = torch.where(episode_lengths < down_threshold, curr - down_step, curr)
        curr = torch.clamp(curr, min=min_scale, max=max_scale)
        scale_buf[env_ids] = curr.unsqueeze(-1)

    def _ensure_index_tensor(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)

    def _update_log_dict(self) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        self.env.log_dict["command_height_scale"] = self.env.command_height_scale.mean().detach().cpu()
        self.env.log_dict["waist_command_scale"] = self.env.waist_command_scale.mean().detach().cpu()
        self.env.log_dict["upper_body_motion_scale"] = self.env.upper_body_motion_scale.mean().detach().cpu()


class MotionTimeTerrainCurriculum(CurriculumTermBase):
    """RPL-style terrain curriculum using remaining motion time."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled = bool(params.get("enabled", True))
        self.move_up_ratio = params.get("terrain_move_up_ratio", None)
        self.move_down_ratio = params.get("terrain_move_down_ratio", None)

    def setup(self) -> None:
        return

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0 or not getattr(self.env, "init_done", False):
            return

        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        terrain_cfg = self.env.terrain_manager.cfg.terrain_term
        motion_len = getattr(self.env, "motion_len", None)
        motion_start_times = getattr(self.env, "motion_start_times", None)
        if (
            terrain_state is None
            or str(terrain_cfg.mesh_type) == "plane"
            or motion_len is None
            or motion_start_times is None
        ):
            return

        move_up_ratio = (
            float(self.move_up_ratio)
            if self.move_up_ratio is not None
            else float(terrain_cfg.terrain_move_up_ratio or 0.3125)
        )
        move_down_ratio = (
            float(self.move_down_ratio)
            if self.move_down_ratio is not None
            else float(terrain_cfg.terrain_move_down_ratio or 0.5)
        )

        distance = torch.norm(
            self.env.simulator.robot_root_states[idx, :2] - terrain_state.env_origins[idx, :2],
            dim=1,
        )
        move_up = distance > terrain_cfg.terrain_length * move_up_ratio
        move_up = move_up & self.env.time_out_buf[idx]

        remaining_motion_time = torch.clamp(motion_len[idx] - motion_start_times[idx], min=0.0)
        if not getattr(self.env, 'motion_reference_enabled', True):
            # Fixed-reference robots have no clip duration. Use the matched
            # episode horizon so the terrain curriculum can still move down.
            remaining_motion_time = torch.full_like(remaining_motion_time, self.env.max_episode_length_s)
        move_down = (
            distance
            < torch.norm(self.env.command_manager.commands[idx, :2], dim=1) * remaining_motion_time * move_down_ratio
        ) & ~move_up

        new_levels = terrain_state.terrain_levels[idx] + move_up.to(torch.long) - move_down.to(torch.long)
        max_level = max(int(terrain_state.max_terrain_level), 1)
        new_levels = torch.where(
            new_levels >= max_level,
            torch.randint_like(new_levels, max_level),
            torch.clamp(new_levels, min=0),
        )
        terrain_state.set_terrain_levels(idx, new_levels)

    def step(self) -> None:
        return

    def _ensure_index_tensor(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


class ForceScaleCurriculum(CurriculumTermBase):
    """Episode-length-driven curriculum for the EE force application scale.

    Mirrors FAR-FALCON-Terrains' ``force_scale_curriculum``: starts each env
    at ``initial_scale`` and adjusts ``env.apply_force_scale`` per env on
    reset based on the just-finished episode length. Envs that survived more
    than ``up_threshold`` steps step up by ``scale_up``; envs that died before
    ``down_threshold`` step down by ``scale_down``. The scale is clamped to
    ``[min_scale, max_scale]``.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled = bool(params.get("enabled", True))
        self.initial_scale = float(params.get("initial_scale", 0.1))
        self.up_threshold = float(params.get("up_threshold", 210.0))
        self.down_threshold = float(params.get("down_threshold", 200.0))
        self.scale_up = float(params.get("scale_up", 0.02))
        self.scale_down = float(params.get("scale_down", 0.02))
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))

    def setup(self) -> None:
        env = self.env
        device = env.device
        n = env.num_envs

        if not hasattr(env, "apply_force_scale"):
            env.apply_force_scale = torch.ones(n, 1, dtype=torch.float32, device=device)

        if self.enabled:
            env.apply_force_scale[:] = self.initial_scale
        else:
            env.apply_force_scale[:] = 1.0

        env.use_force_scale_curriculum = bool(self.enabled)
        self._update_log_dict()

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        pending_lengths = getattr(self.env, "_pending_episode_lengths", None)
        pending_mask = getattr(self.env, "_pending_episode_update_mask", None)
        if pending_lengths is None or pending_mask is None:
            return

        valid_mask = pending_mask.index_select(0, idx)
        if not torch.any(valid_mask):
            return
        active_ids = idx[valid_mask]
        episode_lengths = pending_lengths.index_select(0, active_ids).to(dtype=torch.float32)

        scale = self.env.apply_force_scale[active_ids].squeeze(-1)
        scale = torch.where(episode_lengths > self.up_threshold, scale + self.scale_up, scale)
        scale = torch.where(episode_lengths < self.down_threshold, scale - self.scale_down, scale)
        scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)
        self.env.apply_force_scale[active_ids] = scale.unsqueeze(-1)

        self._update_log_dict()

    def step(self) -> None:
        if not self.enabled:
            return
        self.env.apply_force_scale.clamp_(self.min_scale, self.max_scale)

    def _update_log_dict(self) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        self.env.log_dict["apply_force_scale"] = self.env.apply_force_scale.mean().detach().cpu()

    def _ensure_index_tensor(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


__all__ = ["CommandScaleCurriculum", "MotionTimeTerrainCurriculum", "ForceScaleCurriculum"]
