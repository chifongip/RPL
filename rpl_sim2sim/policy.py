"""RPL / FALCON-Terrains depth student for ``holosoma_inference``.

Runs the decoupled lower/upper-body student exported by DAgger-MA distillation
(``exp:g1-29dof-rpl-distill``) and FALCON-Terrains' own depth exports:

* ONNX inputs ``actor_obs_lower_body = [actor_obs | perception_obs | command_lin_vel_obs]`` and
  ``actor_obs_upper_body = actor_obs``, output ``action`` [29] in motor order;
* observation groups flattened TIME-MAJOR (``[t-N: all terms, ..., t: all terms]``), terms sorted,
  as in training (``ObsGroupCfg(history_time_major=True)``);
* the network runs on every control tick, applied or not, as in FALCON's deployment, so the history
  and the ``actions`` term are warm (not zero frames) when the policy is started;
* per-group push intervals (``RplObservationConfig.obs_intervals``): the depth group takes a new
  frame every 5 policy steps (10 Hz camera at a 50 Hz policy);
* with the Xbox pad, FALCON's end-effector control: ``R1`` / ``X`` + d-pad move both hands' targets
  (pelvis frame) and the arm IK (:mod:`rpl_sim2sim.arm_ik`, own 30 Hz thread) turns them into the
  ``ref_upper_dof_pos`` the student tracks, as FALCON's ``use_upper_body_controller``;
* depth from the sim's ``depth_img_shm`` block, ``[num_cameras, 1, 27, 48]`` normalized to
  [-0.5, 0.5] (front camera first), with the producer heartbeat checked every step.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto

import numpy as np
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.interface import InterfaceInput
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.sensors.depth_heartbeat import STATE_FAILED
from holosoma_inference.sensors.depth_shm import DepthShmSensor
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RplObservationConfig(ObservationConfig):
    """Observation config with FALCON's per-group push intervals."""

    obs_intervals: dict[str, int] | None = None
    """Push a new frame into a group's history every N policy steps (default 1)."""


class FalconCommand(Enum):
    """FALCON loco-manip joystick commands with no ``StateCommand`` equivalent."""

    BASE_HEIGHT_UP = auto()
    BASE_HEIGHT_DOWN = auto()
    WAIST_PITCH_UP = auto()
    WAIST_PITCH_DOWN = auto()
    WAIST_YAW_LEFT = auto()
    WAIST_YAW_RIGHT = auto()
    FIXED_X_UP = auto()
    FIXED_X_DOWN = auto()
    FIXED_X_TOGGLE = auto()
    EE_X_UP = auto()
    EE_X_DOWN = auto()
    EE_Y_OUT = auto()
    EE_Y_IN = auto()
    EE_Z_UP = auto()
    EE_Z_DOWN = auto()


# FAR-FALCON-Terrains' Xbox mapping (``base_policy`` / ``locomotion`` / ``loco_manip_stand_height_waist``
# ``handle_joystick_button``), in the wireless-controller labels the sim bridge publishes.
FALCON_JOYSTICK_COMMANDS: dict[str, StateCommand | FalconCommand] = {
    "start": StateCommand.START,
    "B+Y": StateCommand.STOP,
    "A+X": StateCommand.INIT,
    "L2": StateCommand.ZERO_VELOCITY,
    "Y+left": StateCommand.KP_DOWN,
    "Y+right": StateCommand.KP_UP,
    "A+left": StateCommand.KP_DOWN_FINE,
    "A+right": StateCommand.KP_UP_FINE,
    "A+Y": StateCommand.KP_RESET,
    "A+B": StateCommand.KP_RESET,
    "B+up": FalconCommand.BASE_HEIGHT_UP,
    "B+down": FalconCommand.BASE_HEIGHT_DOWN,
    "Y+up": FalconCommand.WAIST_PITCH_UP,
    "select+up": FalconCommand.WAIST_PITCH_UP,
    "Y+down": FalconCommand.WAIST_PITCH_DOWN,
    "select+down": FalconCommand.WAIST_PITCH_DOWN,
    "select+left": FalconCommand.WAIST_YAW_LEFT,
    "select+right": FalconCommand.WAIST_YAW_RIGHT,
    "A+up": FalconCommand.FIXED_X_UP,
    "A+down": FalconCommand.FIXED_X_DOWN,
    "L1+R1": FalconCommand.FIXED_X_TOGGLE,
    "R1+up": FalconCommand.EE_X_UP,
    "R1+down": FalconCommand.EE_X_DOWN,
    "R1+left": FalconCommand.EE_Y_OUT,
    "R1+right": FalconCommand.EE_Y_IN,
    "X+up": FalconCommand.EE_Z_UP,
    "X+down": FalconCommand.EE_Z_DOWN,
}

# FALCON ``LocoManipStandHeightWaistPolicy``: initial hand targets (pelvis frame, m) and per-press steps.
EE_INIT_LEFT = (0.30, 0.13, 0.08)
EE_INIT_RIGHT = (0.30, -0.13, 0.08)
EE_STEPS: dict[FalconCommand, tuple[float, float, float]] = {
    # (dx, dy_left, dz); the right hand mirrors dy.
    FalconCommand.EE_X_UP: (0.05, 0.0, 0.0),
    FalconCommand.EE_X_DOWN: (-0.05, 0.0, 0.0),
    FalconCommand.EE_Y_OUT: (0.0, 0.02, 0.0),
    FalconCommand.EE_Y_IN: (0.0, -0.02, 0.0),
    FalconCommand.EE_Z_UP: (0.0, 0.0, 0.05),
    FalconCommand.EE_Z_DOWN: (0.0, 0.0, -0.05),
}
UPPER_BODY_CONTROLLER_HZ = 30.0


class RplDepthPolicy(LocomotionPolicy):
    """Decoupled loco-manipulation depth student (RPL / FALCON-Terrains)."""

    enable_arm_ik = True
    # Match training's default stand_vel_threshold (m/s for XY, rad/s for yaw).
    stand_vel_threshold = 0.1

    def __init__(self, config):
        super().__init__(config)
        robot = self.config.robot
        upper_indices = [robot.dof_names.index(n) for n in robot.dof_names_upper_body]
        # Upper-body reference pose: the default angles (FALCON's sim2sim has no motion clips).
        self.ref_upper_dof_pos = np.asarray(robot.default_dof_angles, dtype=np.float32)[upper_indices].reshape(1, -1)
        self.upper_indices = np.asarray(upper_indices)
        self.arm_ik_thread = None
        if self._falcon_joystick and self.enable_arm_ik:
            self._init_upper_body_controller()
        self._init_depth_sensor()

    # --------------------------------------------------------------- joystick
    def _create_input_providers(self):
        self.fixed_x_speed = 0.0
        self.fixed_x_speed_enabled = False
        self._last_velocity_cmd = VelCmd((0.0, 0.0), 0.0)
        super()._create_input_providers()
        self._falcon_joystick = isinstance(self._velocity_input, InterfaceInput)
        for provider in {id(p): p for p in (self._velocity_input, self._command_provider)}.values():
            if isinstance(provider, InterfaceInput):
                provider._mapping = dict(FALCON_JOYSTICK_COMMANDS)

    def _apply_velocity(self, vc: VelCmd) -> None:
        """Derive mode from the requested velocity, before any stance gating."""
        self._last_velocity_cmd = vc
        vx, vy = vc.lin_vel
        if self._falcon_joystick:
            vy = 0.0  # FALCON ignores the stick's lateral axis.
            if self.fixed_x_speed_enabled:
                vx = self.fixed_x_speed
        moving = np.hypot(vx, vy) > self.stand_vel_threshold or abs(vc.ang_vel) > self.stand_vel_threshold
        was_moving = bool(self.stand_command[0, 0])
        self.stand_command[0, 0] = int(moving)
        self.lin_vel_command[0] = (vx, vy)
        self.ang_vel_command[0, 0] = vc.ang_vel
        if moving and not was_moving:
            self.base_height_command[0, 0] = self.desired_base_height

    def _handle_zero_velocity(self):
        # A latched fixed speed must not undo an explicit zero-velocity command.
        self.fixed_x_speed_enabled = False
        super()._handle_zero_velocity()
        self._apply_velocity(VelCmd((0.0, 0.0), 0.0))

    def _print_control_status(self):
        BasePolicy._print_control_status(self)
        mode = "Walking" if self.stand_command[0, 0] else "Standing"
        vx, vy = self.lin_vel_command[0]
        self.logger.info(
            f"{mode} (automatic) | vx {vx:+.2f}, vy {vy:+.2f} m/s | yaw {self.ang_vel_command[0, 0]:+.2f} rad/s"
        )

    def _dispatch_command(self, cmd):
        if cmd in (StateCommand.STAND_TOGGLE, StateCommand.WALK):
            self.logger.info("Stand/walk mode is automatic from commanded velocity.")
            return
        if cmd == StateCommand.STAND:
            self._handle_zero_velocity()
            return
        if not isinstance(cmd, FalconCommand):
            super()._dispatch_command(cmd)
            return
        if cmd is FalconCommand.BASE_HEIGHT_UP:
            self.base_height_command[0, 0] += 0.1
        elif cmd is FalconCommand.BASE_HEIGHT_DOWN:
            self.base_height_command[0, 0] -= 0.1
        elif cmd is FalconCommand.WAIST_PITCH_UP:
            self.waist_dofs_command[0, 2] -= 0.05
        elif cmd is FalconCommand.WAIST_PITCH_DOWN:
            self.waist_dofs_command[0, 2] += 0.05
        elif cmd is FalconCommand.WAIST_YAW_LEFT:
            self.waist_dofs_command[0, 0] -= 0.1
        elif cmd is FalconCommand.WAIST_YAW_RIGHT:
            self.waist_dofs_command[0, 0] += 0.1
        elif cmd is FalconCommand.FIXED_X_UP:
            self.fixed_x_speed += 0.1
        elif cmd is FalconCommand.FIXED_X_DOWN:
            self.fixed_x_speed -= 0.1
        elif cmd is FalconCommand.FIXED_X_TOGGLE:
            self.fixed_x_speed_enabled = not self.fixed_x_speed_enabled
        elif cmd in EE_STEPS:
            self._step_ee_targets(*EE_STEPS[cmd])
            return
        self._apply_velocity(self._last_velocity_cmd)
        self.logger.info(
            f"base height {self.base_height_command[0, 0]:.2f} | waist yaw/pitch {self.waist_dofs_command[0, 0]:+.2f}/"
            f"{self.waist_dofs_command[0, 2]:+.2f} | fixed x {self.fixed_x_speed:+.2f} "
            f"({'on' if self.fixed_x_speed_enabled else 'off'})"
        )

    # ------------------------------------------------------- end-effector IK
    def _init_upper_body_controller(self) -> None:
        from rpl_sim2sim.arm_ik import ArmIK, ArmIKThread

        self.ee_left = np.array(EE_INIT_LEFT, dtype=np.float64)
        self.ee_right = np.array(EE_INIT_RIGHT, dtype=np.float64)
        ik = ArmIK(list(self.config.robot.dof_names_upper_body))
        ik.set_initial_targets(self.ee_left, self.ee_right)
        self.arm_ik_thread = ArmIKThread(ik, self.ref_upper_dof_pos, UPPER_BODY_CONTROLLER_HZ)
        self.arm_ik_thread.start()

    def _step_ee_targets(self, dx: float, dy: float, dz: float) -> None:
        if self.arm_ik_thread is None:
            self.logger.info("Hand target command ignored: ArmIK is disabled.")
            return
        self.ee_left += (dx, dy, dz)
        self.ee_right += (dx, -dy, dz)
        self.logger.info(f"EE target left {np.round(self.ee_left, 2)} | right {np.round(self.ee_right, 2)}")

    def _update_upper_body_reference(self, robot_state_data) -> None:
        """Hand the IK thread the measured arms + targets; adopt its newest solution."""
        q_upper = robot_state_data[0, 7 + self.upper_indices]
        self.arm_ik_thread.submit(q_upper, self.ee_left, self.ee_right)
        self.ref_upper_dof_pos[:] = self.arm_ik_thread.latest()

    # ------------------------------------------------------------------ depth
    def _init_depth_sensor(self) -> None:
        camera = self.config.camera
        if camera is None:
            raise ValueError("RplDepthPolicy needs an inference camera config (num cameras + resized size).")
        shape = (camera.num_cameras, 1, camera.props.resized_height, camera.props.resized_width)
        shm_cfg = self.config.task.depth_shm
        heartbeat = (
            (shm_cfg.heartbeat_endpoint or f"ipc:///tmp/{shm_cfg.name}.heartbeat") if shm_cfg.heartbeat else None
        )
        self.depth_sensor = DepthShmSensor(
            shape=shape, name=shm_cfg.name, required=shm_cfg.required, heartbeat_endpoint=heartbeat
        )
        self.depth_sensor.start()
        self._depth_stale_warned = False
        self._depth_dim = int(np.prod(shape))

    # ------------------------------------------------------------ observations
    def _initialize_history_state(self):
        super()._initialize_history_state()
        intervals = getattr(self.obs_config, "obs_intervals", None) or {}
        self.obs_intervals = {group: int(intervals.get(group, 1)) for group in self.obs_terms_sorted}
        self.obs_interval_counter = 0
        self.group_history: dict[str, deque] = {
            group: deque(maxlen=self.history_length_dict.get(group, 1)) for group in self.obs_terms_sorted
        }

    def get_current_obs_buffer_dict(self, robot_state_data):
        obs = super().get_current_obs_buffer_dict(robot_state_data)
        # Absolute stance height (training's ``command_base_height``; nominal 0.75 m).
        obs["command_base_height"] = self.base_height_command
        obs["command_waist_dofs"] = self.waist_dofs_command
        obs["ref_upper_dof_pos"] = self.ref_upper_dof_pos
        obs["cam_depth"] = self.depth_sensor.get_latest().reshape(1, -1)
        return obs

    def _update_obs_history(self, current_obs_dict):
        """Time-major group history with push intervals; zero frames before the first push."""
        outputs: dict[str, np.ndarray] = {}
        for group, term_dict in current_obs_dict.items():
            frame = np.concatenate(
                [np.asarray(term_dict[t], dtype=np.float32).reshape(1, -1) for t in self.obs_terms_sorted[group]],
                axis=1,
            )
            history = self.group_history[group]
            if self.obs_interval_counter % self.obs_intervals[group] == 0:
                history.append(frame)
            frames = list(history)
            missing = history.maxlen - len(frames)
            frames = [np.zeros_like(frame)] * missing + frames
            outputs[group] = np.concatenate(frames, axis=1)
        self.obs_interval_counter += 1
        self.obs_buf_dict = {group: value.copy() for group, value in outputs.items()}
        return outputs

    def prepare_obs_for_rl(self, robot_state_data):
        groups = self._prepare_group_observations(robot_state_data)
        lower = np.concatenate([groups["actor_obs"], groups["perception_obs"], groups["command_lin_vel_obs"]], axis=1)
        return {
            "actor_obs_lower_body": lower.astype(np.float32, copy=False),
            "actor_obs_upper_body": groups["actor_obs"].astype(np.float32, copy=False),
        }

    def policy_action(self):
        # Before the base class snapshots use_policy_action, so a stalled producer stops the
        # policy on this tick rather than after one more action on a frozen view.
        self._check_depth_liveness()
        if self.arm_ik_thread is not None:
            self._update_upper_body_reference(self.interface.get_low_state())
        if not self.use_policy_action or self.get_ready_state:
            # Not applied: run it anyway to keep the history warm (FALCON's ``rl_inference`` runs
            # every tick). Starting from zero-padded frames makes the first actions a jolt.
            self.rl_inference(self.interface.get_low_state())
        super().policy_action()

    def _handle_init_state(self):
        super()._handle_init_state()
        self.obs_interval_counter = 0  # FALCON's init (A+X) re-phases the depth pushes

    def _check_depth_liveness(self) -> None:
        """Warn on stale depth; past ``stale_stop_s`` (or a failed producer) go back to the init pose."""
        status = self.depth_sensor.liveness()
        if status is None or not status.checked:
            return  # heartbeat disabled, or no heartbeat received yet
        cfg = self.config.task.depth_shm
        failed = status.state == STATE_FAILED
        if failed or status.stale_s > cfg.stale_warn_s:
            if not self._depth_stale_warned:
                logger.warning(
                    f"[RplDepthPolicy] depth frame stale for {status.stale_s:.2f}s (producer: {status.state})"
                )
                self._depth_stale_warned = True
        elif self._depth_stale_warned:
            logger.info("[RplDepthPolicy] depth frames advancing again")
            self._depth_stale_warned = False
        if (failed or status.stale_s > cfg.stale_stop_s) and self.use_policy_action:
            logger.error(
                f"[RplDepthPolicy] depth producer {status.state}, frame {status.stale_s:.2f}s old: stopping policy"
            )
            self._handle_init_state()
