"""Loco-manipulation reward terms (RPL HM / decoupled style)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.managers.reward.terms import locomotion as _loco_reward_terms
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


# ---------------------------------------------------------------------------
# Shared helpers (indices, buffers)
# ---------------------------------------------------------------------------


def _upper_dof_indices(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    rc = env.robot_config
    return torch.tensor([rc.dof_names.index(n) for n in rc.upper_dof_names], device=env.device, dtype=torch.long)


def _lower_dof_indices(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    rc = env.robot_config
    return torch.tensor([rc.dof_names.index(n) for n in rc.lower_dof_names], device=env.device, dtype=torch.long)


def _joint_torques(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return env.action_manager.get_term("joint_control").torques


def _knee_rigid_indices(env: LeggedRobotLocomotionManager) -> tuple[int, int]:
    pat = env.robot_config.knee_name
    knee_bodies = [s for s in env.body_names if pat in s]
    if len(knee_bodies) < 2:
        raise RuntimeError(f"Expected at least two knee bodies matching '{pat}', found {knee_bodies!r}.")
    sim = env.simulator
    return int(sim.find_rigid_body_indice(knee_bodies[0])), int(sim.find_rigid_body_indice(knee_bodies[1]))


def _terrain_type_mask(env, terrain_type_mask) -> torch.Tensor | None:
    """Return a ``[num_envs]`` float mask of envs whose terrain type is in the given list.

    ``terrain_type_mask`` is a list/tuple of terrain-type names (e.g.
    ``["stairs_up", "stairs_down"]``).  Names not present in the current
    terrain are silently skipped.  Returns ``None`` when no filtering should
    be applied (empty list, missing terrain state, or terrain has no type grid).
    """
    if not terrain_type_mask:
        return None
    state = env.terrain_manager.get_state("locomotion_terrain")
    names = getattr(state, "terrain_type_names", None)
    if not names:
        return None
    idx_list = [names.index(n) for n in terrain_type_mask if n in names]
    if not idx_list:
        # All requested types absent → suppress everywhere.
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    allowed = torch.tensor(idx_list, device=env.device, dtype=torch.long)
    per_env = state.terrain_type_idx
    return torch.isin(per_env, allowed).float()


def _waist_yaw_dof_index(env: LeggedRobotLocomotionManager) -> int | None:
    name = getattr(env.robot_config, "waist_yaw_dof_name", None)
    if not name:
        return None
    return env.robot_config.dof_names.index(name)


def _ee_hand_indices(
    env: LeggedRobotLocomotionManager, hand_link_names: tuple[str, str] | None = None
) -> tuple[int, int]:
    sim = env.simulator
    names = env.body_names
    if hand_link_names is not None:
        if len(hand_link_names) != 2 or any(name not in names for name in hand_link_names):
            raise ValueError(f"Expected two end-effector bodies present in the robot: {hand_link_names}")
        return tuple(int(sim.find_rigid_body_indice(name)) for name in hand_link_names)
    li = None
    for link in ("left_hand_link", "left_wrist_yaw_link", "left_wrist_pitch_link"):
        if link in names:
            li = int(sim.find_rigid_body_indice(link))
            break
    if li is None:
        raise RuntimeError("penalty_ee_* could not resolve a left hand / wrist body on this robot.")
    ri = None
    for link in ("right_hand_link", "right_wrist_yaw_link", "right_wrist_pitch_link"):
        if link in names:
            ri = int(sim.find_rigid_body_indice(link))
            break
    if ri is None:
        raise RuntimeError("penalty_ee_* could not resolve a right hand / wrist body on this robot.")
    return li, ri


def _feet_contact_z(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0


def _lower_penalty_dof_indices(env: LeggedRobotLocomotionManager, robot_attr: str) -> torch.Tensor:
    """Resolve :attr:`RobotConfig.lower_body_*_penalty_dof_names` if set; otherwise all ``lower_dof_names``."""
    rc = env.robot_config
    names = getattr(rc, robot_attr, None)
    if not names:
        return _lower_dof_indices(env)
    return torch.tensor([rc.dof_names.index(n) for n in names], device=env.device, dtype=torch.long)


def _hips_roll_yaw_dof_indices(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Hip roll + yaw only: entries ``[1,2]`` and ``[4,5]`` in :attr:`RobotConfig.hips_dof_names` (RPL)."""
    rc = env.robot_config
    h = rc.hips_dof_names
    if len(h) < 6:
        raise RuntimeError(f"hips_dof_names must have at least 6 DOFs for hip roll/yaw penalty, got {len(h)}")
    pick = [h[1], h[2], h[4], h[5]]
    return torch.tensor([rc.dof_names.index(n) for n in pick], device=env.device, dtype=torch.long)


def _hips_roll_dof_indices(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Hip roll only: entries ``[1]`` (left) and ``[4]`` (right) in :attr:`RobotConfig.hips_dof_names`."""
    rc = env.robot_config
    h = rc.hips_dof_names
    if len(h) < 6:
        raise RuntimeError(f"hips_dof_names must have at least 6 DOFs for hip roll penalty, got {len(h)}")
    pick = [h[1], h[4]]
    return torch.tensor([rc.dof_names.index(n) for n in pick], device=env.device, dtype=torch.long)


def _penalty_dof_indices(env: LeggedRobotLocomotionManager, robot_attr: str, use_all_lower_dofs: bool) -> torch.Tensor:
    """Lower-body penalty DOFs; ``use_all_lower_dofs`` ignores the robot's per-penalty subset (FALCON g1)."""
    if use_all_lower_dofs:
        return _lower_dof_indices(env)
    return _lower_penalty_dof_indices(env, robot_attr)


def _falcon_last_dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """FAR-FALCON ``last_dof_vel``: the joint velocity of the previous *control* step.

    FALCON refreshes ``last_dof_vel`` once per control step, after the rewards
    (``_post_compute_observations_callback``), so its dof-acc penalties see
    ``(dof_vel[t-1] - dof_vel[t]) / dt``. Here the snapshot is advanced once per
    ``env.common_step_counter`` (shared by the lower- and upper-body terms) and
    returns the previous step's velocity.

    After a reset FALCON's ``last_dof_vel`` holds the post-reset velocity. The
    lower body is reset with zero velocity, so the first step of an episode uses
    zero; the arms' post-reset velocity (motion velocity when the upper body is
    reset onto the clip) is not recoverable here, so they use zero as well.
    """
    cur = env.simulator.dof_vel
    step = getattr(env, "common_step_counter", None)
    cache = getattr(env, "_falcon_dof_vel_cache", None)
    if cache is not None and step is not None and cache["step"] == step:
        return cache["prev"]
    snap = getattr(env, "_falcon_dof_vel_snapshot", None)
    prev = torch.zeros_like(cur) if snap is None or snap.shape != cur.shape else snap.clone()
    first = env.episode_length_buf == 1
    prev[first] = 0.0
    env._falcon_dof_vel_snapshot = cur.clone()
    env._falcon_dof_vel_cache = {"step": step, "prev": prev}
    return prev


def _feet_contact_filt(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """FAR-FALCON ``contact_filt``: current OR previous-step foot contact (``z > 1``).

    FALCON updates ``last_contacts`` once per step in ``update_contact_info``; the
    filter is cached per ``env.common_step_counter`` so every term sees the same value.
    """
    contact = _feet_contact_z(env)
    step = getattr(env, "common_step_counter", None)
    cache = getattr(env, "_falcon_contact_filt_cache", None)
    if cache is not None and step is not None and cache["step"] == step:
        return cache["filt"]
    last = getattr(env, "_falcon_last_contacts", None)
    if last is None or last.shape != contact.shape:
        last = torch.zeros_like(contact)
    filt = torch.logical_or(contact, last)
    env._falcon_last_contacts = contact.clone()
    env._falcon_contact_filt_cache = {"step": step, "filt": filt}
    return filt


def _feet_contact(env: LeggedRobotLocomotionManager, use_filtered_contact: bool) -> torch.Tensor:
    return _feet_contact_filt(env) if use_filtered_contact else _feet_contact_z(env)


# ---------------------------------------------------------------------------
# Tracking (velocity, heading, height, torso) — RPL decoupled
# ---------------------------------------------------------------------------


def tracking_lin_vel_x(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    cmd = env.command_manager.commands
    err = torch.square(cmd[:, 0] - get_base_lin_vel(env)[:, 0])
    return torch.exp(-err / tracking_sigma)


def tracking_lin_vel_y(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    cmd = env.command_manager.commands
    err = torch.square(cmd[:, 1] - get_base_lin_vel(env)[:, 1])
    return torch.exp(-err / tracking_sigma)


def tracking_heading(
    env,
    tracking_sigma: float = 0.25,
    terrain_type_mask: list[str] | None = None,
) -> torch.Tensor:
    """Heading tracking reward.

    If ``terrain_type_mask`` is provided (e.g. ``["stepping_stones_columns"]``),
    the squared heading error is multiplied by 0/1 mask so the reward collapses
    to ``exp(0)=1`` on other terrains (matching FALCON's ``tracking_heading_mask``
    behavior: only enforce heading on listed terrain types)."""
    cmd = env.command_manager.commands
    forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    err = torch.square(wrap_to_pi(cmd[:, 3] - heading))
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        err = err * mask
    return torch.exp(-err / tracking_sigma)


def tracking_stance_base_height(env, tracking_sigma: float = 0.05) -> torch.Tensor:
    cmd = env.command_manager.commands
    base_h = env.terrain_manager.get_state("locomotion_terrain").base_heights
    target_h = cmd[:, 8]
    if hasattr(env, "command_base_height_nominal"):
        target_h = env.command_base_height_nominal + target_h
    err = torch.abs(target_h - base_h)
    return torch.exp(-err / tracking_sigma) * (1.0 - cmd[:, 4])


def tracking_walk_base_height(
    env,
    tracking_sigma: float = 0.05,
    terrain_type_mask: list[str] | None = None,
    force_saturation: float = 50.0,
) -> torch.Tensor:
    """FAR-FALCON ``_reward_tracking_walk_base_height`` (x walk gate ``commands[:, 4]``).

    The height error is attenuated by the total hand force (``1 - clip(|F_l + F_r| / 50)``)
    and zeroed outside ``terrain_type_mask`` (FALCON: flat only), so on masked-out
    terrains the term reduces to the walk flag.
    """
    cmd = env.command_manager.commands
    base_h = env.terrain_manager.get_state("locomotion_terrain").base_heights
    target_h = cmd[:, 8]
    if hasattr(env, "command_base_height_nominal"):
        target_h = env.command_base_height_nominal + target_h
    err = torch.abs(target_h - base_h)
    force = getattr(env, "apply_force_tensor", None)
    if force is not None and hasattr(env, "left_hand_link_index"):
        total = torch.norm(force[:, env.left_hand_link_index, :] + force[:, env.right_hand_link_index, :], dim=1)
        err = err * (1.0 - torch.clip(total / force_saturation, 0.0, 1.0))
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        err = err * mask
    return torch.exp(-err / tracking_sigma) * cmd[:, 4]


def tracking_torso_orientation_tapping(env, tracking_sigma: float = 0.05) -> torch.Tensor:
    """RPL ``_reward_tracking_torso_orientation_tapping`` (x walk gate ``commands[:, 4]``)."""
    cmd = env.command_manager.commands
    g = gravity_vector(env)
    ref_yaw, ref_roll, ref_pitch = cmd[:, 5], cmd[:, 6], cmd[:, 7]
    ref_q = quat_from_euler_xyz(ref_roll, ref_pitch, ref_yaw)
    ref_pg = quat_rotate_inverse(ref_q, g, w_last=True)
    torso_q = env.simulator._rigid_body_rot[:, env.torso_index]
    pg = quat_rotate_inverse(torso_q, g, w_last=True)
    torso_err = torch.sum(torch.square(pg[:, :2] - ref_pg[:, :2]), dim=1)
    wy = _waist_yaw_dof_index(env)
    if wy is not None:
        torso_err = torso_err + torch.square(env.simulator.dof_pos[:, wy] - ref_yaw)
    return torch.exp(-torso_err / tracking_sigma) * cmd[:, 4]


def tracking_torso_orientation_stance(env, tracking_sigma: float = 0.05) -> torch.Tensor:
    """RPL ``_reward_tracking_torso_orientation_stance`` (x stance gate ``1 - commands[:, 4]``)."""
    cmd = env.command_manager.commands
    g = gravity_vector(env)
    ref_yaw, ref_roll, ref_pitch = cmd[:, 5], cmd[:, 6], cmd[:, 7]
    ref_q = quat_from_euler_xyz(ref_roll, ref_pitch, ref_yaw)
    ref_pg = quat_rotate_inverse(ref_q, g, w_last=True)
    torso_q = env.simulator._rigid_body_rot[:, env.torso_index]
    pg = quat_rotate_inverse(torso_q, g, w_last=True)
    torso_err = torch.sum(torch.square(pg[:, :2] - ref_pg[:, :2]), dim=1)
    wy = _waist_yaw_dof_index(env)
    if wy is not None:
        torso_err = torso_err + torch.square(env.simulator.dof_pos[:, wy] - ref_yaw)
    return torch.exp(-torso_err / tracking_sigma) * (1.0 - cmd[:, 4])


def terrain_progress(env, global_mean: bool = False) -> torch.Tensor:
    """RPL ``_reward_terrain_progress`` based on normalized terrain level.

    ``global_mean=True`` is FAR-FALCON's form: ``2 * mean(levels) / max_level`` for every env.
    """
    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    if terrain_state is None or not hasattr(terrain_state, "terrain_levels"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    max_level = max(int(getattr(terrain_state, "max_terrain_level", 0)), 1)
    levels = terrain_state.terrain_levels.float()
    if global_mean:
        return torch.mean(levels) / float(max_level) * 2.0 * torch.ones(env.num_envs, device=env.device)
    return levels / float(max_level)


def tracking_upper_body_dofs(env, tracking_sigma: float = 0.1) -> torch.Tensor:
    if not hasattr(env, "ref_upper_dof_pos") or not hasattr(env, "upper_dof_indices"):
        raise RuntimeError(
            "tracking_upper_body_dofs requires UpperBodyRefPoseCommand (ref_upper_dof_pos / upper_dof_indices)."
        )
    upper_pos = env.simulator.dof_pos[:, env.upper_dof_indices]
    err = torch.sum(torch.square(upper_pos - env.ref_upper_dof_pos), dim=1)
    return torch.exp(-err / tracking_sigma)


# ---------------------------------------------------------------------------
# Penalties & limits — lower body, feet, contact
# ---------------------------------------------------------------------------


def penalty_lin_vel_z(env, terrain_type_mask: list[str] | None = None) -> torch.Tensor:
    """Walk-gated base z-velocity penalty; zeroed outside ``terrain_type_mask`` if given (FALCON)."""
    vz = torch.square(get_base_lin_vel(env)[:, 2])
    penalty = vz * env.command_manager.commands[:, 4]
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * mask
    return penalty


def penalty_base_roll_orientation(env) -> torch.Tensor:
    return torch.abs(get_projected_gravity(env)[:, 1])


def penalty_base_pitch_orientation(env) -> torch.Tensor:
    pg = get_projected_gravity(env)
    cmd = env.command_manager.commands
    scale = 1.6 - torch.abs(cmd[:, 7])
    return torch.abs(pg[:, 0]) * scale


def penalty_lower_body_torques(env: LeggedRobotLocomotionManager, use_all_lower_dofs: bool = False) -> torch.Tensor:
    idx = _penalty_dof_indices(env, "lower_body_torque_penalty_dof_names", use_all_lower_dofs)
    t = _joint_torques(env)[:, idx]
    return torch.sum(torch.square(t), dim=1)


def penalty_lower_body_dof_vel(env: LeggedRobotLocomotionManager, use_all_lower_dofs: bool = False) -> torch.Tensor:
    idx = _penalty_dof_indices(env, "lower_body_vel_penalty_dof_names", use_all_lower_dofs)
    return torch.sum(torch.square(env.simulator.dof_vel[:, idx]), dim=1)


def penalty_lower_body_dof_acc(
    env: LeggedRobotLocomotionManager,
    use_all_lower_dofs: bool = False,
    falcon_control_step: bool = False,
) -> torch.Tensor:
    """Lower-body dof-acc penalty.

    ``falcon_control_step=True``: FAR-FALCON's ``((last_dof_vel - dof_vel) / dt)^2`` over one
    control step (see :func:`_falcon_last_dof_vel`); never reads ``simulator.dof_acc``.
    """
    idx = _penalty_dof_indices(env, "lower_body_acc_penalty_dof_names", use_all_lower_dofs)
    sim = env.simulator
    if falcon_control_step:
        a = (_falcon_last_dof_vel(env)[:, idx] - sim.dof_vel[:, idx]) / env.dt
        return torch.sum(torch.square(a), dim=1)
    acc = getattr(sim, "dof_acc", None)
    if acc is not None:
        return torch.sum(torch.square(acc[:, idx]), dim=1)
    prev = env.action_manager.get_term("joint_control").get_prev_dof_vel()[:, idx]
    a = (sim.dof_vel[:, idx] - prev) / env.dt
    return torch.sum(torch.square(a), dim=1)


def limits_lower_body_dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _lower_dof_indices(env)
    lo = env.dof_pos_limits[idx, 0]
    hi = env.dof_pos_limits[idx, 1]
    q = env.simulator.dof_pos[:, idx]
    below = -(q - lo).clip(max=0.0)
    above = (q - hi).clip(min=0.0)
    return torch.sum(below + above, dim=1)


def limits_lower_body_dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Lower-body mirror of :func:`limits_upper_body_dof_vel`.

    Penalizes per-joint velocity overshoot above the DOF velocity limit, clipped
    to a max of 1 rad/s per joint so a single large velocity can't dominate the
    reward. Mirrors FAR-FALCON's ``_reward_limits_lower_body_dof_vel``.
    """
    idx = _lower_dof_indices(env)
    lim = env.dof_vel_limits[idx]
    err = (torch.abs(env.simulator.dof_vel[:, idx]) - lim).clip(min=0.0, max=1.0)
    return torch.sum(err, dim=1)


def limits_lower_body_torque(env: LeggedRobotLocomotionManager, torque_softness: float = 1.0) -> torch.Tensor:
    """Lower-body mirror of :func:`limits_upper_body_torque`.

    Penalizes torques above ``torque_softness * torque_limit``. FAR-FALCON's
    ``_reward_limits_lower_body_torque`` uses softness 1.0 (hard limit), while
    the upper-body variant uses 0.9; keep the same defaults here.
    """
    idx = _lower_dof_indices(env)
    t = torch.abs(_joint_torques(env)[:, idx])
    lim = env.torque_limits[idx] * torque_softness
    return torch.sum((t - lim).clip(min=0.0), dim=1)


def penalty_contact_no_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    contact = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :3], dim=2) > 1.0
    fv = env.simulator._rigid_body_vel[:, env.feet_indices]
    penalize = torch.sum(torch.square(fv * contact.unsqueeze(-1)), dim=(1, 2))
    cmd = env.command_manager.commands
    stance = torch.where(cmd[:, 4] == 0)[0]
    if stance.numel() > 0:
        penalize[stance] *= 10.0
    return penalize


def penalty_ankle_roll(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    rc = env.robot_config
    idxs = [
        rc.dof_names.index(names[1]) for names in (rc.left_ankle_dof_names, rc.right_ankle_dof_names) if len(names) > 1
    ]
    if not idxs:
        return torch.zeros(env.num_envs, device=env.device)
    idx = torch.tensor(idxs, device=env.device, dtype=torch.long)
    q = env.simulator.dof_pos[:, idx]
    return torch.sum(torch.abs(q), dim=1)


def feet_air_time(env, use_filtered_contact: bool = False) -> torch.Tensor:
    """Air-time reward on first contact; ``use_filtered_contact`` uses FALCON's ``contact_filt``."""
    cf = _feet_contact(env, use_filtered_contact)
    if not hasattr(env, "_loco_manip_feet_air_time"):
        env._loco_manip_feet_air_time = torch.zeros(env.num_envs, 2, device=env.device)
        env._loco_manip_last_feet_air_time = torch.zeros(env.num_envs, 2, device=env.device)
    fat = env._loco_manip_feet_air_time
    reset = env.episode_length_buf == 1
    if reset.any():
        fat[reset] = 0.0
    first_contact = (fat > 0.0) & cf
    fat += env.dt
    env._loco_manip_last_feet_air_time[first_contact] = fat[first_contact]
    rew = torch.sum((fat - 0.5) * first_contact.float(), dim=1)
    rew *= (torch.norm(env.command_manager.commands[:, :2], dim=1) > 0.1).float()
    fat *= ~cf
    return rew


def feet_heading_alignment(env) -> torch.Tensor:
    fv = base_forward_vector(env)
    left_q = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    right_q = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    fl = quat_apply(left_q, fv, w_last=True)
    fr = quat_apply(right_q, fv, w_last=True)
    hl = torch.atan2(fl[:, 1], fl[:, 0])
    hr = torch.atan2(fr[:, 1], fr[:, 0])
    root = quat_apply(env.base_quat, fv, w_last=True)
    h0 = torch.atan2(root[:, 1], root[:, 0])
    return torch.abs(wrap_to_pi(hl - h0)) + torch.abs(wrap_to_pi(hr - h0))


def penalty_close_knees_xy(env, close_knees_threshold: float = 0.16) -> torch.Tensor:
    lk, rk = _knee_rigid_indices(env)
    lxy = env.simulator._rigid_body_pos[:, lk, :2]
    rxy = env.simulator._rigid_body_pos[:, rk, :2]
    d = torch.norm(lxy - rxy, dim=1)
    cmd = env.command_manager.commands
    return (d < close_knees_threshold).float() * (1.0 - 0.8 * cmd[:, 1])


def penalty_ang_vel_xy_torso(env) -> torch.Tensor:
    torso_q = env.simulator._rigid_body_rot[:, env.torso_index]
    w = env.simulator._rigid_body_ang_vel[:, env.torso_index]
    w_b = quat_rotate_inverse(torso_q, w, w_last=True)
    return torch.sum(torch.square(w_b[:, :2]), dim=1)


def penalty_contact(env) -> torch.Tensor:
    """Stance: penalize if fewer than two feet down; walk: penalize unless exactly one foot is down."""
    cmd = env.command_manager.commands
    contact = _feet_contact_z(env)
    n = contact.sum(dim=1)
    res = torch.zeros(env.num_envs, device=env.device)
    is_stance = cmd[:, 4] == 0
    res[is_stance & (n < 2)] = 1.0
    is_walk = ~is_stance
    res[is_walk & (n != 1)] = 1.0
    return res


def stand_still(env, offset: float = 1.0, command_eps: float = 1e-4) -> torch.Tensor:
    """Penalize lower-body joint deviation from the default pose in *pure* stance.

    Active only when the env is commanded to a plain static stand — i.e. all of:

    * stance mode (``commands[:, 4] == 0``, so velocity is zero), AND
    * no commanded squat (base-height delta ``commands[:, 8] ~= 0``), AND
    * no commanded torso reorientation (waist yaw/roll/pitch ``commands[:, 5:8] ~= 0``).

    Under those conditions there is a single well-defined target — the default
    pose — so anchoring the legs to it suppresses tapping / weight-shifting in
    place. When a squat or torso lean *is* commanded, the gate switches the term
    off so it never fights ``tracking_stance_base_height`` /
    ``tracking_torso_orientation_stance`` (which want the legs/torso to move away
    from default to hit those targets).

    Only the lower-body DOFs are regulated; the upper body is left to track its
    ACCAD reference trajectory via ``tracking_upper_body_dofs``.

    The reward is ``(sum_i |q_i - q_default_i| - offset)`` over the lower-body
    DOFs. With a negative weight this is a bonus while the legs stay within
    ``offset`` (summed abs rad) of default and a penalty once they drift further.

    Args:
        offset: Tolerance subtracted from the summed absolute deviation.
        command_eps: Magnitude below which a base-height / waist command counts
            as zero (commands are curriculum-scaled, so they may not be exactly 0).
    """
    idx = _lower_dof_indices(env)
    q = env.simulator.dof_pos[:, idx]
    q_default = env.default_dof_pos[:, idx]
    dof_error = torch.sum(torch.abs(q - q_default), dim=1)

    cmd = env.command_manager.commands
    is_stance = cmd[:, 4] == 0
    no_height_cmd = torch.abs(cmd[:, 8]) <= command_eps
    no_torso_cmd = torch.all(torch.abs(cmd[:, 5:8]) <= command_eps, dim=1)
    gate = (is_stance & no_height_cmd & no_torso_cmd).float()

    return (dof_error - offset) * gate


def penalty_negative_knee_joint(
    env: LeggedRobotLocomotionManager, knee_joint_min_threshold: float | None = None
) -> torch.Tensor:
    """Count knees below the threshold; ``knee_joint_min_threshold`` overrides the robot config value."""
    rc = env.robot_config
    idx = torch.tensor([rc.dof_names.index(n) for n in rc.knee_dof_names], device=env.device, dtype=torch.long)
    thr = rc.knee_joint_min_threshold if knee_joint_min_threshold is None else knee_joint_min_threshold
    q = env.simulator.dof_pos[:, idx]
    return torch.sum((q < thr).float(), dim=1)


def penalty_diff_feet_air_time(env, terrain_type_mask: list[str] | None = None) -> torch.Tensor:
    if not hasattr(env, "_loco_manip_last_feet_air_time"):
        return torch.zeros(env.num_envs, device=env.device)
    la = env._loco_manip_last_feet_air_time
    err = torch.abs(la[:, 0] - la[:, 1])
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        err = err * mask
    return err


def penalty_feet_edge(
    env,
    terrain_type_mask: list[str] | None = None,
    min_terrain_level: int = 2,
    use_filtered_contact: bool = False,
) -> torch.Tensor:
    """Penalize feet placed on terrain edges (steep slope transitions or void areas).

    Looks up the precomputed edge masks from the terrain manager and checks
    whether each foot in contact sits on an edge cell. Mirrors FAR-FALCON's
    4-point footprint sampling (front/back/left/right offsets in the body
    frame, rotated by base yaw) — a foot fires the penalty if *any* of its
    four sample points lies on an edge cell. Gated by:

    * velocity command ``||cmd[:, :2]|| > 0.1`` — only while locomoting.
    * ``terrain_levels > min_terrain_level`` — mirrors FAR-FALCON's
      ``(self.terrain_levels > 2)`` gate so the penalty doesn't fire on the
      easiest curriculum tiles where the robot hasn't yet learned to walk.
      Set ``min_terrain_level=-1`` (or any negative value) to disable the gate.

    ``use_filtered_contact`` uses FALCON's ``contact_filt`` (current OR previous contact).
    """
    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    if not terrain_state.has_edge_masks:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    terrain = terrain_state.terrain
    x_edge = terrain_state.x_edge_mask
    y_edge = terrain_state.y_edge_mask

    rc = env.robot_config
    num_feet = env.feet_indices.numel()
    # [4, 3] body-frame offsets: front, back, left, right (mirrors FALCON's
    # ``feet_offset_local[:, :, [0,1,2,3], :]`` ordering).
    offsets_local = torch.tensor(
        [
            list(rc.feet_front_offset_in_body),
            list(rc.feet_back_offset_in_body),
            list(rc.feet_left_offset_in_body),
            list(rc.feet_right_offset_in_body),
        ],
        device=env.device,
        dtype=torch.float32,
    )
    # Broadcast to [E, num_feet, 4, 3] then rotate by base quat (yaw) to world frame.
    offsets_local = offsets_local.view(1, 1, 4, 3).expand(env.num_envs, num_feet, 4, 3)
    base_quat = env.base_quat.view(env.num_envs, 1, 1, 4).expand(env.num_envs, num_feet, 4, 4)
    offsets_world = quat_apply(base_quat, offsets_local, w_last=True)  # [E, num_feet, 4, 3]

    feet_xy = env.simulator._rigid_body_pos[:, env.feet_indices, :2]  # [E, num_feet, 2]
    feet_xy_4 = feet_xy.unsqueeze(2).expand(-1, -1, 4, -1) + offsets_world[..., :2]
    # Flatten foot dim and sample dim: [E, num_feet * 4, 2]
    feet_xy_flat = feet_xy_4.reshape(env.num_envs, num_feet * 4, 2)

    # Convert world XY → grid indices (account for border offset)
    grid_pos = ((feet_xy_flat + terrain.border_size) / terrain.horizontal_scale).round().long()
    grid_pos[..., 0] = grid_pos[..., 0].clamp(0, x_edge.shape[0] - 1)
    grid_pos[..., 1] = grid_pos[..., 1].clamp(0, x_edge.shape[1] - 1)

    row = grid_pos[..., 0]  # [E, num_feet * 4]
    col = grid_pos[..., 1]  # [E, num_feet * 4]

    at_edge_x = x_edge[row, col].view(env.num_envs, num_feet, 4)
    at_edge_y = y_edge[row, col].view(env.num_envs, num_feet, 4)
    # FALCON: ``torch.any(...)`` over the 4 sample points per foot.
    foot_on_edge_x = at_edge_x.any(dim=-1)  # [E, num_feet]
    foot_on_edge_y = at_edge_y.any(dim=-1)  # [E, num_feet]

    contact = _feet_contact(env, use_filtered_contact)  # [E, num_feet]
    feet_at_edge = contact & (foot_on_edge_x | foot_on_edge_y)

    commands = env.command_manager.commands
    has_vel_cmd = torch.norm(commands[:, :2], dim=1) > 0.1

    penalty = torch.sum(feet_at_edge.float(), dim=-1) * has_vel_cmd.float()

    # Curriculum gate: only enable once the env has progressed beyond the
    # initial easy levels (matches FAR-FALCON's ``terrain_levels > 2``).
    terrain_levels = getattr(terrain_state, "terrain_levels", None)
    if terrain_levels is not None and min_terrain_level >= 0:
        level_gate = (terrain_levels > min_terrain_level).float()
        penalty = penalty * level_gate

    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * mask
    return penalty


def penalty_foothold(
    env: LeggedRobotLocomotionManager,
    foothold_epsilon: float = 0.05,
    terrain_type_mask: list[str] | None = None,
    falcon_heightfield: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Sole-sample foothold penalty with optional terrain-type mask.

    Delegates to :func:`holosoma.managers.reward.terms.locomotion.penalty_foothold`
    (terrain height by mesh ray-cast). ``falcon_heightfield`` instead reads the
    raw heightfield cell under each sample like FAR-FALCON's ``_get_terrain_heights``
    (truncated cell index, no interpolation); requires ``height_scan_mode="falcon"``.
    The term must stay named ``penalty_foothold``: the env builds the sole sample
    grid from that term's params (``foothold_num_samples_long`` / ``_wide``)."""
    if not falcon_heightfield:
        penalty = _loco_reward_terms.penalty_foothold(env, foothold_epsilon=foothold_epsilon)
    else:
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        if getattr(terrain_state, "_height_samples", None) is None:
            raise RuntimeError("penalty_foothold(falcon_heightfield=True) requires height_scan_mode='falcon'.")
        contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E, 2]
        penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        for foot in range(2):
            rb_idx = env.feet_indices[foot]
            foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]
            foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]
            pts_local = env.foot_samples_local[foot].unsqueeze(0).expand(env.num_envs, -1, -1)
            pts_world = quat_apply(
                foot_quat_w.unsqueeze(1).expand(-1, pts_local.shape[1], -1), pts_local, w_last=True
            ) + foot_pos_w.unsqueeze(1)
            terrain_h = terrain_state._falcon_heightfield_heights(pts_world, None)
            bad = (pts_world[:, :, 2] - terrain_h > foothold_epsilon).float()
            penalty += torch.sum(bad * contact[:, foot].unsqueeze(1).float(), dim=1)
        penalty = penalty / env.num_foot_samples
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * mask
    return penalty


def penalty_lower_body_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _lower_dof_indices(env)
    a = env.action_manager.action[:, idx]
    p = env.action_manager.prev_action[:, idx]
    return torch.sum(torch.square(p - a), dim=1)


def penalty_upper_body_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    a = env.action_manager.action[:, idx]
    p = env.action_manager.prev_action[:, idx]
    return torch.sum(torch.square(p - a), dim=1)


def base_height(
    env,
    desired_base_height: float = 0.89,
    reduce_terrain_type_mask: list[str] | None = None,
    reduction_factor: float = 0.75,
    max_delta_height_range: float = 0.6,
) -> torch.Tensor:
    """Penalize base height away from target.

    For 9-D loco-manip commands with ``env.command_base_height_nominal`` set, the per-env target is
    ``nominal + commands[:, 8]`` (delta command); otherwise ``desired_base_height`` is used.

    ``reduce_terrain_type_mask`` (FAR-FALCON ``reduce_base_height_mask``, stairs): on the listed
    terrains the penalty is scaled by ``1 - reduction_factor * slope_factor`` with
    ``slope_factor = clip(max_i |h_i - base_height| / max_delta_height_range, 0, 1)``, where
    ``h_i`` are the scan-grid terrain heights (FALCON ``measured_heights``, world z) and
    ``base_height`` the base height above terrain — the same (dimensionally mixed) expression
    FALCON evaluates in ``_pre_compute_observations_callback``.

    Args:
        env: The environment instance
        desired_base_height: Target base height when not using delta height commands

    Returns:
        Reward tensor [num_envs]
    """
    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    base_heights = terrain_state.base_heights
    commands = env.command_manager.commands
    if commands.shape[1] > 8 and hasattr(env, "command_base_height_nominal"):
        desired_h = env.command_base_height_nominal + commands[:, 8]
        penalty = torch.square(base_heights - desired_h)
    else:
        penalty = torch.square(base_heights - desired_base_height)
    mask = _terrain_type_mask(env, reduce_terrain_type_mask)
    if mask is not None:
        heights = terrain_state.height_measurements()
        height_range = torch.max(torch.abs(heights - base_heights.unsqueeze(1)), dim=1)[0]
        slope_factor = torch.clamp(height_range / max_delta_height_range, 0.0, 1.0)
        penalty = penalty * (1.0 - mask * slope_factor * reduction_factor)
    return penalty


def penalty_hip_pos(
    env: LeggedRobotLocomotionManager,
    terrain_type_mask: list[str] | None = None,
    reduction_scale: float = 0.25,
) -> torch.Tensor:
    """RPL decoupled ``_reward_penalty_hip_pos`` (hip roll/yaw only, gated by walk / height cmd).

    With ``terrain_type_mask`` (FAR-FALCON ``penalty_hip_pos_mask``) the penalty is kept in full on
    the listed terrains and scaled by ``reduction_scale`` elsewhere (stepping stones / gap).
    """
    idx = _hips_roll_yaw_dof_indices(env)
    hip_pos = env.simulator.dof_pos[:, idx]
    penalty = torch.sum(torch.square(hip_pos), dim=1)
    cmd = env.command_manager.commands
    desired_h = cmd[:, 8]
    if hasattr(env, "command_base_height_nominal"):
        desired_h = env.command_base_height_nominal + desired_h
    desired_h = torch.clamp(desired_h, min=0.0)
    penalty = penalty * (cmd[:, 4] + (1.0 - cmd[:, 4]) * desired_h)
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * (mask + (1.0 - mask) * reduction_scale)
    return penalty


def penalty_hip_roll(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize hip-roll joint deviation from zero (sum of squares, both legs).

    Discourages the legs splaying outward / crossing inward at the hips, keeping
    the feet under the body for a more natural, stable gait and stance.
    """
    idx = _hips_roll_dof_indices(env)
    hip_roll = env.simulator.dof_pos[:, idx]
    return torch.sum(torch.square(hip_roll), dim=1)


def penalty_stumble(env, terrain_type_mask: list[str] | None = None, stumble_ratio: float = 0.5) -> torch.Tensor:
    """Stumble penalty (feet hitting vertical surfaces) with optional terrain-type mask.

    Delegates to :func:`holosoma.managers.reward.terms.locomotion.penalty_stumble`
    then zeroes the result on terrain types not in ``terrain_type_mask``."""
    penalty = _loco_reward_terms.penalty_stumble(env, stumble_ratio=stumble_ratio).float()
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * mask
    return penalty


def penalty_feet_ori(
    env: LeggedRobotLocomotionManager,
    terrain_type_mask: list[str] | None = None,
    feet_ori_multiply_scale: float = 1.0,
    use_filtered_contact: bool = False,
) -> torch.Tensor:
    """Feet-orientation penalty with optional terrain-type mask.

    Mirrors FAR-FALCON's masked behaviour: on listed terrain types the penalty
    sums both feet unconditionally; on other terrains the penalty is only
    applied to feet in contact and scaled by ``feet_ori_multiply_scale``
    (default 1.0 — no scaling when not configured). ``use_filtered_contact``
    uses FALCON's ``contact_filt`` (current OR previous contact)."""
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    left_penalty = torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
    right_penalty = torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5

    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is None:
        return left_penalty + right_penalty

    contact = _feet_contact(env, use_filtered_contact)
    left_contact = contact[:, 0].float()
    right_contact = contact[:, 1].float()
    return (
        mask * (left_penalty + right_penalty)
        + (1.0 - mask) * (left_penalty * left_contact + right_penalty * right_contact) * feet_ori_multiply_scale
    )


def penalty_feet_contact_forces(
    env: LeggedRobotLocomotionManager,
    locomotion_max_contact_force: float = 350.0,
    terrain_type_mask: list[str] | None = None,
    add_ee_force_z: bool = False,
) -> torch.Tensor:
    """Penalize foot contact force norm above a cap (RPL ``_reward_penalty_feet_contact_forces``).

    ``add_ee_force_z``: FAR-FALCON terrain variant, which adds the base-frame z of the
    applied hand forces (``left/right_ee_apply_force``) to each foot's excess before the
    clip, so a downward hand load raises the allowed foot force.
    """
    excess = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1) - locomotion_max_contact_force
    if add_ee_force_z and hasattr(env, "left_ee_apply_force"):
        excess = excess + (env.left_ee_apply_force[:, 2] + env.right_ee_apply_force[:, 2]).unsqueeze(1)
    excess = excess.clip(min=0.0)
    penalty = torch.sum(excess, dim=1)
    mask = _terrain_type_mask(env, terrain_type_mask)
    if mask is not None:
        penalty = penalty * mask
    return penalty


def penalty_upper_body_torques(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    return torch.sum(torch.square(_joint_torques(env)[:, idx]), dim=1)


def penalty_upper_body_dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    return torch.sum(torch.square(env.simulator.dof_vel[:, idx]), dim=1)


def penalty_upper_body_dof_acc(env: LeggedRobotLocomotionManager, falcon_control_step: bool = False) -> torch.Tensor:
    """Upper-body dof-acc penalty; ``falcon_control_step`` as in :func:`penalty_lower_body_dof_acc`."""
    idx = _upper_dof_indices(env)
    sim = env.simulator
    if falcon_control_step:
        a = (_falcon_last_dof_vel(env)[:, idx] - sim.dof_vel[:, idx]) / env.dt
        return torch.sum(torch.square(a), dim=1)
    acc = getattr(sim, "dof_acc", None)
    if acc is not None:
        return torch.sum(torch.square(acc[:, idx]), dim=1)
    prev = env.action_manager.get_term("joint_control").get_prev_dof_vel()[:, idx]
    a = (sim.dof_vel[:, idx] - prev) / env.dt
    return torch.sum(torch.square(a), dim=1)


def limits_upper_body_dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    lo = env.dof_pos_limits[idx, 0]
    hi = env.dof_pos_limits[idx, 1]
    q = env.simulator.dof_pos[:, idx]
    below = -(q - lo).clip(max=0.0)
    above = (q - hi).clip(min=0.0)
    return torch.sum(below + above, dim=1)


def limits_upper_body_dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    lim = env.dof_vel_limits[idx]
    err = (torch.abs(env.simulator.dof_vel[:, idx]) - lim).clip(min=0.0, max=1.0)
    return torch.sum(err, dim=1)


def limits_upper_body_torque(env: LeggedRobotLocomotionManager, torque_softness: float = 0.9) -> torch.Tensor:
    idx = _upper_dof_indices(env)
    t = torch.abs(_joint_torques(env)[:, idx])
    lim = env.torque_limits[idx] * torque_softness
    return torch.sum((t - lim).clip(min=0.0), dim=1)


def penalty_ee_lin_acc(
    env: LeggedRobotLocomotionManager, hand_link_names: tuple[str, str] | None = None
) -> torch.Tensor:
    li, ri = _ee_hand_indices(env, hand_link_names)
    v_l = env.simulator._rigid_body_vel[:, li, :3]
    v_r = env.simulator._rigid_body_vel[:, ri, :3]
    pl = getattr(env, "_loco_manip_prev_left_ee_lin_vel", None)
    pr = getattr(env, "_loco_manip_prev_right_ee_lin_vel", None)
    if pl is None or pr is None or pl.shape != v_l.shape:
        env._loco_manip_prev_left_ee_lin_vel = v_l.clone()
        env._loco_manip_prev_right_ee_lin_vel = v_r.clone()
        return torch.zeros(env.num_envs, device=env.device, dtype=v_l.dtype)
    first = (env.episode_length_buf == 1).unsqueeze(1)
    d_l = torch.where(first.expand_as(v_l), torch.zeros_like(v_l), v_l - pl)
    d_r = torch.where(first.expand_as(v_r), torch.zeros_like(v_r), v_r - pr)
    pl.copy_(v_l)
    pr.copy_(v_r)
    return torch.norm(d_l, dim=1) + torch.norm(d_r, dim=1)


def penalty_ee_ang_acc(
    env: LeggedRobotLocomotionManager, hand_link_names: tuple[str, str] | None = None
) -> torch.Tensor:
    li, ri = _ee_hand_indices(env, hand_link_names)
    w_l = env.simulator._rigid_body_ang_vel[:, li, :3]
    w_r = env.simulator._rigid_body_ang_vel[:, ri, :3]
    pl = getattr(env, "_loco_manip_prev_left_ee_ang_vel", None)
    pr = getattr(env, "_loco_manip_prev_right_ee_ang_vel", None)
    if pl is None or pr is None or pl.shape != w_l.shape:
        env._loco_manip_prev_left_ee_ang_vel = w_l.clone()
        env._loco_manip_prev_right_ee_ang_vel = w_r.clone()
        return torch.zeros(env.num_envs, device=env.device, dtype=w_l.dtype)
    first = (env.episode_length_buf == 1).unsqueeze(1)
    d_l = torch.where(first.expand_as(w_l), torch.zeros_like(w_l), w_l - pl)
    d_r = torch.where(first.expand_as(w_r), torch.zeros_like(w_r), w_r - pr)
    pl.copy_(w_l)
    pr.copy_(w_r)
    return torch.norm(d_l, dim=1) + torch.norm(d_r, dim=1)


__all__ = [
    "feet_air_time",
    "feet_heading_alignment",
    "limits_lower_body_dof_pos",
    "limits_upper_body_dof_pos",
    "limits_upper_body_dof_vel",
    "limits_upper_body_torque",
    "penalty_ang_vel_xy_torso",
    "penalty_ankle_roll",
    "penalty_base_pitch_orientation",
    "penalty_base_roll_orientation",
    "penalty_close_knees_xy",
    "penalty_contact",
    "penalty_contact_no_vel",
    "penalty_diff_feet_air_time",
    "penalty_ee_ang_acc",
    "penalty_ee_lin_acc",
    "penalty_feet_contact_forces",
    "penalty_feet_edge",
    "penalty_hip_pos",
    "penalty_hip_roll",
    "penalty_lin_vel_z",
    "penalty_lower_body_action_rate",
    "penalty_lower_body_dof_acc",
    "penalty_lower_body_dof_vel",
    "penalty_lower_body_torques",
    "penalty_negative_knee_joint",
    "penalty_upper_body_action_rate",
    "penalty_upper_body_dof_acc",
    "penalty_upper_body_dof_vel",
    "penalty_upper_body_torques",
    "stand_still",
    "terrain_progress",
    "tracking_heading",
    "tracking_lin_vel_x",
    "tracking_lin_vel_y",
    "tracking_stance_base_height",
    "tracking_torso_orientation_stance",
    "tracking_torso_orientation_tapping",
    "tracking_upper_body_dofs",
    "tracking_walk_base_height",
]
