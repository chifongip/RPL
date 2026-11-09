"""Verify the X2 motion library against its provenance manifest.

Requires NumPy. Checks every file and stored trajectory, not just sampled clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rpl_training.x2 import DOF_NAMES, MODEL, POSITION_LIMITS  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_library(manifest_path, source=None, target=None):
    manifest = json.loads(Path(manifest_path).read_text())
    source = Path(source) if source is not None else ROOT / manifest["source_dataset"]
    target = Path(target) if target is not None else ROOT / manifest["target_dataset"]
    records = manifest["files"]
    require(bool(records), "Manifest contains no clips")
    names = sorted(r["clip"] for r in records)
    require(len(set(names)) == len(names) == manifest["clips"], "Duplicate/missing manifest clip entries")
    require(names == sorted(p.name for p in source.glob("*.npz")), "Source clip set differs from manifest")
    require(names == sorted(p.name for p in target.glob("*.npz")), "Target clip set differs from manifest")
    limits = np.asarray(POSITION_LIMITS)
    speeds = np.asarray([joint["velocity"] for joint in MODEL["dofs"]])
    max_violation = max_speed_ratio = max_velocity_error = 0.0
    total_frames = 0
    for record in records:
        name = record["clip"]
        require(Path(name).name == name and name.endswith(".npz"), f"Invalid clip filename: {name}")
        require(digest(source / name) == record["source_sha256"], f"Source hash mismatch: {name}")
        require(digest(target / name) == record["output_sha256"], f"Output hash mismatch: {name}")
        with np.load(source / name, allow_pickle=False) as original:
            require(len(original["joint_pos"]) == record["frames"], f"Source frame count mismatch: {name}")
            source_fps = float(original["fps"].item())
            require(np.isfinite(source_fps) and source_fps > 0, f"Invalid source FPS: {name}")
        with np.load(target / name, allow_pickle=False) as data:
            n = record["frames"]
            total_frames += n
            fps = float(data["fps"].item())
            require(n >= 2 and np.isfinite(fps) and fps > 0, f"Invalid frame count/FPS: {name}")
            require(fps == source_fps, f"Source/output FPS mismatch: {name}")
            require(record["native_fps"] == source_fps, f"Manifest FPS mismatch: {name}")
            require(abs(record["duration_s"] - (n - 1) / source_fps) < 1e-5, f"Duration mismatch: {name}")
            require(list(data["dof_names"]) == DOF_NAMES, f"Joint order mismatch: {name}")
            require(list(data["body_names"]) == MODEL["expected_collapsed_body_names"], f"Body order mismatch: {name}")
            expected = dict(
                joint_pos=(n, 36),
                joint_vel=(n, 35),
                body_pos_w=(n, 30, 3),
                body_quat_w=(n, 30, 4),
                body_lin_vel_w=(n, 30, 3),
                body_ang_vel_w=(n, 30, 3),
            )
            for key, shape in expected.items():
                require(data[key].shape == shape and np.isfinite(data[key]).all(), f"Invalid {key}: {name}")
            for quaternions in (data["joint_pos"][:, 3:7], data["body_quat_w"]):
                require(
                    np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=2e-6, rtol=0),
                    f"Nonunit quaternion: {name}",
                )
            joints = data["joint_pos"][:, 7:].astype(float)
            violation = float(np.maximum(np.maximum(limits[:, 0] - joints, joints - limits[:, 1]), 0).max())
            require(violation < 2e-6, f"Joint limit violation: {name}")
            velocity = np.diff(joints, axis=0) * fps
            ratio = float((np.abs(velocity) / speeds).max())
            require(ratio <= manifest["joint_velocity_limit_margin"] + 2e-5, f"Joint speed violation: {name}")
            error = float(np.abs(data["joint_vel"][:-1, 6:] - velocity).max())
            require(error < 2e-4, f"Position/velocity inconsistency: {name}")
            max_violation = max(max_violation, violation)
            max_speed_ratio = max(max_speed_ratio, ratio)
            max_velocity_error = max(max_velocity_error, error)
    require(total_frames == manifest["frames"], "Total frame count differs from manifest")
    return dict(
        clips=len(records),
        frames=total_frames,
        source_and_output_hashes_verified=True,
        all_finite=True,
        source_timing_preserved=True,
        max_joint_limit_violation_rad=max_violation,
        max_joint_speed_ratio=max_speed_ratio,
        max_velocity_roundoff_error_rad_s=max_velocity_error,
    )


def validate_kinematics(target):
    """Recompute every frame with the canonical model and independent SO(3) differences."""
    import mujoco
    from scipy.spatial.transform import Rotation

    from rpl_training.x2 import ASSET_DIR

    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
    require([model.joint(i).name for i in range(1, model.njnt)] == DOF_NAMES, "Canonical model joint order mismatch")
    body_ids = [model.body(name).id for name in MODEL["expected_collapsed_body_names"]]
    data = mujoco.MjData(model)
    maxima = dict(
        body_position_m=0.0,
        body_rotation_rad=0.0,
        joint_velocity=0.0,
        body_linear_velocity_m_s=0.0,
        body_angular_velocity_rad_s=0.0,
    )
    frames = 0
    for path in sorted(Path(target).glob("*.npz")):
        with np.load(path, allow_pickle=False) as motion:
            q = motion["joint_pos"].astype(float)
            fps = float(motion["fps"].item())
            positions, quaternions = [], []
            for frame in q:
                data.qpos[:] = frame
                data.qpos[3:7] = frame[[6, 3, 4, 5]]
                data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
                mujoco.mj_forward(model, data)
                positions.append(data.xpos[body_ids].copy())
                quaternions.append(data.xquat[body_ids].copy())
            positions, quaternions = np.asarray(positions), np.asarray(quaternions)
            n, bodies = positions.shape[:2]
            rotations = Rotation.from_quat(quaternions[..., [1, 2, 3, 0]].reshape(-1, 4))
            stored_rotations = Rotation.from_quat(motion["body_quat_w"][..., [1, 2, 3, 0]].reshape(-1, 4))
            root = Rotation.from_quat(q[:, 3:7])
            local_root_velocity = (root[:-1].inv() * root[1:]).as_rotvec() * fps
            velocity = np.concatenate(
                (np.diff(q[:, :3], axis=0) * fps, local_root_velocity, np.diff(q[:, 7:], axis=0) * fps), axis=1
            )
            # Body angular velocities are world-frame rotations, unlike root tangent velocities.
            quat_xyzw = quaternions[..., [1, 2, 3, 0]]
            previous = Rotation.from_quat(quat_xyzw[:-1].reshape(-1, 4))
            following = Rotation.from_quat(quat_xyzw[1:].reshape(-1, 4))
            angular = (following * previous.inv()).as_rotvec().reshape(n - 1, bodies, 3) * fps
            errors = dict(
                body_position_m=float(np.abs(positions - motion["body_pos_w"]).max()),
                body_rotation_rad=float((stored_rotations.inv() * rotations).magnitude().max()),
                joint_velocity=float(np.abs(motion["joint_vel"][:-1] - velocity).max()),
                body_linear_velocity_m_s=float(
                    np.abs(motion["body_lin_vel_w"] - np.gradient(positions, 1 / fps, axis=0)).max()
                ),
                body_angular_velocity_rad_s=float(np.abs(motion["body_ang_vel_w"][:-1] - angular).max()),
            )
            for key, error in errors.items():
                tolerance = 2e-6 if key in ("body_position_m", "body_rotation_rad") else 2e-4
                require(error < tolerance, f"FK/velocity mismatch for {key}: {path.name} ({error})")
                maxima[key] = max(maxima[key], error)
            require(
                np.allclose(motion["joint_vel"][-1], motion["joint_vel"][-2], atol=1e-6),
                f"Final joint velocity mismatch: {path.name}",
            )
            require(
                np.allclose(motion["body_ang_vel_w"][-1], motion["body_ang_vel_w"][-2], atol=1e-6),
                f"Final body angular velocity mismatch: {path.name}",
            )
            frames += n
    return dict(frames=frames, canonical_model_fk_verified=True, max_errors=maxima)


def validate_loader(target):
    """Read-only compatibility check through the production motion loader."""
    from holosoma.config_types.motion import MotionLibConfig
    from holosoma.utils.motion_lib import MotionLibRobot
    from holosoma.utils.safe_torch_import import torch

    paths = sorted(Path(target).glob("*.npz"))
    config = MotionLibConfig(motion_file=str(Path(target).resolve()), standardize_motion_length=False)
    lib = MotionLibRobot(config, num_envs=len(paths), device="cpu")
    loaded = lib.load_motions(random_sample=False)
    require([Path(p).name for p in loaded] == [p.name for p in paths], "Loader clip ordering mismatch")
    limits = np.asarray(POSITION_LIMITS)
    speed_limits = np.asarray([joint["velocity"] for joint in MODEL["dofs"]])
    checks = 0
    interpolation_errors = dict(dof_pos=0.0, dof_vel=0.0)
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as motion:
            fps = float(motion["fps"].item())
            frames = len(motion["joint_pos"])
            require(
                abs(float(lib.get_motion_length()[i]) - (frames - 1) / fps) < 1e-5,
                f"Loader duration mismatch: {path.name}",
            )
            require(abs(float(lib._motion_fps[i]) - fps) < 1e-5, f"Loader FPS mismatch: {path.name}")
            # Test every interpolation interval, endpoints, and before/after-clip clamping.
            frame_times = np.r_[-1, 0, np.arange(frames - 1) + 0.5, frames - 1, frames + 1]
            times = torch.tensor(frame_times / fps, dtype=torch.float32)
            ids = torch.full((len(times),), i, dtype=torch.long)
            state = lib.get_motion_state(ids, times)
            for value in state.values():
                require(bool(torch.isfinite(value).all()), f"Nonfinite loaded motion: {path.name}")
            for key, source, offset in (("dof_pos", "joint_pos", 7), ("dof_vel", "joint_vel", 6)):
                expected = np.stack(
                    [
                        np.interp(times.numpy().astype(float) * fps, np.arange(frames), channel)
                        for channel in motion[source][:, offset:].T
                    ],
                    axis=1,
                )
                error = float(np.abs(state[key].numpy() - expected).max())
                interpolation_errors[key] = max(interpolation_errors[key], error)
                # Float32 frame-time arithmetic loses precision in long clips.
                tolerance = 2e-4 if key == "dof_pos" else 2e-3
                require(error < tolerance, f"Loader interpolation mismatch: {path.name} {key} ({error})")
            positions, velocities = state["dof_pos"].numpy(), state["dof_vel"].numpy()
            require(
                positions.shape == velocities.shape == (len(times), len(DOF_NAMES)),
                f"Loader joint dimensions mismatch: {path.name}",
            )
            require(
                np.all(positions >= limits[:, 0] - 2e-6) and np.all(positions <= limits[:, 1] + 2e-6),
                f"Loaded positions exceed joint limits: {path.name}",
            )
            require(
                np.all(np.abs(velocities) <= speed_limits * 0.95 + 2e-4),
                f"Loaded velocities exceed limits: {path.name}",
            )
            checks += len(times)
    return dict(
        clips=len(paths),
        interpolation_samples=checks,
        max_interpolation_errors=interpolation_errors,
        native_timing=True,
        joint_channels=len(DOF_NAMES),
        arm_channels=len(DOF_NAMES[15:]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "rpl_training/data/motions/x2/retarget_manifest.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--kinematics", action="store_true", help="Recompute all frames with MuJoCo and SciPy")
    parser.add_argument(
        "--loader", action="store_true", help="Read-only production loader check; run in the training environment"
    )
    parser.add_argument("--report", type=Path, help="Save validation results as JSON")
    args = parser.parse_args()
    result = validate_library(args.manifest, args.source, args.target)
    manifest = json.loads(args.manifest.read_text())
    target = args.target or ROOT / manifest["target_dataset"]
    if args.kinematics:
        result["kinematics"] = validate_kinematics(target)
    if args.loader:
        result["loader"] = validate_loader(target)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
