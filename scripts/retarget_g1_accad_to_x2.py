"""Retarget packaged G1 ACCAD clips using an installed robot_retargeter kit.

Run with the kit's Python environment. The source dataset and kit are read-only.
Output NPZ files retain all source frames and the source frame rate,
and use the Holosoma motion schema. Raw native-rate kit CSVs are also retained.
The script does not enable the new library in X2 training presets.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rpl_training.x2 import ASSET_DIR, DOF_NAMES, HOME, MODEL  # noqa: E402

SPEED_LIMIT_MARGIN = 0.95

AUX = {
    "hips_sphere",
    "neck_sphere",
    "head_sphere",
    "left_foot_end_link",
    "left_toe_link",
    "right_foot_end_link",
    "right_toe_link",
    "head_yaw_link",
    "head_pitch_link",
}


def validate_paths(source: Path, output: Path, work: Path, kit: Path) -> None:
    """Reject output locations that could modify source data or the kit."""
    source, output, work, kit = (p.resolve() for p in (source, output, work, kit))
    for path in (output, work):
        if path == source or source in path.parents or path in source.parents:
            raise ValueError(f"Output/work path overlaps the read-only source directory: {path}")
        if path == kit or kit in path.parents:
            raise ValueError(f"Output/work path is inside the read-only retargeting kit: {path}")


def adapt_model(source: Path, template: Path, output: Path, sole_z: float) -> None:
    """Add only kit FK markers; retain the actual training robot kinematics."""
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = (source.parent / compiler.get("meshdir", "meshes")).resolve()
    compiler.set("meshdir", str(meshdir))
    other = ET.parse(template).getroot()
    parents = {c: p for p in other.iter() for c in p}
    for node in other.iter("body"):
        if node.get("name") not in AUX or root.find(f".//body[@name='{node.get('name')}']") is not None:
            continue
        parent_name = parents[node].get("name")
        parent = root.find(f".//body[@name='{parent_name}']")
        if parent is None:
            raise ValueError(f"Missing auxiliary parent {parent_name}")
        marker = ET.SubElement(parent, "body", dict(node.attrib))
        if "foot_end" in node.get("name") or "toe_link" in node.get("name"):
            pos = [float(v) for v in node.get("pos").split()]
            pos[2] = sole_z
            marker.set("pos", " ".join(str(v) for v in pos))
    tree.write(output, encoding="utf-8", xml_declaration=True)


def prepare(kit: Path, work: Path, arm_position_cost: float = 100.0) -> None:
    work.mkdir(parents=True, exist_ok=True)
    g1_dir = ROOT / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1"
    source_xml, target_xml = work / "g1_fk.xml", work / "x2_fk.xml"
    adapt_model(g1_dir / "g1_29dof_rev_1_0.xml", kit / "asset/robot/g1_description/mjcf/g1.xml", source_xml, -0.035)
    adapt_model(ASSET_DIR / "x2_ultra_29dof.xml", kit / "asset/robot/agibot_x2/x2_ultra.xml", target_xml, -0.073)
    for role, name, model in [("source", "g1", source_xml), ("target", "agibot_x2", target_xml)]:
        config = yaml.safe_load((kit / f"config/robot/{name}.yaml").read_text())
        config["robot_xml_path"] = str(model)
        config["verbose"] = False
        config["render_debug"] = False
        if role == "target":
            # The RPL robot has a fixed head; do not ask the waist to track head motions.
            config["ik_match_table"].pop("head", None)
            for key in ["left_arm", "left_fore_arm", "right_arm", "right_fore_arm"]:
                config["ik_match_table"][key][1] = arm_position_cost
        path = work / f"{role}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=None))
    source_model = mujoco.MjModel.from_xml_path(str(source_xml))
    target_model = mujoco.MjModel.from_xml_path(str(target_xml))
    if source_model.nq != 36 or target_model.nq != 36:
        raise ValueError("Expected 29-DOF source and target models with floating bases")
    if [target_model.joint(i).name for i in range(1, target_model.njnt)] != DOF_NAMES:
        raise ValueError("Target model joint order differs from the RPL X2 model")


def kit_modules(kit: Path):
    sys.path.insert(0, str(kit / "scripts"))
    import robot_replay as replay
    import robot_retarget as retarget

    return replay, retarget


def generate_keypoints(source: Path, source_config: Path, target_config: Path, kit: Path, dest: Path):
    rr, _ = kit_modules(kit)
    with np.load(source, allow_pickle=False) as data:
        q = data["joint_pos"].astype(np.float64)
        fps = float(data["fps"].item())
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid source poses or FPS: {source}")
    norms = np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError(f"Invalid source root quaternion: {source}")
    # Packaged NPZ root quaternions are xyzw; MuJoCo requires wxyz.
    q[:, 3:7] = q[:, [6, 3, 4, 5]]
    q[:, 3:7] /= norms
    src = yaml.safe_load(source_config.read_text())
    tgt = yaml.safe_load(target_config.read_text())
    source_xml, target_xml = Path(src["robot_xml_path"]), Path(tgt["robot_xml_path"])
    links = rr.load_robot_links_config(source_config)
    contact_links = tuple(src["contact_links"])
    contact_names = tuple(rr.canonicalize_contact_name(n) for n in contact_links)
    pos, quat, contact_pos = rr.build_robot_replay_buffers(q, source_xml, links, contact_links)
    speeds, states = rr.compute_robot_contact_sequence(
        contact_pos,
        fps,
        src["contact_vel_calculate_window"],
        src["contact_vel_threshold"],
        src["contact_height_threshold"],
    )
    lengths = rr.compute_robot_link_lengths(target_config, target_xml)
    kp, kq, raw_quat, vectors, _, _ = rr.build_robot_retarget_keypoints(
        pos, quat, lengths, source_config, target_config
    )
    kp, kq, extra = rr.append_target_contact_keypoints(
        kp, kq, raw_quat, rr.load_robot_links_config(target_config), target_config, target_xml
    )
    leg_scale, _, _ = rr.compute_leg_displacement_scale(
        lengths, vectors, rr.load_effective_knee_angle_offset_degrees(target_config)
    )
    kp = rr.scale_keypoint_frame_displacements(kp, leg_scale, root_keypoint_idx=0)
    names = ["hips_mean", *tgt["robot_links"], *extra]
    kp, _ = rr.offset_keypoints_by_contact_height(
        kp, names, contact_names, contact_pos, states, src["contact_height_lpf_alpha"]
    )
    rr.save_keypoints_pkl(dest, names, kp, kq, fps, contact_names, contact_pos, speeds, states)
    return q, fps, kp, names


def constrain_joint_speed(joints, fps, limits):
    """Symmetric forward/backward velocity projection within a convex joint box.

    Each sweep bounds consecutive differences. Their mean is also bounded,
    avoids a purely causal lag, and remains within the input position bounds.
    """
    joints, limits = np.asarray(joints, dtype=float), np.asarray(limits, dtype=float)
    if (
        joints.ndim != 2
        or len(joints) < 2
        or limits.shape != (joints.shape[1],)
        or not np.isfinite(joints).all()
        or not np.isfinite(limits).all()
        or (limits <= 0).any()
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Expected finite joint trajectories, positive FPS, and positive per-joint velocity limits")
    max_step = SPEED_LIMIT_MARGIN * limits / fps
    forward, backward = joints.copy(), joints.copy()
    for i in range(1, len(joints)):
        forward[i] = np.clip(forward[i], forward[i - 1] - max_step, forward[i - 1] + max_step)
    for i in range(len(joints) - 2, -1, -1):
        backward[i] = np.clip(backward[i], backward[i + 1] - max_step, backward[i + 1] + max_step)
    return 0.5 * (forward + backward)


def export_motion(engine, path: Path, fps: float):
    q = np.array(engine.result_pos, dtype=np.float64, copy=True)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected at least two finite 36-coordinate poses and a positive source FPS")
    norms = np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Root quaternions must be nonzero")
    q[:, 3:7] /= norms
    model = engine.model
    # The solver's limits include the kit's knee/elbow safety margins.
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    max_violation = float(np.maximum(np.maximum(lo - q[:, 7:], q[:, 7:] - hi), 0).max())
    if max_violation > 1e-5:
        raise ValueError(f"IK exceeded joint limits by {max_violation}")
    q[:, 7:] = np.clip(q[:, 7:], lo, hi)
    original_joints = q[:, 7:].copy()
    velocity_limits = np.asarray([j["velocity"] for j in MODEL["dofs"]])
    q[:, 7:] = constrain_joint_speed(original_joints, fps, velocity_limits)
    adjustment = np.abs(q[:, 7:] - original_joints)
    speed_stats = dict(
        max_joint_adjustment_rad=float(adjustment.max()),
        adjusted_joint_frame_fraction=float((adjustment > 1e-6).mean()),
        output_fps=fps,
    )
    # Consistent quaternion signs avoid artificial derivative jumps.
    for i in range(1, len(q)):
        if q[i, 3:7] @ q[i - 1, 3:7] < 0:
            q[i, 3:7] *= -1
    body_ids = [model.body(n).id for n in MODEL["expected_collapsed_body_names"]]
    data = mujoco.MjData(model)
    body_pos, body_quat = [], []
    for frame in q:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[body_ids].copy())
        body_quat.append(data.xquat[body_ids].copy())
    body_pos, body_quat = np.asarray(body_pos), np.asarray(body_quat)
    qvel = np.zeros((len(q), 35))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(model, qvel[i], 1 / fps, q[i], q[i + 1])
    qvel[-1] = qvel[-2]
    # MuJoCo body quaternions are wxyz, as expected by Holosoma's motion body fields.
    linear = np.gradient(body_pos, 1 / fps, axis=0)
    angular = np.zeros_like(linear)
    for i in range(len(q) - 1):
        for b in range(len(body_ids)):
            local_velocity = np.zeros(3)
            mujoco.mju_subQuat(local_velocity, body_quat[i + 1, b], body_quat[i, b])
            mujoco.mju_rotVecQuat(angular[i, b], local_velocity, body_quat[i, b])
    angular[:-1] *= fps
    angular[-1] = angular[-2]
    packed = q.copy()
    packed[:, 3:7] = q[:, [4, 5, 6, 3]]  # NPZ root: xyzw, matching the source schema.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.array([fps], dtype=np.float64),
        joint_pos=packed.astype(np.float32),
        joint_vel=qvel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=linear.astype(np.float32),
        body_ang_vel_w=angular.astype(np.float32),
        dof_names=np.asarray(DOF_NAMES),
        body_names=np.asarray(MODEL["expected_collapsed_body_names"]),
    )
    return q, body_pos, max_violation, speed_stats


def process_clip(args):
    source, output, work, kit = (Path(v) for v in args)
    start = time.monotonic()
    log = work / "clip_logs" / f"{source.stem}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        _, rt = kit_modules(kit)
        source_config, target_config = work / "source.yaml", work / "target.yaml"
        keypoints = work / "keypoints" / f"{source.stem}.pkl"
        keypoints.parent.mkdir(exist_ok=True)
        src_q, fps, targets, target_names = generate_keypoints(source, source_config, target_config, kit, keypoints)
        config = yaml.safe_load(target_config.read_text())
        engine = rt.RobotRetarget(
            config["robot_xml_path"],
            str(keypoints),
            config["ik_match_table"],
            solver="daqp",
            verbose=False,
            render_debug=False,
            joints_limit_offset_degrees=config["joints_limit_offset_degrees"],
            contact_body_names=config["contact_links"],
            contact_position_cost=config["contact_pos_fixed_factor"],
        )
        seed = engine.configuration.q.copy()
        seed[:3] = targets[0, 0]
        seed[3:7] = src_q[0, 3:7]
        seed[7:] = HOME
        engine.configuration.update(seed)
        engine.retarget()
        raw_csv_dir = work / "raw_robot_motion"
        raw_csv_dir.mkdir(exist_ok=True)
        raw_q = np.asarray(engine.result_pos).copy()
        raw_q[:, 3:7] = raw_q[:, [4, 5, 6, 3]]
        np.savetxt(raw_csv_dir / f"{source.stem}_x2.csv", raw_q, delimiter=",", fmt="%.8f")
        q, body_pos, violation, speed_stats = export_motion(engine, output / source.name, fps)
        errors = {}
        for key, body in [
            ("left_fore_arm", "left_wrist_roll_link"),
            ("right_fore_arm", "right_wrist_roll_link"),
            ("left_arm", "left_elbow_link"),
            ("right_arm", "right_elbow_link"),
        ]:
            actual = body_pos[:, MODEL["expected_collapsed_body_names"].index(body)]
            error = np.linalg.norm(actual - targets[:, target_names.index(key)], axis=1)
            errors[key] = dict(
                mean_m=float(error.mean()), p95_m=float(np.quantile(error, 0.95)), max_m=float(error.max())
            )
        return dict(
            clip=source.name,
            frames=len(q),
            native_fps=fps,
            duration_s=(len(q) - 1) / fps,
            solve_seconds=time.monotonic() - start,
            joint_limit_violation_rad=violation,
            arm_position_errors=errors,
            velocity_projection=speed_stats,
            source_sha256=digest(source),
        )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_path(path: Path) -> str:
    """Keep in-repository manifest paths portable across checkouts."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def cached_record(source: Path, output: Path, metrics: Path, signature: str):
    """Return matching cached metrics, or rerun missing, stale, or corrupt outputs."""
    if not output.is_file() or not metrics.is_file():
        return None
    try:
        record = json.loads(metrics.read_text())
    except (ValueError, OSError):
        return None
    if (
        isinstance(record, dict)
        and record.get("clip") == source.name
        and record.get("run_signature") == signature
        and record.get("source_sha256") == digest(source)
        and record.get("output_sha256") == digest(output)
    ):
        return record
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path("/home/ubuntu/robot_retargeter"))
    parser.add_argument("--source", type=Path, default=ROOT / "rpl_training/data/motions/g1/falcon_accad_all")
    parser.add_argument("--output", type=Path, default=ROOT / "rpl_training/data/motions/x2/falcon_accad_all")
    parser.add_argument("--work", type=Path, default=ROOT / "logs/retargeting/g1_to_x2_accad/final")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, help="Also write provenance to this path (use for a full library run)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clips", nargs="+", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="Skip clips only when inputs, generator, settings and output hashes match"
    )
    parser.add_argument("--arm-position-cost", type=float, default=100.0)
    args = parser.parse_args()
    for field in ("kit", "source", "output", "work"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    validate_paths(args.source, args.output, args.work, args.kit)
    if args.manifest is not None:
        args.manifest = args.manifest.expanduser().resolve()
        for protected in (args.source, args.kit, args.output):
            if args.manifest == protected or protected in args.manifest.parents:
                parser.error(f"Manifest must be outside source, kit, and motion output directories: {args.manifest}")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    if not np.isfinite(args.arm_position_cost) or args.arm_position_cost <= 0:
        parser.error("--arm-position-cost must be finite and positive")
    files = sorted(args.source.glob("*.npz"))
    if args.clips:
        files = [p for p in files if int(p.name[:3]) in args.clips]
    if args.limit:
        files = files[: args.limit]
    if not files:
        parser.error("No source clips match the requested selection")
    prepare(args.kit, args.work, args.arm_position_cost)
    generator_hash = digest(Path(__file__))
    kit_commit = subprocess.check_output(["git", "-C", str(args.kit), "rev-parse", "HEAD"], text=True).strip()
    run_signature = hashlib.sha256(
        json.dumps(
            dict(
                generator=generator_hash,
                kit_commit=kit_commit,
                inputs={
                    name: digest(args.work / name) for name in ("source.yaml", "target.yaml", "g1_fk.xml", "x2_fk.xml")
                },
                robot_parameters=digest(ROOT / "rpl_training/x2.py"),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    records = []
    record_dir = args.work / "metrics"
    record_dir.mkdir(exist_ok=True)
    pending = []
    for source in files:
        record_path = record_dir / f"{source.stem}.json"
        output_path = args.output / source.name
        record = cached_record(source, output_path, record_path, run_signature) if args.resume else None
        if record is not None:
            records.append(record)
            continue
        pending.append((str(source), str(args.output), str(args.work), str(args.kit)))
    print(f"Retargeting {len(pending)} clips ({len(records)} complete)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_clip, item): item for item in pending}
        for future in as_completed(jobs):
            record = future.result()
            record["run_signature"] = run_signature
            record["output_sha256"] = digest(args.output / record["clip"])
            records.append(record)
            (record_dir / f"{Path(record['clip']).stem}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(
                f"[{len(records)}/{len(files)}] {record['clip']} {record['frames']} frames {record['solve_seconds']:.1f}s",
                flush=True,
            )
    manifest = dict(
        source_dataset=dataset_path(args.source),
        target_dataset=dataset_path(args.output),
        clips=len(records),
        frames=sum(r["frames"] for r in records),
        kit=str(args.kit),
        toolkit_commit=kit_commit,
        generator_sha256=generator_hash,
        head_fixed=True,
        timing="source frame rate and frame count preserved",
        arm_position_cost=args.arm_position_cost,
        joint_velocity_limit_margin=SPEED_LIMIT_MARGIN,
        root_quaternion_order="xyzw",
        body_quaternion_order="wxyz",
        dof_names=DOF_NAMES,
        files=sorted(records, key=lambda r: r["clip"]),
    )
    (args.work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {manifest['clips']} clips, {manifest['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
