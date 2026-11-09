"""Render side-by-side G1/X2 retargeted clips; requires MuJoCo and ffmpeg.

Videos resample each clip to ten seconds by default for comparison. This
presentation setting does not change the motion-library frame rate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rpl_training.x2 import HOME, MODEL  # noqa: E402


def load_q(path):
    with np.load(path, allow_pickle=False) as d:
        q = d["joint_pos"].astype(float)
    q[:, 3:7] = q[:, [6, 3, 4, 5]]
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return q


def interpolate(q, count):
    old, new = np.linspace(0, 1, len(q)), np.linspace(0, 1, count)
    result = np.stack([np.interp(new, old, q[:, k]) for k in range(q.shape[1])], axis=1)
    quat = q[:, [4, 5, 6, 3]]
    result[:, 3:7] = Slerp(old, Rotation.from_quat(quat))(new).as_quat()[:, [3, 0, 1, 2]]
    return result


def frame_image(renderer, model, data, q, camera, option):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render().copy()


def write_png(path, image):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{image.shape[1]}x{image.shape[0]}",
            "-i",
            "-",
            "-frames:v",
            "1",
            str(path),
        ],
        input=image.tobytes(),
        check=True,
    )


def playback_caption(mode, seconds):
    title = "FULL BODY IK REPLAY" if mode == "full_body" else "ARM REFERENCES ON HOME LEGS / TORSO"
    return f"{title} - {seconds:g}s clip playback"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "logs/retargeting/g1_to_x2_accad/final")
    parser.add_argument("--source", type=Path, default=ROOT / "rpl_training/data/motions/g1/falcon_accad_all")
    parser.add_argument("--target", type=Path, default=ROOT / "rpl_training/data/motions/x2/falcon_accad_all")
    parser.add_argument("--kit", type=Path, default=Path("/home/ubuntu/robot_retargeter"))
    parser.add_argument("--clips", nargs="+", type=int, default=[0, 4, 74, 110, 120, 158, 177, 205, 217, 220])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--seconds", type=float, default=10, help="Video playback duration per clip; independent of dataset timing"
    )
    parser.add_argument("--thumbnail-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.fps <= 0 or round(args.fps * args.seconds) < 2:
        parser.error("FPS and duration must be positive and produce at least two frames")
    for field in ("kit", "work", "source", "target"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    files = {int(p.name.split("_", 1)[0]): p for p in args.source.glob("*.npz")}
    missing = [i for i in args.clips if i not in files]
    if missing:
        parser.error(f"Source clips not found: {missing}")
    chosen = [files[i] for i in args.clips]
    for clip in chosen:
        if not (args.target / clip.name).is_file():
            parser.error(f"Target clip not found: {clip.name}")
    sys.path.insert(0, str(args.kit / "scripts"))
    from robot_retarget import RobotRetarget

    helper = object.__new__(RobotRetarget)
    models = [helper._load_model_with_ground(str(args.work / f"{role}_fk.xml")) for role in ["g1", "x2"]]
    size = 640
    for model in models:
        model.vis.global_.offwidth = size
        model.vis.global_.offheight = size
        model.vis.headlight.ambient[:] = [0.55] * 3
        model.vis.headlight.diffuse[:] = [0.6] * 3
    datas = [mujoco.MjData(m) for m in models]
    renderers = [mujoco.Renderer(m, height=size, width=size) for m in models]
    from OpenGL import GL

    print("OpenGL renderer:", GL.glGetString(GL.GL_RENDERER), flush=True)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.75]
    camera.distance, camera.azimuth, camera.elevation = 2.5, 20, -15
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    out = args.output_dir.expanduser().resolve() if args.output_dir else args.work / "videos"
    out.mkdir(parents=True, exist_ok=True)
    count = round(args.fps * args.seconds)
    source_home = np.zeros(29)
    for side in ("left", "right"):
        for joint, value in [("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2)]:
            jid = models[0].joint(f"{side}_{joint}_joint").id
            source_home[models[0].jnt_qposadr[jid] - 7] = value
    home_pose = models[0].qpos0.copy()
    home_pose[:3] = 0
    home_pose[3:7] = [1, 0, 0, 0]
    home_pose[7:] = source_home
    datas[0].qpos[:] = home_pose
    mujoco.mj_forward(models[0], datas[0])
    g1_home_z = -min(
        datas[0].xpos[models[0].body(f"{side}_ankle_roll_link").id, 2] - 0.035 for side in ("left", "right")
    )
    thumbnails = []
    try:
        for mode in ["full_body", "upper_body_references"]:
            segments = []
            for clip in chosen:
                qs = [interpolate(load_q(p), count) for p in [clip, args.target / clip.name]]
                for i in range(2):
                    qs[i][:, :2] = 0  # Center root translation to keep the comparison visible.
                if mode == "upper_body_references":
                    qs[0][:, 7:22] = source_home[:15]
                    qs[1][:, 7:22] = np.asarray(HOME[:15])
                    qs[0][:, 2], qs[1][:, 2] = g1_home_z, MODEL["geometric_root_z_for_first_foot_contact_m"]
                    for q in qs:
                        q[:, 3:7] = [1, 0, 0, 0]
                segment = out / f"{mode}_{clip.stem[:3]}.mp4"
                label = clip.stem[6:].replace("_", " ")[:88]
                title = playback_caption(mode, args.seconds)
                filt = (
                    f"drawtext=text='G1 SOURCE':x=20:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6,"
                    f"drawtext=text='X2 RETARGETED':x=660:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6,"
                    f"drawtext=text='{title}':x=20:y=570:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6,"
                    f"drawtext=text='{clip.stem[:3]} {label}':x=20:y=606:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.6"
                )
                if args.thumbnail_only:
                    indices = [int(0.5 * count)]
                    encoder = None
                else:
                    indices = range(count)
                    encoder = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            "1280x640",
                            "-framerate",
                            str(args.fps),
                            "-i",
                            "-",
                            "-vf",
                            filt,
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-threads",
                            "2",
                            "-filter_threads",
                            "1",
                            "-crf",
                            "20",
                            "-pix_fmt",
                            "yuv420p",
                            str(segment),
                        ],
                        stdin=subprocess.PIPE,
                    )
                try:
                    for frame in indices:
                        images = [
                            frame_image(r, m, d, q[frame], camera, option)
                            for r, m, d, q in zip(renderers, models, datas, qs)
                        ]
                        image = np.concatenate(images, axis=1)
                        if encoder:
                            encoder.stdin.write(image.tobytes())
                        if frame == int(0.5 * count):
                            thumb = out / f"{mode}_{clip.stem[:3]}.png"
                            write_png(thumb, image)
                            thumbnails.append(str(thumb))
                finally:
                    if encoder:
                        try:
                            encoder.stdin.close()
                        finally:
                            returncode = encoder.wait()
                if encoder:
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, encoder.args)
                    segments.append(segment)
                print(f"Rendered {mode} {clip.name}", flush=True)
            if segments:
                listing = out / f"{mode}_concat.txt"
                listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(listing),
                        "-c",
                        "copy",
                        str(out / f"{mode}_comparison.mp4"),
                    ],
                    check=True,
                )
        (out / "video_manifest.json").write_text(
            json.dumps(
                dict(
                    clips=[p.name for p in chosen],
                    mode="kinematic replay; root XY centered",
                    clip_playback_seconds=args.seconds,
                    fps=args.fps,
                    thumbnails=thumbnails,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        for r in renderers:
            r.close()


if __name__ == "__main__":
    main()
