"""Run with the robot_retargeter environment: python scripts/test_retarget_g1_accad_to_x2.py."""

import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import retarget_g1_accad_to_x2 as converter
from render_retargeted_accad import interpolate, playback_caption
from retarget_g1_accad_to_x2 import (
    ASSET_DIR,
    DOF_NAMES,
    HOME,
    MODEL,
    constrain_joint_speed,
    export_motion,
    prepare,
    validate_paths,
)
from scipy.spatial.transform import Rotation
from validate_x2_motion_library import validate_kinematics, validate_library


class RetargetConversionTests(unittest.TestCase):
    def test_speed_projection_bounds_and_time_symmetry(self):
        joints = np.random.default_rng(71).uniform(-1, 1, (100, 29))
        original = joints.copy()
        limits = np.linspace(0.5, 5, 29)
        result = constrain_joint_speed(joints, 30, limits)
        self.assertTrue(np.all(np.abs(np.diff(result, axis=0)) * 30 <= 0.95 * limits + 1e-12))
        self.assertGreaterEqual(result.min(), joints.min())
        self.assertLessEqual(result.max(), joints.max())
        np.testing.assert_array_equal(joints, original)
        np.testing.assert_allclose(result, constrain_joint_speed(joints[::-1], 30, limits)[::-1])

    def test_speed_projection_accepts_integer_inputs(self):
        joints = np.array([[0], [1], [0]])
        result = constrain_joint_speed(joints, 10, np.array([1]))
        self.assertGreater(float(result.max()), 0)
        self.assertTrue(np.all(np.abs(np.diff(result, axis=0)) * 10 <= 0.95 + 1e-12))
        np.testing.assert_array_equal(joints[:, 0], [0, 1, 0])

    def test_already_feasible_motion_is_unchanged(self):
        joints = np.linspace(0, 0.1, 20)[:, None] * np.ones((1, 29))
        np.testing.assert_allclose(constrain_joint_speed(joints, 30, np.ones(29)), joints)

    def test_source_and_kit_are_protected_from_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, kit, output, work = [root / p for p in ("source", "kit", "output", "work")]
            validate_paths(source, output, work, kit)
            for unsafe in (source, source / "output", root, kit, kit / "output"):
                with self.subTest(path=unsafe), self.assertRaises(ValueError):
                    validate_paths(source, unsafe, work, kit)
                with self.subTest(work=unsafe), self.assertRaises(ValueError):
                    validate_paths(source, output, unsafe, kit)
            source.mkdir()
            alias = root / "alias"
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_paths(source, alias, work, kit)

    def test_playback_duration_and_quaternion_interpolation(self):
        self.assertIn("5s clip playback", playback_caption("full_body", 5))
        self.assertNotIn("10s", playback_caption("full_body", 5))
        q = np.zeros((2, 36))
        q[:, 3:7] = Rotation.from_euler("z", [[0], [90]], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        q[1, 7] = 1
        middle = interpolate(q, 3)[1]
        np.testing.assert_allclose(middle[3:7], [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-12)
        self.assertEqual(middle[7], 0.5)

    def test_library_validator_detects_corruption_and_retiming(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        q = np.tile(model.qpos0, (21, 1))
        q[:, 7:] = HOME
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, target = work / "source", work / "target"
            source.mkdir()
            target.mkdir()
            name = "clip.npz"
            np.savez(source / name, joint_pos=q, fps=np.array([30], dtype=np.float32))
            output = target / name
            export_motion(SimpleNamespace(model=model, result_pos=q), output, 30)
            record = dict(
                clip=name,
                frames=len(q),
                native_fps=30,
                duration_s=(len(q) - 1) / 30,
                source_sha256=hashlib.sha256((source / name).read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            manifest = dict(files=[record], clips=1, frames=len(q), joint_velocity_limit_margin=0.95)
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(validate_library(manifest_path, source, target)["clips"], 1)
            with output.open("ab") as file:
                file.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Output hash mismatch"):
                validate_library(manifest_path, source, target)
            # Even an updated provenance hash must not hide the old ten-second timing bug.
            with np.load(output) as data:
                fields = dict(data)
            fields["fps"] = np.array([(len(q) - 1) / 10], dtype=np.float32)
            np.savez_compressed(output, **fields)
            record["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Source/output FPS mismatch"):
                validate_library(manifest_path, source, target)

    def test_kinematics_validator_rejects_inconsistent_body_data(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        q = np.tile(model.qpos0, (21, 1))
        q[:, 7:] = HOME
        q[:, 0] = np.linspace(0, 0.1, len(q))
        q[:, 7 + DOF_NAMES.index("waist_yaw_joint")] = np.linspace(0, 0.2, len(q))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "clip.npz"
            export_motion(SimpleNamespace(model=model, result_pos=q), path, 30)
            self.assertTrue(validate_kinematics(directory)["canonical_model_fk_verified"])
            with np.load(path) as motion:
                fields = dict(motion)
            for key in ("body_pos_w", "body_quat_w", "joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                with self.subTest(field=key):
                    corrupt = {name: value.copy() for name, value in fields.items()}
                    corrupt[key].flat[1 if key == "body_quat_w" else 0] += 0.1
                    np.savez_compressed(path, **corrupt)
                    with self.assertRaisesRegex(ValueError, "FK/velocity mismatch"):
                        validate_kinematics(directory)

    def test_export_preserves_native_rates_and_uses_them_for_speed_limits(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        q = np.tile(model.qpos0, (51, 1))
        q[:, 7:] = HOME
        joint = DOF_NAMES.index("waist_yaw_joint")
        q[25:, 7 + joint] += 0.5
        limit = MODEL["dofs"][joint]["velocity"]
        with tempfile.TemporaryDirectory() as tmp:
            for fps in (24, 29.97, 30, 60):
                with self.subTest(fps=fps):
                    path = Path(tmp) / "clip.npz"
                    exported, _, _, stats = export_motion(SimpleNamespace(model=model, result_pos=q), path, fps)
                    self.assertEqual(stats["output_fps"], fps)
                    self.assertGreater(stats["max_joint_adjustment_rad"], 0)
                    speed = np.diff(exported[:, 7 + joint]) * fps
                    self.assertLessEqual(np.abs(speed).max(), 0.95 * limit + 1e-12)
                    with np.load(path) as data:
                        self.assertEqual(float(data["fps"].item()), fps)
                        self.assertAlmostEqual((len(data["joint_pos"]) - 1) / fps, 50 / fps)
                        np.testing.assert_allclose(data["joint_vel"][:-1, 6 + joint], speed, atol=2e-6)

    def test_cli_replaces_legacy_exports_and_resumes_valid_outputs(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        q = np.tile(model.qpos0, (21, 1))
        q[:, 7:] = HOME
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, work, kit = [root / p for p in ("source", "target", "work", "kit")]
            for directory in (source, target, work / "metrics", kit):
                directory.mkdir(parents=True)
            name = "000_clip.npz"
            np.savez(source / name, joint_pos=q, fps=np.array([30], dtype=np.float32))
            export_motion(SimpleNamespace(model=model, result_pos=q), target / name, 2)
            (work / "metrics" / "000_clip.json").write_text(json.dumps(dict(clip=name, native_fps=30)))
            manifest_path = root / "published.json"

            def prepare_stub(*args):
                for name in ("source.yaml", "target.yaml", "g1_fk.xml", "x2_fk.xml"):
                    path = work / name
                    if not path.exists():
                        path.write_text("test configuration")

            def submit_stub(function, args):
                export_motion(SimpleNamespace(model=model, result_pos=q), target / name, 30)
                future = Future()
                future.set_result(
                    dict(
                        clip=name,
                        frames=len(q),
                        native_fps=30,
                        duration_s=(len(q) - 1) / 30,
                        solve_seconds=0,
                        source_sha256=hashlib.sha256((source / name).read_bytes()).hexdigest(),
                    )
                )
                return future

            argv = [
                "retarget",
                "--source",
                str(source),
                "--output",
                str(target),
                "--work",
                str(work),
                "--kit",
                str(kit),
                "--manifest",
                str(manifest_path),
                "--resume",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(converter, "prepare", side_effect=prepare_stub))
                stack.enter_context(patch.object(converter.subprocess, "check_output", return_value="test-commit"))
                pool = stack.enter_context(patch.object(converter, "ProcessPoolExecutor"))
                submit = pool.return_value.__enter__.return_value.submit
                submit.side_effect = submit_stub
                converter.main()
                self.assertEqual(submit.call_count, 1)  # Legacy metrics cannot resume.
                self.assertTrue(validate_library(manifest_path)["source_timing_preserved"])
                converter.main()
                self.assertEqual(submit.call_count, 1)  # Correctly signed output resumes.
                with (target / name).open("ab") as file:
                    file.write(b"corruption")
                converter.main()
                self.assertEqual(submit.call_count, 2)
                self.assertTrue(validate_library(manifest_path)["source_timing_preserved"])
                (work / "x2_fk.xml").write_text("changed model")
                converter.main()
                self.assertEqual(submit.call_count, 3)  # Model changes invalidate cached IK.
                (work / "metrics" / "000_clip.json").write_text("{interrupted write")
                converter.main()
                self.assertEqual(submit.call_count, 4)

    def test_invalid_exports_and_speed_inputs_are_rejected(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            for poses in (np.zeros((1, 36)), np.zeros((2, 35)), np.zeros((2, 36))):
                with self.subTest(shape=poses.shape), self.assertRaises(ValueError):
                    export_motion(SimpleNamespace(model=model, result_pos=poses), path, 30)
            for fps in (0, -1, float("nan")):
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    constrain_joint_speed(np.zeros((2, 29)), fps, np.ones(29))

    @unittest.skipUnless(Path("/home/ubuntu/robot_retargeter").is_dir(), "Retargeting kit is not installed")
    def test_markers_preserve_training_robot_kinematics(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            prepare(Path("/home/ubuntu/robot_retargeter"), work)
            original = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
            adapted = mujoco.MjModel.from_xml_path(str(work / "x2_fk.xml"))
            self.assertEqual(adapted.nq, 36)
            self.assertEqual([adapted.joint(i).name for i in range(1, adapted.njnt)], DOF_NAMES)
            q = original.qpos0.copy()
            q[7:] = HOME
            datas = [mujoco.MjData(m) for m in (original, adapted)]
            for model, data in zip((original, adapted), datas):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
            for body in MODEL["expected_collapsed_body_names"]:
                np.testing.assert_allclose(
                    datas[0].xpos[original.body(body).id], datas[1].xpos[adapted.body(body).id], atol=1e-12
                )
                np.testing.assert_allclose(
                    datas[0].xquat[original.body(body).id], datas[1].xquat[adapted.body(body).id], atol=1e-12
                )
            np.testing.assert_array_equal(original.jnt_range, adapted.jnt_range)

    def test_npz_quaternions_duration_velocities_and_fk(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "x2_ultra_29dof.xml"))
        q = np.tile(model.qpos0, (21, 1))
        q[:, 7:] = HOME
        angles = np.linspace(0, 0.3, len(q))
        q[:, 3:7] = Rotation.from_euler("xyz", np.column_stack((angles * 0.2, angles * 0.3, angles))).as_quat()[
            :, [3, 0, 1, 2]
        ]
        q[1::2, 3:7] *= -1
        q[:, 7 + DOF_NAMES.index("waist_yaw_joint")] = angles * 0.1
        original = q.copy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            exported, pos, violation, stats = export_motion(SimpleNamespace(model=model, result_pos=q), path, 30)
            np.testing.assert_array_equal(q, original)
            self.assertEqual(violation, 0)
            self.assertEqual(stats["max_joint_adjustment_rad"], 0)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(float(data["fps"].item()), 30)
                self.assertAlmostEqual((len(q) - 1) / float(data["fps"].item()), (len(q) - 1) / 30)
                self.assertEqual(list(data["dof_names"]), DOF_NAMES)
                self.assertEqual(data["joint_pos"].shape, (21, 36))
                np.testing.assert_allclose(data["joint_pos"][:, 3:7], exported[:, [4, 5, 6, 3]], atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(data["joint_pos"][:, 3:7], axis=1), 1, atol=1e-7)
                # Independent FK of the first exported pose.
                check = mujoco.MjData(model)
                check.qpos[:] = exported[0]
                mujoco.mj_forward(model, check)
                indices = [model.body(n).id for n in MODEL["expected_collapsed_body_names"]]
                np.testing.assert_allclose(pos[0], check.xpos[indices], atol=1e-12)
                # Finite differences of WORLD rotation matrices verify the angular-velocity frame.
                for body_index in [0, MODEL["expected_collapsed_body_names"].index("torso_link")]:
                    rotations = Rotation.from_quat(data["body_quat_w"][:, body_index, [1, 2, 3, 0]])
                    expected = (rotations[1:] * rotations[:-1].inv()).as_rotvec() * float(data["fps"].item())
                    # Differentiating stored float32 quaternions amplifies roundoff by FPS.
                    np.testing.assert_allclose(data["body_ang_vel_w"][:-1, body_index], expected, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
