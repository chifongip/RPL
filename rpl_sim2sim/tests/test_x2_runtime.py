"""Runtime assets and joint/camera mappings, independent of trained weights."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rpl_sim2sim.joint_mapping import JointMapping  # noqa: E402
from rpl_training.x2 import ASSET_DIR, DOF_NAMES, EFFORT, HOME, KD, KP, MODEL, POSITION_LIMITS  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))


def test_asset_mass_limits_and_order(model):
    assert (model.nq, model.nv, model.nu, model.nbody) == (36, 35, 29, 31)
    assert model.body_mass.sum() == pytest.approx(MODEL["urdf_total_mass"], abs=1e-9)
    assert [model.body(i).name for i in range(1, model.nbody)] == MODEL["expected_collapsed_body_names"]
    assert [model.joint(i).name for i in range(1, model.njnt)] == DOF_NAMES
    np.testing.assert_allclose(model.jnt_range[1:], POSITION_LIMITS)
    urdf = ET.parse(ASSET_DIR / "x2_ultra_29dof.urdf").getroot()
    for name, limits in zip(DOF_NAMES, POSITION_LIMITS):
        limit = urdf.find(f"joint[@name='{name}']/limit")
        np.testing.assert_allclose([float(limit.get("lower")), float(limit.get("upper"))], limits)
    np.testing.assert_allclose(model.actuator_ctrlrange[:, 1], EFFORT)
    np.testing.assert_allclose(model.dof_armature[6:], 0.03)
    np.testing.assert_allclose(model.dof_frictionloss[6:], 0.3)


def test_collisions_are_from_training_urdf(model):
    urdf = ET.parse(ASSET_DIR / "x2_ultra_29dof.urdf").getroot()
    assert len(urdf.findall(".//collision")) == int((model.geom_contype == 1).sum())
    for side in ("left", "right"):
        body = model.body(side + "_ankle_roll_link").id
        indices = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype == 1))
        assert len(indices) == 12
        spheres = MODEL["feet"][side + "_ankle_roll_link"]["collision_spheres"]
        np.testing.assert_allclose(model.geom_pos[indices], [s["center"] for s in spheres])
        np.testing.assert_allclose(model.geom_size[indices, 0], 0.005)


def test_camera_geometry_and_initial_state(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    mapping = JointMapping(model, DOF_NAMES)
    np.testing.assert_allclose(data.qpos[mapping.qpos], HOME)
    for name, expected in MODEL["g1_camera_derived_optics"]["camera_frames"].items():
        camera = model.camera(f"cam_{name}_depth").id
        np.testing.assert_allclose(data.cam_xpos[camera], expected["home_position_world"], atol=2e-6)
        np.testing.assert_allclose(data.cam_xmat[camera].reshape(3, 3), expected["mujoco_rotation_in_torso"], atol=2e-6)


def test_mapping_handles_actuator_reordering():
    xml = ET.parse(ASSET_DIR / "x2_ultra_29dof.xml").getroot()
    xml.find("compiler").set("meshdir", str(ASSET_DIR / "meshes"))
    actuator = xml.find("actuator")
    actuator[:] = list(reversed(list(actuator)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    mapping = JointMapping(model, DOF_NAMES)
    np.testing.assert_array_equal(mapping.actuators, np.arange(29)[::-1])
    np.testing.assert_array_equal(mapping.qpos, np.arange(7, 36))
    data = mujoco.MjData(model)
    data.ctrl[mapping.actuators] = np.arange(29) * 0.01
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[mapping.qvel], np.arange(29) * 0.01)


def test_named_joint_states_and_commands_pass_through_bridge(model):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from rpl_sim2sim.bridge import LowState_default, UnitreeSdk2Bridge
    from rpl_sim2sim.simulator import FalconLocoManipCamSimulator

    data = mujoco.MjData(model)
    bridge = object.__new__(UnitreeSdk2Bridge)
    bridge.mj_data, bridge.joints = data, JointMapping(model, DOF_NAMES)
    bridge.num_motor, bridge.torque_limit = 29, np.array(EFFORT)
    bridge.low_state, bridge.crc, bridge.low_state_puber = LowState_default(), SimpleNamespace(Crc=lambda _: 0), Mock()
    commands = []
    for index, name in enumerate(DOF_NAMES):
        # Fill native state addresses independently of the bridge mapping.
        qpos = int(model.joint(name).qposadr[0])
        qvel = int(model.joint(name).dofadr[0])
        actuator = model.actuator("motor_" + name).id
        value = (index + 1) * 0.001
        data.qpos[qpos], data.qvel[qvel], data.qacc[qvel], data.actuator_force[actuator] = (
            value,
            value * 2,
            value * 3,
            value * 4,
        )
        commands.append(SimpleNamespace(q=value + 0.001, dq=value * 2, kp=KP[index], kd=KD[index], tau=0.0))
    bridge.PublishLowState()
    for field, multiplier in [("q", 1), ("dq", 2), ("ddq", 3), ("tau_est", 4)]:
        np.testing.assert_allclose(
            [getattr(motor, field) for motor in bridge.low_state.motor_state[:29]],
            np.arange(1, 30) * 0.001 * multiplier,
        )
    bridge.low_cmd = SimpleNamespace(motor_cmd=commands)
    simulator = object.__new__(FalconLocoManipCamSimulator)
    simulator.robot_bridge, simulator.mj_data = bridge, data
    simulator.torques, simulator.logger = np.zeros(29), Mock()
    simulator.compute_torques()
    np.testing.assert_allclose(simulator.torques, np.array(KP) * 0.001)


def test_pd_control_dynamics_and_clipping(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mapping = JointMapping(model, DOF_NAMES)
    peak = np.zeros(29)
    for _ in range(400):
        data.ctrl[mapping.actuators] = np.clip(
            np.array(KP) * (HOME - data.qpos[mapping.qpos]) - np.array(KD) * data.qvel[mapping.qvel],
            -np.array(EFFORT),
            EFFORT,
        )
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        peak = np.maximum(peak, np.abs(data.ctrl[mapping.actuators]))
    assert np.all(peak <= np.array(EFFORT) + 1e-9)
    assert data.time == pytest.approx(2.0)
    # HOME PD is an actuator test, not an untrained whole-body balance policy.
    assert peak.max() > 0


def test_g1_legacy_base_actuators_still_map():
    from rpl_sim2sim.run_sim import load_config

    model = mujoco.MjModel.from_xml_path(load_config()["ROBOT_SCENE"])
    mapping = JointMapping(model)
    assert len(mapping.names) == 29
    np.testing.assert_array_equal(mapping.qpos, np.arange(7, 36))
    np.testing.assert_array_equal(mapping.actuators, np.arange(6, 35))


def test_releasing_band_clears_applied_force(model):
    from types import SimpleNamespace

    from rpl_sim2sim.simulator import FalconLocoManipCamSimulator

    simulator = object.__new__(FalconLocoManipCamSimulator)
    simulator.config = {"ENABLE_ELASTIC_BAND": True}
    simulator.mj_model = model
    simulator.mj_data = mujoco.MjData(model)
    simulator.band_attached_link = model.body("torso_link").id
    simulator.elastic_band = SimpleNamespace(enable=False)
    simulator.robot_bridge = SimpleNamespace(
        PublishLowState=lambda: None, joystick=None, joints=JointMapping(model, DOF_NAMES)
    )
    simulator.compute_torques = lambda: None
    simulator.torques = np.zeros(29)
    simulator.mj_data.xfrc_applied[simulator.band_attached_link, :3] = [1, 2, 400]
    simulator.sim_step()
    np.testing.assert_allclose(simulator.mj_data.xfrc_applied[simulator.band_attached_link], 0)


@pytest.mark.parametrize("missing_frame", [False, True])
def test_camera_failure_after_first_frame_stops_simulation(monkeypatch, missing_frame):
    from multiprocessing import shared_memory
    from threading import Thread
    from types import SimpleNamespace
    from uuid import uuid4

    from rpl_sim2sim.image_server import ImageServer, MujocoCamera
    from rpl_sim2sim.run_sim import run_simulation

    name = "x2_failure_" + uuid4().hex
    config = {"id_numbers": {"camera": 0}, "image_shape": [2, 2], "fps": 100, "mj_model": None, "mj_data": None}
    server = ImageServer(config, name)

    def get_frame(camera):
        if server.frame_count:
            if missing_frame:
                return None
            raise RuntimeError("test camera failure")
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(MujocoCamera, "get_frame", get_frame)
    simulator = SimpleNamespace(
        server=server,
        record_thread=None,
        viewer=SimpleNamespace(close=lambda: None),
        sim_thread=Thread(target=lambda: server.stop_event.wait(5)),
        cam_thread=Thread(target=server.send_process),
    )
    with pytest.raises(RuntimeError, match="Camera producer failed") as error:
        run_simulation(simulator, require_camera=True)
    assert str(error.value.__cause__) == ("No frame from depth camera 0" if missing_frame else "test camera failure")
    assert server.frame_count == 1
    assert not simulator.sim_thread.is_alive() and not simulator.cam_thread.is_alive()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_depth_producers_cannot_share_a_memory_block():
    from multiprocessing import shared_memory
    from uuid import uuid4

    from rpl_sim2sim.image_server import ImageServer

    name = "x2_duplicate_" + uuid4().hex
    owner = shared_memory.SharedMemory(create=True, size=16, name=name)
    try:
        owner.buf[:] = bytes(range(16))
        config = {"id_numbers": {"camera": 0}, "image_shape": [2, 2], "mj_model": None, "mj_data": None}
        with pytest.raises(FileExistsError, match="another --shm-name"):
            ImageServer(config, name)
        # Rejecting another producer must neither overwrite nor unlink the owner.
        attachment = shared_memory.SharedMemory(name=name)
        try:
            assert bytes(attachment.buf) == bytes(range(16))
        finally:
            attachment.close()
    finally:
        owner.close()
        owner.unlink()


def test_all_cameras_and_shared_memory_close_on_renderer_failure():
    from multiprocessing import shared_memory
    from unittest.mock import Mock
    from uuid import uuid4

    from rpl_sim2sim.image_server import ImageServer

    server = object.__new__(ImageServer)
    name = "x2_cleanup_" + uuid4().hex
    server.image_shm = shared_memory.SharedMemory(create=True, size=16, name=name)
    good, bad = Mock(), Mock()
    bad.release.side_effect = RuntimeError("renderer close failure")
    server.cameras = [good, bad]
    with pytest.raises(RuntimeError, match="renderer close failure"):
        server._close()
    good.release.assert_called_once_with()
    bad.release.assert_called_once_with()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_x2_sim_config_matches_training_constants():
    from pathlib import Path

    from rpl_sim2sim.run_sim import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "config/x2_loco_manip_cam.yaml")
    robot = config["ROBOT_CFG"]
    assert robot["dof_names"] == DOF_NAMES
    np.testing.assert_array_equal(robot["default_joint_angles"], HOME)
    np.testing.assert_array_equal(robot["motor_kp"], KP)
    np.testing.assert_array_equal(robot["motor_kd"], KD)
    np.testing.assert_array_equal(robot["motor_effort_limit_list"], EFFORT)


@pytest.mark.parametrize("headless", [True, False])
def test_egl_priming_detects_depth_cameras_without_video(monkeypatch, headless):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from rpl_sim2sim.simulator import FalconLocoManipCamSimulator

    monkeypatch.setenv("MUJOCO_GL", "egl")
    renderer = Mock()
    factory = Mock(return_value=renderer)
    monkeypatch.setattr(mujoco, "Renderer", factory)
    simulator = object.__new__(FalconLocoManipCamSimulator)
    simulator.config = {"HEADLESS": headless, "USE_RENDERER": False}
    simulator.robot_config = {"cameras": {"poses": {"cam_front_depth": {}}}}
    simulator.mj_model, simulator.mj_data, simulator.logger = object(), object(), SimpleNamespace(info=Mock())
    simulator._prime_offscreen_gl()
    assert factory.call_count == (0 if headless else 1)
    if not headless:
        renderer.update_scene.assert_called_once_with(simulator.mj_data)
        renderer.render.assert_called_once_with()
