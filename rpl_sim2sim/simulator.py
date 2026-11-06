"""FAR-FALCON-Terrains ``sim2real/sim_env``: ``BaseSimulator`` + ``LocoManipSimulator`` + ``LocoManipCamSimulator``.

Merged into one class with the code paths FALCON's ``task=loco_manip/loco_manip_cam`` sim2sim runs:
MuJoCo at ``SIMULATE_DT`` with PD on the latest ``rt/lowcmd`` every step, the elastic band on the
torso, the passive viewer synced every ``VIEWER_DT``, the follow-camera video thread
(``USE_RENDERER``), and the depth ``ImageServer`` thread.
"""

from __future__ import annotations

import os
import time
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from rpl_sim2sim.bridge import ElasticBand, UnitreeSdk2Bridge
from rpl_sim2sim.image_server import ImageServer
from rpl_sim2sim.rate_limiter import RateLimiter


class FalconLocoManipCamSimulator:
    def __init__(self, config, video_path: str = "simulation.mp4"):
        self.config = config
        self.video_path = video_path
        self.init_config()
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()

        self.sim_thread = Thread(target=self.simulation_thread)
        self.record_thread = None
        if self.config.get("USE_RENDERER", False):
            self.record_thread = Thread(target=self.record_video, args=(self.video_path,), daemon=True)
        self.init_cams()

    # --------------------------------------------------------------- BaseSimulator
    def init_config(self):
        self.robot_config = self.config["ROBOT_CFG"]
        self.num_dof = self.robot_config["NUM_JOINTS"]
        self.num_motor = self.robot_config["NUM_MOTORS"]
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_motor)
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"])

    def _prime_offscreen_gl(self):
        """Create an EGL offscreen context BEFORE the GLFW viewer (NVIDIA: a worker-thread EGL context
        fails if it is first created after the main-thread GLX context). Kept alive for the process."""
        if os.environ.get("MUJOCO_GL", "").lower() != "egl":
            return
        if self.config.get("HEADLESS", False):
            return
        has_cameras = bool(self.robot_config.get("cameras", {}).get("poses"))
        if not (self.config.get("USE_RENDERER", False) or has_cameras):
            return
        try:
            self._gl_prime_renderer = mujoco.Renderer(self.mj_model, height=64, width=64)
            self._gl_prime_renderer.update_scene(self.mj_data)
            self._gl_prime_renderer.render()
            self.logger.info("Primed EGL offscreen context before viewer launch.")
        except Exception as e:
            self.logger.warning(f"Failed to prime EGL context: {e}")

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        if self.config.get("INITIAL_KEYFRAME"):
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.key(self.config["INITIAL_KEYFRAME"]).id
            )
            mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mj_model.opt.timestep = self.sim_dt
        self._prime_offscreen_gl()
        self.elastic_band = ElasticBand()
        if self.config.get("HOLD_HOME", False):
            # Support the untrained robot while the policy initializes.
            self.elastic_band.point = self.mj_data.qpos[:3].copy() + [0, 0, 2.1]
            self.elastic_band.length = 0.0
        if self.config.get("HEADLESS", False):
            self.band_attached_link = self.mj_model.body(self.config.get("BAND_ATTACHED_LINK", "torso_link")).id
            self.viewer = _HeadlessViewer()
            return
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.band_attached_link = self.mj_model.body(self.config.get("BAND_ATTACHED_LINK", "torso_link")).id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        # LocoManipCamSimulator / LocoManipSimulator viewer options
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    def init_factory(self):
        if self.config.get("INTERFACE", None):
            ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
        else:
            ChannelFactoryInitialize(self.config["DOMAIN_ID"])

    def init_robot_bridge(self):
        self.robot_bridge = UnitreeSdk2Bridge(self.mj_model, self.mj_data, self.robot_config)
        if self.config.get("HOLD_HOME", False):
            for i, motor in enumerate(self.robot_bridge.low_cmd.motor_cmd[: self.num_motor]):
                motor.q = self.robot_config["default_joint_angles"][i]
                motor.kp = self.robot_config["motor_kp"][i]
                motor.kd = self.robot_config["motor_kd"][i]
        if self.config["USE_JOYSTICK"]:
            self.robot_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[self.robot_bridge.joints.qpos[i]])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[self.robot_bridge.joints.qvel[i]])
                    )
            except Exception as e:
                self.logger.error(f"Joint {i} not found in motor_cmd: {e}")
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    # --------------------------------------------------------- LocoManipSimulator
    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.mj_data.xfrc_applied[self.band_attached_link, :3] = 0
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        self.mj_data.ctrl[:] = 0
        self.mj_data.ctrl[self.robot_bridge.joints.actuators] = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

    def simulation_thread(self):
        try:
            self._simulation_loop()
        except BaseException as error:
            self.simulation_error = error

    def _simulation_loop(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running() and not self.server.stop_event.is_set():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            sim_cnt += 1
            if self.config.get("MAX_STEPS") and sim_cnt >= self.config["MAX_STEPS"]:
                break
            if sim_cnt % 1000 == 0:
                end_time = time.time()
                self.logger.info(f"FPS: {1000 / (end_time - start_time):.2f}")
                start_time = end_time
            self.rate.sleep()
        if not np.isfinite(self.mj_data.qpos).all() or not np.isfinite(self.mj_data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo state")
        self.logger.info(f"Simulation finished: {sim_cnt} steps, pelvis z={self.mj_data.qpos[2]:.4f}")

    def record_video(self, filename="simulation.mp4"):
        """Follow camera on the torso at VIEWER_DT; frames are written while the band's ``R`` toggle is on."""
        import imageio

        renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        fps = int(1 / self.viewer_dt)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        root_id = self.mj_model.body("torso_link").id
        self.logger.info("Video recording thread started.")
        with imageio.get_writer(filename, fps=fps) as video:
            while self.viewer.is_running():
                cam.lookat[:] = self.mj_data.xpos[root_id].copy()
                cam.distance = 3.0
                cam.azimuth = 90.0
                cam.elevation = -20.0
                renderer.update_scene(self.mj_data, camera=cam)
                frame = renderer.render()
                if getattr(self.elastic_band, "video_recording", False):
                    video.append_data(frame)
                time.sleep(self.viewer_dt)
        self.logger.info(f"Video saved to {filename}")

    # ------------------------------------------------------ LocoManipCamSimulator
    def init_cams(self):
        camera_config = self.robot_config["cameras"]
        props = camera_config["props"]
        image_config = {
            "mj_model": self.mj_model,
            "mj_data": self.mj_data,
            "fps": props["frame_rate"],
            "image_shape": [props["height"], props["width"]],
            "if_resize": props["if_resize"],
            "resized_image_shape": [props["resized_height"], props["resized_width"]],
            "near_clip": props["near_clip"],
            "far_clip": props["far_clip"],
            "depth_delay": props.get("depth_delay", 0),
            "clip_before_resize": props.get("clip_before_resize", False),
            "native_renderer": props.get("native_renderer", False),
            "id_numbers": {},
        }
        for camera_name in camera_config["poses"]:
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self.logger.info(f"Camera ID: {cam_id}")
            image_config["id_numbers"][camera_name] = cam_id
        self.server = ImageServer(image_config, img_shm_name=self.config.get("IMG_SHM_NAME", "depth_img_shm"))
        self.cam_thread = Thread(target=self.server.send_process)
        self.cam_thread.daemon = True


class _HeadlessViewer:
    def is_running(self):
        return True

    def sync(self):
        pass
