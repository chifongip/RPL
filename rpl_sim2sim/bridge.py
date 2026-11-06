"""FAR-FALCON-Terrains ``sim2real/utils/sdk2py_bridge``: the Unitree bridge and the elastic band.

``BasicSdk2Bridge`` + ``UnitreeSdk2Bridge`` merged (G1 / HG messages only) and ``ElasticBand``,
unchanged otherwise.
"""

from __future__ import annotations

import sys

import glfw
import numpy as np
import pygame
from loguru import logger
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from rpl_sim2sim.joint_mapping import JointMapping

# Wireless-controller bit order of the Unitree remote.
KEY_MAP = {
    "R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
    "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15,
}  # fmt: skip

# Xbox pad under Linux (pygame ids).
XBOX_AXIS_ID = {"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5, "DX": 6, "DY": 7}
XBOX_BUTTON_ID = {
    "X": 2,
    "Y": 3,
    "B": 1,
    "A": 0,
    "LB": 4,
    "RB": 5,
    "SELECT": 6,
    "START": 7,
    "XBOX": 8,
    "LSB": 9,
    "RSB": 10,
}


class UnitreeSdk2Bridge:
    """Publishes ``rt/lowstate`` / ``rt/wirelesscontroller`` and receives ``rt/lowcmd`` (unitree_sdk2py)."""

    def __init__(self, mj_model, mj_data, robot_config):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.free_base = robot_config["FREE_BASE"]
        self.joints = JointMapping(mj_model, robot_config.get("dof_names"))
        self.num_motor = len(self.joints.names)
        self.torques = np.zeros(self.num_motor)
        self.torque_limit = np.array(robot_config["motor_effort_limit_list"])
        self.use_sensor = robot_config["USE_SENSOR"]  # True: use sensor data; False: use ground truth data
        if self.use_sensor:
            raise NotImplementedError("FALCON's G1 sim2sim reads ground-truth state (USE_SENSOR: false)")
        self.joystick = None

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()
        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)
        self.crc = CRC()
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_puber.Init()

    def LowCmdHandler(self, msg):
        if msg:
            self.low_cmd = msg

    def PublishLowState(self):
        if self.mj_data is None:
            return
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qacc = self.mj_data.qacc
        actuator_force = self.mj_data.actuator_force
        imu = self.low_state.imu_state
        motor_state = self.low_state.motor_state
        for i in range(self.num_motor):
            m = motor_state[i]
            m.q = qpos[self.joints.qpos[i]]
            m.dq = qvel[self.joints.qvel[i]]
            m.ddq = qacc[self.joints.qvel[i]]
            m.tau_est = actuator_force[self.joints.actuators[i]]
        imu.quaternion[:] = qpos[3:7]
        imu.gyroscope[:] = qvel[3:6]
        imu.accelerometer[:] = qacc[0:3]
        self.low_state.tick = int(self.mj_data.time * 1e3)
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)

    def PublishWirelessController(self):
        if self.joystick is None:
            return
        pygame.event.get()
        key_state = [0] * 16
        key_state[KEY_MAP["R1"]] = self.joystick.get_button(self.button_id["RB"])
        key_state[KEY_MAP["L1"]] = self.joystick.get_button(self.button_id["LB"])
        key_state[KEY_MAP["start"]] = self.joystick.get_button(self.button_id["START"])
        key_state[KEY_MAP["select"]] = self.joystick.get_button(self.button_id["SELECT"])
        key_state[KEY_MAP["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
        key_state[KEY_MAP["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
        key_state[KEY_MAP["A"]] = self.joystick.get_button(self.button_id["A"])
        key_state[KEY_MAP["B"]] = self.joystick.get_button(self.button_id["B"])
        key_state[KEY_MAP["X"]] = self.joystick.get_button(self.button_id["X"])
        key_state[KEY_MAP["Y"]] = self.joystick.get_button(self.button_id["Y"])
        key_state[KEY_MAP["up"]] = self.joystick.get_hat(0)[1] > 0
        key_state[KEY_MAP["right"]] = self.joystick.get_hat(0)[0] > 0
        key_state[KEY_MAP["down"]] = self.joystick.get_hat(0)[1] < 0
        key_state[KEY_MAP["left"]] = self.joystick.get_hat(0)[0] < 0
        self.wireless_controller.keys = sum(int(key_state[i]) << i for i in range(16))
        self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
        self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
        self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
        self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])
        self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        if js_type != "xbox" or not sys.platform.startswith("linux"):
            raise NotImplementedError("Only an Xbox pad on Linux is mapped")
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Exiting program.")
            sys.exit()
        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()
        self.axis_id = XBOX_AXIS_ID
        self.button_id = XBOX_BUTTON_ID


class ElasticBand:
    """ref: https://github.com/unitreerobotics/unitree_mujoco"""

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([2.5, -2.0, 3.5])
        self.length = 0
        self.enable = True
        self.video_recording = 0
        self.logger = logger

    def Advance(self, x, dx):
        """x: current position, dx: current velocity."""
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        return (self.stiffness * (distance - self.length) - self.damping * v) * direction

    def MujuocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
        if key == glfw.KEY_R:
            self.logger.info("Start recording video..." if self.video_recording == 0 else "Stop recording video...")
            self.video_recording = 1 - self.video_recording
        if key == glfw.KEY_3:
            self.point[0] = self.point[0] - 0.25
        if key == glfw.KEY_4:
            self.point[0] = self.point[0] + 0.25
        if key == glfw.KEY_5:
            self.point[1] = self.point[1] - 0.25
        if key == glfw.KEY_6:
            self.point[1] = self.point[1] + 0.25
