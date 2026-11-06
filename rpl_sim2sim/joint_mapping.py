"""Map policy joints to MuJoCo state and actuator addresses by name."""

import mujoco
import numpy as np


class JointMapping:
    def __init__(self, model, names=None):
        if names is None:
            names = [
                model.joint(int(j)).name
                for j in model.actuator_trnid[:, 0]
                if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            ]
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate motor joints")
        ids = [model.joint(name).id for name in self.names]
        self.qpos = np.array([model.jnt_qposadr[j] for j in ids], dtype=int)
        self.qvel = np.array([model.jnt_dofadr[j] for j in ids], dtype=int)
        actuators = []
        for joint in ids:
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError(f"Expected one actuator for {model.joint(joint).name}, got {matches}")
            actuators.append(matches[0])
        self.actuators = np.array(actuators, dtype=int)
