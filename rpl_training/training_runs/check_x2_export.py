"""Check a real X2 checkpoint against its ONNX export on identical inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_student_teacher_actor_module
from holosoma.config_types.algo import ModuleConfig

from rpl_training.x2 import DOF_NAMES, HOME, KD, KP, POSITION_LIMITS


def check(checkpoint):
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["experiment_config"]
    assert config["robot"]["dof_names"] == DOF_NAMES
    robot = config["robot"]
    assert robot["lower_dof_names"] + robot["upper_dof_names"] == DOF_NAMES
    np.testing.assert_allclose(robot["dof_pos_lower_limit_list"], [limits[0] for limits in POSITION_LIMITS])
    np.testing.assert_allclose(robot["dof_pos_upper_limit_list"], [limits[1] for limits in POSITION_LIMITS])
    np.testing.assert_allclose([robot["init_state"]["default_joint_angles"][name] for name in DOF_NAMES], HOME)
    session = ort.InferenceSession(str(checkpoint.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert json.loads(meta["dof_names"]) == DOF_NAMES
    np.testing.assert_allclose(json.loads(meta["kp"]), KP)
    np.testing.assert_allclose(json.loads(meta["kd"]), KD)
    distill = "DAgger" in config["algo"]["_target_"]
    lower_dim = session.get_inputs()[0].shape[1]
    dimensions = {
        "actor_state_obs": 575,
        "perception_obs": 5184 if distill else lower_dim - 575,
        "command_lin_vel_obs": 2,
    }
    history = {"actor_state_obs": 5, "perception_obs": 2 if distill else 1, "command_lin_vel_obs": 1}
    inputs, outputs = {}, []
    builder = setup_student_teacher_actor_module if distill else setup_ppo_actor_module
    rng = np.random.default_rng(7)
    for key, size in [("lower_body", 15), ("upper_body", 14)]:
        actor_config = ModuleConfig(**config["algo"]["config"]["module_dict"]["actor_" + key])
        actor = builder(dimensions, actor_config, size, 0.8, "cpu", history)
        actor.load_state_dict(saved["actor_model_state_dict"][key])
        actor.eval()
        assert all(torch.isfinite(p).all() for p in actor.parameters())
        name = "actor_obs_" + key
        width = next(i.shape[1] for i in session.get_inputs() if i.name == name)
        inputs[name] = rng.normal(0, 0.2, (1, width)).astype(np.float32)
        with torch.no_grad():
            outputs.append(actor.act_inference({"actor_obs": torch.from_numpy(inputs[name])}).numpy())
    expected = np.concatenate(outputs, axis=1)
    actual = session.run(None, inputs)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    result = {
        "checkpoint": str(checkpoint),
        "max_export_error": float(np.abs(actual - expected).max()),
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "output_shape": list(actual.shape),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    check(args.checkpoint)
