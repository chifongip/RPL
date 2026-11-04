"""Exercise the front-only student with the depth stack emitted by its camera rig."""

from dataclasses import replace

import torch
from holosoma.agents.modules.modules import BaseModule

from rpl_training.config_values.experiment import g1_29dof_rpl_distill_front


def test_front_student_forward():
    preset = g1_29dof_rpl_distill_front
    term = preset.camera.terms["warp_depth"]
    props = term.params["props"]
    depth_dim = len(term.params["poses"]) * 2 * props.resized_height * props.resized_width
    cfg = replace(preset.algo.config.module_dict.actor_lower_body, output_dim=[15])
    obs_dims = {
        "actor_state_obs": 575,
        "perception_obs": depth_dim,
        "command_lin_vel_obs": 2,
    }
    history = {"actor_state_obs": 5, "perception_obs": 2, "command_lin_vel_obs": 1}
    student = BaseModule(obs_dims, cfg, {k: history[k] for k in cfg.input_dim})
    assert list(term.params["poses"]) == ["front_depth"]
    assert depth_dim == 2592
    # One camera has 18 spatial cells per channel token, versus 36 for the dual rig.
    assert student.encoder.vision_feat_dim == 18
    with torch.no_grad():
        out = student.encoder(torch.rand(3, depth_dim), torch.randn(3, 575), v_cmd=torch.randn(3, 2))
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()
