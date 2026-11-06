"""Portable X2 model/control constants, shared by training and inference."""

import json
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "data/robots/x2"
PARAMETERS = json.loads((ASSET_DIR / "parameters.json").read_text(encoding="utf-8"))
MODEL = PARAMETERS["derived"]
BASELINE = PARAMETERS["selected_or_inherited_baseline"]
DOF_NAMES = MODEL["dof_names"]
HOME = [j["home"] for j in MODEL["dofs"]]
KP = [j["kp"] for j in MODEL["dofs"]]
KD = [j["kd"] for j in MODEL["dofs"]]
EFFORT = [j["effort"] for j in MODEL["dofs"]]
# Keep vendor values in MODEL; these adaptations belong to simulation assets.
POSITION_LIMITS = [
    BASELINE["joint_position_limit_overrides_rad"].get(joint["name"], [joint["lower"], joint["upper"]])
    for joint in MODEL["dofs"]
]
