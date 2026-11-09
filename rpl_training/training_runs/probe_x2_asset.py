"""Check the unrandomized IsaacGym asset's names, mass and authored dynamics."""

import json

import numpy as np
from isaacgym import gymapi

from rpl_training.x2 import ASSET_DIR, DOF_NAMES, MODEL, POSITION_LIMITS


def main():
    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.dt = 0.005
    parameters.physx.use_gpu = True
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    if sim is None:
        raise RuntimeError("Could not create IsaacGym simulation")
    try:
        options = gymapi.AssetOptions()
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = False
        options.armature = 0.0
        asset = gym.load_asset(sim, str(ASSET_DIR), "x2_ultra_29dof.urdf", options)
        assert gym.get_asset_dof_names(asset) == DOF_NAMES
        assert gym.get_asset_rigid_body_names(asset) == MODEL["expected_collapsed_body_names"]
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
        actor = gym.create_actor(env, asset, gymapi.Transform(), "x2", 0, 1)
        bodies = gym.get_actor_rigid_body_properties(env, actor)
        mass = sum(b.mass for b in bodies)
        assert abs(mass - MODEL["urdf_total_mass"]) < 1e-5
        torso_mass = bodies[gym.get_asset_rigid_body_names(asset).index("torso_link")].mass
        assert abs(torso_mass - MODEL["merged_body_inertials"]["torso_link"]["mass"]) < 1e-5
        properties = gym.get_actor_dof_properties(env, actor)
        for field in ("lower", "upper", "velocity", "effort"):
            expected = (
                [limit[0 if field == "lower" else 1] for limit in POSITION_LIMITS]
                if field in ("lower", "upper")
                else [joint[field] for joint in MODEL["dofs"]]
            )
            np.testing.assert_allclose(properties[field], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(properties["friction"], 0.0, atol=1e-6)
        np.testing.assert_allclose(properties["damping"], 0.0, atol=1e-6)
        print(
            json.dumps(
                dict(
                    mass_kg=mass,
                    torso_mass_kg=torso_mass,
                    bodies=len(bodies),
                    dofs=len(DOF_NAMES),
                    joint_order=gym.get_asset_dof_names(asset),
                    joint_friction=properties["friction"].tolist(),
                ),
                indent=2,
            )
        )
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
