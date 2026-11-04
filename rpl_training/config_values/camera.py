"""Warp ray-cast depth camera presets (camera manager) for RPL / FALCON-Terrains."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg

_g1_dynamic_meshes = {
    "enable": True,
    "body_meshes": {
        # Lower body
        "pelvis_contour_link.STL": "pelvis",
        "pelvis.STL": "pelvis",
        "left_hip_roll_link.STL": "left_hip_roll_link",
        "left_hip_yaw_link.STL": "left_hip_yaw_link",
        "left_knee_link.STL": "left_knee_link",
        "left_ankle_roll_link.STL": "left_ankle_roll_link",
        "right_hip_roll_link.STL": "right_hip_roll_link",
        "right_hip_yaw_link.STL": "right_hip_yaw_link",
        "right_knee_link.STL": "right_knee_link",
        "right_ankle_roll_link.STL": "right_ankle_roll_link",
        # Upper body
        "left_shoulder_yaw_link.STL": "left_shoulder_yaw_link",
        "left_elbow_link.STL": "left_elbow_link",
        "left_wrist_roll_link.STL": "left_wrist_roll_link",
        "left_wrist_pitch_link.STL": "left_wrist_pitch_link",
        "left_wrist_yaw_link.STL": "left_wrist_yaw_link",
        "left_rubber_hand.STL": "left_rubber_hand",
        "right_shoulder_yaw_link.STL": "right_shoulder_yaw_link",
        "right_elbow_link.STL": "right_elbow_link",
        "right_wrist_roll_link.STL": "right_wrist_roll_link",
        "right_wrist_pitch_link.STL": "right_wrist_pitch_link",
        "right_wrist_yaw_link.STL": "right_wrist_yaw_link",
        "right_rubber_hand.STL": "right_rubber_hand",
    },
    "meshes_root": "@holosoma/data/robots/g1/meshes",
}

_g1_random_depth_occlusion = {
    "enable": True,
    "mask": [0, 1, 2],
    "box": [90, 50, 0, 0],
    "prob": 1.0,
    "diff": 30,
    "terrain_bias": 0.15,
    "p_none": 0.1,
    "p_small": 0.45,
    "p_large": 0.45,
}


# FAR-FALCON-Terrains parity (g1_29dof_rev_1_0 + exp/falcon_hm_dagger_moe):
# ray-cast at 240x135 -> noise -> occlusion -> clip [0.1, 2.0] -> bicubic resize
# to 48x27 -> 4-frame ring buffer read at delay index 2.
warp_depth_camera_props_falcon = CameraProps(
    image_type="depth",
    width=240,
    height=135,
    resized_width=48,
    resized_height=27,
    horizontal_fov=101.41,
    vertical_fov=69.00,  # unused by the ray caster: vfov follows from hfov + aspect, as in FALCON
    near_clip=0.1,
    far_clip=2.0,
    image_show=False,
    frame_rate=10,
    depth_delay=2,
    depth_buffer_len=4,
    depth_delay_range=[2, 2],
)

warp_dual_depth_cameras_falcon = CameraManagerCfg(
    terms={
        "warp_depth": CameraTermCfg(
            func="holosoma.managers.camera.terms.warp_depth:WarpDepthCamera",
            params={
                "poses": {
                    "front_depth": CameraPose(
                        camera_body_link="torso_link",
                        camera_offset=[0.135, 0.06, 0.02],
                        camera_rotation=[0.0, 75.0, 0.0],
                    ),
                    "back_depth": CameraPose(
                        camera_body_link="torso_link",
                        camera_offset=[-0.125, -0.06, 0.02],
                        camera_rotation=[0.0, 75.0, 180.0],
                    ),
                },
                "props": warp_depth_camera_props_falcon,
                "ray_cast_width": 240,
                "ray_cast_height": 135,
                "ray_far_plane": 2.5,  # FALCON max_range = far_clip + 0.5
                "noise": {
                    "enable": True,
                    "std_multiplier": 0.1,
                    "dropout_prob": 0.05,
                    "dropout_value": -2.5,  # FALCON near_out_of_range_value, clipped to near_clip later
                },
                "pose_randomization": {
                    "enable": True,
                    "translation_min": [-0.025, -0.025, -0.025],
                    "translation_max": [0.025, 0.025, 0.025],
                    "rotation_min_deg": [-2.5, -3.0, -2.5],
                    "rotation_max_deg": [2.5, 3.0, 2.5],
                },
                "fov_randomization": {
                    "enable": True,
                    "min_fov_deg": 101.41 - 5.0,
                    "max_fov_deg": 101.41 + 5.0,
                },
                "resize": {"enable": True, "mode": "bicubic", "antialias": True},
                "dynamic_meshes": _g1_dynamic_meshes,
                # ``mask`` indexes ``terrain_locomotion_multi_falcon``'s order (stones, slopes,
                # stairs_up, stairs_down): every terrain but stepping stones, as FALCON's
                # ``random_depth_occlusion_mask`` [0, 1, 3, 4, 5, 6, 7] (all but index 2).
                "random_depth_occlusion": {
                    **_g1_random_depth_occlusion,
                    "mask": [1, 2, 3],
                    "share_across_cameras": True,
                },
            },
        ),
    }
)

# Preserve front-camera geometry, noise and timing from the dual-camera rig.
_dual_depth_term = warp_dual_depth_cameras_falcon.terms["warp_depth"]
warp_front_depth_camera_falcon = replace(
    warp_dual_depth_cameras_falcon,
    terms={
        "warp_depth": replace(
            _dual_depth_term,
            params={
                **_dual_depth_term.params,
                "poses": {"front_depth": _dual_depth_term.params["poses"]["front_depth"]},
            },
        ),
    },
)
