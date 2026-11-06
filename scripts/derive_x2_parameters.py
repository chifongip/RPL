"""Derive X2 preparation values without changing assets or training presets.

Run with a Python environment containing numpy. The source constants are parsed
as data: their imports and executable code are never run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path('/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0')
DEFAULT_CONSTANTS = Path('/home/ubuntu/unitree_rl_mjlab/src/assets/robots/agibot_x2/x2_constants.py')
HEAD_JOINTS = {'head_yaw_joint', 'head_pitch_joint'}


def assignment(tree, name):
    for node in tree.body:
        names = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(n, ast.Name) and n.id == name for n in names):
            return node.value
    raise ValueError(f'Missing constant: {name}')


def vector(text, default='0 0 0'):
    return np.array([float(x) for x in (text or default).split()])


def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def transform(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, 3] = vector(origin.get('xyz'))
        roll, pitch, yaw = vector(origin.get('rpy'))
        t[:3, :3] = rotation(np.array([0, 0, 1]), yaw) @ rotation(np.array([0, 1, 0]), pitch) @ rotation(np.array([1, 0, 0]), roll)
    return t


def derive(asset_dir=DEFAULT_ASSETS, constants=DEFAULT_CONSTANTS):
    asset_dir, constants = Path(asset_dir), Path(constants)
    urdf_path = asset_dir / 'x2_ultra_simple_collision.urdf'
    urdf = ET.parse(urdf_path).getroot()
    xml_path = asset_dir / 'x2_ultra.xml'
    xml = ET.parse(xml_path).getroot()
    tree = ast.parse(constants.read_text(encoding='utf-8'))
    gains = ast.literal_eval(assignment(tree, 'X2_GAIN_PRESETS'))['agibot_stiff']
    home = {kw.arg: ast.literal_eval(kw.value) for kw in assignment(tree, 'HOME_KEYFRAME').keywords}
    links = {link.get('name'): link for link in urdf.findall('link')}
    joints = urdf.findall('joint')
    children = {}
    for j in joints:
        children.setdefault(j.find('parent').get('link'), []).append(j)
    roots = set(links) - {j.find('child').get('link') for j in joints}
    assert roots == {'pelvis'}, roots
    positions, owners, bodies, ordered = {}, {}, [], []
    zero_positions, zero_owners = {}, {}

    def walk(link, pose, owner, at_home):
        pos = positions if at_home else zero_positions
        own = owners if at_home else zero_owners
        pos[link], own[link] = pose, owner
        if at_home and link == owner:
            bodies.append(link)
        for j in children.get(link, []):
            name = j.get('name')
            active = j.get('type') != 'fixed' and name not in HEAD_JOINTS
            child = j.find('child').get('link')
            angle = 0.0
            if active and at_home:
                matches = [v for p, v in home['joint_pos'].items() if re.fullmatch(p, name)]
                assert len(matches) <= 1, name
                angle = matches[0] if matches else 0.0
                ordered.append((j, angle))
            relative = transform(j.find('origin'))
            if active:
                rot = np.eye(4)
                rot[:3, :3] = rotation(vector(j.find('axis').get('xyz')), angle)
                relative = relative @ rot
            walk(child, pose @ relative, child if active else owner, at_home)

    walk('pelvis', np.eye(4), 'pelvis', True)
    walk('pelvis', np.eye(4), 'pelvis', False)
    dofs = []
    for i, (j, angle) in enumerate(ordered):
        name = j.get('name')
        matches = [v for p, v in gains.items() if re.fullmatch(p, name)]
        assert len(matches) == 1, name
        kp, kd = matches[0]
        limits = {k: float(v) for k, v in j.find('limit').attrib.items()}
        assert limits['lower'] <= angle <= limits['upper'], name
        mid = (limits['lower'] + limits['upper']) / 2
        half = (limits['upper'] - limits['lower']) / 2
        mirror = name.replace('left_', 'right_') if name.startswith('left_') else name.replace('right_', 'left_') if name.startswith('right_') else name
        sign = -1 if any(s in name for s in ('_roll_', '_yaw_')) else 1
        dofs.append(dict(index=i, name=name, child_link=j.find('child').get('link'),
                         axis=vector(j.find('axis').get('xyz')).tolist(), home=angle, kp=kp, kd=kd, **limits,
                         soft_position_limits_095=[mid - 0.95 * half, mid + 0.95 * half],
                         raw_action_interval_without_position_clipping=[(limits['lower'] - angle) / 0.25, (limits['upper'] - angle) / 0.25],
                         mirror_joint=mirror, mirror_sign=sign))
    names = [d['name'] for d in dofs]
    assert len(names) == 29
    xml_names = [j.get('name') for j in xml.findall('worldbody/.//joint') if j.get('name') not in HEAD_JOINTS]
    motor_names = [j.get('joint') for j in xml.findall('actuator/*') if j.get('joint') not in HEAD_JOINTS]
    assert names == xml_names == motor_names

    mass, weighted_com = 0.0, np.zeros(3)
    lumped = {b: [] for b in bodies}
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None:
            continue
        m = float(inertial.find('mass').get('value'))
        values = inertial.find('inertia').attrib
        inertia = np.array([
            [float(values['ixx']), float(values['ixy']), float(values['ixz'])],
            [float(values['ixy']), float(values['iyy']), float(values['iyz'])],
            [float(values['ixz']), float(values['iyz']), float(values['izz'])]])
        assert np.linalg.eigvalsh(inertia).min() > 0, name
        principal = np.linalg.eigvalsh(inertia)
        assert principal[-1] <= principal[:2].sum() + 1e-10, name
        world = positions[name] @ transform(inertial.find('origin'))
        mass += m
        weighted_com += m * world[:3, 3]
        owner = zero_owners[name]
        local = np.linalg.inv(zero_positions[owner]) @ zero_positions[name] @ transform(inertial.find('origin'))
        lumped[owner].append((m, local[:3, 3], local[:3, :3] @ inertia @ local[:3, :3].T))
    merged_inertials = {}
    for name, parts in lumped.items():
        total = sum(p[0] for p in parts)
        com = sum(m * p for m, p, _ in parts) / total
        inertia = sum(i + m * (np.dot(p - com, p - com) * np.eye(3) - np.outer(p - com, p - com)) for m, p, i in parts)
        assert np.linalg.eigvalsh(inertia).min() > 0, name
        merged_inertials[name] = dict(mass=total, com=com.tolist(), inertia_matrix=inertia.tolist())

    feet = {}
    for side in ('left', 'right'):
        name = side + '_ankle_roll_link'
        spheres = []
        for collision in links[name].findall('collision'):
            sphere = collision.find('geometry/sphere')
            assert sphere is not None
            spheres.append(dict(center=transform(collision.find('origin'))[:3, 3].tolist(), radius=float(sphere.get('radius'))))
        centers = np.array([s['center'] for s in spheres])
        radii = np.array([s['radius'] for s in spheres])
        bottom = centers[:, 2] - radii
        lowest = np.isclose(bottom, bottom.min())
        support = centers[lowest]
        world_centers = centers @ positions[name][:3, :3].T + positions[name][:3, 3] + np.array(home['pos'])
        feet[name] = dict(collision_spheres=spheres, collision_bounds_min=(centers - radii[:, None]).min(axis=0).tolist(),
                          collision_bounds_max=(centers + radii[:, None]).max(axis=0).tolist(),
                          sole_z=float(bottom.min()), lowest_support_sphere_count=int(lowest.sum()),
                          support_centers_xy=support[:, :2].tolist(),
                          home_link_position_world=(positions[name][:3, 3] + np.array(home['pos'])).tolist(),
                          home_lowest_collision_z_world=float((world_centers[:, 2] - radii).min()),
                          home_rotation_world=positions[name][:3, :3].tolist())

    meshes = []
    for name, link in links.items():
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            path = asset_dir / mesh.get('filename')
            assert path.is_file(), path
            meshes.append(dict(link=name, retained_body=owners[name], file=mesh.get('filename'),
                               scale=vector(mesh.get('scale'), '1 1 1').tolist(),
                               transform_in_retained_body=(np.linalg.inv(positions[owners[name]]) @ positions[name] @ transform(visual.find('origin'))).tolist()))
    kinematics = {name: (np.array(home['pos']) + pose[:3, 3]).tolist() for name, pose in positions.items()}
    camera_optics = {}
    frame_rotation = rotation(np.array([0, 0, 1]), -np.pi / 2) @ rotation(np.array([1, 0, 0]), -np.pi / 2)
    focal_px = 120 / np.tan(np.deg2rad(101.41 / 2))
    for name, offset, yaw in [('front', [0.135, 0.06, 0.02], 0), ('back', [-0.125, -0.06, 0.02], np.pi)]:
        local_rotation = rotation(np.array([0, 0, 1]), yaw) @ rotation(np.array([0, 1, 0]), np.deg2rad(75))
        optical = local_rotation @ frame_rotation
        mj_camera = optical @ np.diag([1, -1, -1])
        camera_optics[name] = dict(
            optical_rotation_in_torso=optical.tolist(), mujoco_rotation_in_torso=mj_camera.tolist(),
            mujoco_xyaxes_in_torso=np.concatenate([mj_camera[:, 0], mj_camera[:, 1]]).tolist(),
            optical_forward_in_torso=optical[:, 2].tolist(),
            home_position_world=(positions['torso_link'][:3, :3] @ offset + positions['torso_link'][:3, 3] + np.array(home['pos'])).tolist(),
        )
    baseline_files = ['rpl_training/config_values/robot.py', 'rpl_training/config_values/camera.py',
                      'rpl_training/config_values/simulator.py', 'rpl_training/config_values/command.py',
                      'thirdparty/holosoma/src/holosoma/holosoma/config_values/robot.py',
                      'thirdparty/holosoma/src/holosoma/holosoma/config_values/simulator.py',
                      'thirdparty/holosoma/src/holosoma/holosoma/managers/camera/terms/warp_depth.py']
    source_paths = [urdf_path, xml_path, constants] + [ROOT / p for p in baseline_files]
    return dict(
        provenance={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        units=dict(position='m', angle='rad', mass='kg', inertia='kg*m^2', effort='Nm', velocity='rad/s', kp='Nm/rad', kd='Nm*s/rad'),
        derived=dict(dofs=dofs, dof_names=names, lower_dof_names=names[:15], upper_dof_names=names[15:],
                     expected_collapsed_body_names=bodies, expected_num_bodies=len(bodies),
                     fixed_head_joint_names=sorted(HEAD_JOINTS), fixed_joint_body_owner=owners,
                     floating_mujoco_dimensions=dict(nq=36, nv=35, nu=29),
                     urdf_total_mass=mass, head_mass=sum(float(links[n].find('inertial/mass').get('value')) for n in ('head_yaw_link', 'head_pitch_link')),
                     merged_body_inertials=merged_inertials,
                     home_root_position=list(home['pos']), home_root_quaternion_xyzw=[0, 0, 0, 1], home_joint_velocities=[0.0]*29,
                     home_com_world=(weighted_com / mass + np.array(home['pos'])).tolist(), home_link_positions_world=kinematics,
                     nominal_foot_separation=abs(positions['left_ankle_roll_link'][1, 3] - positions['right_ankle_roll_link'][1, 3]),
                     nominal_knee_separation=abs(positions['left_knee_link'][1, 3] - positions['right_knee_link'][1, 3]),
                     home_nominal_drop_to_first_foot_contact_m=min(f['home_lowest_collision_z_world'] for f in feet.values()),
                     geometric_root_z_for_first_foot_contact_m=home['pos'][2] - min(f['home_lowest_collision_z_world'] for f in feet.values()),
                     robot_role_names=dict(root='pelvis', torso='torso_link', foot_pattern='ankle_roll_link',
                                           feet=['left_ankle_roll_link', 'right_ankle_roll_link'], knee_pattern='knee_link',
                                           knees=['left_knee_link', 'right_knee_link'],
                                           arm_terminal_bodies=['left_wrist_roll_link', 'right_wrist_roll_link']),
                     joint_subsets=dict(waist=names[12:15], left_arm=names[15:22], right_arm=names[22:],
                                        knees=[n for n in names if '_knee_' in n],
                                        hips=[n for n in names if '_hip_' in n],
                                        left_ankles=[n for n in names if n.startswith('left_ankle')],
                                        right_ankles=[n for n in names if n.startswith('right_ankle')]),
                     feet=feet, visual_mesh_instances=meshes,
                     g1_camera_derived_optics=dict(camera_frames=camera_optics, focal_xy_pixels=[float(focal_px)]*2,
                                                  image_center_convention='Use the Warp ray caster pixel-center convention; do not assume this equals a calibrated hardware intrinsic matrix.'),
                     observation_dimensions=dict(actor_frame=115, actor_history_frames=5, actor_state=575,
                                                 student_depth=5184, student_lower_input=5761, student_upper_input=575,
                                                 lower_actions=15, upper_actions=14, actions=29)),
        selected_or_inherited_baseline=dict(
            note='These are simulation starting settings, not measurements or values implied by kinematics.',
            action_scale_rad=0.25, action_clip_value=100.0, clip_torques=True,
            action_scales_by_effort_limit_over_p_gain=False,
            torque_equation='clip(Kp * (q_home + 0.25 * clip(a, -100, 100) - q) - Kd * dq, -effort, effort)',
            target_position_clipping='Unmodified G1 joint-position action term; no position-target clamp. Generated simulation models use symmetric waist-yaw hard stops.',
            armature_kg_m2=[0.03]*29, frictionloss_Nm=[0.3]*29, passive_joint_damping=0.0,
            friction_mapping_note='MuJoCo Coulomb frictionloss and PhysX joint friction are not guaranteed equivalent; match the nominal coefficient and validate.',
            physics_dt_s=0.005, policy_dt_s=0.02, pd_updates_per_policy_step=4, pd_update_dt_s=0.005,
            baseline_action_delay_steps=0, optional_g1_upper_body_delay_policy_steps=[0, 1],
            nominal_ground_friction=1.0, restitution=0.0, self_collisions=False,
            preserve_cylinders=True, collision_model='x2_ultra_simple_collision.urdf',
            soft_position_limit_fraction=0.95, knee_joint_min_threshold_rad=0.2,
            command_nominal_base_height_m=0.68, lidar_height_offset_m=0.68,
            torso_pitch_command_range_rad=[-0.5, 1.57], torso_yaw_command_range_rad=[-1.0, 1.0],
            torso_roll_command_range_rad=[0.0, 0.0], base_height_delta_range_m=[-0.3, 0.0],
            initial_upper_body_reference='HOME_KEYFRAME arms, motion=None', initial_hand_forces=False,
            symmetry_enabled_initially=True, symmetry_scope='PPO flat and terrain experts; DAgger unchanged',
            symmetry_actor_coef=1.0, symmetry_critic_coef=0.0,
            joint_position_limit_overrides_rad={'waist_yaw_joint': [-2.2078, 2.2078]},
            joint_limit_override_note='Apply only to generated training URDF/MuJoCo and runtime configuration; source vendor limits are preserved in derived.dofs.',
            symmetry_implementation='Unmodified G1 PPO and SymmetryUtils; no sample masking or custom distributed loss handling.',
            four_point_foot_edge_offsets_m=dict(front=[0.11, 0, 0], back=[-0.065, 0, 0], left=[0.0225, 0.06, 0], right=[0.0225, -0.06, 0]),
            foot_edge_offsets_note='Approximation derived from the lowest support-sphere center extents; excludes raised toe spheres, ignores the 5 mm radius, and uses existing base-oriented reward sampling.',
            low_height_termination_m=0.3, gravity_tilt_termination_xy=[0.8, 0.8],
            contact_penalty_body_patterns=['pelvis', 'shoulder', 'hip'],
            physx=dict(solver_type=1, position_iterations=4, velocity_iterations=0, bounce_threshold_velocity=0.5, contact_offset=0.01, rest_offset=0.0, max_depenetration_velocity=1.0),
            camera=dict(parent='torso_link', front_offset_m=[0.135, 0.06, 0.02], front_rotation_deg=[0, 75, 0],
                        back_offset_m=[-0.125, -0.06, 0.02], back_rotation_deg=[0, 75, 180],
                        render_size_wh=[240, 135], policy_size_wh=[48, 27], horizontal_fov_deg=101.41,
                        vertical_fov_from_aspect_deg=float(np.rad2deg(2*np.arctan(np.tan(np.deg2rad(101.41/2))*135/240))),
                        frame_rate_hz=10, policy_history_frames=2, buffer_frames=4, delay_index=2,
                        near_clip_m=0.1, far_clip_m=2.0, ray_far_plane_m=2.5,
                        noise_std_multiplier=0.1, dropout_probability=0.05, dropout_value=-2.5,
                        translation_randomization_m=[-0.025, 0.025], rotation_randomization_deg=[[-2.5, -3, -2.5], [2.5, 3, 2.5]],
                        horizontal_fov_randomization_deg=[96.41, 106.41])),
        requires_runtime_validation=['IsaacGym loaded body and DOF order after fixed-joint collapsing',
                                     'Settled standing/contact and controller stability',
                                     'PhysX versus MuJoCo contact/friction and actuator behavior',
                                     'Torso/base-height command reachability',
                                     'Camera self-occlusion and depth pipeline parity'],
    )


def report(parameters):
    d = parameters['derived']
    baseline = parameters['selected_or_inherited_baseline']
    text = [
        '# X2 Ultra parameters and simulation baseline', '',
        'Generated by `scripts/derive_x2_parameters.py`; update the generator and regenerate this file. See [runtime integration status](x2_ultra_training_review.md) for implementation, launch commands and validation results.', '',
        '[Full-precision JSON](x2_ultra_parameters.json) contains ordered arrays, source hashes, merged body inertias, mesh-instance transforms and symmetry candidates. Model-derived values and simulation starting settings are stored separately. The runtime packages a snapshot in `rpl_training/data/robots/x2/parameters.json`; regenerate runtime assets after changing this reference.', '',
        'The source model is `/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf`. The vendor `x2_ultra.xml` supplies an independent kinematic reference. Only `agibot_stiff` PD gains and `HOME_KEYFRAME` are read from `/home/ubuntu/unitree_rl_mjlab/src/assets/robots/agibot_x2/x2_constants.py`; other settings from that file are not adopted.', '',
        '## Values determined by the selected model and references', '',
        f'- Articulated DOFs: **29**; expected collapsed robot bodies: **{d["expected_num_bodies"]}**. Floating-base MuJoCo dimensions: `nq=36`, `nv=35`, `nu=29`.',
        '- Lower body: 12 leg joints + waist yaw/pitch/roll (15). Upper body: 14 arm joints.',
        f'- URDF mass: **{d["urdf_total_mass"]:.6f} kg**, including **{d["head_mass"]:.6f} kg** of fixed head mass.',
        f'- Head-fixed torso body mass after merging: **{d["merged_body_inertials"]["torso_link"]["mass"]:.6f} kg**.',
        f'- HOME pelvis height: **0.68 m**; HOME COM: `{[round(x, 6) for x in d["home_com_world"]]}` m.',
        f'- HOME foot-origin separation: **{d["nominal_foot_separation"]:.6f} m**; knee-origin separation: **{d["nominal_knee_separation"]:.6f} m**.',
        '- Feet: 12 spheres per foot, radius **0.005 m**. Ten share the lowest local sole level, **−0.073 m** relative to the ankle-roll link.',
        '- Collision bounds per foot: `[-0.070, -0.065, -0.073]` to `[0.144, 0.065, -0.015]` m. The raised toe spheres contribute to these bounds but not the lowest flat-ground support level.',
        f'- At HOME, the lowest foot surface is **{d["home_nominal_drop_to_first_foot_contact_m"]:.9f} m above flat ground**. With joint angles fixed, geometric first contact occurs at pelvis z **{d["geometric_root_z_for_first_foot_contact_m"]:.9f} m**. Keep the selected 0.68 m initialization; allow settling. This geometric contact height is not a dynamically settled equilibrium.',
        '- With the retained G1/RPL observation organization: state frame **115**, five-frame history **575**, two-camera/two-frame depth **5184**; depth-student lower input **5761**, upper input **575**; lower/upper actions **15/14**. The flat policy uses **575** inputs for each actor.', '',
        '### Ordered joint values', '',
        'The table records original vendor limits; generated simulation waist-yaw stops are adapted as described below. All angles are radians, velocity limits rad/s, effort limits Nm. Kp and Kd are from `agibot_stiff`; HOME values are from `HOME_KEYFRAME`. Head joints are excluded.', '',
        '| Index | Joint | HOME | Lower | Upper | Velocity | Effort | Kp | Kd |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for j in d['dofs']:
        text.append('| ' + ' | '.join([str(j['index']), j['name']] + [f'{j[k]:g}' for k in ('home', 'lower', 'upper', 'velocity', 'effort', 'kp', 'kd')]) + ' |')
    text += ['', '## Simulation starting settings', '',
             'These inherited settings and preparation choices define the nominal simulation baseline; they are not calibrated hardware measurements. Training retains the corresponding G1 randomization and task settings around this baseline. The status page describes runtime validation and the limits of short smoke runs.', '',
             '| Setting | Baseline | Source / interpretation |',
             '| --- | --- | --- |',
             '| Position-action scale | 0.25 rad per action unit | Existing G1/RPL method; no effort/Kp scaling |',
             '| Action clipping | ±100 | Existing G1 setting; this does **not** guarantee joint-position feasibility |',
             '| Torque clipping | Per-joint vendor effort limits above | Applied after explicit PD |',
             '| Armature | 0.03 kg·m² on each joint | Vendor XML default |',
             '| Joint friction loss / passive damping | 0.3 Nm / 0 | Vendor XML; PhysX friction behavior still requires comparison |',
             '| Physics / policy / PD | 200 Hz / 50 Hz / 200 Hz | G1 training baseline, four torque updates per action |',
             '| Baseline command delay | 0 | Deterministic first comparison; G1 upper-body 0–1 policy-step randomization can follow |',
             '| Ground friction / restitution | 1.0 / 0 | RPL terrain baseline |',
             '| Self-collisions | Disabled initially | G1 asset setting |',
             '| Collision shapes | Preserve vendor simplified URDF primitives | Preserve cylinders rather than silently expanding them into capsules |',
             '| Joint soft-limit fraction | 0.95 | G1 configuration; symmetric generated waist-yaw stops yield soft limits +/-2.09741 rad |',
             '| Simulation waist-yaw hard stops | +/-2.2078 rad | Applied to generated URDF, MuJoCo and runtime config; original vendor files retain [-3.43,2.2078] |',
             '| PPO symmetry | Actor coefficient 1, critic coefficient 0 | Unmodified G1 trainer and transforms; no X2 loss override |',
             '| Nominal base height / height-scan offset | 0.68 m / 0.68 m | HOME pelvis height; height-scan offset is a normalization choice |',
             '| Foot-edge samples | front `(0.11,0,0)`, rear `(-0.065,0,0)`, sides `(0.0225,±0.06,0)` m | Approximation from lowest support-sphere centers; existing reward uses base orientation |',
             '| Knee minimum penalty threshold | 0.2 rad | G1 task preference; hard knee minimum remains 0 |',
             '| Low-height termination / gravity XY threshold | 0.3 m / 0.8 | Existing RPL termination baseline |',
             '| Initial arms / hand forces | HOME references / disabled | No motion library needed for the baseline |',
             '| Torso pitch / yaw / roll range | `[-0.5,1.57]` / `[-1,1]` / `[0,0]` rad | RPL stance-gated command curriculum; torso pitch is not a waist-joint target |',
             '| Base-height command delta | `[-0.3,0]` m | RPL stance-gated curriculum, relative to 0.68 m |', '',
             'Per-joint raw-action intervals that keep targets within hard limits are included in the JSON. For example, waist pitch permits about ±1.256 raw action units around HOME at the 0.25 scale. G1\'s ±100 global clip is only an outlier guard; it must not be described as joint-limit protection.', '',
             '## Camera values', '',
             'Retain the selected G1 training rig on `torso_link`: front offset `(0.135,0.06,0.02)` m with rotation `(0,75,0)` degrees; back offset `(-0.125,-0.06,0.02)` m with rotation `(0,75,180)` degrees. Render at 240×135, resize to 48×27, 10 Hz, two-frame policy history, four-frame delay buffer at index 2, clip 0.1–2 m, ray far plane 2.5 m. Preserve the existing noise/pose/FOV randomization and use the X2 mesh instances in the JSON.', '',
             f'Horizontal FOV 101.41° implies vertical FOV **{baseline["camera"]["vertical_fov_from_aspect_deg"]:.6f}°** at this aspect ratio (the preset\'s 69° is its rounded description). MuJoCo camera conversion must also honor Warp\'s optical-frame rotation; copying camera Euler numbers into MJCF directly is insufficient. G1 sim2sim currently uses slightly different offsets and delay, so the selected training rig must be applied explicitly on the X2 sim2sim side.', '',
             'The JSON provides optical rotation matrices, HOME camera positions and MuJoCo `xyaxes` for both views, including the Warp `[-90,0,-90]` degree frame conversion and the MuJoCo −Z viewing direction. Fixed head/sensor mesh-instance transforms are also provided; bake these transforms into meshes or add transform support when attaching them to retained bodies in the current Warp loader.', '',
             '## Validation boundary', '',
             'This report records numerical inputs, not a locomotion-performance result. The runtime status page records which import, controller, export and transport checks have passed. Whole-body balance and terrain traversal require trained policies; joint PD at HOME alone is insufficient. Reward weights and domain-randomization ranges remain G1-derived task hyperparameters. Hardware identification is outside the current scope.', '',
             '## Regeneration and tests', '',
             'The derivation checks gain coverage, HOME within hard limits, agreement of head-excluded URDF/XML joint and actuator order, physical source inertia tensors and referenced visual meshes. Seven tests cover artifact regeneration, joint/pose/action consistency, inertial mass conservation, foot geometry, analytical camera-frame conversion, independent MuJoCo HOME forward kinematics/COM, and in-memory head-fixed model dimensions/mass. These checks establish numerical and model-structure consistency; they do not establish standing stability or policy performance.', '',
             '```bash',
             'python scripts/derive_x2_parameters.py',
             'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest rpl_training/tests/test_x2_parameters.py -q -p no:cacheprovider',
             '```', '',
             'Run from the repository root in the existing `rplmujoco` environment (NumPy, pytest and MuJoCo); see the status page for its absolute interpreter path. External source files are needed for regeneration. Derivation tests skip when the default local sources are absent; MuJoCo-dependent tests skip when MuJoCo is unavailable. Source paths and hashes in the JSON describe this local checkout.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--constants', type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/x2_ultra_parameters.json')
    args = parser.parse_args()
    result = derive(args.assets, args.constants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    args.output.with_suffix('.md').write_text(report(result), encoding='utf-8')
    print(f'Wrote {args.output}: 29 DOFs, {result["derived"]["expected_num_bodies"]} expected collapsed bodies')


if __name__ == '__main__':
    main()
