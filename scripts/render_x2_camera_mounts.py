"""Capture the selected X2 torso camera rig using the vendor MuJoCo meshes.

Requires the rplmujoco environment (MuJoCo, NumPy, Pillow) and an EGL renderer.
Camera housings and frustums are illustrative; positions and orientations use
the derived preparation parameters. No robot assets or presets are modified.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size, bold=False):
    suffix = '-Bold' if bold else ''
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf', size)


def add_geometry(scene, kind, size, pos, mat, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos), np.asarray(mat).reshape(9), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1
    return geom


def connector(scene, start, end, color, width=0.002, arrow=False):
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    geom = add_geometry(scene, kind, [0, 0, 0], [0, 0, 0], np.eye(3), color)
    mujoco.mjv_connector(geom, kind, width, np.asarray(start), np.asarray(end))


def capture(parameters, assets):
    tree = ET.parse(assets / 'x2_ultra.xml')
    xml = tree.getroot()
    xml.find('compiler').set('meshdir', str(assets / 'meshes'))
    head = set(parameters['derived']['fixed_head_joint_names'])
    for parent in xml.iter():
        for node in list(parent):
            if node.get('joint') in head or (node.tag == 'joint' and node.get('name') in head):
                parent.remove(node)
    world = xml.find('worldbody')
    ET.SubElement(world, 'geom', name='preview_floor', type='plane', size='3 3 0.1', rgba='0.91 0.93 0.95 1')
    ET.SubElement(world, 'light', pos='2 -2 3', dir='-1 1 -1', diffuse='0.8 0.8 0.8')
    visual = xml.find('visual')
    if visual is None:
        visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='800', offheight='800')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    data = mujoco.MjData(model)
    data.qpos[:3] = parameters['derived']['home_root_position']
    for joint in parameters['derived']['dofs']:
        data.qpos[model.jnt_qposadr[model.joint(joint['name']).id]] = joint['home']
    mujoco.mj_forward(model, data)
    torso = model.body('torso_link').id
    body_rotation = data.xmat[torso].reshape(3, 3)
    config = parameters['selected_or_inherited_baseline']['camera']
    colors = {'front': [0.0, 0.65, 0.85, 1.0], 'back': [1.0, 0.43, 0.08, 1.0]}
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Collision shapes are hidden; show vendor visuals.
    panels = []
    with mujoco.Renderer(model, height=760, width=620) as renderer:
        for azimuth in [215, 90, 35]:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0, 0, 0.69]
            camera.distance = 1.95
            camera.azimuth = azimuth
            camera.elevation = -8
            renderer.update_scene(data, camera=camera, scene_option=option)
            for name in ['front', 'back']:
                optics = parameters['derived']['g1_camera_derived_optics']['camera_frames'][name]
                rotation = body_rotation @ np.array(optics['optical_rotation_in_torso'])
                point = data.xpos[torso] + body_rotation @ np.array(config[name + '_offset_m'])
                color = colors[name]
                add_geometry(renderer.scene, mujoco.mjtGeom.mjGEOM_BOX, [0.027, 0.012, 0.012], point, rotation, color)
                add_geometry(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.009, 0, 0], point + rotation[:, 2] * 0.014, np.eye(3), [0.08, 0.12, 0.16, 1])
                connector(renderer.scene, point, point + rotation[:, 2] * 0.38, color, width=0.006, arrow=True)
                length = 0.27
                half_x = length * np.tan(np.deg2rad(config['horizontal_fov_deg'] / 2))
                half_y = length * np.tan(np.deg2rad(config['vertical_fov_from_aspect_deg'] / 2))
                corners = [point + rotation @ [x * half_x, y * half_y, length] for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
                for i, corner in enumerate(corners):
                    connector(renderer.scene, point, corner, color)
                    connector(renderer.scene, corner, corners[(i + 1) % 4], color)
            panels.append(Image.fromarray(renderer.render().copy()))
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0'))
    parser.add_argument('--parameters', type=Path, default=ROOT / 'docs/x2_ultra_parameters.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/assets/x2_camera_mounts.png')
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text())
    panels = capture(parameters, args.assets)
    canvas = Image.new('RGB', (1920, 1080), '#f7f9fc')
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 20), 'X2 ULTRA  |  SELECTED TORSO CAMERA RIG', font=font(34, True), fill='#172b40')
    draw.text((40, 67), 'HOME standing pose  /  head fixed  /  front + back depth cameras', font=font(23), fill='#536779')
    for i, (panel, label) in enumerate(zip(panels, ['FRONT THREE-QUARTER', 'SIDE  /  FRONT IS RIGHT', 'REAR THREE-QUARTER'])):
        x = 20 + 640 * i
        canvas.paste(panel, (x, 125))
        draw.text((x + 12, 101), label, font=font(19, True), fill='#172b40')
    draw.rounded_rectangle((35, 898, 64, 927), radius=5, fill='#00a6d9')
    draw.text((78, 898), 'FRONT: torso offset (+0.135, +0.060, +0.020) m', font=font(22), fill='#172b40')
    draw.rounded_rectangle((980, 898, 1009, 927), radius=5, fill='#ff6e14')
    draw.text((1023, 898), 'BACK: torso offset (-0.125, -0.060, +0.020) m', font=font(22), fill='#172b40')
    draw.text((40, 947), 'Arrows: optical axes, 75 deg below horizontal.  Wireframes: 101.41 deg H x 69.01 deg V field of view.', font=font(23), fill='#172b40')
    draw.text((40, 986), 'Offsets use torso axes: +X forward, +Y left, +Z up.  Both mounts are about 0.855 m above ground at HOME.', font=font(22), fill='#536779')
    draw.text((40, 1025), 'MuJoCo capture of vendor meshes. Camera housings and wireframe length are illustrative; this is the proposed simulation rig.', font=font(20), fill='#536779')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
