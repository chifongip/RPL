"""Independent checks of X2 parameter derivation; no training runtime needed."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('derive_x2_parameters', ROOT / 'scripts/derive_x2_parameters.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DEFAULT_ASSETS, derive = _module.DEFAULT_ASSETS, _module.derive


@pytest.fixture(scope='module')
def parameters():
    if not DEFAULT_ASSETS.is_dir() or not _module.DEFAULT_CONSTANTS.is_file():
        pytest.skip('Local vendor X2 assets and gain/pose references are needed for derivation tests')
    return derive()


def test_committed_parameters_are_reproducible(parameters):
    assert parameters == json.loads((ROOT / 'docs/x2_ultra_parameters.json').read_text())
    assert _module.report(parameters) == (ROOT / 'docs/x2_ultra_parameters.md').read_text()


def test_limits_home_and_name_mapping(parameters):
    derived = parameters['derived']
    dofs = derived['dofs']
    assert len(dofs) == len(set(derived['dof_names'])) == 29
    assert derived['expected_num_bodies'] == 30
    assert len(derived['lower_dof_names']) == 15
    assert len(derived['upper_dof_names']) == 14
    assert derived['lower_dof_names'] + derived['upper_dof_names'] == derived['dof_names']
    by_name = {j['name']: j for j in dofs}
    for j in dofs:
        assert j['lower'] <= j['home'] <= j['upper']
        assert j['effort'] > 0 and j['velocity'] > 0
        assert j['kp'] > 0 and j['kd'] > 0
        other = by_name[j['mirror_joint']]
        assert other['mirror_joint'] == j['name']
        assert other['home'] == pytest.approx(j['mirror_sign'] * j['home'])
        np.testing.assert_allclose(np.array(j['raw_action_interval_without_position_clipping']) * 0.25 + j['home'], [j['lower'], j['upper']])
    assert by_name['waist_pitch_joint']['kp'] == 200
    assert by_name['waist_yaw_joint']['kp'] == 40.18
    assert by_name['right_ankle_pitch_joint']['velocity'] == 13.088
    assert by_name['left_ankle_pitch_joint']['velocity'] == 13.087


def test_head_mass_and_inertial_merge(parameters):
    derived = parameters['derived']
    assert derived['head_mass'] == pytest.approx(1.719334)
    assert derived['urdf_total_mass'] == pytest.approx(41.966521)
    assert derived['fixed_joint_body_owner']['head_pitch_link'] == 'torso_link'
    inertials = derived['merged_body_inertials']
    assert sum(i['mass'] for i in inertials.values()) == pytest.approx(derived['urdf_total_mass'])
    for inertial in inertials.values():
        assert np.linalg.eigvalsh(inertial['inertia_matrix']).min() > 0


def test_support_geometry(parameters):
    feet = list(parameters['derived']['feet'].values())
    for foot in feet:
        assert len(foot['collision_spheres']) == 12
        assert foot['lowest_support_sphere_count'] == 10
        assert foot['sole_z'] == pytest.approx(-0.073)
        np.testing.assert_allclose(foot['collision_bounds_min'], [-0.07, -0.065, -0.073])
        np.testing.assert_allclose(foot['collision_bounds_max'], [0.144, 0.065, -0.015])
    assert feet[0]['home_lowest_collision_z_world'] == pytest.approx(feet[1]['home_lowest_collision_z_world'], abs=1e-7)


def test_camera_frame_conversion(parameters):
    for name, camera in parameters['derived']['g1_camera_derived_optics']['camera_frames'].items():
        optical = np.array(camera['optical_rotation_in_torso'])
        mj_camera = np.array(camera['mujoco_rotation_in_torso'])
        np.testing.assert_allclose(mj_camera.T @ mj_camera, np.eye(3), atol=1e-14)
        assert np.linalg.det(mj_camera) == pytest.approx(1)
        # MuJoCo looks along local -Z; Warp looks along local +Z.
        np.testing.assert_allclose(-mj_camera[:, 2], optical[:, 2], atol=1e-14)
        forward = np.array([np.cos(np.deg2rad(75)), 0, -np.sin(np.deg2rad(75))])
        if name == 'back':
            forward[0] *= -1
        np.testing.assert_allclose(optical[:, 2], forward, atol=1e-14)


def test_fk_against_vendor_mujoco(parameters):
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_ASSETS / 'x2_ultra.xml'))
    data = mujoco.MjData(model)
    data.qpos[:3] = parameters['derived']['home_root_position']
    for j in parameters['derived']['dofs']:
        jid = model.joint(j['name']).id
        data.qpos[model.jnt_qposadr[jid]] = j['home']
    mujoco.mj_forward(model, data)
    # Independent MJCF model checks the URDF transform composition and joint ordering.
    for name, expected in parameters['derived']['home_link_positions_world'].items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0:
            np.testing.assert_allclose(data.xpos[body], expected, atol=2e-6, err_msg=name)
    np.testing.assert_allclose(data.subtree_com[model.body('pelvis').id], parameters['derived']['home_com_world'], atol=2e-5)
    assert model.body_mass.sum() == pytest.approx(parameters['derived']['urdf_total_mass'], abs=5e-5)


def test_head_exclusion_dimensions_and_mass(parameters):
    mujoco = pytest.importorskip('mujoco')
    xml = ET.parse(DEFAULT_ASSETS / 'x2_ultra.xml').getroot()
    xml.find('compiler').set('meshdir', str(DEFAULT_ASSETS / 'meshes'))
    xml.find('compiler').set('fusestatic', 'true')
    for parent in xml.iter():
        for node in list(parent):
            if node.get('joint') in _module.HEAD_JOINTS or (node.tag == 'joint' and node.get('name') in _module.HEAD_JOINTS):
                parent.remove(node)
    # In-memory feasibility check only: the vendor and repository assets are not edited.
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.body_mass.sum() == pytest.approx(parameters['derived']['urdf_total_mass'], abs=5e-5)
