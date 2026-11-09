"""Build portable, head-fixed X2 assets from the vendor simplified URDF.

Run in rplmujoco; source files remain unchanged. Regenerate parameter preparation
before this command if the selected model/gains/HOME have changed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def build(source, destination):
    parameters = json.loads((ROOT / "docs/x2_ultra_parameters.json").read_text())
    d = parameters["derived"]
    destination.mkdir(parents=True, exist_ok=True)
    mesh_dir = destination / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    urdf = ET.parse(source / "x2_ultra_simple_collision.urdf").getroot()
    urdf.set("name", "x2_ultra_29dof")
    urdf.find("mujoco/compiler").set("meshdir", ".")
    for joint in urdf.findall("joint"):
        if joint.get("name") in d["fixed_head_joint_names"]:
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                element = joint.find(tag)
                if element is not None:
                    joint.remove(element)
        elif joint.get("type") != "fixed":
            limits = parameters["selected_or_inherited_baseline"]["joint_position_limit_overrides_rad"].get(
                joint.get("name")
            )
            if limits is not None:
                joint.find("limit").set("lower", str(limits[0]))
                joint.find("limit").set("upper", str(limits[1]))
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", "0")
            dynamics.set("friction", str(parameters["selected_or_inherited_baseline"]["physx_joint_friction"]))
    for mesh in urdf.findall(".//mesh"):
        filename = Path(mesh.get("filename")).name
        shutil.copy2(source / "meshes" / filename, mesh_dir / filename)
        mesh.set("filename", "meshes/" + filename)
    license_source = source.parent / "LICENSE"
    shutil.copy2(license_source, destination / "LICENSE")
    ET.indent(urdf)
    urdf_path = destination / "x2_ultra_29dof.urdf"
    ET.ElementTree(urdf).write(urdf_path, encoding="utf-8", xml_declaration=True)

    # Convert the same collision topology and restore full inertias below.
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.body("pelvis").add_freejoint(name="floating_base_joint")
    spec.body("pelvis").pos = d["home_root_position"]
    # to_xml() compiles internally. Compiling first mutates fixed-link fusion
    # state, so a second compilation loses head/sensor/hand geom transforms.
    xml = ET.fromstring(spec.to_xml())
    # Serialize full-precision merged URDF inertias explicitly. The URDF
    # importer/to_xml path can omit fixed-head inertia and rounds other values.
    for body in xml.findall("worldbody/.//body"):
        inertia = d["merged_body_inertials"].get(body.get("name"))
        if inertia is None:
            continue
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        tensor = np.array(inertia["inertia_matrix"])
        ET.SubElement(
            body,
            "inertial",
            mass=str(inertia["mass"]),
            pos=" ".join(map(str, inertia["com"])),
            fullinertia=" ".join(
                map(str, [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])
            ),
        )
    xml.find("compiler").set("meshdir", "meshes")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    option = ET.SubElement(xml, "option", timestep="0.005", integrator="implicitfast")
    option.set("iterations", "50")
    for joint in xml.findall("worldbody/.//joint"):
        if joint.get("type") == "free":
            continue
        joint.set("armature", "0.03")
        joint.set("frictionloss", "0.3")
        joint.set("damping", "0")
    for geom in xml.findall("worldbody/.//geom"):
        if geom.get("contype") != "0":
            geom.set("group", "3")
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
            geom.set("friction", "1 0.005 0.0001")
    actuator = ET.SubElement(xml, "actuator")
    for j in d["dofs"]:
        ET.SubElement(
            actuator,
            "motor",
            name="motor_" + j["name"],
            joint=j["name"],
            ctrllimited="true",
            ctrlrange=f"-{j['effort']} {j['effort']}",
        )
    torso = xml.find("worldbody/.//body[@name='torso_link']")
    config = parameters["selected_or_inherited_baseline"]["camera"]
    for name, camera in d["g1_camera_derived_optics"]["camera_frames"].items():
        ET.SubElement(
            torso,
            "camera",
            name=f"cam_{name}_depth",
            pos=" ".join(map(str, config[name + "_offset_m"])),
            xyaxes=" ".join(map(str, camera["mujoco_xyaxes_in_torso"])),
            fovy=str(config["vertical_fov_from_aspect_deg"]),
        )
    ET.SubElement(
        ET.SubElement(xml, "keyframe"),
        "key",
        name="home",
        qpos=" ".join(map(str, d["home_root_position"] + [1, 0, 0, 0] + [j["home"] for j in d["dofs"]])),
    )
    ET.indent(xml)
    ET.ElementTree(xml).write(destination / "x2_ultra_29dof.xml", encoding="utf-8", xml_declaration=True)
    scene = ET.Element("mujoco", model="x2_flat")
    ET.SubElement(scene, "include", file="x2_ultra_29dof.xml")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        contype="2",
        conaffinity="1",
        friction="1 0.005 0.0001",
        rgba="0.8 0.85 0.9 1",
    )
    ET.SubElement(world, "light", pos="0 0 3")
    ET.indent(scene)
    ET.ElementTree(scene).write(destination / "scene.xml", encoding="utf-8")

    # Warp's existing dynamic-mesh loader takes one untransformed mesh per body.
    # Bake visual offsets/scales and fixed-head transforms into each body mesh.
    visuals = {}
    for instance in d["visual_mesh_instances"]:
        mesh = trimesh.load(source / instance["file"], force="mesh")
        mesh.apply_scale(instance["scale"])
        mesh.apply_transform(np.array(instance["transform_in_retained_body"]))
        visuals.setdefault(instance["retained_body"], []).append(mesh)
    camera_mesh_dir = destination / "camera_meshes"
    camera_mesh_dir.mkdir(exist_ok=True)
    for body, meshes in visuals.items():
        trimesh.util.concatenate(meshes).export(camera_mesh_dir / (body + ".stl"))
    (destination / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# X2 Ultra runtime assets\n\n"
        "Generated with `scripts/build_x2_assets.py` from X2_URDF-v1.3.0 simplified collisions. "
        "Head joints are fixed at zero with mass/inertia preserved. MuJoCo is converted from "
        "the same training URDF with symmetric waist-yaw stops at +/-2.2078 rad; "
        "the source vendor files are unchanged. Camera meshes bake fixed-link transforms. Vendor assets are under "
        "the accompanying Mulan PSL v2 LICENSE. Gains and HOME are selected separately in parameters.json.\n"
    )
    model = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert abs(model.body_mass.sum() - d["urdf_total_mass"]) < 1e-8
    assert [model.joint(i).name for i in range(1, model.njnt)] == d["dof_names"]
    print(f"Built {destination}: {model.nbody - 1} robot bodies, {model.nu} motors, mass={model.body_mass.sum():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0"))
    parser.add_argument("--destination", type=Path, default=ROOT / "rpl_training/data/robots/x2")
    args = parser.parse_args()
    build(args.source, args.destination)
