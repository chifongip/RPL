"""Plot configured G1/X2 force segments over projected hand mesh envelopes.

Run with the training environment from the repository root:
    python scripts/plot_force_application_geometry.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull

from rpl_training.config_values.experiment import g1_29dof_rpl_expert
from rpl_training.config_values.x2 import x2_29dof_rpl_expert
from rpl_training.envs.loco_manip_force_manager import LeggedRobotLocoManipForceManager

matplotlib.use("Agg")


def main():
    root = Path(__file__).resolve().parents[1]
    g1_dir = root / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1"
    urdf = ET.parse(g1_dir / "g1_29dof_rev_1_0.urdf").getroot()
    hand_origin = np.fromstring(urdf.find("joint[@name='left_hand_palm_joint']/origin").get("xyz"), sep=" ")
    configs = [
        LeggedRobotLocoManipForceManager._resolve_force_config(cfg)
        for cfg in (g1_29dof_rpl_expert, x2_29dof_rpl_expert)
    ]
    origins = [hand_origin, np.zeros(3)]
    x2_urdf = ET.parse(Path(x2_29dof_rpl_expert.robot.asset.asset_root) / x2_29dof_rpl_expert.robot.asset.urdf_file)
    collision = x2_urdf.find("link[@name='left_wrist_roll_link']/collision")
    box_center = np.fromstring(collision.find("origin").get("xyz"), sep=" ") * 1000
    box_size = np.fromstring(collision.find("geometry/box").get("size"), sep=" ") * 1000
    paths = [
        g1_dir / "meshes/left_rubber_hand.STL",
        root / "rpl_training/data/robots/x2/meshes/left_wrist_roll_link.STL",
    ]
    titles = ["G1 · final wrist-yaw frame", "X2 · final wrist-roll frame"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), facecolor="#f5f7fb")
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.24, top=0.82, wspace=0.18)
    fig.suptitle("G1 and X2: force application sampling", fontsize=22, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.9,
        "Left wrist local XZ projection • identical scale • dimensions in millimetres",
        ha="center",
        fontsize=12,
    )
    for i, (ax, path, title, origin, cfg) in enumerate(zip(axes, paths, titles, origins, configs)):
        hand = origin * 1000
        extension = np.array(cfg["extend_offset"]) * 1000
        lo, hi = cfg["apply_force_pos_ratio_range"]
        start = hand + lo * (extension - hand)
        end = hand + hi * (extension - hand)
        ax.set(xlim=(-65, 220), ylim=(-205, 65), xlabel="local x (mm)", ylabel="local z (mm)")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
        ax.grid(alpha=0.13)
        for spine in ax.spines.values():
            spine.set_visible(False)
        verts = (np.asarray(trimesh.load(path, force="mesh").vertices) + origin) * 1000
        projected = verts[:, [0, 2]]
        hull = ConvexHull(projected)
        ax.add_patch(Polygon(projected[hull.vertices], facecolor="#cdd4df", edgecolor="#a6b1c2", alpha=0.7))
        if i == 1:
            corner = (box_center - box_size / 2)[[0, 2]]
            ax.add_patch(
                Rectangle(corner, box_size[0], box_size[2], facecolor="none", edgecolor="#8b98ab", ls="--", lw=1.4)
            )
            ax.text(43, -108, "Hand collision\nbox (dashed)", fontsize=9, color="#65748c")
        ax.plot([start[0], end[0]], [start[2], end[2]], color="#2374cc", lw=5)
        points = hand + np.linspace(lo, hi, 5)[:, None] * (extension - hand)
        ax.scatter(points[:, 0], points[:, 2], s=32, color="#2374cc", edgecolor="white", linewidth=0.6, zorder=5)
        ax.scatter(0, 0, s=70, color="#17243b", zorder=6)
        ax.annotate(
            "O: wrist", (0, 0), xytext=(-50, 37), fontsize=10, arrowprops={"arrowstyle": "-", "color": "#8290a5"}
        )
        labels = [
            f"{name} = ({point[0]:g}, {point[1]:g}, {point[2]:g})" for name, point in zip(("S", "E"), (start, end))
        ]
        positions = [(38, -40), (112, 30)] if i == 0 else [(45, -24), (45, -181)]
        for point, label, position in zip([start, end], labels, positions):
            ax.annotate(
                label,
                point[[0, 2]],
                xytext=position,
                fontsize=10,
                color="#235d9d",
                arrowprops={"arrowstyle": "-", "color": "#2374cc"},
            )
        if i == 0:
            ax.text(
                50,
                -130,
                "Blue: sampled segment\nGrey: projected hand mesh envelope",
                fontsize=10,
                ha="center",
                color="#536176",
            )
        else:
            length = np.linalg.norm(end - start)
            ax.text(48, -65, f"{length:g} mm sampling segment\ninside the collision box", fontsize=10, color="#536176")
    ranges = [cfg["apply_force_pos_ratio_range"] for cfg in configs]
    fig.text(
        0.07, 0.155, f"P = H + α(E − H);  G1: α ∼ U{ranges[0]},  X2: α ∼ U{ranges[1]}", fontsize=15, color="#235d9d"
    )
    fig.text(
        0.07, 0.105, "α is sampled once per hand per environment; both endpoints follow the wrist pose.", fontsize=11
    )
    fig.text(
        0.07,
        0.065,
        "H: hand-link origin; S: sampling start; E: endpoint. Local y coordinates are shown in the labels.",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        target = root / f"docs/assets/g1_x2_force_geometry.{suffix}"
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
