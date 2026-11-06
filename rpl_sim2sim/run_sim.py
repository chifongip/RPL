"""Run FAR-FALCON-Terrains' sim2sim environment (vendored) for the RPL depth student.

Usage::

    python -m rpl_sim2sim.run_sim                     # FALCON course, Xbox pad on
    python -m rpl_sim2sim.run_sim --no-joystick       # drive the policy from its terminal
    python -m rpl_sim2sim.run_sim --spawn 0 -2 0.8    # start on the floor facing the stairs

Equivalent to FALCON's ``sim2real/run_sim.py task=loco_manip/loco_manip_cam robot=g1/g1_29dof
+obs=loco_manip/g1_29dof_depth task.USE_JOYSTICK=true task.sim_env.config.ROBOT_SCENE=...
scene_g1_29dof_terrains_freebase_real.xml``. Pair it with ``python -m rpl_sim2sim.run_policy``
(DDS domain 0 on ``lo``, depth in ``depth_img_shm``). MuJoCo window: 7/8 raise/lower the band,
9 releases it, 3-6 move its anchor, R toggles video recording.
"""

from __future__ import annotations

import os

# Offscreen (camera / video) renderers on EGL, the passive viewer on GLFW, as FALCON's run_sim.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # type: ignore[import-untyped]  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "loco_manip_cam.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    scene = Path(config["ROBOT_SCENE"])
    config["ROBOT_SCENE"] = str(scene if scene.is_absolute() else PACKAGE_DIR / scene)
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--headless", action="store_true", help="run without a viewer (use --steps for a bounded test)")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many physics steps; 0 runs continuously")
    ap.add_argument("--no-joystick", action="store_true", help="no Xbox pad (USE_JOYSTICK: false)")
    ap.add_argument("--domain-id", type=int, default=None, help="DDS domain (default: the config's, 0)")
    ap.add_argument("--shm-name", default="depth_img_shm", help="depth shared-memory block")
    ap.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="pelvis start position; the band anchor moves above it (FALCON: 2.5 -2 1.893)")  # fmt: skip
    ap.add_argument("--video", default="logs/sim2sim/simulation.mp4", help="where R-toggled video is written")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.headless:
        config["HEADLESS"] = True
        config["USE_RENDERER"] = False
    config["MAX_STEPS"] = args.steps
    if args.no_joystick:
        config["USE_JOYSTICK"] = False
    if args.domain_id is not None:
        config["DOMAIN_ID"] = args.domain_id
    config["IMG_SHM_NAME"] = args.shm_name
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    import mujoco

    from rpl_sim2sim.simulator import FalconLocoManipCamSimulator

    simulator = FalconLocoManipCamSimulator(config, video_path=args.video)
    if args.spawn is not None:
        simulator.mj_data.qpos[:3] = args.spawn
        mujoco.mj_forward(simulator.mj_model, simulator.mj_data)
        simulator.elastic_band.point = np.array([args.spawn[0], args.spawn[1], simulator.elastic_band.point[2]])

    run_simulation(simulator, require_camera=bool(args.steps))


def run_simulation(simulator, *, require_camera=False):
    """Run physics/cameras together and propagate worker failures to the caller."""
    workers = [simulator.sim_thread, simulator.cam_thread]
    if simulator.record_thread:
        workers.append(simulator.record_thread)
    started = []
    try:
        for worker in workers:
            worker.start()
            started.append(worker)
        simulator.sim_thread.join()
    finally:
        simulator.server.stop_event.set()
        if hasattr(simulator.viewer, "close"):
            simulator.viewer.close()
        for worker in started:
            worker.join(timeout=10)
        renderer = getattr(simulator, "_gl_prime_renderer", None)
        if renderer is not None:
            renderer.close()
    if any(worker.is_alive() for worker in started):
        raise RuntimeError("Simulator worker did not stop")
    if hasattr(simulator, "simulation_error"):
        raise simulator.simulation_error
    if simulator.server.error is not None:
        raise RuntimeError("Camera producer failed") from simulator.server.error
    if require_camera and simulator.server.frame_count == 0:
        raise RuntimeError("No camera frames produced during the bounded simulation")


if __name__ == "__main__":
    main()
