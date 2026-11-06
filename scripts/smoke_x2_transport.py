"""Run real X2 policy/simulator processes with local DDS and shared depth."""

import argparse
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--conda-root", type=Path, default=Path.home() / ".holosoma_deps/miniconda3")
    args = parser.parse_args()
    for path in (args.flat, args.depth):
        if not path.with_suffix(".onnx").is_file():
            raise FileNotFoundError(path.with_suffix(".onnx"))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    for kind, checkpoint in [("flat", args.flat), ("depth", args.depth)]:
        with (args.report_dir / f"{kind}-sim.log").open("w") as sim_log, (args.report_dir / f"{kind}-policy.log").open(
            "w"
        ) as policy_log:
            sim = subprocess.Popen(
                [
                    str(args.conda_root / "envs/rplmujoco/bin/python"),
                    "-m",
                    "rpl_sim2sim.run_sim",
                    "--config",
                    "rpl_sim2sim/config/x2_loco_manip_cam.yaml",
                    "--headless",
                    "--steps",
                    "1600",
                    "--domain-id",
                    "0",
                    "--shm-name",
                    "x2_runtime_smoke",
                ],
                cwd=repo,
                stdout=sim_log,
                stderr=subprocess.STDOUT,
            )
            try:
                time.sleep(2)
                subprocess.run(
                    [
                        str(args.conda_root / "envs/rplinference/bin/python"),
                        "-m",
                        "rpl_sim2sim.smoke_x2_policy",
                        "--kind",
                        kind,
                        "--model",
                        str(checkpoint.resolve().with_suffix(".onnx")),
                        "--steps",
                        "50",
                        "--report",
                        str((args.report_dir / f"{kind}-sim2sim.json").resolve()),
                    ],
                    cwd=repo,
                    stdout=policy_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=40,
                )
                if sim.wait(timeout=30) != 0:
                    raise RuntimeError(f"{kind} simulator failed; see {sim_log.name}")
            finally:
                if sim.poll() is None:
                    sim.terminate()
                    sim.wait(timeout=10)
        print(f"{kind} ONNX/DDS/depth smoke passed", flush=True)


if __name__ == "__main__":
    main()
