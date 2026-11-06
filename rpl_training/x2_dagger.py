"""G1 DAgger-MoE with deterministic X2-only teacher checkpoint selection."""

from pathlib import Path

from holosoma.agents.dagger.dagger_ma_moe import DAggerMultiActorMoE
from holosoma.utils.safe_torch_import import torch

from rpl_training.x2 import DOF_NAMES


class X2DAggerMoE(DAggerMultiActorMoE):
    def _resolve_per_terrain_ckpt(self, ckpt_path, terrain_type, wandb_map):
        if wandb_map.get(terrain_type):
            path = super()._resolve_per_terrain_ckpt(ckpt_path, terrain_type, wandb_map)
        else:
            candidates = sorted((Path(ckpt_path) / terrain_type).glob("*/model_*.pt")) if ckpt_path else []
            if not candidates:
                raise FileNotFoundError(f"Provide an X2 teacher at {ckpt_path}/{terrain_type}/RUN/model_XXXXX.pt")
            path = candidates[-1]
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        metadata = (loaded.get("experiment_config") or {}).get("robot", {})
        if metadata.get("dof_names") != DOF_NAMES or metadata.get("asset", {}).get("robot_type") != "x2_29dof":
            raise ValueError(f"{path} is not a compatible X2 teacher; G1 checkpoints cannot train this student")
        return path
