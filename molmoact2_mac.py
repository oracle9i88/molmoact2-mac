"""Shared loader: MolmoAct2 (LeRobot checkpoint) on Apple Silicon via PyTorch MPS.

No model code is modified. Three things differ from the stock CUDA setup:
  1. float32 only  (bfloat16 autocast on MPS gives numerically wrong actions, see README)
  2. the CUDA-graph fast path is switched off
  3. the base checkpoint is read from a local directory instead of the Hub
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # fail loudly instead of silently downloading 22 GB

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import (
    make_molmoact2_pre_post_processors_from_pretrained,
)


def load(ckpt: str, base: str, device: str = "mps", dtype: torch.dtype = torch.float32):
    """Return (policy, preprocessor, postprocessor, config).

    ckpt: local copy of lerobot/MolmoAct2-SO100_101-LeRobot (after prepare.py)
    base: local base directory produced by prepare.py
    """
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.checkpoint_path = base
    cfg.device = device
    cfg.dtype = dtype
    cfg.enable_inference_cuda_graph = False
    policy = MolmoAct2Policy.from_pretrained(ckpt, config=cfg).eval()
    pre, post = make_molmoact2_pre_post_processors_from_pretrained(
        cfg,
        ckpt,
        preprocessor_overrides={
            "molmoact2_pack_inputs": {"checkpoint_path": base},
            "device_processor": {"device": device},
        },
    )
    return policy, pre, post, cfg


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
