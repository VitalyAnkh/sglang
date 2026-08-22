# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_t2i_generate_kwargs(batch: Req) -> dict[str, Any]:
    """Map SGLang sampling fields onto official t2i_generate kwargs.

    Official 8-step LoRA recipe is cfg_scale=1.0, timestep_shift=3.0,
    num_steps=8. Method defaults (1 / 1 / 30) are not used.
    """
    seed = batch.seed
    if isinstance(seed, list):
        seed = seed[0]
    return {
        "cfg_scale": float(batch.guidance_scale),
        "timestep_shift": float(batch.timestep_shift),
        "enable_timestep_shift": True,
        "cfg_norm": str(batch.cfg_norm),
        "image_size": (int(batch.width), int(batch.height)),
        "num_steps": int(batch.num_inference_steps),
        "seed": int(seed),
        "think_mode": False,
        "cfg_interval": (0.0, 1.0),
        "batch_size": 1,
    }


def invert_n11_to_01(x: torch.Tensor) -> torch.Tensor:
    """Invert official (img-mean)/std to [0, 1] NCHW."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W] or [3,H,W], got {tuple(x.shape)}")
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def denorm_n11_chw_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = invert_n11_to_01(x).permute(0, 2, 3, 1).detach().cpu().float().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in denorm_n11_chw_to_uint8_hwc(batch)]


def _save_if_requested(batch: Req, tensor: torch.Tensor) -> list[str] | None:
    if not batch.save_output:
        return None
    output_path = batch.output_file_path()
    if not output_path:
        raise ValueError("SenseNova U1 T2I requires an output file path")
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _to_pil(tensor)[0].save(output_path)
    logger.info("Saved SenseNova U1 image to %s", output_path)
    return [output_path]


def pack_t2i_output(batch: Req, tensor: torch.Tensor) -> OutputBatch:
    output_paths = _save_if_requested(batch, tensor)
    if batch.save_output and batch.return_file_paths_only:
        return OutputBatch(output_file_paths=output_paths, metrics=batch.metrics)
    return OutputBatch(
        output=invert_n11_to_01(tensor),
        output_file_paths=output_paths,
        metrics=batch.metrics,
    )


class SenseNovaU1T2IStage(PipelineStage):
    """Single T2I stage wrapping official NEOChatModel.t2i_generate."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @property
    def role_affinity(self):
        return RoleType.MONOLITHIC

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        if batch.is_warmup:
            return OutputBatch(output_file_paths=[], metrics=batch.metrics)

        kwargs = build_t2i_generate_kwargs(batch)
        prompt = batch.prompt
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("SenseNova U1 T2I requires a non-empty string prompt")
        tensor = self.model.t2i_generate(self.tokenizer, prompt, **kwargs)
        return pack_t2i_output(batch, tensor)
