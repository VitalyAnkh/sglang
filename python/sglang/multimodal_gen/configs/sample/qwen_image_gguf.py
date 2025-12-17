# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImageGGUFSamplingParams(SamplingParams):
    """Default sampling params for Qwen-Image GGUF transformer checkpoints.

    Real-Qwen-Image recommends relatively low CFG (≈1.0) and 20-30 steps.
    """

    negative_prompt: str = " "
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 25

