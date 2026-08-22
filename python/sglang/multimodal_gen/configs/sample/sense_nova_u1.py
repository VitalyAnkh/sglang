# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

# Official examples/t2i/inference.py SUPPORTED_RESOLUTIONS values (width, height).
SENSENOVA_U1_SUPPORTED_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (2048, 2048),
    (2720, 1536),
    (1536, 2720),
    (2496, 1664),
    (1664, 2496),
    (2368, 1760),
    (1760, 2368),
    (1440, 2880),
    (2880, 1440),
    (1152, 3456),
    (3456, 1152),
)


@dataclass
class SenseNovaU1SamplingParams(SamplingParams):
    data_type: DataType = DataType.IMAGE
    num_frames: int = 1
    negative_prompt: str = ""
    guidance_scale: float = 1.0
    num_inference_steps: int = 8
    timestep_shift: float = 3.0
    cfg_norm: str = "none"
    _default_width: ClassVar[int | None] = 2048
    _default_height: ClassVar[int | None] = 2048
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(SENSENOVA_U1_SUPPORTED_RESOLUTIONS)
    )
