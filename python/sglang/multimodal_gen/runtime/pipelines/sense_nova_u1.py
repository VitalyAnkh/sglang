# SPDX-License-Identifier: Apache-2.0
"""Wrap official SenseNova-U1 T2I (OpenSenseNova/SenseNova-U1 @ f71dfb0)."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sense_nova_u1 import (
    SenseNovaU1PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sense_nova_u1 import SenseNovaU1SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sense_nova_u1 import (
    SenseNovaU1T2IStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_REQUIRED_PARAM_PREFIXES = (
    "language_model",
    "fm_modules.fm_head",
    "fm_modules.vision_model_mot_gen",
)


def _assert_generation_prefixes(model: torch.nn.Module) -> None:
    names = [name for name, _ in model.named_parameters()]
    missing = [
        prefix
        for prefix in _REQUIRED_PARAM_PREFIXES
        if not any(name == prefix or name.startswith(prefix + ".") for name in names)
    ]
    if missing:
        raise RuntimeError(
            "SenseNova U1 checkpoint is missing generation prefixes "
            f"{missing}. Preview GGUF must match this config; do not load "
            "U1 Merger Q4 under a U1.5 Preview config."
        )


class SenseNovaU1Pipeline(ComposedPipelineBase):
    pipeline_name = "SenseNovaU1Pipeline"
    pipeline_config_cls = SenseNovaU1PipelineConfig
    sampling_params_cls = SenseNovaU1SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "SenseNovaU1Pipeline v1 supports same-process execution only."
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules

        import sensenova_u1
        from sensenova_u1.utils import load_model_and_tokenizer

        sensenova_u1.set_attn_backend("sdpa")
        vram_mode = server_args.vram_mode
        if vram_mode is None:
            vram_mode = "balanced" if server_args.gguf_checkpoint else "full"
        for_offload = vram_mode != "full"
        logger.info(
            "Loading SenseNova U1 via official loader (gguf=%s, vram_mode=%s, for_offload=%s)",
            server_args.gguf_checkpoint,
            vram_mode,
            for_offload,
        )
        model, tokenizer = load_model_and_tokenizer(
            server_args.model_path,
            dtype=torch.bfloat16,
            device="cuda",
            gguf_checkpoint=server_args.gguf_checkpoint,
            for_offload=for_offload,
        )
        _assert_generation_prefixes(model)
        return {"model": model, "tokenizer": tokenizer}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            SenseNovaU1T2IStage(
                model=self.get_module("model"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "sense_nova_u1_t2i",
        )


EntryClass = SenseNovaU1Pipeline
