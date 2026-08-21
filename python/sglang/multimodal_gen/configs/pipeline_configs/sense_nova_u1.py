# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class SenseNovaU1PipelineConfig(PipelineConfig):
    """NEO-Unify T2I wrap: no DiT/VAE contract."""

    task_type: ModelTaskType = ModelTaskType.T2I
    should_use_guidance: bool = True
    enable_autocast: bool = False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            keep_resident_components=(),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )

    def check_pipeline_config(self) -> None:
        return
