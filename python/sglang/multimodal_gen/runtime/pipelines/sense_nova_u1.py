# SPDX-License-Identifier: Apache-2.0
"""Official SenseNova-U1.5 T2I: 8-step LoRA, no Preview/GGUF/layer-offload."""

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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sense_nova_u1_attn import (
    install_fast_attn_backend,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

OFFICIAL_MODEL_ID = "sensenova/SenseNova-U1.5-8B-MoT"
OFFICIAL_LORA_REPO = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"
OFFICIAL_LORA_FILE = "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"

_REQUIRED_PARAM_PREFIXES = (
    "language_model",
    "fm_modules.fm_head",
    "fm_modules.vision_model_mot_gen",
)
# ~35 GiB bf16 weights; 40 GiB is the resident-GPU cutoff.
_STATIC_MAP_GPU_BYTES = 40 * (1024**3)
_ACTIVATION_HEADROOM_GIB = 4
_MIN_GPU_BUDGET_GIB = 8
_CPU_BUDGET = "48GiB"


def reject_preview_checkpoint(model_path: str) -> None:
    lowered = model_path.lower().replace("_", "-")
    if "u1.5-8b-mot-preview" in lowered:
        raise ValueError(
            "SenseNova-U1.5-8B-MoT-Preview is not supported. "
            f"Use {OFFICIAL_MODEL_ID} with the official 8-step LoRA "
            f"({OFFICIAL_LORA_REPO}/{OFFICIAL_LORA_FILE})."
        )


def resolve_official_8step_lora(lora_path: str | None) -> str | None:
    if lora_path is not None and lora_path.strip().lower() in {"none", "off", ""}:
        return None
    if lora_path:
        return lora_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=OFFICIAL_LORA_REPO, filename=OFFICIAL_LORA_FILE)


def probe_cuda_memory() -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    total = int(torch.cuda.get_device_properties(0).total_memory)
    free, _reported_total = torch.cuda.mem_get_info(0)
    return total, int(free)


def needs_accelerate_dispatch(total_memory_bytes: int | None) -> bool:
    return (
        total_memory_bytes is not None and total_memory_bytes < _STATIC_MAP_GPU_BYTES
    )


def gpu_weight_budget_gib(free_memory_bytes: int | None) -> int:
    if free_memory_bytes is None:
        return _MIN_GPU_BUDGET_GIB
    free_gib = int(free_memory_bytes / (1024**3))
    # Do not floor at 8 GiB: if the card is already occupied, that OOM's dispatch.
    return max(free_gib - _ACTIVATION_HEADROOM_GIB, 1)


def build_hf_load_kwargs(
    *,
    total_memory_bytes: int | None,
    free_memory_bytes: int | None,
) -> dict:
    del free_memory_bytes
    kwargs: dict = {"dtype": torch.bfloat16}
    if total_memory_bytes is None:
        return kwargs
    # Load onto real tensors so LoRA merge can write .data.
    # from_pretrained(device_map="auto") leaves CPU shards on the meta device.
    if needs_accelerate_dispatch(total_memory_bytes):
        kwargs["device"] = "cpu"
        return kwargs
    kwargs["device"] = "cuda"
    return kwargs


def count_matched_lora_targets(
    param_names: list[str], lora_keys: set[str]
) -> tuple[int, int]:
    downs = {key for key in lora_keys if key.endswith(".lora_down.weight")}
    native_prefix = (
        "diffusion_model." if any("diffusion_model." in key for key in lora_keys) else ""
    )
    matched_downs: set[str] = set()
    for name in param_names:
        if not name.endswith(".weight"):
            continue
        mapped = native_prefix + name.replace(".weight", ".lora_down.weight")
        if mapped in downs:
            matched_downs.add(mapped)
    return len(matched_downs), len(downs)


def assert_no_meta_parameters(model: torch.nn.Module) -> None:
    meta = [
        name
        for name, param in model.named_parameters()
        if param.device.type == "meta"
    ]
    if meta:
        raise RuntimeError(
            "Cannot merge LoRA into meta tensors "
            f"({len(meta)} parameters). Load on CPU first."
        )


def load_official_model_and_tokenizer(model_path: str, **load_kwargs):
    from sensenova_u1.utils import load_model_and_tokenizer

    return load_model_and_tokenizer(model_path, **load_kwargs)


def merge_official_lora(model: torch.nn.Module, lora_path: str) -> int:
    from safetensors.torch import safe_open

    assert_no_meta_parameters(model)
    lora_state: dict[str, torch.Tensor] = {}
    with safe_open(lora_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            lora_state[key] = handle.get_tensor(key)
    matched, total = count_matched_lora_targets(
        [name for name, _ in model.named_parameters()],
        set(lora_state),
    )
    if total == 0 or matched != total:
        raise RuntimeError(
            f"8-step LoRA matched {matched}/{total} lora_down tensors; "
            "refusing a partial merge."
        )
    from sensenova_u1.utils.lora import load_and_merge_lora_weight

    load_and_merge_lora_weight(model, lora_state)
    return matched


def colocate_device_map_prefix(
    device_map: dict, prefix: str, *, prefer_gpu: bool = True
) -> dict:
    keys = [key for key in device_map if key == prefix or key.startswith(prefix + ".")]
    if len(keys) <= 1:
        return device_map
    devices = [device_map[key] for key in keys]
    gpuish = {0, "cuda", "cuda:0"}
    if prefer_gpu and any(device in gpuish for device in devices):
        target: int | str = 0
    else:
        target = devices[0]
    colocated = dict(device_map)
    for key in keys:
        colocated[key] = target
    return colocated


def dispatch_after_lora(
    model: torch.nn.Module, *, gpu_budget_gib: int
) -> torch.nn.Module:
    from accelerate import dispatch_model, infer_auto_device_map

    model.tie_weights()
    no_split = list(model._no_split_modules)
    max_memory = {0: f"{gpu_budget_gib}GiB", "cpu": _CPU_BUDGET}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    # t2i_generate adds vision_model_mot_gen embeds to timestep_embedder.
    device_map = colocate_device_map_prefix(device_map, "fm_modules")
    return dispatch_model(model, device_map=device_map)


def _assert_generation_prefixes(model: torch.nn.Module) -> None:
    names = [name for name, _ in model.named_parameters()]
    missing = [
        prefix
        for prefix in _REQUIRED_PARAM_PREFIXES
        if not any(name == prefix or name.startswith(prefix + ".") for name in names)
    ]
    if missing:
        raise RuntimeError(
            "SenseNova U1.5 checkpoint is missing generation prefixes "
            f"{missing}."
        )


class SenseNovaU1Pipeline(ComposedPipelineBase):
    pipeline_name = "SenseNovaU1Pipeline"
    pipeline_config_cls = SenseNovaU1PipelineConfig
    sampling_params_cls = SenseNovaU1SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "SenseNovaU1Pipeline supports same-process execution only."
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules

        reject_preview_checkpoint(server_args.model_path)

        attn_name = install_fast_attn_backend()
        lora_path = resolve_official_8step_lora(server_args.lora_path)
        total_bytes, free_bytes = probe_cuda_memory()
        load_kwargs = build_hf_load_kwargs(
            total_memory_bytes=total_bytes,
            free_memory_bytes=free_bytes,
        )
        logger.info(
            "Loading %s (attn=%s, lora=%s, load_device=%s)",
            server_args.model_path,
            attn_name,
            lora_path,
            load_kwargs.get("device"),
        )
        model, tokenizer = load_official_model_and_tokenizer(
            server_args.model_path, **load_kwargs
        )
        if lora_path:
            n_merged = merge_official_lora(model, lora_path)
            logger.info(
                "Merged %s official 8-step LoRA tensors from %s",
                n_merged,
                lora_path,
            )
        if needs_accelerate_dispatch(total_bytes):
            budget = gpu_weight_budget_gib(free_bytes)
            logger.info(
                "Dispatching after LoRA merge (static device_map, %sGiB GPU)",
                budget,
            )
            model = dispatch_after_lora(model, gpu_budget_gib=budget)
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
