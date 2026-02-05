import copy
import logging
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    check_gguf_file,
    get_diffusers_component_config,
    get_gguf_architecture,
    resolve_gguf_diffusion_base_model_id,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


def _server_args_for_transformer_component(
    server_args: ServerArgs, component_name: str
) -> ServerArgs:
    """Mask global quantized override flags for secondary transformer components."""
    if component_name != "transformer_2":
        return server_args

    if (
        server_args.transformer_weights_path is None
        and server_args.nunchaku_config is None
    ):
        return server_args

    component_server_args = copy.copy(server_args)
    component_server_args.transformer_weights_path = None
    component_server_args.nunchaku_config = None
    logger.info(
        "Ignoring global transformer_weights_path for %s; keep it on the base "
        "checkpoint unless a per-component override path is provided.",
        component_name,
    )
    return component_server_args


def _maybe_strip_state_dict_prefix(
    state_dict: dict[str, Any],
    expected_keys: set[str],
) -> dict[str, Any]:
    if not state_dict:
        return state_dict

    preview_keys = list(state_dict.keys())[:50]
    if preview_keys and all(k in expected_keys for k in preview_keys):
        return state_dict

    prefixes = ("transformer.", "model.transformer.", "model.")
    best: tuple[int, dict[str, Any]] | None = None
    for prefix in prefixes:
        stripped = {
            k[len(prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if not stripped:
            continue
        score = sum(1 for k in stripped.keys() if k in expected_keys)
        if best is None or score > best[0]:
            best = (score, stripped)

    if best is None or best[0] <= 0:
        return state_dict

    logger.info(
        "Aligned GGUF state_dict by stripping prefix; matched %s keys.",
        best[0],
    )
    return best[1]


class _QwenImageDiffusersTransformerWrapper(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.Tensor | None = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: torch.Tensor | None = None,
        attention_kwargs=None,
        freqs_cis=None,
        **_kwargs,
    ) -> torch.Tensor:
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_mask, list):
            encoder_hidden_states_mask = encoder_hidden_states_mask[0]

        if timestep is not None:
            timestep = timestep.to(dtype=hidden_states.dtype) / 1000.0

        out = self.inner(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )
        return out[0]


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        component_server_args = _server_args_for_transformer_component(
            server_args, component_name
        )
        if check_gguf_file(component_model_path):
            normalized_component_name = _normalize_component_type(component_name)
            if normalized_component_name != "transformer" or component_name != "transformer":
                raise ValueError(
                    "GGUF diffusion checkpoints are only supported for the base "
                    f"transformer component, got: {component_name}"
                )
            return self._load_transformer_gguf(component_model_path, server_args)

        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        safetensors_list = resolve_transformer_safetensors_to_load(
            component_server_args, component_model_path
        )

        # 2. dit config
        # Config from Diffusers supersedes sgl_diffusion's model config
        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path
        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        cls_name = config.pop("_class_name")
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=component_server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            model_cls=model_cls,
            cls_name=cls_name,
        )

        logger.info(
            "Loading %s from %s safetensors file(s) %s, param_dtype: %s",
            cls_name,
            len(safetensors_list),
            f": {safetensors_list}" if get_log_level() == logging.DEBUG else "",
            quant_spec.param_dtype,
        )
        # prepare init_param
        init_params: dict[str, Any] = {
            "config": dit_config,
            "hf_config": config,
            "quant_config": quant_spec.runtime_quant_config,
        }
        if (
            init_params["quant_config"] is None
            and component_server_args.transformer_weights_path is not None
        ):
            logger.warning(
                f"transformer_weights_path provided, but quantization config not resolved, which is unexpected and likely to cause errors"
            )
        else:
            logger.debug("quantization config: %s", init_params["quant_config"])

        # Load the model using FSDP loader
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=component_server_args.dit_cpu_offload,
            pin_cpu_memory=component_server_args.pin_cpu_memory,
            fsdp_inference=component_server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        # post-hooks (e.g., patch scales (nunchaku))
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        # considering the existent of mixed-precision models (e.g., nunchaku)
        if (
            next(model.parameters()).dtype != quant_spec.param_dtype
            and quant_spec.param_dtype
        ):
            logger.warning(
                "Model dtype does not match expected param dtype, %s vs %s",
                next(model.parameters()).dtype,
                quant_spec.param_dtype,
            )

        return model

    def _load_transformer_gguf(
        self, gguf_file: str, server_args: ServerArgs
    ) -> torch.nn.Module:
        """Load a diffusion transformer from a GGUF single file.

        Currently only supports Qwen-Image transformer checkpoints (architecture=qwen_image).
        """
        server_args.model_paths["transformer"] = gguf_file

        arch = get_gguf_architecture(gguf_file)
        base_model_id = resolve_gguf_diffusion_base_model_id(arch, model_ref=gguf_file)
        precision = getattr(server_args.pipeline_config, "dit_precision", "bf16")
        compute_dtype = PRECISION_TO_TYPE.get(str(precision), torch.bfloat16)
        if (
            compute_dtype == torch.bfloat16
            and torch.cuda.is_available()
            and (not torch.cuda.is_bf16_supported())
        ):
            compute_dtype = torch.float16

        from accelerate import init_empty_weights

        from diffusers import QwenImageTransformer2DModel
        from diffusers.models.model_loading_utils import (
            load_gguf_checkpoint,
            load_model_dict_into_meta,
        )
        from diffusers.quantizers.auto import DiffusersAutoQuantizer
        from diffusers.quantizers.gguf.utils import _replace_with_gguf_linear
        from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

        logger.info("Loading Qwen-Image transformer GGUF (arch=%s): %s", arch, gguf_file)
        try:
            state_dict = load_gguf_checkpoint(gguf_file)
        except ModuleNotFoundError as e:  # pragma: no cover
            if getattr(e, "name", None) == "gguf":
                raise ImportError(
                    "GGUF diffusion loading requires the `gguf` Python package."
                ) from e
            raise

        config_dict = QwenImageTransformer2DModel.load_config(
            base_model_id, subfolder="transformer"
        )

        with init_empty_weights():
            transformer = QwenImageTransformer2DModel.from_config(config_dict)

        expected_keys = set(transformer.state_dict().keys())
        state_dict = _maybe_strip_state_dict_prefix(state_dict, expected_keys)

        # Replace Linear layers that have GGUFParameters in the checkpoint.
        _replace_with_gguf_linear(transformer, compute_dtype, state_dict)

        # Prefer GPU for the transformer weights when available.
        device_map = {"": "cuda"} if torch.cuda.is_available() else {"": "cpu"}

        quant_cfg = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        hf_quantizer = DiffusersAutoQuantizer.from_config(quant_cfg, pre_quantized=True)
        hf_quantizer.validate_environment(
            torch_dtype=compute_dtype, from_flax=False, device_map=device_map
        )

        load_model_dict_into_meta(
            transformer,
            state_dict,
            dtype=compute_dtype,
            model_name_or_path=str(gguf_file),
            hf_quantizer=hf_quantizer,
            device_map=device_map,
        )
        del state_dict

        transformer.eval()
        return _QwenImageDiffusersTransformerWrapper(transformer)
