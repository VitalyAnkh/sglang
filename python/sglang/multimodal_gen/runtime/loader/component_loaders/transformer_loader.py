from copy import deepcopy
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    check_gguf_file,
    get_diffusers_component_config,
    get_gguf_architecture,
    resolve_gguf_diffusion_base_model_id,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        if check_gguf_file(component_model_path):
            component_name = _normalize_component_type(component_name)
            if component_name != "transformer":
                raise ValueError(
                    f"GGUF diffusion checkpoints are only supported for transformer, got: {component_name}"
                )
            return self._load_transformer_gguf(component_model_path, server_args)

        config = get_diffusers_component_config(model_path=component_model_path)
        hf_config = deepcopy(config)
        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path

        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        logger.info(
            "Loading %s from %s safetensors files, default_dtype: %s",
            cls_name,
            len(safetensors_list),
            default_dtype,
        )

        # Load the model using FSDP loader
        assert server_args.hsdp_shard_dim is not None
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": hf_config},
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            # TODO(will): make these configurable
            default_dtype=default_dtype,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        assert (
            next(model.parameters()).dtype == default_dtype
        ), "Model dtype does not match default dtype"

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

        def _maybe_strip_state_dict_prefix(sd: dict[str, Any]) -> dict[str, Any]:
            # Fast path: already aligned.
            if sd and all(k in expected_keys for k in list(sd.keys())[:50]):
                return sd

            prefixes = ("transformer.", "model.transformer.", "model.")
            best: tuple[int, dict[str, Any]] | None = None
            for prefix in prefixes:
                stripped = {
                    k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)
                }
                if not stripped:
                    continue
                score = sum(1 for k in stripped.keys() if k in expected_keys)
                if best is None or score > best[0]:
                    best = (score, stripped)
            if best is not None and best[0] > 0:
                logger.info(
                    "Aligned GGUF state_dict by stripping prefix; matched %s keys.",
                    best[0],
                )
                return best[1]
            return sd

        state_dict = _maybe_strip_state_dict_prefix(state_dict)

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

        class _QwenImageDiffusersTransformerWrapper(nn.Module):
            def __init__(self, inner: QwenImageTransformer2DModel):
                super().__init__()
                self.inner = inner

            def forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
                encoder_hidden_states_mask: torch.Tensor
                | list[torch.Tensor]
                | None = None,
                timestep: torch.Tensor | None = None,
                img_shapes=None,
                txt_seq_lens=None,
                guidance: torch.Tensor | None = None,
                attention_kwargs=None,
                freqs_cis=None,
                **kwargs,
            ) -> torch.Tensor:
                del kwargs, freqs_cis

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

        return _QwenImageDiffusersTransformerWrapper(transformer)
