import contextlib
from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.denoising as denoising_mod
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


def test_cfg_parallel_applies_cfg_postprocess_hook(monkeypatch):
    # Simulate CFG-parallel rank 0 (cond pass) and ensure postprocess_cfg_noise_pred
    # runs even when CFG-parallel is enabled.
    cond = torch.ones((1, 2), dtype=torch.float32)
    uncond = torch.full((1, 2), 2.0, dtype=torch.float32)
    guidance_scale = 7.0
    expected_cfg = guidance_scale * cond + (1.0 - guidance_scale) * uncond

    class FakeCFGGroup:
        world_size = 2

        def broadcast(
            self, input_: torch.Tensor, src: int = 0, async_op: bool = False
        ):
            del async_op
            if src == 0:
                return input_
            if src == 1:
                return uncond
            raise AssertionError(f"Unexpected src={src}")

    class DummyPipelineConfig:
        def __init__(self):
            self.called = 0

        def slice_noise_pred(self, noise_pred: torch.Tensor, latents: torch.Tensor):
            del latents
            return noise_pred

        def postprocess_cfg_noise_pred(
            self,
            noise_pred: torch.Tensor,
            noise_pred_text: torch.Tensor,
            noise_pred_uncond: torch.Tensor,
            guidance_scale: float,
            *,
            batch=None,
            server_args=None,
        ) -> torch.Tensor:
            del guidance_scale, batch, server_args
            self.called += 1
            # Make the output depend on both cond and uncond to validate that
            # CFG-parallel broadcasts are wired correctly.
            return noise_pred + noise_pred_text + noise_pred_uncond

    pipeline_config = DummyPipelineConfig()
    server_args = SimpleNamespace(
        enable_cfg_parallel=True, pipeline_config=pipeline_config
    )
    batch = SimpleNamespace(
        do_classifier_free_guidance=True,
        is_cfg_negative=False,
        cfg_normalization=0.0,
        guidance_rescale=0.0,
    )

    class DummyStage:
        def _predict_noise(
            self,
            *,
            current_model,
            latent_model_input: torch.Tensor,
            timestep,
            target_dtype,
            guidance: torch.Tensor,
            **kwargs,
        ):
            del (
                current_model,
                latent_model_input,
                timestep,
                target_dtype,
                guidance,
                kwargs,
            )
            return cond

    monkeypatch.setattr(denoising_mod, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(denoising_mod, "get_cfg_group", lambda: FakeCFGGroup())
    monkeypatch.setattr(
        denoising_mod,
        "cfg_model_parallel_all_reduce",
        lambda _partial: expected_cfg,
    )
    monkeypatch.setattr(
        denoising_mod,
        "set_forward_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )

    out = DenoisingStage._predict_noise_with_cfg(
        DummyStage(),
        current_model=None,
        latent_model_input=torch.zeros_like(cond),
        timestep=torch.tensor([0]),
        batch=batch,
        timestep_index=0,
        attn_metadata=None,
        target_dtype=None,
        current_guidance_scale=guidance_scale,
        image_kwargs={},
        pos_cond_kwargs={},
        neg_cond_kwargs={},
        server_args=server_args,
        guidance=torch.tensor([0.0]),
        latents=torch.zeros_like(cond),
    )

    assert pipeline_config.called == 1
    assert torch.allclose(out, expected_cfg + cond + uncond)
