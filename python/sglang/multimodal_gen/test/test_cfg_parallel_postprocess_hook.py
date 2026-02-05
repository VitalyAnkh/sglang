import contextlib
from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.denoising as denoising_mod
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


def test_cfg_parallel_applies_cfg_postprocess_hook(monkeypatch):
    cond = torch.ones((1, 2), dtype=torch.float32)
    uncond = torch.full((1, 2), 2.0, dtype=torch.float32)
    guidance_scale = 7.0
    expected_cfg = guidance_scale * cond + (1.0 - guidance_scale) * uncond

    class FakeCFGGroup:
        world_size = 2

        def broadcast(
            self, _input: torch.Tensor, src: int = 0, _async_op: bool = False
        ):
            if src == 0:
                return cond
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
            return noise_pred + noise_pred_text + noise_pred_uncond

    class DummyStage:
        def _predict_noise(self, **kwargs):
            return uncond if kwargs.get("_which") == "uncond" else cond

    monkeypatch.setattr(denoising_mod, "get_cfg_group", lambda: FakeCFGGroup())
    monkeypatch.setattr(
        denoising_mod,
        "set_forward_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )

    for cfg_rank, expected_partial in (
        (0, guidance_scale * cond),
        (1, (1.0 - guidance_scale) * uncond),
    ):
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

        captured: dict[str, torch.Tensor] = {}

        def fake_all_reduce(partial: torch.Tensor) -> torch.Tensor:
            captured["partial"] = partial
            return expected_cfg

        monkeypatch.setattr(
            denoising_mod,
            "get_classifier_free_guidance_rank",
            lambda cfg_rank=cfg_rank: cfg_rank,
        )
        monkeypatch.setattr(
            denoising_mod,
            "cfg_model_parallel_all_reduce",
            fake_all_reduce,
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
            pos_cond_kwargs={"_which": "cond"},
            neg_cond_kwargs={"_which": "uncond"},
            server_args=server_args,
            guidance=torch.tensor([0.0]),
            latents=torch.zeros_like(cond),
        )

        assert torch.allclose(captured["partial"], expected_partial)
        assert pipeline_config.called == 1
        assert torch.allclose(out, expected_cfg + cond + uncond)
