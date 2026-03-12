from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader as text_encoder_loader_mod
import sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader as transformer_loader_mod


def test_text_encoder_loader_customized_path_cleans_config(monkeypatch):
    loader = text_encoder_loader_mod.TextEncoderLoader()
    cleaned: dict[str, object] = {}

    class DummyEncoderConfig:
        def __init__(self):
            self.arch_config = SimpleNamespace()

        def update_model_arch(self, model_config):
            cleaned["updated_model_config"] = dict(model_config)

    monkeypatch.setattr(
        text_encoder_loader_mod,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(model_type="clip_text_model", foo="bar"),
    )
    monkeypatch.setattr(
        text_encoder_loader_mod,
        "get_diffusers_component_config",
        lambda component_path: {"architectures": ["FakeEncoderArch"]},
    )

    def fake_clean_hf_config_inplace(model_config):
        model_config["cleaned"] = True
        cleaned["clean_called"] = True

    monkeypatch.setattr(
        text_encoder_loader_mod,
        "_clean_hf_config_inplace",
        fake_clean_hf_config_inplace,
    )

    captured: dict[str, object] = {}

    def fake_load_model(
        model_path, model_config, server_args, dtype, cpu_offload_flag=None
    ):
        captured["model_path"] = model_path
        captured["model_config"] = model_config
        captured["server_args"] = server_args
        captured["dtype"] = dtype
        captured["cpu_offload_flag"] = cpu_offload_flag
        return "loaded"

    monkeypatch.setattr(loader, "load_model", fake_load_model)

    server_args = SimpleNamespace(
        text_encoder_cpu_offload=False,
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[DummyEncoderConfig()],
            text_encoder_precisions=["bf16"],
        ),
    )

    result = loader.load_customized("/tmp/text-encoder", server_args, "text_encoder")

    assert result == "loaded"
    assert cleaned["clean_called"] is True
    assert cleaned["updated_model_config"] == {
        "architectures": ["FakeEncoderArch"],
        "cleaned": True,
    }
    assert captured["model_path"] == "/tmp/text-encoder"
    assert captured["dtype"] == "bf16"
    assert captured["cpu_offload_flag"] is None
    assert getattr(captured["model_config"].arch_config, "foo") == "bar"


def test_qwen_image_gguf_wrapper_uses_raw_timestep_and_precomputed_freqs():
    class FakePosEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, img_shapes, txt_seq_lens, device=None):
            self.calls.append((img_shapes, txt_seq_lens, device))
            return "generated-pos-embed"

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = FakePosEmbed()
            self.calls = []

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            timestep=None,
            img_shapes=None,
            txt_seq_lens=None,
            guidance=None,
            attention_kwargs=None,
            return_dict=True,
        ):
            rotary = self.pos_embed(
                img_shapes,
                txt_seq_lens,
                device=hidden_states.device,
            )
            self.calls.append(
                {
                    "timestep": timestep,
                    "guidance": guidance,
                    "rotary": rotary,
                    "return_dict": return_dict,
                }
            )
            return (hidden_states + 1.0,)

    inner = FakeInner()
    wrapper = transformer_loader_mod._QwenImageDiffusersTransformerWrapper(inner)

    hidden_states = torch.zeros((1, 2, 3), dtype=torch.float32)
    timestep = torch.tensor([1000], dtype=torch.int64)
    freqs_cis = (torch.randn(4, 8), torch.randn(2, 8))

    out = wrapper(
        hidden_states=hidden_states,
        encoder_hidden_states=torch.zeros((1, 5, 3)),
        encoder_hidden_states_mask=torch.ones((1, 5), dtype=torch.long),
        timestep=timestep,
        img_shapes=[[(1, 16, 16)]],
        txt_seq_lens=[5],
        guidance=torch.tensor([4.0]),
        attention_kwargs={"scale": 1.0},
        freqs_cis=freqs_cis,
    )

    assert torch.equal(out, hidden_states + 1.0)
    assert torch.equal(inner.calls[0]["timestep"], timestep)
    assert inner.calls[0]["rotary"] is freqs_cis
    assert inner.calls[0]["return_dict"] is False
    assert inner.pos_embed.calls == []

    assert inner.pos_embed([[(1, 8, 8)]], [3], device=hidden_states.device) == "generated-pos-embed"
    assert len(inner.pos_embed.calls) == 1
