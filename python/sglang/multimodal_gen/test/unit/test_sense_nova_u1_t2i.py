# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sense_nova_u1 import (
    SENSENOVA_U1_SUPPORTED_RESOLUTIONS,
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines.sense_nova_u1 import (
    OFFICIAL_LORA_FILE,
    OFFICIAL_LORA_REPO,
    SenseNovaU1Pipeline,
    assert_no_meta_parameters,
    build_hf_load_kwargs,
    colocate_device_map_prefix,
    count_matched_lora_targets,
    gpu_weight_budget_gib,
    merge_official_lora,
    needs_accelerate_dispatch,
    reject_preview_checkpoint,
    resolve_official_8step_lora,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sense_nova_u1 import (
    build_t2i_generate_kwargs,
    denorm_n11_chw_to_uint8_hwc,
    pack_t2i_output,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sense_nova_u1_attn import (
    AttnDispatch,
    list_attn_candidates,
)


class TestSenseNovaU1PreviewRejected(unittest.TestCase):
    def test_preview_id_raises(self):
        with self.assertRaisesRegex(ValueError, "Preview is not supported"):
            reject_preview_checkpoint("sensenova/SenseNova-U1.5-8B-MoT-Preview")

    def test_hf_cache_preview_raises(self):
        with self.assertRaisesRegex(ValueError, "Preview is not supported"):
            reject_preview_checkpoint(
                "models--sensenova--SenseNova-U1.5-8B-MoT-Preview/snapshots/abc"
            )

    def test_preview_substring_in_unrelated_path_is_ok(self):
        reject_preview_checkpoint(
            "/preview/weights/sensenova/SenseNova-U1.5-8B-MoT"
        )

    def test_official_id_ok(self):
        reject_preview_checkpoint("sensenova/SenseNova-U1.5-8B-MoT")


class TestSenseNovaU1Buckets(unittest.TestCase):
    def test_matches_official_inference_table(self):
        official = {
            "1:1": (2048, 2048),
            "16:9": (2720, 1536),
            "9:16": (1536, 2720),
            "3:2": (2496, 1664),
            "2:3": (1664, 2496),
            "4:3": (2368, 1760),
            "3:4": (1760, 2368),
            "1:2": (1440, 2880),
            "2:1": (2880, 1440),
            "1:3": (1152, 3456),
            "3:1": (3456, 1152),
        }
        self.assertEqual(
            set(SENSENOVA_U1_SUPPORTED_RESOLUTIONS),
            set(official.values()),
        )


class TestSenseNovaU1Kwargs(unittest.TestCase):
    def test_maps_cli_defaults_not_method_defaults(self):
        sampling = SenseNovaU1SamplingParams(
            prompt="a portrait",
            width=1536,
            height=2720,
            seed=42,
        )
        batch = Req(sampling_params=sampling)
        kwargs = build_t2i_generate_kwargs(batch)
        self.assertEqual(kwargs["cfg_scale"], 1.0)
        self.assertEqual(kwargs["timestep_shift"], 3.0)
        self.assertEqual(kwargs["num_steps"], 8)
        self.assertEqual(kwargs["image_size"], (1536, 2720))
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["cfg_norm"], "none")
        self.assertFalse(kwargs["think_mode"])


class TestSenseNovaU1Denorm(unittest.TestCase):
    def test_n11_chw_round_trips_to_uint8_hwc(self):
        chw = torch.tensor(
            [
                [[-1.0, 1.0], [0.0, -1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, -1.0]],
            ]
        )
        hwc = denorm_n11_chw_to_uint8_hwc(chw)
        self.assertEqual(hwc.shape, (1, 2, 2, 3))
        self.assertEqual(hwc.dtype, np.uint8)
        # mean=std=0.5: (-1 -> 0), (0 -> 127.5->128), (1 -> 255)
        np.testing.assert_array_equal(hwc[0, 0, 0], [0, 0, 255])
        np.testing.assert_array_equal(hwc[0, 0, 1], [255, 0, 128])


class TestSenseNovaU1LoRAResolve(unittest.TestCase):
    def test_none_off_empty_skip_lora(self):
        for value in ("none", "off", "OFF", "  ", ""):
            self.assertIsNone(resolve_official_8step_lora(value))

    def test_explicit_path_kept(self):
        self.assertEqual(
            resolve_official_8step_lora("/tmp/custom.safetensors"),
            "/tmp/custom.safetensors",
        )

    def test_default_downloads_official_8step_file(self):
        with patch(
            "huggingface_hub.hf_hub_download",
            return_value="/cache/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors",
        ) as download:
            path = resolve_official_8step_lora(None)
        self.assertEqual(
            path, "/cache/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"
        )
        download.assert_called_once_with(
            repo_id=OFFICIAL_LORA_REPO, filename=OFFICIAL_LORA_FILE
        )


class TestSenseNovaU1LoadKwargs(unittest.TestCase):
    def test_cpu_has_no_gguf_or_offload_flags(self):
        kwargs = build_hf_load_kwargs(
            total_memory_bytes=None, free_memory_bytes=None
        )
        self.assertEqual(kwargs["dtype"], torch.bfloat16)
        self.assertNotIn("gguf_checkpoint", kwargs)
        self.assertNotIn("for_offload", kwargs)
        self.assertNotIn("device_map", kwargs)

    def test_under_40gib_loads_cpu_then_dispatches(self):
        kwargs = build_hf_load_kwargs(
            total_memory_bytes=16 * (1024**3),
            free_memory_bytes=14 * (1024**3),
        )
        self.assertEqual(kwargs["device"], "cpu")
        self.assertNotIn("device_map", kwargs)
        self.assertNotIn("gguf_checkpoint", kwargs)
        self.assertNotIn("for_offload", kwargs)
        self.assertTrue(needs_accelerate_dispatch(16 * (1024**3)))
        self.assertEqual(gpu_weight_budget_gib(14 * (1024**3)), 10)
        self.assertEqual(gpu_weight_budget_gib(6 * (1024**3)), 2)

    def test_40gib_plus_stays_resident_on_cuda(self):
        kwargs = build_hf_load_kwargs(
            total_memory_bytes=80 * (1024**3),
            free_memory_bytes=70 * (1024**3),
        )
        self.assertEqual(kwargs["device"], "cuda")
        self.assertNotIn("device_map", kwargs)
        self.assertFalse(needs_accelerate_dispatch(80 * (1024**3)))


class TestSenseNovaU1LoRAMergeGuard(unittest.TestCase):
    def test_partial_lora_match_is_detectable(self):
        """A missed mot_gen down tensor is not hidden by extra non-LoRA weights."""
        names = [
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
            "language_model.model.embed_tokens.weight",
        ]
        keys = {
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_down.weight",
            "language_model.model.layers.1.self_attn.q_proj_mot_gen.lora_down.weight",
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_up.weight",
            "language_model.model.layers.1.self_attn.q_proj_mot_gen.lora_up.weight",
        }
        matched, total = count_matched_lora_targets(names, keys)
        self.assertEqual((matched, total), (1, 2))
        self.assertNotEqual(len(names), matched)

    def test_native_diffusion_model_prefix_is_mapped(self):
        names = ["layers.0.self_attn.q_proj_mot_gen.weight"]
        keys = {
            "diffusion_model.layers.0.self_attn.q_proj_mot_gen.lora_down.weight",
            "diffusion_model.layers.0.self_attn.q_proj_mot_gen.lora_up.weight",
        }
        matched, total = count_matched_lora_targets(names, keys)
        self.assertEqual((matched, total), (1, 1))

    def test_meta_parameters_refuse_merge(self):
        """8-step merge on meta/Accelerate shards skips weights and ghosts."""
        layer = torch.nn.Linear(2, 2, device="meta")
        with self.assertRaisesRegex(RuntimeError, "meta tensors"):
            assert_no_meta_parameters(layer)
        with self.assertRaisesRegex(RuntimeError, "meta tensors"):
            merge_official_lora(layer, "/unused.safetensors")

    def test_partial_lora_file_refuses_merge(self):
        class _Partial:
            def keys(self):
                return [
                    "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_down.weight",
                    "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_up.weight",
                    "language_model.model.layers.0.self_attn.q_proj_mot_gen.alpha",
                ]

            def get_tensor(self, key):
                return torch.zeros(1)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        model = torch.nn.Linear(2, 2)
        with patch("safetensors.torch.safe_open", return_value=_Partial()):
            with self.assertRaisesRegex(RuntimeError, "matched 0/1"):
                merge_official_lora(model, "/partial.safetensors")

    def test_empty_lora_file_refuses_merge(self):
        class _Empty:
            def keys(self):
                return []

            def get_tensor(self, key):
                raise KeyError(key)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        model = torch.nn.Linear(2, 2)
        with patch("safetensors.torch.safe_open", return_value=_Empty()):
            with self.assertRaisesRegex(RuntimeError, "matched 0/0"):
                merge_official_lora(model, "/empty.safetensors")

    def test_colocate_keeps_fm_modules_on_one_device(self):
        colocated = colocate_device_map_prefix(
            {
                "fm_modules.vision_model_mot_gen": "cpu",
                "fm_modules.timestep_embedder": 0,
                "language_model": "cpu",
            },
            "fm_modules",
        )
        self.assertEqual(colocated["fm_modules.vision_model_mot_gen"], 0)
        self.assertEqual(colocated["fm_modules.timestep_embedder"], 0)
        self.assertEqual(colocated["language_model"], "cpu")

    def test_load_modules_merges_before_dispatch(self):
        """Ghosting came from dispatch-then-merge; load_modules must not swap that."""
        order: list[str] = []
        model = torch.nn.Linear(2, 2)
        tokenizer = object()

        def _load(_path, **_kwargs):
            order.append("load")
            return model, tokenizer

        def _merge(loaded, _path):
            order.append("merge")
            return loaded

        def _dispatch(loaded, *, gpu_budget_gib):
            del gpu_budget_gib
            order.append("dispatch")
            return loaded

        server_args = type("Args", (), {})()
        server_args.model_path = "sensenova/SenseNova-U1.5-8B-MoT"
        server_args.lora_path = "/lora.safetensors"
        pipeline = SenseNovaU1Pipeline.__new__(SenseNovaU1Pipeline)
        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "install_fast_attn_backend",
                return_value="sdpa",
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "resolve_official_8step_lora",
                return_value="/lora.safetensors",
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "probe_cuda_memory",
                return_value=(16 * (1024**3), 14 * (1024**3)),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "load_official_model_and_tokenizer",
                side_effect=_load,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "merge_official_lora",
                side_effect=_merge,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "dispatch_after_lora",
                side_effect=_dispatch,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines.sense_nova_u1."
                "_assert_generation_prefixes",
            ),
        ):
            out = SenseNovaU1Pipeline.load_modules(pipeline, server_args)
        self.assertEqual(order, ["load", "merge", "dispatch"])
        self.assertIs(out["model"], model)
        self.assertIs(out["tokenizer"], tokenizer)


class TestSenseNovaU1AttnSelect(unittest.TestCase):
    def test_falls_to_sdpa_when_flash_factories_empty(self):
        def sdpa(*_args, **_kwargs):
            return "sdpa"

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_lightx2v_fa3",
                return_value=None,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_sglang_fa",
                return_value=None,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_fa2",
                return_value=None,
            ),
        ):
            names = [name for name, _ in list_attn_candidates(sdpa)]
        self.assertEqual(names, ["sdpa"])

    def test_lightx2v_fa3_ranks_first(self):
        def sdpa(*_args, **_kwargs):
            return "sdpa"

        def fa3(*_args, **_kwargs):
            return "fa3"

        def sglang_fa3(*_args, **_kwargs):
            return "sglang_fa3"

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_lightx2v_fa3",
                return_value=fa3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_sglang_fa",
                side_effect=lambda ver: sglang_fa3 if ver == 3 else None,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_fa2",
                return_value=None,
            ),
        ):
            names = [name for name, _ in list_attn_candidates(sdpa)]
        self.assertEqual(names[0], "flash_attn3")
        self.assertEqual(names, ["flash_attn3", "sglang_fa3", "sdpa"])

    def test_candidate_list_keeps_later_fa_for_runtime_fallback(self):
        def sdpa(*_args, **_kwargs):
            return "sdpa"

        def sglang_fa3(*_args, **_kwargs):
            return "sglang_fa3"

        def fa2(*_args, **_kwargs):
            return "fa2"

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_lightx2v_fa3",
                return_value=None,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_sglang_fa",
                side_effect=lambda ver: sglang_fa3 if ver == 3 else None,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.sense_nova_u1_attn._try_fa2",
                return_value=fa2,
            ),
        ):
            names = [name for name, _ in list_attn_candidates(sdpa)]
        self.assertEqual(names, ["sglang_fa3", "flash_attn2", "sdpa"])

    def test_kernel_exception_tries_next_fa_not_sdpa(self):
        calls = []

        def fa3(*_args, **_kwargs):
            calls.append("fa3")
            raise RuntimeError("no hopper")

        def fa2(*_args, **_kwargs):
            calls.append("fa2")
            return "ok"

        def sdpa(*_args, **_kwargs):
            calls.append("sdpa")
            return "sdpa"

        dummy = torch.zeros(1, 1, 1, 1)
        dispatch = AttnDispatch(
            [("flash_attn3", fa3), ("flash_attn2", fa2), ("sdpa", sdpa)],
        )
        out = dispatch(dummy, dummy, dummy)
        self.assertEqual(out, "ok")
        self.assertEqual(calls, ["fa3", "fa2"])
        self.assertEqual(dispatch.name, "flash_attn2")
        out2 = dispatch(dummy, dummy, dummy)
        self.assertEqual(out2, "ok")
        self.assertEqual(calls, ["fa3", "fa2", "fa2"])


class TestSenseNovaU1PackOutput(unittest.TestCase):
    def test_file_path_only_omits_output_tensor(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sampling = SenseNovaU1SamplingParams(
                prompt="p",
                width=2,
                height=2,
                seed=0,
                save_output=True,
                return_file_paths_only=True,
                output_path=tmp,
                output_file_name="out.png",
            )
            batch = Req(sampling_params=sampling)
            packed = pack_t2i_output(batch, torch.zeros(1, 3, 2, 2))
        self.assertIsNone(packed.output)
        self.assertEqual(len(packed.output_file_paths), 1)

    def test_frames_return_fills_output_when_not_file_path_only(self):
        sampling = SenseNovaU1SamplingParams(
            prompt="p",
            width=2,
            height=2,
            seed=0,
            save_output=False,
            return_file_paths_only=False,
        )
        batch = Req(sampling_params=sampling)
        packed = pack_t2i_output(batch, torch.zeros(1, 3, 2, 2))
        self.assertIsNotNone(packed.output)
        self.assertEqual(tuple(packed.output.shape), (1, 3, 2, 2))
        self.assertIsNone(packed.output_file_paths)
        self.assertTrue(torch.all(packed.output == 0.5))
