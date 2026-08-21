# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sense_nova_u1 import (
    SENSENOVA_U1_SUPPORTED_RESOLUTIONS,
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sense_nova_u1 import (
    build_t2i_generate_kwargs,
    denorm_n11_chw_to_uint8_hwc,
)


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
        self.assertEqual(kwargs["cfg_scale"], 4.0)
        self.assertEqual(kwargs["timestep_shift"], 3.0)
        self.assertEqual(kwargs["num_steps"], 50)
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
