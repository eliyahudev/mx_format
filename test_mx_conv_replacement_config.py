import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch.nn as nn
except ModuleNotFoundError:
    nn = None


if nn is None:

    class MXConvReplacementConfigTest(unittest.TestCase):
        @unittest.skip("torch is required for MX conv replacement tests")
        def test_torch_required(self):
            pass

else:

    class FakeMXConv1d(nn.Conv1d):
        def __init__(self, *args, mx_specs, name, **kwargs):
            super().__init__(*args, **kwargs)
            self.mx_specs = mx_specs
            self.mx_name = name


    class FakeMXConv2d(nn.Conv2d):
        def __init__(self, *args, mx_specs, name, **kwargs):
            super().__init__(*args, **kwargs)
            self.mx_specs = mx_specs
            self.mx_name = name


    class FakeMXConv3d(nn.Conv3d):
        def __init__(self, *args, mx_specs, name, **kwargs):
            super().__init__(*args, **kwargs)
            self.mx_specs = mx_specs
            self.mx_name = name


    def fake_finalize_mx_specs(mx_specs):
        finalized = dict(mx_specs)
        finalized["_finalized"] = True
        return finalized


    fake_mx = types.ModuleType("mx")
    fake_mx.Conv1d = FakeMXConv1d
    fake_mx.Conv2d = FakeMXConv2d
    fake_mx.Conv3d = FakeMXConv3d
    fake_mx.finalize_mx_specs = fake_finalize_mx_specs
    sys.modules["mx"] = fake_mx

    if "mx_conv_replacement" in sys.modules:
        del sys.modules["mx_conv_replacement"]
    mx_conv_replacement = importlib.import_module("mx_conv_replacement")


    class TinyConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.features = nn.Sequential(
                nn.Conv2d(4, 5, kernel_size=1),
                nn.ReLU(),
            )


    class MXConvReplacementConfigTest(unittest.TestCase):
        def write_config(self, config):
            temp_dir = tempfile.TemporaryDirectory()
            path = Path(temp_dir.name) / "mx_conv_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.addCleanup(temp_dir.cleanup)
            return path

        def test_config_default_and_exact_layer_overrides(self):
            config_path = self.write_config(
                {
                    "default": {
                        "block_size": 32,
                        "w_elem_format": "fp6_e3m2",
                        "a_elem_format": "fp6_e3m2",
                    },
                    "layers": {
                        "conv1": {"w_elem_format": "fp8_e4m3"},
                        "features.0": {"block_size": 16},
                    },
                }
            )

            converted = mx_conv_replacement.replace_conv_layers_with_mx(
                TinyConvModel(),
                config_path=config_path,
            )

            self.assertIsInstance(converted.conv1, FakeMXConv2d)
            self.assertEqual(converted.conv1.mx_specs["w_elem_format"], "fp8_e4m3")
            self.assertEqual(converted.conv1.mx_specs["block_size"], 32)
            self.assertEqual(converted.conv1.mx_name, "conv1")

            self.assertIsInstance(converted.features[0], FakeMXConv2d)
            self.assertEqual(converted.features[0].mx_specs["w_elem_format"], "fp6_e3m2")
            self.assertEqual(converted.features[0].mx_specs["block_size"], 16)
            self.assertEqual(converted.features[0].mx_name, "0")
            self.assertTrue(converted.features[0].mx_specs["_finalized"])

        def test_unmatched_layer_override_raises(self):
            config_path = self.write_config({"layers": {"missing.conv": {"block_size": 16}}})

            with self.assertRaisesRegex(ValueError, "missing.conv"):
                mx_conv_replacement.replace_conv_layers_with_mx(
                    TinyConvModel(),
                    config_path=config_path,
                )

        def test_mx_specs_api_still_applies_to_all_convs(self):
            converted = mx_conv_replacement.replace_conv_layers_with_mx(
                TinyConvModel(),
                mx_specs={"block_size": 8},
            )

            self.assertEqual(converted.conv1.mx_specs["block_size"], 8)
            self.assertEqual(converted.features[0].mx_specs["block_size"], 8)

        def test_mx_specs_and_config_path_are_mutually_exclusive(self):
            config_path = self.write_config({})

            with self.assertRaisesRegex(ValueError, "either mx_specs or config_path"):
                mx_conv_replacement.replace_conv_layers_with_mx(
                    TinyConvModel(),
                    mx_specs={"block_size": 8},
                    config_path=config_path,
                )


if __name__ == "__main__":
    unittest.main()
