import unittest

try:
    import torch
    from microxcaling.mx.convolution import Conv1d, Conv2d
except ModuleNotFoundError:
    torch = None
    Conv1d = None
    Conv2d = None


INT_OPS_SPECS = {
    "scale_bits": 8,
    "w_elem_format": "int8",
    "a_elem_format": "int8",
    "block_size": 32,
    "bfloat": 16,
    "custom_cuda": False,
    "quantize_backprop": False,
    "int_ops": True,
}

INT16_OPS_SPECS = dict(INT_OPS_SPECS)
INT16_OPS_SPECS["w_elem_format"] = "int16"
INT16_OPS_SPECS["a_elem_format"] = "int16"


@unittest.skipIf(torch is None, "torch is required for INT_OPS Conv2d tests")
class IntOpsConv2dTest(unittest.TestCase):
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_int_ops_conv2d_forward_and_backward_error(self):
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=INT_OPS_SPECS).cuda()
        x = torch.randn(2, 32, 16, 16, device="cuda", requires_grad=True)

        y = conv(x)

        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertEqual(y.dtype, torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "INT_OPS Conv2d backward"):
            y.sum().backward()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_int16_ops_conv2d_forward_and_backward_error(self):
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=INT16_OPS_SPECS).cuda()
        x = torch.randn(2, 32, 16, 16, device="cuda", requires_grad=True)

        y = conv(x)

        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertEqual(y.dtype, torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "INT_OPS Conv2d backward"):
            y.sum().backward()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_int_ops_rejects_unsupported_conv1d(self):
        conv = Conv1d(32, 8, kernel_size=3, padding=1, mx_specs=INT_OPS_SPECS).cuda()
        x = torch.randn(2, 32, 16, device="cuda")

        with self.assertRaisesRegex(ValueError, "Conv2d only"):
            conv(x)

    def test_int_ops_rejects_cpu_conv2d(self):
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=INT_OPS_SPECS)
        x = torch.randn(2, 32, 16, 16)

        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            conv(x)

    def test_int_ops_rejects_non_integer_formats(self):
        specs = dict(INT_OPS_SPECS)
        specs["a_elem_format"] = "fp6_e3m2"
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=specs)
        x = torch.randn(2, 32, 16, 16)

        with self.assertRaisesRegex(ValueError, "only int8 or int16"):
            conv(x)

    def test_int_ops_rejects_mixed_integer_widths(self):
        specs = dict(INT_OPS_SPECS)
        specs["w_elem_format"] = "int16"
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=specs)
        x = torch.randn(2, 32, 16, 16)

        with self.assertRaisesRegex(ValueError, "matching activation and weight"):
            conv(x)

    def test_int_ops_rejects_non_max_shared_exponent(self):
        specs = dict(INT_OPS_SPECS)
        specs["shared_exp_method"] = "none"
        conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=specs)
        x = torch.randn(2, 32, 16, 16)

        with self.assertRaisesRegex(ValueError, "shared_exp_method='max'"):
            conv(x)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_int_ops_rejects_groups_and_dilation(self):
        x = torch.randn(2, 32, 16, 16, device="cuda")

        grouped = Conv2d(32, 32, kernel_size=3, padding=1, groups=32, mx_specs=INT_OPS_SPECS).cuda()
        with self.assertRaisesRegex(ValueError, "groups == 1"):
            grouped(x)

        dilated = Conv2d(32, 8, kernel_size=3, padding=2, dilation=2, mx_specs=INT_OPS_SPECS).cuda()
        with self.assertRaisesRegex(ValueError, "dilation == 1"):
            dilated(x)


if __name__ == "__main__":
    unittest.main()
