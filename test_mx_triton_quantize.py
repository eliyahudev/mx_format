import unittest

try:
    import torch
    import torch.nn.functional as F
    from microxcaling.mx.formats import _get_format_params
    from microxcaling.mx.mx_ops import _quantize_mx, _reshape_to_blocks, _shared_exponents
    from mx_triton_conv2d import (
        mxint8_conv2d_triton,
        quantize_mxint8_last_axis_blocks,
        quantize_mxint8_channel_blocks,
        quantize_mxint_channel_blocks,
    )
except ModuleNotFoundError:
    torch = None
    F = None
    _get_format_params = None
    _quantize_mx = None
    _reshape_to_blocks = None
    _shared_exponents = None
    mxint8_conv2d_triton = None
    quantize_mxint8_last_axis_blocks = None
    quantize_mxint_channel_blocks = None
    quantize_mxint8_channel_blocks = None


@unittest.skipIf(torch is None, "torch is required for MXINT8 quantizer tests")
class MXTritonQuantizeTest(unittest.TestCase):
    def test_int16_format_params(self):
        ebits, mbits, emax, _, _ = _get_format_params("int16")

        self.assertEqual(ebits, 0)
        self.assertEqual(mbits, 16)
        self.assertEqual(emax, 0)

    def test_physical_last_axis_shared_exponent_differs_by_layout(self):
        x_chw = torch.tensor([
            [[1.0, 12.0], [1.0, 12.0]],
            [[4069.0, 4080.0], [4069.0, 4080.0]],
        ])
        x_hwc = x_chw.permute(1, 2, 0).contiguous()

        chw_max = self._blocked_abs_max_values(x_chw, axis=-1, block_size=2)
        hwc_max = self._blocked_abs_max_values(x_hwc, axis=-1, block_size=2)

        expected_chw = torch.tensor([
            [[12.0], [12.0]],
            [[4080.0], [4080.0]],
        ])
        expected_hwc = torch.tensor([
            [[4069.0], [4080.0]],
            [[4069.0], [4080.0]],
        ])

        self.assertTrue(torch.equal(chw_max, expected_chw))
        self.assertTrue(torch.equal(hwc_max, expected_hwc))

    def _blocked_abs_max_values(self, x, *, axis, block_size):
        blocked, blocked_axes, _, _ = _reshape_to_blocks(x, [axis], block_size)
        shared_exp_axes = [blocked_axis + 1 for blocked_axis in blocked_axes]
        max_values = blocked.abs().max(dim=shared_exp_axes[0], keepdim=True).values

        shared_exp = _shared_exponents(
            blocked,
            method="max",
            axes=shared_exp_axes,
            ebits=0,
        )
        expected_exp = torch.floor(torch.log2(max_values))
        self.assertTrue(torch.equal(shared_exp, expected_exp))

        return max_values.squeeze(shared_exp_axes[0])

    def _reconstruct_last_axis_mxint(self, raw):
        scale_index = torch.arange(raw.elements.shape[-1], device=raw.elements.device) // raw.block_size
        expanded_exponents = raw.scales.index_select(-1, scale_index)
        expanded_scales = torch.exp2(expanded_exponents.float() - (raw.elem_mbits - 2))
        return raw.elements.float() * expanded_scales

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_mxint8_raw_payload_reconstructs_microxcaling_quantized_values(self):
        torch.manual_seed(0)
        x = torch.randn(2, 35, 4, 4, device="cuda")

        reference = _quantize_mx(
            x,
            scale_bits=8,
            elem_format="int8",
            axes=[-1],
            block_size=32,
            round="nearest",
            custom_cuda=False,
        )
        raw = quantize_mxint8_channel_blocks(
            x,
            axis=-1,
            block_size=32,
            scale_bits=8,
            round="nearest",
        )

        scale_index = torch.arange(x.shape[-1], device=x.device) // raw.block_size
        expanded_exponents = raw.scales.index_select(-1, scale_index)
        expanded_scales = torch.exp2(expanded_exponents.float() - 6.0)
        reconstructed = raw.elements.float() * expanded_scales

        self.assertEqual(raw.elements.dtype, torch.int8)
        self.assertFalse(raw.scales.dtype.is_floating_point)
        self.assertEqual(tuple(raw.scales.shape), (2, 35, 4, 1))
        self.assertTrue(torch.allclose(reconstructed, reference, equal_nan=True))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_mxint16_raw_payload_reconstructs_microxcaling_quantized_values(self):
        torch.manual_seed(0)
        x = torch.randn(2, 35, 4, 4, device="cuda")

        reference = _quantize_mx(
            x,
            scale_bits=8,
            elem_format="int16",
            axes=[-1],
            block_size=32,
            round="nearest",
            custom_cuda=False,
        )
        raw = quantize_mxint_channel_blocks(
            x,
            axis=-1,
            elem_format="int16",
            block_size=32,
            scale_bits=8,
            round="nearest",
        )

        scale_index = torch.arange(x.shape[-1], device=x.device) // raw.block_size
        expanded_exponents = raw.scales.index_select(-1, scale_index)
        expanded_scales = torch.exp2(expanded_exponents.float() - 14.0)
        reconstructed = raw.elements.float() * expanded_scales

        self.assertEqual(raw.elements.dtype, torch.int16)
        self.assertFalse(raw.scales.dtype.is_floating_point)
        self.assertEqual(tuple(raw.scales.shape), (2, 35, 4, 1))
        self.assertEqual(raw.elem_format, "int16")
        self.assertEqual(raw.elem_mbits, 16)
        self.assertTrue(torch.allclose(reconstructed, reference, equal_nan=True))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_triton_conv2d_matches_last_axis_reconstruction_nchw(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 5, 5, device="cuda")
        weight = torch.randn(3, 4, 3, 3, device="cuda")
        bias = torch.randn(3, device="cuda")

        x_raw = quantize_mxint8_last_axis_blocks(x, block_size=2)
        w_raw = quantize_mxint8_last_axis_blocks(weight, block_size=2)

        actual = mxint8_conv2d_triton(x_raw, w_raw, bias, padding=1, input_layout="nchw")
        expected = F.conv2d(
            self._reconstruct_last_axis_mxint(x_raw),
            self._reconstruct_last_axis_mxint(w_raw),
            bias,
            padding=1,
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-3, atol=1e-3))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_triton_conv2d_matches_last_axis_reconstruction_nhwc(self):
        torch.manual_seed(0)
        x_nchw = torch.randn(1, 4, 5, 5, device="cuda")
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
        weight = torch.randn(3, 4, 3, 3, device="cuda")
        bias = torch.randn(3, device="cuda")

        x_raw = quantize_mxint8_last_axis_blocks(x_nhwc, block_size=2)
        w_raw = quantize_mxint8_last_axis_blocks(weight, block_size=2)

        actual = mxint8_conv2d_triton(x_raw, w_raw, bias, padding=1, input_layout="nhwc")
        expected = F.conv2d(
            self._reconstruct_last_axis_mxint(x_raw).permute(0, 3, 1, 2).contiguous(),
            self._reconstruct_last_axis_mxint(w_raw),
            bias,
            padding=1,
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
