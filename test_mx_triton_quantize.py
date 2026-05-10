import unittest

try:
    import torch
    from microxcaling.mx.formats import _get_format_params
    from microxcaling.mx.mx_ops import _quantize_mx
    from mx_triton_conv2d import quantize_mxint8_channel_blocks, quantize_mxint_channel_blocks
except ModuleNotFoundError:
    torch = None
    _get_format_params = None
    _quantize_mx = None
    quantize_mxint_channel_blocks = None
    quantize_mxint8_channel_blocks = None


@unittest.skipIf(torch is None, "torch is required for MXINT8 quantizer tests")
class MXTritonQuantizeTest(unittest.TestCase):
    def test_int16_format_params(self):
        ebits, mbits, emax, _, _ = _get_format_params("int16")

        self.assertEqual(ebits, 0)
        self.assertEqual(mbits, 16)
        self.assertEqual(emax, 0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_mxint8_raw_payload_reconstructs_microxcaling_quantized_values(self):
        torch.manual_seed(0)
        x = torch.randn(2, 35, 4, 4, device="cuda")

        reference = _quantize_mx(
            x,
            scale_bits=8,
            elem_format="int8",
            axes=[1],
            block_size=32,
            round="nearest",
            custom_cuda=False,
        )
        raw = quantize_mxint8_channel_blocks(
            x,
            axis=1,
            block_size=32,
            scale_bits=8,
            round="nearest",
        )

        scale_index = torch.arange(x.shape[1], device=x.device) // raw.block_size
        expanded_exponents = raw.scales.index_select(1, scale_index)
        expanded_scales = torch.exp2(expanded_exponents.float() - 6.0)
        reconstructed = raw.elements.float() * expanded_scales

        self.assertEqual(raw.elements.dtype, torch.int8)
        self.assertFalse(raw.scales.dtype.is_floating_point)
        self.assertTrue(torch.allclose(reconstructed, reference, equal_nan=True))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_mxint16_raw_payload_reconstructs_microxcaling_quantized_values(self):
        torch.manual_seed(0)
        x = torch.randn(2, 35, 4, 4, device="cuda")

        reference = _quantize_mx(
            x,
            scale_bits=8,
            elem_format="int16",
            axes=[1],
            block_size=32,
            round="nearest",
            custom_cuda=False,
        )
        raw = quantize_mxint_channel_blocks(
            x,
            axis=1,
            elem_format="int16",
            block_size=32,
            scale_bits=8,
            round="nearest",
        )

        scale_index = torch.arange(x.shape[1], device=x.device) // raw.block_size
        expanded_exponents = raw.scales.index_select(1, scale_index)
        expanded_scales = torch.exp2(expanded_exponents.float() - 14.0)
        reconstructed = raw.elements.float() * expanded_scales

        self.assertEqual(raw.elements.dtype, torch.int16)
        self.assertFalse(raw.scales.dtype.is_floating_point)
        self.assertEqual(raw.elem_format, "int16")
        self.assertEqual(raw.elem_mbits, 16)
        self.assertTrue(torch.allclose(reconstructed, reference, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
