import unittest

try:
    import torch
    from microxcaling.mx.mx_ops import _quantize_mx
    from mx_triton_conv2d import quantize_mxint8_channel_blocks
except ModuleNotFoundError:
    torch = None
    _quantize_mx = None
    quantize_mxint8_channel_blocks = None


@unittest.skipIf(torch is None, "torch is required for MXINT8 quantizer tests")
class MXTritonQuantizeTest(unittest.TestCase):
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
        expanded_scales = raw.scales.index_select(1, scale_index)
        reconstructed = raw.elements.float() * expanded_scales

        self.assertTrue(torch.allclose(reconstructed, reference, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
