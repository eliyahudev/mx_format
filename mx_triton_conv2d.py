"""Triton Conv2d over raw MX-style quantized tensors.

This example is intentionally narrower than Microsoft's `microxcaling` layers:
it keeps activations and weights as integer MX elements plus shared exponent
blocks, then applies those exponents inside a Triton convolution accumulator.

The implemented formats are MXINT8/MXINT12/MXINT16-style:
    real_value ~= int_element * 2 ** (shared_exp - (element_bits - 2))

Supported convolution shape for this example:
    - NCHW or NHWC activations
    - OIHW weights
    - groups == 1
    - dilation == 1
    - square or tuple stride/padding

Install requirements in a CUDA environment:
    pip install torch triton
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - import guard for CPU-only envs
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


@dataclass(frozen=True)
class MXIntTensor:
    """Raw MXINT-style tensor: integer elements plus grouped exponents.

    `elements` is the low-precision integer tensor. `scales` keeps its name for
    compatibility, but stores integer shared exponents. A real value is
    represented conceptually as `element * 2 ** (shared_exp - (elem_mbits - 2))`.

    Keeping these pieces separate is the whole point of this experiment: the
    Triton kernel can load raw elements and exponents, apply them inside the
    accumulator, and avoid dequantizing the full tensor before convolution.
    """

    elements: torch.Tensor
    scales: torch.Tensor
    block_size: int
    axis: int
    elem_format: str
    elem_mbits: int


MXInt8Tensor = MXIntTensor


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize Conv2d arguments so later code can always unpack H/W values."""
    if isinstance(value, tuple):
        return value
    return value, value


def quantize_mxint_channel_blocks(
    tensor: torch.Tensor,
    *,
    axis: int,
    elem_format: str = "int8",
    block_size: int = 32,
    scale_bits: int = 8,
    round: str = "nearest",
    shared_exp_method: str = "max",
    flush_fp32_subnorms: bool = False,
) -> MXIntTensor:
    """Quantize a tensor to raw MX integer elements and shared exponents.

    This mirrors microxcaling's `_quantize_mx` path for integer element formats,
    but stops before the final dequantized-float reconstruction. It returns the
    raw signed integer payload and the integer shared exponent needed by the float
    accumulator:

        microxcaling_mx_value == raw_int_element * 2 ** (shared_exp - (mbits - 2))

    The function returns raw integer elements and exponents, not a dequantized
    float tensor. The later Triton kernel tests the accumulator path by doing:

        acc += x_elem * w_elem * exp2(x_exp - offset) * exp2(w_exp - offset)

    For physical-last grouping of NCHW activations use axis=-1, producing
    scales of shape [N, C, H, ceil(W / block_size)].

    For physical-last grouping of internally-transposed NHWC activations use
    axis=-1, producing scales of shape [N, H, W, ceil(C / block_size)].

    For physical-last grouping of OIHW weights use axis=-1, producing scales
    of shape [O, I, H, ceil(W / block_size)].

    Args:
        tensor: CUDA activation or weight tensor to quantize.
        axis: Physical tensor axis to block.
        elem_format: Integer MX element format. Supported: "int8", "int12", or "int16".
        block_size: Number of values sharing one scale along `axis`.
        scale_bits: Number of bits used by the shared MX exponent.
        round: Rounding mode passed to microxcaling's elementwise core.
        shared_exp_method: Shared exponent method; currently expected to be "max".
        flush_fp32_subnorms: Match microxcaling's optional subnormal flush.

    Returns:
        `MXIntTensor` with integer elements and int16 shared exponents.
    """
    if tensor.device.type != "cuda":
        raise ValueError("MX Triton quantization expects a CUDA tensor")
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError("MXINT quantization axis is out of range")
    if elem_format not in ("int8", "int12", "int16"):
        raise ValueError("MX Triton quantization currently supports elem_format 'int8', 'int12', or 'int16'")
    if block_size <= 0:
        raise ValueError("MXINT quantization expects block_size > 0")
    if scale_bits <= 0:
        raise ValueError("MXINT quantization expects scale_bits > 0")

    from microxcaling.mx.elemwise_ops import _quantize_elemwise_core
    from microxcaling.mx.formats import _get_format_params
    from microxcaling.mx.mx_ops import _reshape_to_blocks, _shared_exponents, _undo_reshape_to_blocks

    tensor = tensor.contiguous()
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    blocked, blocked_axes, orig_shape, padded_shape = _reshape_to_blocks(
        tensor,
        [axis],
        block_size,
    )
    shared_exp_axes = [x + 1 for x in blocked_axes]
    shared_exp = _shared_exponents(
        blocked,
        method=shared_exp_method,
        axes=shared_exp_axes,
        ebits=0,
    )

    if flush_fp32_subnorms:
        from microxcaling.mx.formats import FP32_EXPONENT_BIAS

        blocked = blocked * (shared_exp > -FP32_EXPONENT_BIAS).type(blocked.dtype)

    shared_exp = shared_exp - emax
    scale_emax = 2 ** (scale_bits - 1) - 1
    if torch.any(shared_exp > scale_emax):
        raise ValueError("MXINT shared exponent overflow for configured scale_bits")
    shared_exp = torch.clamp(shared_exp, min=-scale_emax)

    normalized = blocked / (2**shared_exp)
    quantized_mx_float = _quantize_elemwise_core(
        normalized,
        mbits,
        ebits,
        max_norm,
        round=round,
        allow_denorm=True,
        saturate_normals=True,
        custom_cuda=False,
    )

    int_scale = 2 ** (mbits - 2)
    raw_min = -(2 ** (mbits - 1) - 1)
    raw_max = 2 ** (mbits - 1) - 1
    raw_dtype = torch.int8 if elem_format == "int8" else torch.int16
    raw_int = torch.clamp(torch.round(quantized_mx_float * int_scale), raw_min, raw_max).to(raw_dtype)
    raw_int = _undo_reshape_to_blocks(raw_int, padded_shape, orig_shape, blocked_axes).contiguous()

    shared_exp = shared_exp.squeeze(shared_exp_axes[0]).to(torch.int16).contiguous()

    return MXIntTensor(
        elements=raw_int,
        scales=shared_exp,
        block_size=block_size,
        axis=axis,
        elem_format=elem_format,
        elem_mbits=mbits,
    )


def quantize_mxint8_channel_blocks(
    tensor: torch.Tensor,
    *,
    axis: int,
    block_size: int = 32,
    scale_bits: int = 8,
    round: str = "nearest",
    shared_exp_method: str = "max",
    flush_fp32_subnorms: bool = False,
) -> MXIntTensor:
    """Backward-compatible wrapper for MXINT8 quantization."""
    return quantize_mxint_channel_blocks(
        tensor,
        axis=axis,
        elem_format="int8",
        block_size=block_size,
        scale_bits=scale_bits,
        round=round,
        shared_exp_method=shared_exp_method,
        flush_fp32_subnorms=flush_fp32_subnorms,
    )


def quantize_mxint_last_axis_blocks(
    tensor: torch.Tensor,
    *,
    elem_format: str = "int8",
    block_size: int = 32,
    scale_bits: int = 8,
    round: str = "nearest",
    shared_exp_method: str = "max",
    flush_fp32_subnorms: bool = False,
) -> MXIntTensor:
    """Quantize raw MXINT blocks over the tensor's physical last axis."""
    return quantize_mxint_channel_blocks(
        tensor,
        axis=-1,
        elem_format=elem_format,
        block_size=block_size,
        scale_bits=scale_bits,
        round=round,
        shared_exp_method=shared_exp_method,
        flush_fp32_subnorms=flush_fp32_subnorms,
    )


def quantize_mxint8_last_axis_blocks(
    tensor: torch.Tensor,
    *,
    block_size: int = 32,
    scale_bits: int = 8,
    round: str = "nearest",
    shared_exp_method: str = "max",
    flush_fp32_subnorms: bool = False,
) -> MXIntTensor:
    """Quantize raw MXINT8 blocks over the tensor's physical last axis."""
    return quantize_mxint_last_axis_blocks(
        tensor,
        elem_format="int8",
        block_size=block_size,
        scale_bits=scale_bits,
        round=round,
        shared_exp_method=shared_exp_method,
        flush_fp32_subnorms=flush_fp32_subnorms,
    )


if triton is not None:

    @triton.jit
    def _mxint8_conv2d_kernel(
        x_q,
        x_s,
        w_q,
        w_s,
        bias,
        out,
        total_outputs: tl.constexpr,
        n: tl.constexpr,
        c: tl.constexpr,
        h: tl.constexpr,
        width: tl.constexpr,
        oc: tl.constexpr,
        kh: tl.constexpr,
        kw: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        pad_h: tl.constexpr,
        pad_w: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
        x_elem_mbits: tl.constexpr,
        w_elem_mbits: tl.constexpr,
        INPUT_IS_NHWC: tl.constexpr,
        ACC_BITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        # Kernel step: dequantize each integer product with its MX block exponents while accumulating.
        # Each Triton program owns BLOCK_M flattened output elements. The next
        # few lines decode each flat output index into N, output-channel,
        # output-row, and output-column coordinates.
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_outputs

        ow_idx = offsets % out_w
        tmp = offsets // out_w
        oh_idx = tmp % out_h
        tmp = tmp // out_h
        oc_idx = tmp % oc
        n_idx = tmp // oc

        acc = tl.zeros((BLOCK_M,), tl.float32)
        acc_min = -(2 ** (ACC_BITS - 1))
        acc_max = (2 ** (ACC_BITS - 1)) - 1

        # Walk over the convolution window and input channels. Padding is
        # handled with masks so invalid input positions contribute zero.
        for r in range(0, kh):
            ih_idx = oh_idx * stride_h + r - pad_h
            valid_h = (ih_idx >= 0) & (ih_idx < h)

            for s in range(0, kw):
                iw_idx = ow_idx * stride_w + s - pad_w
                valid_w = (iw_idx >= 0) & (iw_idx < width)
                spatial_mask = mask & valid_h & valid_w

                for ci in range(0, c):
                    c_block = ci // block_size
                    w_block = s // block_size

                    if INPUT_IS_NHWC:
                        x_offset = ((n_idx * h + ih_idx) * width + iw_idx) * c + ci
                        x_scale_offset = ((n_idx * h + ih_idx) * width + iw_idx) * tl.cdiv(c, block_size) + c_block
                    else:
                        x_offset = ((n_idx * c + ci) * h + ih_idx) * width + iw_idx
                        x_scale_offset = ((n_idx * c + ci) * h + ih_idx) * tl.cdiv(width, block_size) + (iw_idx // block_size)
                    w_offset = ((oc_idx * c + ci) * kh + r) * kw + s

                    w_scale_offset = ((oc_idx * c + ci) * kh + r) * tl.cdiv(kw, block_size) + w_block

                    x_elem = tl.load(x_q + x_offset, mask=spatial_mask, other=0).to(tl.float32)
                    w_elem = tl.load(w_q + w_offset, mask=spatial_mask, other=0).to(tl.float32)
                    x_exp = tl.load(x_s + x_scale_offset, mask=spatial_mask, other=0).to(tl.float32)
                    w_exp = tl.load(w_s + w_scale_offset, mask=spatial_mask, other=0).to(tl.float32)
                    x_scale = tl.exp((x_exp - (x_elem_mbits - 2)) * 0.6931471805599453)
                    w_scale = tl.exp((w_exp - (w_elem_mbits - 2)) * 0.6931471805599453)

                    # Core accumulator behavior:
                    #   real_x ~= x_elem * exp2(x_shared_exp - (x_elem_mbits - 2))
                    #   real_w ~= w_elem * exp2(w_shared_exp - (w_elem_mbits - 2))
                    #   acc += real_x * real_w
                    acc_man = x_elem * w_elem
                    tl.device_assert(
                        (~spatial_mask) | ((acc_man >= acc_min) & (acc_man <= acc_max)),
                        "MXINT Conv2d product accumulator overflow",
                    )
                    acc_next = acc + acc_man * x_scale * w_scale
                    tl.device_assert(
                        (~spatial_mask) | ((acc_next >= acc_min) & (acc_next <= acc_max)),
                        "MXINT Conv2d running accumulator overflow",
                    )
                    acc = acc_next

        if has_bias:
            acc += tl.load(bias + oc_idx, mask=mask, other=0.0)

        tl.store(out + offsets, acc, mask=mask)


def mxint8_conv2d_triton(
    x_mx: MXIntTensor,
    w_mx: MXIntTensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    acc_bits: int = 32,
    block_m: int = 128,
    input_layout: str = "nchw",
) -> torch.Tensor:
    """Run Conv2d on raw MX integer elements and scales with a Triton accumulator.

    At this point activations and weights are already quantized. This function
    validates the MX tensors, computes the output shape, launches the Triton
    kernel, and returns a float32 output tensor.

    This is the public low-level API for accumulator experiments. It does not
    call `torch.nn.functional.conv2d`, and it does not reconstruct full float
    activation or weight tensors before computing.

    Args:
        x_mx: Quantized activation tensor. Public Conv2d inputs are NCHW;
            `input_layout="nhwc"` expects this tensor to have been internally
            transposed to NHWC before quantization. Activations and weights
            must be grouped over their physical last dimension.
        w_mx: Quantized OIHW weight tensor.
        bias: Optional float bias with one value per output channel.
        stride: Conv2d stride as an int or `(height, width)` tuple.
        padding: Conv2d zero padding as an int or `(height, width)` tuple.
        acc_bits: Signed accumulator bit width checked by Triton device asserts.
        block_m: Number of output elements computed by one Triton program.
        input_layout: Activation layout, either `"nchw"` or `"nhwc"`.

    Returns:
        Float32 convolution output with shape `[N, out_channels, out_h, out_w]`.
    """
    if triton is None:
        raise ImportError("Triton is required for mxint8_conv2d_triton") from _TRITON_IMPORT_ERROR
    if x_mx.elements.device.type != "cuda" or w_mx.elements.device.type != "cuda":
        raise ValueError("MX Triton convolution expects CUDA tensors")
    x_expected_dtype = torch.int8 if x_mx.elem_format == "int8" else torch.int16
    w_expected_dtype = torch.int8 if w_mx.elem_format == "int8" else torch.int16
    if x_mx.elements.dtype != x_expected_dtype:
        raise ValueError(f"MX Triton convolution expects activation {x_mx.elem_format} element tensors")
    if w_mx.elements.dtype != w_expected_dtype:
        raise ValueError(f"MX Triton convolution expects weight {w_mx.elem_format} element tensors")
    if x_mx.scales.dtype.is_floating_point or w_mx.scales.dtype.is_floating_point:
        raise ValueError("MX Triton convolution expects integer shared exponent tensors")
    if x_mx.block_size != w_mx.block_size:
        raise ValueError("Activation and weight block sizes must match")
    if input_layout not in ("nchw", "nhwc"):
        raise ValueError("MX Triton convolution input_layout must be 'nchw' or 'nhwc'")
    input_is_nhwc = input_layout == "nhwc"
    if x_mx.axis != x_mx.elements.ndim - 1 or w_mx.axis != w_mx.elements.ndim - 1:
        raise ValueError("MX Triton convolution expects tensors quantized along the physical last axis")
    if acc_bits <= 0:
        raise ValueError("MX Triton convolution expects acc_bits > 0")

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    if input_is_nhwc:
        n, h, width, c = x_mx.elements.shape
    else:
        n, c, h, width = x_mx.elements.shape
    oc, weight_c, kh, kw = w_mx.elements.shape
    if c != weight_c:
        raise ValueError(f"Input channels ({c}) must match weight channels ({weight_c})")

    out_h = (h + 2 * pad_h - kh) // stride_h + 1
    out_w = (width + 2 * pad_w - kw) // stride_w + 1
    out = torch.empty((n, oc, out_h, out_w), device=x_mx.elements.device, dtype=torch.float32)
    total_outputs = out.numel()
    bias_arg = bias if bias is not None else out

    grid = (triton.cdiv(total_outputs, block_m),)
    _mxint8_conv2d_kernel[grid](
        x_mx.elements,
        x_mx.scales,
        w_mx.elements,
        w_mx.scales,
        bias_arg,
        out,
        total_outputs,
        n,
        c,
        h,
        width,
        oc,
        kh,
        kw,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        bias is not None,
        x_mx.block_size,
        x_mx.elem_mbits,
        w_mx.elem_mbits,
        INPUT_IS_NHWC=input_is_nhwc,
        ACC_BITS=acc_bits,
        BLOCK_M=block_m,
    )
    return out


class MXInt8TritonConv2d(nn.Module):
    """Drop-in Conv2d wrapper that quantizes inputs/weights before Triton conv.

    Use `from_conv2d` to replace a supported `nn.Conv2d` while preserving the
    original layer's weights, bias, stride, padding, and train/eval state.

    The module stores normal floating-point PyTorch parameters. During
    `forward`, it quantizes the current activation tensor and current weight
    tensor to `MXInt8Tensor`, calls the Triton accumulator kernel, and returns
    the final float32 output. This is meant for inference-style accumulator
    experiments, not as a full training-ready layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        block_size: int = 32,
    ) -> None:
        """Create the wrapper module with Conv2d-shaped parameters.

        The parameters remain floating-point here. Quantization happens inside
        `forward` so the wrapper can be built directly from an existing Conv2d.
        """
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, *, block_size: int = 32) -> "MXInt8TritonConv2d":
        """Convert one supported `nn.Conv2d` into the MXINT8 Triton wrapper.

        Replacement step for a single layer:
        validate the Conv2d options this example supports, construct the wrapper,
        copy weights/bias, and keep the source module's training mode.

        Supported options are intentionally narrow: `groups == 1`, dilation of
        1, and zero padding. Unsupported settings raise immediately so a model
        conversion does not silently change convolution semantics.
        """
        if conv.groups != 1:
            raise ValueError("MXInt8TritonConv2d currently supports groups == 1")
        if _pair(conv.dilation) != (1, 1):
            raise ValueError("MXInt8TritonConv2d currently supports dilation == 1")
        if getattr(conv, "padding_mode", "zeros") != "zeros":
            raise ValueError("MXInt8TritonConv2d currently supports zero padding only")

        module = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
            block_size=block_size,
        )
        module.to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(conv.weight)
            module.weight.requires_grad_(conv.weight.requires_grad)
            if conv.bias is not None and module.bias is not None:
                module.bias.copy_(conv.bias)
                module.bias.requires_grad_(conv.bias.requires_grad)
        module.train(conv.training)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation and weight tensors, then run the MXINT8 conv.

        The output is already finalized as float32 because the Triton kernel
        accumulates into float32 and stores a float output. The intermediate
        activation and weight tensors remain raw int8+scale pairs.
        """
        # Step 1: quantize the current activation tensor into int8 blocks.
        x_mx = quantize_mxint8_last_axis_blocks(x, block_size=self.block_size)
        # Step 2: quantize this layer's floating-point weights the same way.
        w_mx = quantize_mxint8_last_axis_blocks(self.weight, block_size=self.block_size)
        # Step 3: call the Triton convolution that applies scales during accumulation.
        return mxint8_conv2d_triton(
            x_mx,
            w_mx,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


def replace_conv2d_with_mxint8_triton(
    model: nn.Module,
    *,
    block_size: int = 32,
    inplace: bool = False,
) -> nn.Module:
    """Replace supported `nn.Conv2d` layers with Triton MXINT8 accumulator layers.

    Model replacement flow:
    optionally deep-copy the model, walk every child module recursively, replace
    each supported Conv2d with `MXInt8TritonConv2d.from_conv2d`, and leave all
    non-convolution modules unchanged.

    Args:
        model: Any PyTorch module tree.
        block_size: Number of values sharing one MX scale along the last axis.
        inplace: If true, mutate `model`; otherwise convert a deep copy.

    Returns:
        A model with supported Conv2d layers replaced by Triton MX wrappers.
    """
    converted = model if inplace else deepcopy(model)

    for module_name, child in list(converted.named_children()):
        if type(child) is nn.Conv2d:
            setattr(
                converted,
                module_name,
                MXInt8TritonConv2d.from_conv2d(child, block_size=block_size),
            )
        else:
            replace_conv2d_with_mxint8_triton(child, block_size=block_size, inplace=True)

    return converted


def example_usage() -> None:
    """Minimal smoke-test showing quantize -> MX conv -> compare with fp32 conv.

    This is a quick sanity check for a CUDA environment. It creates one input,
    one weight tensor, and one bias vector; runs the raw MXINT8 Triton path; and
    compares the final float output to ordinary PyTorch fp32 convolution.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")

    torch.manual_seed(0)
    x = torch.randn(1, 32, 16, 16, device="cuda")
    weight = torch.randn(8, 32, 3, 3, device="cuda")
    bias = torch.randn(8, device="cuda")

    x_mx = quantize_mxint8_last_axis_blocks(x, block_size=32)
    w_mx = quantize_mxint8_last_axis_blocks(weight, block_size=32)

    y_mx = mxint8_conv2d_triton(x_mx, w_mx, bias, stride=1, padding=1)
    y_ref = F.conv2d(x, weight, bias, stride=1, padding=1)

    print("output shape:", tuple(y_mx.shape))
    print("mean abs diff vs fp32 conv:", (y_ref - y_mx).abs().mean().item())


if __name__ == "__main__":
    example_usage()
