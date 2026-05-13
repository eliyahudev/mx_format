"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import packaging.version as version

from torch.nn import grad
from torch.nn.modules.utils import _single, _pair, _triple

from .mx_ops import _physical_last_axis, quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from mx_triton_conv2d import quantize_mxint_last_axis_blocks, mxint8_conv2d_triton

f_conv1d = torch.nn.functional.conv1d
f_conv2d = torch.nn.functional.conv2d
f_conv3d = torch.nn.functional.conv3d


def conv_weight(
    input, weight_shape, grad_output, stride=1, padding=0, dilation=1, groups=1
):
    """Computes the gradient of conv2d wrt the weight.
    nn.grad.conv2d_weight is bugged in Pytorch < v1.13.0
    This function implements a fix.
    See https://github.com/pytorch/pytorch/issues/51430
    and https://github.com/geohot/tinygrad/commit/8864b373338886a9173d3f823154815535104f28
    """
    num_spatial_dims = input.ndim - 2
    if num_spatial_dims == 1:
        _p = _single
        _conv = f_conv1d
        _conv_weight = grad.conv1d_weight
    elif num_spatial_dims == 2:
        _p = _pair
        _conv = f_conv2d
        _conv_weight = grad.conv2d_weight
    elif num_spatial_dims == 3:
        _p = _triple
        _conv = f_conv3d
        _conv_weight = grad.conv3d_weight
    else:
        raise ValueError(
            "conv_weight does not work with " "input with ndim=%d" % input.ndims
        )

    # For pytorch v1.13.0+, use the built-in convNd_weight.
    # Otherwise use our function
    if version.parse(torch.__version__) >= version.parse("1.13.dev0"):
        return _conv_weight(
            input,
            weight_shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    stride = _p(stride)
    padding = _p(padding)
    dilation = _p(dilation)

    bs = input.shape[0]
    cin = weight_shape[1]
    cout = weight_shape[0]
    assert grad_output.shape[0] == bs
    assert cout % groups == 0

    # Get the spatial dims for each tensor
    sdin = list(input.shape[2:])
    sdw = list(weight_shape[2:])
    sdout = list(grad_output.shape[2:])
    sd1s = [1] * len(sdout)

    grad_output = grad_output.reshape(bs, groups, cout // groups, *sdout).repeat(
        1, 1, cin, *sd1s
    )
    grad_output = grad_output.view(bs * cout * cin, 1, *sdout)

    input = input.reshape(1, bs * groups * cin, *sdin)

    grad_weight = _conv(
        input,
        grad_output,
        stride=dilation,
        padding=padding,
        dilation=stride,
        groups=bs * groups * cin,
    )

    # Sum over the batch dim, preserve current spatial dims
    sdgw = list(grad_weight.shape[2:])
    grad_weight = grad_weight.reshape(bs, -1, *sdgw).sum(dim=0)

    # If stride > 1, we only need to keep a subset
    # of the grad_weight spatial dims
    for i in range(num_spatial_dims):
        if stride[i] > 1:
            grad_weight = grad_weight.narrow(i + 1, 0, sdw[i])

    # Transpose and reshape to final shape
    grad_weight = grad_weight.view(groups, cin, cout // groups, *sdw).transpose(2, 1)
    grad_weight = grad_weight.contiguous().view(groups * cout // groups, cin, *sdw)

    return grad_weight


class ConvFunction(torch.autograd.Function):
    """Note that stride, padding, etc will be stored as
    tuples in torch.nn.Conv2d/Conv3d"""

    @staticmethod
    def _validate_int_ops_conv2d(input, weight, stride, padding, dilation, groups, mx_specs):
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("INT_OPS convolution currently supports Conv2d only")
        input_layout = mx_specs["conv2d_input_layout"]
        if input_layout not in ("nchw", "nhwc"):
            raise ValueError("INT_OPS Conv2d mx_specs['conv2d_input_layout'] must be 'nchw' or 'nhwc'")
        supported_int_formats = ("int8", "int16")
        if mx_specs["a_elem_format"] not in supported_int_formats or mx_specs["w_elem_format"] not in supported_int_formats:
            raise ValueError("INT_OPS Conv2d currently supports only int8 or int16 element formats")
        if mx_specs["shared_exp_method"] != "max":
            raise ValueError("INT_OPS Conv2d currently supports shared_exp_method='max'")
        if input.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("INT_OPS Conv2d requires CUDA input and weight tensors")
        if groups != 1:
            raise ValueError("INT_OPS Conv2d currently supports groups == 1")
        if _pair(dilation) != (1, 1):
            raise ValueError("INT_OPS Conv2d currently supports dilation == 1")
        if mx_specs["block_size"] <= 0:
            raise ValueError("INT_OPS Conv2d requires mx_specs['block_size'] > 0")
        if mx_specs["acc_bits"] <= 0:
            raise ValueError("INT_OPS Conv2d requires mx_specs['acc_bits'] > 0")
        input_channels = input.shape[1]
        if input_channels != weight.shape[1]:
            raise ValueError(f"Input channels ({input_channels}) must match weight channels ({weight.shape[1]})")

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        mx_specs=None,
        name=None,
    ):
        # input: input tensor (minibatch x in_channels x ...)
        # weight: weight tensor (out_channels x in_channels/groups x ...)
        # bias: optional bias tensor of shape (out_channels)

        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name
        ctx.int_ops_forward_only = False

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)
        if num_spatial_dims == 1:
            fwd_func = f_conv1d
            ctx.conv_input = grad.conv1d_input
        elif num_spatial_dims == 2:
            fwd_func = f_conv2d
            ctx.conv_input = grad.conv2d_input
        elif num_spatial_dims == 3:
            fwd_func = f_conv3d
            ctx.conv_input = grad.conv3d_input

        # round with mx_specs['round_output']
        bf_in = quantize_elemwise_op(
            input, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            weight, mx_specs=mx_specs, round=mx_specs["round_weight"]
        )

        if bias is not None:
            bf_bias = quantize_elemwise_op(
                bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )
        else:
            bf_bias = None

        if mx_specs.get("int_ops", False):
            ConvFunction._validate_int_ops_conv2d(
                input, weight, stride, padding, dilation, groups, mx_specs
            )
            input_layout = mx_specs["conv2d_input_layout"]
            bf_in_for_int_ops = bf_in.permute(0, 2, 3, 1).contiguous() if input_layout == "nhwc" else bf_in

            if mx_specs["quantize_backprop"]:
                ctx.save_for_backward(bf_in, bf_weight)
            else:
                ctx.save_for_backward(input, weight)

            x_mx = quantize_mxint_last_axis_blocks(
                bf_in_for_int_ops,
                elem_format=mx_specs["a_elem_format"],
                block_size=mx_specs["block_size"],
                scale_bits=mx_specs["scale_bits"],
                round=mx_specs["round_mx_output"],
                shared_exp_method=mx_specs["shared_exp_method"],
                flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            )
            w_mx = quantize_mxint_last_axis_blocks(
                bf_weight,
                elem_format=mx_specs["w_elem_format"],
                block_size=mx_specs["block_size"],
                scale_bits=mx_specs["scale_bits"],
                round=mx_specs["round_mx_output"],
                shared_exp_method=mx_specs["shared_exp_method"],
                flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            )
            output = mxint8_conv2d_triton(
                x_mx,
                w_mx,
                bf_bias,
                stride=stride,
                padding=padding,
                acc_bits=mx_specs["acc_bits"],
                input_layout=input_layout,
            )
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )
            ctx.int_ops_forward_only = True
            ctx.mx_specs = get_backwards_mx_specs(mx_specs)
            ctx.conv2d_input_layout = input_layout
            return output

        assert input.shape[1] % groups == 0

        # save context after quantize
        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        #####################################################
        # MX conv for output
        #####################################################
        #   input is (batch, in_channels, ...)
        #   weight is (out_channels, in_channels/groups, ..)
        # quantize over the physical last dimension so layout changes affect
        # which values share an MX exponent.
        qid_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs['a_elem_format'],
            axes=[_physical_last_axis(bf_in)],
        )
        qid_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            axes=[_physical_last_axis(bf_weight)],
        )

        # compute output
        output = fwd_func(
            qid_input, qid_weight, bf_bias, stride, padding, dilation, groups
        )

        # element-wise quantize for output
        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        assert grad_output.shape[1] % ctx.groups == 0

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # MX conv for grad_weight
        #####################################################
        #   input is  (batch, in_channels, ...)
        #   output is (batch, out_channels, ...)
        # Quantize over the physical last dimension for layout-dependent MX grouping.
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[_physical_last_axis(input)],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[_physical_last_axis(grad_output)],
        )

        # compute grad_weight
        # don't use nn.grad.conv2d_weight because it is bugged
        grad_weight = conv_weight(
            qex_input,
            weight.shape,
            qex_grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        # element-wise quantize for grad_weight
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # MX conv_transpose for grad_input
        #####################################################
        # grad_input = conv_transpose2d(output, weight)
        #   weight is (out_channels, in_channels/groups, ...)
        #   output is (batch, out_channels, ...)
        # Quantize over the physical last dimension for layout-dependent MX grouping.
        qod_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format'],
            axes=[_physical_last_axis(weight)],
        )
        qod_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[_physical_last_axis(grad_output)],
        )

        # compute grad_input
        grad_input = ctx.conv_input(
            input.shape,
            qod_weight,
            qod_grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        # element-wise quantize for grad_input
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)


def conv1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv1d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


def conv3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv3d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


class Conv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )


class Conv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )


class Conv3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )
