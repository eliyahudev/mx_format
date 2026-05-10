"""Replace PyTorch convolution layers with Microsoft microxcaling MX layers.

This module expects PyTorch and Microsoft's `microxcaling` package to be
installed. Install the MX package from:

    https://github.com/microsoft/microxcaling
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from mx import Conv1d, Conv2d, Conv3d, finalize_mx_specs


DEFAULT_MX_SPECS = {
    "scale_bits": 8,
    "w_elem_format": "fp6_e3m2",
    "a_elem_format": "fp6_e3m2",
    "block_size": 32,
    "bfloat": 16,
    "custom_cuda": True,
    "quantize_backprop": False,
}


_CONV_REPLACEMENTS = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.Conv3d: Conv3d,
}


def make_mx_specs(overrides: dict | None = None) -> dict:
    """Create a finalized MX config for convolution layer replacement."""
    mx_specs = dict(DEFAULT_MX_SPECS)
    if overrides:
        mx_specs.update(overrides)
    return finalize_mx_specs(mx_specs)


def _copy_conv_parameters(source: nn.Module, target: nn.Module) -> None:
    """Copy weights and bias from a PyTorch conv layer to an MX conv layer."""
    target.to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)
            target.bias.requires_grad_(source.bias.requires_grad)


def _build_mx_conv(source: nn.Module, mx_specs: dict, name: str) -> nn.Module:
    """Build the matching MX convolution layer for one PyTorch conv module."""
    if getattr(source, "padding_mode", "zeros") != "zeros":
        raise ValueError(
            "microxcaling Conv layers do not expose PyTorch's non-zero "
            f"padding_mode behavior; layer {name!r} uses "
            f"padding_mode={source.padding_mode!r}."
        )

    mx_cls = _CONV_REPLACEMENTS[type(source)]
    target = mx_cls(
        in_channels=source.in_channels,
        out_channels=source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        mx_specs=mx_specs,
        name=name,
    )
    _copy_conv_parameters(source, target)
    target.train(source.training)
    return target


def replace_conv_layers_with_mx(
    model: nn.Module,
    mx_specs: dict | None = None,
    *,
    inplace: bool = False,
) -> nn.Module:
    """Replace all Conv1d/Conv2d/Conv3d modules in `model` with MX layers.

    Args:
        model: Any PyTorch module tree.
        mx_specs: A finalized or unfinalized microxcaling config dictionary.
        inplace: If True, mutate `model`. If False, work on a deep copy.

    Returns:
        A model with convolution layers replaced by `mx.Conv*` modules.
    """
    converted = model if inplace else deepcopy(model)
    finalized_specs = finalize_mx_specs(mx_specs or DEFAULT_MX_SPECS)

    for module_name, child in list(converted.named_children()):
        if type(child) in _CONV_REPLACEMENTS:
            mx_child = _build_mx_conv(child, finalized_specs, name=module_name)
            setattr(converted, module_name, mx_child)
        else:
            replace_conv_layers_with_mx(child, finalized_specs, inplace=True)

    return converted
