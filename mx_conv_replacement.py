"""Replace PyTorch convolution layers with Microsoft microxcaling MX layers.

This module expects PyTorch and Microsoft's `microxcaling` package to be
installed. Install the MX package from:

    https://github.com/microsoft/microxcaling
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

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
    """Create the finalized MX config used when building replacement convs.

    Start from the default MX settings, apply any caller overrides, then run
    `finalize_mx_specs` so the Microsoft microxcaling layers receive the format
    they expect.
    """
    mx_specs = dict(DEFAULT_MX_SPECS)
    if overrides:
        mx_specs.update(overrides)
    return finalize_mx_specs(mx_specs)


def _load_mx_config(config_path: str | Path) -> tuple[dict, dict[str, dict]]:
    """Load default MX specs and per-layer overrides from a JSON config file."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("MX config JSON must contain an object at the top level")

    default_overrides = config.get("default", {})
    if not isinstance(default_overrides, dict):
        raise ValueError("MX config field 'default' must be an object")

    layer_overrides = config.get("layers", {})
    if not isinstance(layer_overrides, dict):
        raise ValueError("MX config field 'layers' must be an object")

    for layer_name, layer_specs in layer_overrides.items():
        if not isinstance(layer_name, str):
            raise ValueError("MX config layer names must be strings")
        if not isinstance(layer_specs, dict):
            raise ValueError(f"MX config override for layer {layer_name!r} must be an object")

    default_specs = dict(DEFAULT_MX_SPECS)
    default_specs.update(default_overrides)
    return default_specs, layer_overrides


def _copy_conv_parameters(source: nn.Module, target: nn.Module) -> None:
    """Copy learned Conv parameters from the PyTorch layer to the MX layer.

    This keeps replacement behavior close to the original model: same device,
    dtype, weight values, optional bias values, and `requires_grad` flags.
    """
    target.to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)
            target.bias.requires_grad_(source.bias.requires_grad)


def _build_mx_conv(source: nn.Module, mx_specs: dict, name: str) -> nn.Module:
    """Build the matching MX convolution layer for one PyTorch conv module.

    Single-layer replacement step:
    validate unsupported PyTorch options, choose Conv1d/Conv2d/Conv3d from the
    source layer type, construct the MX layer with the same convolution settings,
    copy parameters, and preserve train/eval mode.
    """
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


def _join_module_path(parent_path: str, module_name: str) -> str:
    """Build the full module path used for JSON layer override matching."""
    if parent_path:
        return f"{parent_path}.{module_name}"
    return module_name


def _collect_conv_module_paths(model: nn.Module, *, parent_path: str = "") -> set[str]:
    """Find full module paths for all supported conv layers before replacement."""
    conv_paths: set[str] = set()
    for module_name, child in model.named_children():
        module_path = _join_module_path(parent_path, module_name)
        if type(child) in _CONV_REPLACEMENTS:
            conv_paths.add(module_path)
        else:
            conv_paths.update(_collect_conv_module_paths(child, parent_path=module_path))
    return conv_paths


def _replace_conv_layers_with_mx(
    model: nn.Module,
    *,
    parent_path: str,
    default_specs: dict,
    layer_overrides: dict[str, dict],
) -> nn.Module:
    """Walk a module tree and replace each supported conv with its resolved specs."""
    for module_name, child in list(model.named_children()):
        module_path = _join_module_path(parent_path, module_name)
        if type(child) in _CONV_REPLACEMENTS:
            raw_specs = dict(default_specs)
            if module_path in layer_overrides:
                raw_specs.update(layer_overrides[module_path])
            mx_child = _build_mx_conv(child, finalize_mx_specs(raw_specs), name=module_name)
            setattr(model, module_name, mx_child)
        else:
            _replace_conv_layers_with_mx(
                child,
                parent_path=module_path,
                default_specs=default_specs,
                layer_overrides=layer_overrides,
            )

    return model


def replace_conv_layers_with_mx(
    model: nn.Module,
    mx_specs: dict | None = None,
    *,
    config_path: str | Path | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Replace all Conv1d/Conv2d/Conv3d modules in `model` with MX layers.

    Full model replacement flow:
    optionally deep-copy the model, load one default MX spec set, walk children
    recursively, replace supported convolution modules, and apply exact
    full-path layer overrides from JSON when `config_path` is provided.

    Args:
        model: Any PyTorch module tree.
        mx_specs: A finalized or unfinalized microxcaling config dictionary.
        config_path: Optional JSON file with `default` specs and `layers`
            overrides keyed by full module path, such as `features.0`.
        inplace: If True, mutate `model`. If False, work on a deep copy.

    Returns:
        A model with convolution layers replaced by `mx.Conv*` modules.
    """
    if mx_specs is not None and config_path is not None:
        raise ValueError("Pass either mx_specs or config_path, not both")

    if config_path is None:
        default_specs = mx_specs or DEFAULT_MX_SPECS
        layer_overrides: dict[str, dict] = {}
    else:
        default_specs, layer_overrides = _load_mx_config(config_path)

    unmatched_layers = sorted(set(layer_overrides) - _collect_conv_module_paths(model))
    if unmatched_layers:
        raise ValueError(
            "MX config contains layer overrides that did not match Conv modules: "
            + ", ".join(unmatched_layers)
        )

    converted = model if inplace else deepcopy(model)
    _replace_conv_layers_with_mx(
        converted,
        parent_path="",
        default_specs=default_specs,
        layer_overrides=layer_overrides,
    )

    return converted
