import torch
from microxcaling.mx.convolution import Conv2d

mx_specs = {
    "scale_bits": 8,
    "w_elem_format": "int8",
    "a_elem_format": "int8",
    "block_size": 32,
    "bfloat": 16,
    "custom_cuda": False,
    "quantize_backprop": False,
    "int_ops": True,
    "acc_bits": 32,
}

conv = Conv2d(32, 8, kernel_size=3, padding=1, mx_specs=mx_specs).cuda()
x = torch.randn(2, 32, 16, 16, device="cuda")

y = conv(x)
print(y.shape, y.dtype)
