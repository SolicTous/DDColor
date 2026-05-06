import numpy as np
import tensorflow as tf
import torch
from torch import nn, Tensor
from typing import Optional

from basicsr.archs.ddcolor_arch_utils.unet import NormType, custom_conv_layer

class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.refine_net = nn.Sequential(custom_conv_layer(100 + 3, 2, ks=1, use_activ=False,
                                                          norm_type=NormType.Spectral))
    def forward(self, x, out_feat):
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)
        return out

x = torch.randn(1, 3, 512, 512, requires_grad=True)
out_feat = torch.randn(1, 100, 512, 512, requires_grad=True)
model = Example()
c = model(x, out_feat)
print('torch', c.shape)

inputs = ['input']
outputs = ['output']
torch.onnx.export(model, (x, out_feat), './final.onnx',
                  export_params=True, do_constant_folding=True,  # dynamic_axes=dynamic_axes,
                  input_names=inputs, output_names=outputs, opset_version=12,  # 14
                  verbose=True)