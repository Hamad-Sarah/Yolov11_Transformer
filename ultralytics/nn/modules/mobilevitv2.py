# ultralytics/nn/modules/mobilevitv2.py
import torch
import torch.nn as nn
from cvnets.cvnets.modules.mobilenetv2 import InvertedResidual as CVNetsInvRes
from cvnets.cvnets.modules.mobilevit_block import MobileViTBlockv2 as CVNetsMV2
from ultralytics.nn.modules.conv import Conv

class CVNetsConvLayer2d(nn.Module):
    """Alias Ultralytics Conv → BN → SiLU block for YOLO-style parsing."""
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        # Conv(in_channels, out_channels, kernel, stride, groups=1, act=True)
        self.layer = Conv(c1, c2, k, s)
    def forward(self, x):
        return self.layer(x)


class CVNetsInvertedResidual(nn.Module):
    def __init__(self, c1, c2, stride, expand_ratio, dilation=1, skip=True):
        super().__init__()
        # Wrap CVNets MobilenetV2 InvertedResidual
        self.block = CVNetsInvRes(
            opts=None, 
            in_channels=c1, out_channels=c2,
            stride=stride, expand_ratio=expand_ratio,
            dilation=dilation, skip_connection=skip
        )
    def forward(self, x):
        return self.block(x)

class CVNetsMobileViTBlockv2(nn.Module):
    def __init__(self, c1, attn_dim, ffn_mult=2.0, n_blocks=2, 
                 patch_h=2, patch_w=2, conv_ksize=3, dilation=1):
        super().__init__()
        # Wrap CVNets MobileViTBlockv2
        self.block = CVNetsMV2(
            opts=None,
            in_channels=c1,
            attn_unit_dim=attn_dim,
            ffn_multiplier=ffn_mult,
            n_attn_blocks=n_blocks,
            patch_h=patch_h, patch_w=patch_w,
            conv_ksize=conv_ksize, dilation=dilation
        )
    def forward(self, x):
        return self.block(x)
