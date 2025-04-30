# ultralytics/nn/modules/mobilevitv2.py
import torch
import torch.nn as nn
from cvnets.cvnets.modules.mobilenetv2 import InvertedResidual as CVNetsInvRes
from cvnets.cvnets.modules.mobilevit_block import MobileViTBlockv2 as CVNetsMV2
from ultralytics.nn.modules.conv import Conv

# ----------------------------------------------------------------------------
# Dummy opts so all CVNets modules don’t crash when they try opts.model.normalization
class _Opts:
    def __init__(self):
        self.model = {
            "normalization": {
                "name": "batchnorm2d",  # Default normalization type
                "groups": 32,          # Default number of groups for group normalization
                "momentum": 0.1        # Default momentum for normalization layers
            }
        }

DUMMY_OPTS = _Opts()
# ----------------------------------------------------------------------------

class CVNetsConvLayer2d(nn.Module):
    """Wrap Ultralytics Conv, swallowing any extra args the parser may give."""
    def __init__(self, c1, c2, k=1, s=1, *args, **kwargs):
        super().__init__()
        # k, s come from the first two positional extra args; 
        # any further args are ignored.
        self.layer = Conv(c1, c2, k, s)

    def forward(self, x):
        return self.layer(x)


class CVNetsInvertedResidual(nn.Module):
    def __init__(self, c1, c2, stride, expand_ratio, dilation=1, skip=True):
        super().__init__()
        # Wrap CVNets MobilenetV2 InvertedResidual
        self.block = CVNetsInvRes(
            opts=DUMMY_OPTS,
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
            opts=DUMMY_OPTS,
            in_channels=c1,
            attn_unit_dim=attn_dim,
            ffn_multiplier=ffn_mult,
            n_attn_blocks=n_blocks,
            patch_h=patch_h, patch_w=patch_w,
            conv_ksize=conv_ksize, dilation=dilation
        )

    def forward(self, x):
        return self.block(x)
