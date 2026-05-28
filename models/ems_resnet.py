"""EMS-ResNet10 backbone for COCO.

ResNet10 = stem + 4 stages of one BasicBlock each (= 8 conv layers in stages
+ 1 stem conv + 1 final conv-style head = 10 weight layers). Channel widths
follow the standard ResNet schedule (64, 128, 256, 512); the paper reports
6.20 M params for the Gen1 variant with reduced channels, and uses a wider
config on COCO. The defaults below give ~9.3 M params, close to the COCO
EMS-Res18 reported (9.34 M); halve `widths` for Gen1 parity.

The backbone returns two feature maps at strides 16 and 32 to feed the
2-head detector (Sec. 5.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ems_blocks import EMSBlock1, EMSBlock2, LCB, TConv2d
from .neurons import TDBN


class EMSResNet10(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: tuple[int, ...] = (64, 128, 256, 512),
        tau: float = 0.25,
        v_th: float = 0.5,
        alpha: float = 1.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = widths
        # Stem: dense conv -> tdBN. This is the only non-spike conv (analog
        # input pixels) and is excluded from the energy accounting in the paper.
        self.stem_conv = TConv2d(in_ch, c1, k=3, s=2, p=1)
        self.stem_bn = TDBN(c1, v_th=v_th, alpha=alpha)
        # Stage 1: stride 2  -> /4
        self.stage1 = EMSBlock1(c1, c1, stride=2, decay=tau)
        # Stage 2: stride 2  -> /8
        self.stage2 = EMSBlock2(c1, c2, stride=2, decay=tau)
        # Stage 3: stride 2  -> /16   (feature P4)
        self.stage3 = EMSBlock2(c2, c3, stride=2, decay=tau)
        # Stage 4: stride 2  -> /32   (feature P5)
        self.stage4 = EMSBlock2(c3, c4, stride=2, decay=tau)

        self.out_channels = (c3, c4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, 3, H, W) -> (P4, P5) each (T, B, C, H', W')."""
        x = self.stem_bn(self.stem_conv(x))
        x = self.stage1(x)
        x = self.stage2(x)
        p4 = self.stage3(x)
        p5 = self.stage4(p4)
        return p4, p5
