"""EMS residual blocks.

Reference: Section 4.2 and Figure 1c / Figure 2 of the EMS-YOLO paper.

Two block shapes:

EMS-Block1  (constant or decreasing channels)
    residual:  LIF -> Conv -> BN -> LIF -> Conv -> BN
    shortcut:  identity                              (no conv needed)
    output  :  residual + shortcut                   (sum of two spike trains -> integer)
                                                     ^ consumed by next LIF, full-spike preserved

EMS-Block2  (increasing channels)
    residual:  LIF -> Conv(stride=2 if downsample) -> BN -> LIF -> Conv -> BN
    shortcut:  LIF -> MaxPool -> Conv1x1 -> BN       (LIF first => sparse spikes for conv)
                                                       and concat the input with a maxpool branch
    output  :  residual + concat-based shortcut
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import LIF, LIFNeuron, TDBN


# ---------------------------------------------------------------------------
# Time-distributed conv helpers
# ---------------------------------------------------------------------------
class TConv2d(nn.Module):
    """Conv2d applied to (T, B, C, H, W) by folding T into the batch axis."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None,
                 bias: bool = False, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                              bias=bias, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        y = self.conv(x.view(T * B, C, H, W))
        Co, Ho, Wo = y.shape[-3], y.shape[-2], y.shape[-1]
        return y.view(T, B, Co, Ho, Wo)


class TMaxPool2d(nn.Module):
    def __init__(self, k: int = 2, s: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        y = self.pool(x.view(T * B, C, H, W))
        return y.view(T, B, C, y.size(-2), y.size(-1))


class SnnConv2d(nn.Module):
    """Conv2d applied independently at each timestep. Input/output: (T, B, C, H, W)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.conv(x[t]) for t in range(x.shape[0])])


class SnnMaxPool2d(nn.Module):
    """MaxPool2d applied independently at each timestep. Input/output: (T, B, C, H, W)."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.pool(x[t]) for t in range(x.shape[0])])


class LCB(nn.Module):
    """LIF -> Conv -> BN, the basic "spike conv" unit used everywhere."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None,
                 tau: float = 0.25, v_th: float = 0.5, alpha: float = 1.0):
        super().__init__()
        self.lif = LIF(tau=tau, v_th=v_th)
        self.conv = TConv2d(in_ch, out_ch, k=k, s=s, p=p)
        self.bn = TDBN(out_ch, v_th=v_th, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(self.lif(x)))


# ---------------------------------------------------------------------------
# EMS-Block1 — constant or decreasing channels, identity or projected shortcut
# ---------------------------------------------------------------------------
class EMSBlock1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, decay: float = 0.25):
        super().__init__()
        self.lif1 = LIFNeuron(decay=decay)
        self.conv1 = SnnConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = TDBN(out_ch)

        self.lif2 = LIFNeuron(decay=decay)
        self.conv2 = SnnConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = TDBN(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                SnnMaxPool2d(stride, stride),
                LIFNeuron(decay=decay),
                SnnConv2d(in_ch, out_ch, kernel_size=1),
                TDBN(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(self.lif1(x)))
        out = self.bn2(self.conv2(self.lif2(out)))
        return out + self.shortcut(x)


# ---------------------------------------------------------------------------
# EMS-Block2 — increasing channels, concat-based feature reuse on shortcut
# ---------------------------------------------------------------------------
class EMSBlock2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, decay: float = 0.25):
        super().__init__()
        assert out_ch > in_ch, "EMSBlock2 is for increasing channels only"

        self.lif1 = LIFNeuron(decay=decay)
        self.conv1 = SnnConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = TDBN(out_ch)

        self.lif2 = LIFNeuron(decay=decay)
        self.conv2 = SnnConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = TDBN(out_ch)

        self.pool = SnnMaxPool2d(kernel_size=stride, stride=stride)
        self.shortcut_lif = LIFNeuron(decay=decay)
        self.shortcut_conv = SnnConv2d(in_ch, out_ch - in_ch, kernel_size=1)
        self.shortcut_bn = TDBN(out_ch - in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(self.lif1(x)))
        out = self.bn2(self.conv2(self.lif2(out)))

        pooled = self.pool(x)
        shortcut = self.shortcut_bn(self.shortcut_conv(self.shortcut_lif(pooled)))
        sc = torch.cat([shortcut, pooled], dim=2)

        return out + sc


# ---------------------------------------------------------------------------
# MS-Block (used in the detection head — straight residual, no shortcut conv)
# ---------------------------------------------------------------------------
class MSBlock(nn.Module):
    def __init__(self, ch: int, tau: float = 0.25, v_th: float = 0.5, alpha: float = 1.0):
        super().__init__()
        self.conv1 = LCB(ch, ch, k=3, s=1, tau=tau, v_th=v_th, alpha=alpha)
        self.conv2 = LCB(ch, ch, k=3, s=1, tau=tau, v_th=v_th, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))
