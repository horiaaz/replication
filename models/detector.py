"""Spiking detection head.

Two output scales (P4 stride 16, P5 stride 32), each predicting `na` anchors of
`(x, y, w, h, obj, *cls)`. The paper takes the yolov3-tiny detection head as a
starting point and replaces its multi-layer direct-connected convs with
EMS-Blocks (Sec. 4.3). We mirror that here.

Per Sec. 4.3, the *final* projection that maps spike features to continuous
bbox/cls predictions uses the **last membrane potential** of a LIF neuron,
not its spike output — this is the bridge between the spike domain and the
continuous regression targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ems_blocks import EMSBlock1, LCB, MSBlock, TConv2d
from .neurons import LIF, TDBN


def _upsample_time(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Nearest-neighbour upsample on a (T, B, C, H, W) tensor."""
    T, B, C, H, W = x.shape
    y = nn.functional.interpolate(x.view(T * B, C, H, W), scale_factor=scale, mode="nearest")
    return y.view(T, B, C, y.size(-2), y.size(-1))


class MembraneHead(nn.Module):
    """Final detection projection that reads out the LIF *membrane potential*
    instead of its spike output, then applies a Conv to produce the (B, A*(5+nc), H, W)
    raw prediction grid.

    Note: the conv input here is *not* a spike — it's a continuous membrane
    potential — so this conv is unavoidably a MAC. It's a single small conv
    per scale, which is why the paper still claims near-full-spike behavior.
    """

    def __init__(self, in_ch: int, na: int, nc: int, tau: float = 0.25, v_th: float = 0.5):
        super().__init__()
        self.lif = LIF(tau=tau, v_th=v_th)
        self.out = nn.Conv2d(in_ch, na * (5 + nc), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lif.forward_last_v(x)  # (B, C, H, W)
        return self.out(v)              # (B, A*(5+nc), H, W)


class EMSYOLOHead(nn.Module):
    """2-scale spiking head."""

    def __init__(
        self,
        in_channels: tuple[int, int],     # (P4_ch, P5_ch) from backbone
        num_classes: int = 80,
        num_anchors: int = 3,
        head_ch: int = 256,
        tau: float = 0.25,
        v_th: float = 0.5,
        alpha: float = 1.0,
    ):
        super().__init__()
        c4, c5 = in_channels
        self.nc = num_classes
        self.na = num_anchors

        # P5 path: a couple of MS-Blocks at head_ch
        self.p5_in = LCB(c5, head_ch, k=1, s=1, p=0, tau=tau, v_th=v_th, alpha=alpha)
        self.p5_blocks = nn.Sequential(
            MSBlock(head_ch, tau=tau, v_th=v_th, alpha=alpha),
            MSBlock(head_ch, tau=tau, v_th=v_th, alpha=alpha),
        )

        # Upsample P5 -> concat with P4 path
        self.p5_reduce = LCB(head_ch, head_ch // 2, k=1, s=1, p=0, tau=tau, v_th=v_th, alpha=alpha)
        self.p4_in = LCB(c4, head_ch // 2, k=1, s=1, p=0, tau=tau, v_th=v_th, alpha=alpha)
        self.p4_blocks = nn.Sequential(
            MSBlock(head_ch, tau=tau, v_th=v_th, alpha=alpha),
            MSBlock(head_ch, tau=tau, v_th=v_th, alpha=alpha),
        )

        # Output heads (membrane-potential readouts)
        self.head_p4 = MembraneHead(head_ch, num_anchors, num_classes, tau=tau, v_th=v_th)
        self.head_p5 = MembraneHead(head_ch, num_anchors, num_classes, tau=tau, v_th=v_th)

    def forward(self, p4: torch.Tensor, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_p4, pred_p5) each (B, A*(5+nc), H, W)."""
        p5 = self.p5_blocks(self.p5_in(p5))
        pred_p5 = self.head_p5(p5)

        # Top-down
        p5_up = _upsample_time(self.p5_reduce(p5), scale=2)
        p4 = self.p4_in(p4)
        p4 = torch.cat([p4, p5_up], dim=2)         # along channel dim
        p4 = self.p4_blocks(p4)
        pred_p4 = self.head_p4(p4)

        return pred_p4, pred_p5
