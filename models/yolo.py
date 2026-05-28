"""Full EMS-YOLO model.

Wraps the static-image -> (T, B, C, H, W) lift, runs the backbone for T steps,
applies the spiking head, and returns raw prediction grids ready for the loss
function or NMS post-processing.

The 'lift' from static images to a spike train is the standard repeat-input
encoding (Sec. 4.1): the same image is fed in every timestep, and the first
conv + LIF stage converts analog pixels into a spike train.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .detector import EMSYOLOHead
from .ems_resnet import EMSResNet10


class EMSYOLO(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        T: int = 4,
        widths: tuple[int, ...] = (64, 128, 256, 512),
        head_ch: int = 256,
        num_anchors: int = 3,
        tau: float = 0.25,
        v_th: float = 0.5,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.T = T
        self.nc = num_classes
        self.na = num_anchors
        self.backbone = EMSResNet10(in_ch=3, widths=widths, tau=tau, v_th=v_th, alpha=alpha)
        self.head = EMSYOLOHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            head_ch=head_ch,
            tau=tau,
            v_th=v_th,
            alpha=alpha,
        )
        # Strides corresponding to head outputs (P4, P5)
        self.strides = (16, 32)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (T, B, C, H, W) by direct repeat."""
        return x.unsqueeze(0).expand(self.T, *x.shape).contiguous()

    def set_T(self, T: int) -> None:
        """Allow swapping the inference timesteps post-load (e.g. T=1 -> T=3)."""
        self.T = T

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W)

        Returns: list of two prediction grids
            pred_p4: (B, A*(5+nc), H/16, W/16)
            pred_p5: (B, A*(5+nc), H/32, W/32)
        """
        x_t = self.lift(x)
        p4, p5 = self.backbone(x_t)
        return self.head(p4, p5)


def build_model(cfg: dict) -> EMSYOLO:
    """Build from a config dict (see configs/*.yaml)."""
    model_cfg = cfg["model"]
    return EMSYOLO(
        num_classes=model_cfg.get("num_classes", 80),
        T=model_cfg["T"],
        widths=tuple(model_cfg.get("widths", (64, 128, 256, 512))),
        head_ch=model_cfg.get("head_ch", 256),
        num_anchors=model_cfg.get("num_anchors", 3),
        tau=model_cfg.get("tau", 0.25),
        v_th=model_cfg.get("v_th", 0.5),
        alpha=model_cfg.get("alpha", 1.0),
    )
