"""LIF spiking neuron, surrogate gradient, and threshold-dependent BatchNorm.

Implements:
- Eq. (1)-(2): iterative LIF (Wu et al., 2019)
- Eq. (3):     rectangular surrogate gradient
- Eq. (4)-(5): TDBN normalization (Zheng et al., 2021)

All tensors carry an explicit leading time dimension T, except inside the LIF
forward where time is iterated step-by-step. The convention used everywhere
in this repo is:

    x: (T, B, C, H, W)

so that a single `for t in range(T)` loop drives the network.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Surrogate gradient
# ---------------------------------------------------------------------------
class RectSurrogate(torch.autograd.Function):
    """Rectangular surrogate, Eq. (3).

    dH/dV = (1/a) * 1{ |V - Vth| <= a/2 }
    """

    @staticmethod
    def forward(ctx, v_minus_vth: torch.Tensor, a: float) -> torch.Tensor:
        ctx.save_for_backward(v_minus_vth)
        ctx.a = a
        return (v_minus_vth >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_vth,) = ctx.saved_tensors
        a = ctx.a
        grad_input = grad_output * (v_minus_vth.abs() <= a / 2).float() / a
        return grad_input, None


def spike_fn(v: torch.Tensor, v_th: float, a: float = 1.0) -> torch.Tensor:
    return RectSurrogate.apply(v - v_th, a)


# ---------------------------------------------------------------------------
# LIF neuron
# ---------------------------------------------------------------------------
class LIF(nn.Module):
    """Iterative LIF neuron with hard reset.

    The neuron operates on an input that already contains a time dimension:
        x: (T, B, ...) -> out: (T, B, ...)

    Internally:
        V_{t+1} = tau * V_t * (1 - X_t) + I_{t+1}
        X_{t+1} = H(V_{t+1} - V_th)
    """

    def __init__(self, tau: float = 0.25, v_th: float = 0.5, a: float = 1.0):
        super().__init__()
        self.tau = tau
        self.v_th = v_th
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        v = torch.zeros_like(x[0])
        s = torch.zeros_like(x[0])
        out = []
        for t in range(T):
            v = self.tau * v * (1.0 - s) + x[t]
            s = spike_fn(v, self.v_th, self.a)
            out.append(s)
        return torch.stack(out, dim=0)

    def forward_last_v(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final membrane potential V_T instead of spike train.

        Used at the detector head per Sec. 4.3: 'we feed last membrane
        potential of the neurons into each detector to generate anchors'.
        """
        T = x.size(0)
        v = torch.zeros_like(x[0])
        s = torch.zeros_like(x[0])
        for t in range(T):
            v = self.tau * v * (1.0 - s) + x[t]
            s = spike_fn(v, self.v_th, self.a)
        return v


class LIFNeuron(LIF):
    """LIF with `decay` parameter name (alias for tau)."""
    def __init__(self, decay: float = 0.25, v_th: float = 0.5, a: float = 1.0):
        super().__init__(tau=decay, v_th=v_th, a=a)


# ---------------------------------------------------------------------------
# tdBN (threshold-dependent BN)
# ---------------------------------------------------------------------------
class TDBN(nn.Module):
    """Threshold-dependent BatchNorm over (T, B, C, H, W).

    Normalises across the (T, B, H, W) axes per channel, then scales by
    alpha * V_th (Eq. 5). The trainable affine params (lambda, beta) are the
    standard BN weight/bias.
    """

    def __init__(self, num_features: int, v_th: float = 0.5, alpha: float = 1.0, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.v_th = v_th
        self.alpha = alpha
        # We reuse BatchNorm3d's running stats by treating (T*B) as the batch
        # axis. Simpler: collapse to BN2d.
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        # Re-scale BN gamma so the effective output multiplier is alpha * V_th * lambda.
        nn.init.constant_(self.bn.weight, alpha * v_th)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W) -> (T*B, C, H, W) -> BN -> reshape back
        T, B, C, H, W = x.shape
        y = self.bn(x.view(T * B, C, H, W))
        return y.view(T, B, C, H, W)
