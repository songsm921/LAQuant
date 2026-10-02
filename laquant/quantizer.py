"""
STE-based integer asymmetric weight quantizer with learnable scales/zeros.
"""
import torch
import torch.nn as nn

SCALE_CLAMP_MIN = 1e-5
SCALE_CLAMP_MAX = 1e4


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min_val, max_val):
    return (x.clamp(min_val, max_val) - x).detach() + x


class SteIntAsymQuantizer(nn.Module):
    """Straight-Through Estimator based per-group integer asymmetric quantizer.

    Scales and zeros are learnable parameters of shape [out_features * n_groups, 1].
    They are initialized from the per-group min/max of `weight`, or from
    `init_scales` / `init_zeros` when given (e.g. to resume from a checkpoint).
    """

    def __init__(
        self,
        bit=4,
        w_group_size=128,
        weight=None,
        init_scales=None,
        init_zeros=None,
    ):
        super().__init__()
        assert w_group_size > 0, "only group-wise quantization is supported"
        self.w_group_size = w_group_size
        self.bit = bit
        self._enabled = bit < 16
        self.max_int = 2 ** self.bit - 1
        self.min_int = 0

        if init_scales is not None and init_zeros is not None:
            self.scales = nn.Parameter(init_scales.clamp(min=1e-6).clone())
            self.zeros = nn.Parameter(init_zeros.clone())
        else:
            x = weight.reshape(-1, self.w_group_size)
            xmin = x.amin([-1], keepdim=True)
            xmax = x.amax([-1], keepdim=True)
            scales = (xmax - xmin) / (2 ** self.bit - 1)
            scales = scales.clamp(min=SCALE_CLAMP_MIN, max=SCALE_CLAMP_MAX)
            zeros = -(xmin / scales).clamp(min=-1e4, max=1e4)
            self.scales = nn.Parameter(scales.clamp(min=1e-6))
            self.zeros = nn.Parameter(zeros.round())

    @property
    def enabled(self):
        return self._enabled

    def disable(self):
        self._enabled = False

    def get_scales(self, for_backward=True):
        if for_backward:
            return clamp_ste(self.scales, SCALE_CLAMP_MIN, SCALE_CLAMP_MAX)
        return self.scales.clamp(SCALE_CLAMP_MIN, SCALE_CLAMP_MAX)

    def get_int_zeros(self):
        """Integer zero points (for saving/export)."""
        return torch.round(self.zeros).clamp(self.min_int, self.max_int)

    def forward(self, x):
        if not self._enabled:
            return x

        org_w_shape = x.shape
        assert org_w_shape[-1] % self.w_group_size == 0
        x = x.reshape(-1, self.w_group_size)

        scales = self.get_scales(for_backward=True)
        zeros = clamp_ste(round_ste(self.zeros), self.min_int, self.max_int)
        x = ((round_ste(x / scales) + zeros).clamp(self.min_int, self.max_int) - zeros) * scales

        return x.reshape(org_w_shape)

    @torch.no_grad()
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization without STE (for baking weights)."""
        if not self._enabled:
            return weight.clone()

        org_shape = weight.shape
        org_dtype = weight.dtype
        assert org_shape[-1] % self.w_group_size == 0
        w = weight.reshape(-1, self.w_group_size)

        scales = self.get_scales(for_backward=False)
        zeros = self.get_int_zeros()

        w_int = torch.round(w / scales) + zeros
        w_int = w_int.clamp(self.min_int, self.max_int)
        w_quant = (w_int - zeros) * scales

        return w_quant.reshape(org_shape).to(org_dtype)

    def extra_repr(self):
        return f'bit={self.bit}, group_size={self.w_group_size}, enabled={self._enabled}'
