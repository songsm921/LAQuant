"""
QLinear for weight-only fake quantization with learnable scales/zeros,
Learnable Weight Clipping (LWC) and learnable per-channel scaling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from quantizer import SteIntAsymQuantizer

LWC_DEFAULT_ALPHA = 4.0


class QLinear(nn.Linear):
    """Linear layer with weight-only fake quantization via SteIntAsymQuantizer.

    Optional features:
    - LWC (Learnable Weight Clipping): per-group symmetric clip with sigmoid
      reparameterization, applied before quantization.
    - Per-channel scaling: equivalent transform W*s, x/s applied before quantization.
      The channel_scale parameter is set externally (shared across modules).
    """

    def __init__(
        self, in_features, out_features, bias=True,
        w_bit=4, w_group_size=128,
        weight=None,
        init_scales=None, init_zeros=None,
        enable_lwc=False,
    ):
        super().__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.w_group_size = w_group_size

        self.lwc_alpha = None
        self._channel_scale = None  # Set externally for shared per-channel scaling

        # Init LWC from weight: clip bound c = absmax * sigmoid(alpha) per group
        w_for_quant = weight
        if enable_lwc:
            n_groups = in_features // w_group_size
            w_grouped = weight.data.float().reshape(-1, w_group_size)
            w_absmax = w_grouped.abs().amax(dim=-1)  # [out*n_groups]
            w_absmax = w_absmax.clamp(min=1e-6)
            self.register_buffer('lwc_w_absmax', w_absmax.to(weight.dtype))
            self.lwc_alpha = nn.Parameter(
                torch.full((out_features * n_groups,), LWC_DEFAULT_ALPHA, dtype=weight.dtype))

            with torch.no_grad():
                c = self.lwc_w_absmax.unsqueeze(-1) * torch.sigmoid(
                    self.lwc_alpha).unsqueeze(-1)
                w_for_quant = torch.clamp(w_grouped, -c, c).reshape(weight.shape)
                w_for_quant = w_for_quant.to(weight.dtype)

        # Init quantizer from (clipped) weight
        self.weight_quantizer = SteIntAsymQuantizer(
            bit=w_bit,
            w_group_size=w_group_size,
            weight=w_for_quant,
            init_scales=init_scales,
            init_zeros=init_zeros,
        )

    def _clip(self, w):
        if self.lwc_alpha is None:
            return w
        w_grouped = w.reshape(-1, self.w_group_size)
        c = self.lwc_w_absmax.unsqueeze(-1) * torch.sigmoid(
            self.lwc_alpha).unsqueeze(-1)
        return torch.clamp(w_grouped, -c, c).reshape(self.weight.shape)

    def forward(self, x: torch.Tensor):
        # 1. LWC clipping (per-group)
        w = self._clip(self.weight)
        # 2. Per-channel scaling (per-input-channel)
        if self._channel_scale is not None:
            s = self._channel_scale
            w = w * s.unsqueeze(0)
            x = x / s.unsqueeze(0)
        # 3. Quantize
        quantized_weight = self.weight_quantizer(w)
        return F.linear(x, quantized_weight, self.bias)

    @torch.no_grad()
    def bake_weights(self):
        """Apply LWC clip + channel scale + fake quantization permanently."""
        if self.weight_quantizer.enabled:
            w = self._clip(self.weight.data)
            if self._channel_scale is not None:
                w = w * self._channel_scale.unsqueeze(0)
            quantized = self.weight_quantizer.quantize_weight(w)
            self.weight.data.copy_(quantized.to(self.weight.dtype))
        self.weight_quantizer.disable()

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, w_bit={self.w_bit}')


def wrap_layer_with_qlinear(
    layer: nn.Module,
    layer_name_prefix: str,
    w_bit: int = 4,
    w_group_size: int = 128,
    enable_lwc: bool = False,
    init_qparams: Optional[dict] = None,
):
    """
    Replace all nn.Linear modules in a layer with QLinear.

    Args:
        layer: transformer layer module
        layer_name_prefix: e.g. "model.layers.0" (used for init_qparams lookup)
        w_bit: quantization bits
        w_group_size: group size
        enable_lwc: enable Learnable Weight Clipping
        init_qparams: optional quant_params.pt dict ({full_name: {"scale", "zero", ...}})
            to initialize scales/zeros from. If None, scales/zeros use per-group min/max.
    """
    for name, module in list(layer.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, QLinear):
            continue

        full_name = f"{layer_name_prefix}.{name}" if layer_name_prefix else name
        weight = module.weight

        init_scales, init_zeros = None, None
        if init_qparams is not None:
            if full_name not in init_qparams:
                raise KeyError(f"{full_name} not found in the checkpoint quant params")
            entry = init_qparams[full_name]
            init_scales = entry["scale"].to(weight.dtype).reshape(-1, 1)
            init_zeros = entry["zero"].to(weight.dtype).reshape(-1, 1)

        qlinear = QLinear(
            module.in_features, module.out_features,
            bias=(module.bias is not None),
            w_bit=w_bit, w_group_size=w_group_size,
            weight=weight,
            init_scales=init_scales, init_zeros=init_zeros,
            enable_lwc=enable_lwc,
        )
        qlinear.weight = module.weight
        qlinear.bias = module.bias

        _set_module_by_name(layer, name, qlinear)

    return layer


def bake_layer_weights(layer: nn.Module):
    """Bake (fake-quantize permanently) all QLinear weights in a layer."""
    for name, module in layer.named_modules():
        if isinstance(module, QLinear):
            module.bake_weights()


def convert_qlinear_to_linear(layer: nn.Module):
    """Convert all QLinear modules back to nn.Linear (strips quantizer state)."""
    replacements = []
    for name, module in layer.named_modules():
        if isinstance(module, QLinear):
            linear = nn.Linear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            linear.weight = module.weight
            linear.bias = module.bias
            replacements.append((name, linear))
    for name, linear in replacements:
        _set_module_by_name(layer, name, linear)


def _set_module_by_name(parent: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by dotted name path."""
    parts = name.split('.')
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def get_quant_params(layer: nn.Module):
    """Get all quantization parameters (scales, zeros, LWC alphas, channel scales)."""
    params = []
    for name, module in layer.named_modules():
        if isinstance(module, QLinear):
            params.append(module.weight_quantizer.scales)
            params.append(module.weight_quantizer.zeros)
            if module.lwc_alpha is not None:
                params.append(module.lwc_alpha)
    # Collect layer-level channel_scale params (registered on layer, not on QLinear)
    for name, param in layer.named_parameters():
        if 'channel_scale' in name:
            params.append(param)
    return params


def get_weight_params(layer: nn.Module):
    """Get all weight parameters from QLinear modules + normalization layer parameters."""
    params = []
    for name, module in layer.named_modules():
        if isinstance(module, QLinear):
            params.append(module.weight)
        elif 'norm' in type(module).__name__.lower():
            for p in module.parameters():
                params.append(p)
    return params


def set_quant_params_trainable(layer: nn.Module, trainable: bool):
    """Set requires_grad for quantization parameters."""
    for p in get_quant_params(layer):
        p.requires_grad_(trainable)


def set_weight_params_trainable(layer: nn.Module, trainable: bool):
    """Set requires_grad for weight parameters."""
    for p in get_weight_params(layer):
        p.requires_grad_(trainable)


# ── Learnable channel scaling helpers ────────────────────────────────────
def setup_channel_scales(layer):
    """Register shared per-channel scale params on layer, assign refs to QLinear modules."""
    groups = {}
    for name, module in layer.named_modules():
        if not isinstance(module, QLinear):
            continue
        if any(p in name for p in ('q_proj', 'k_proj', 'v_proj')):
            key = 'attn'
        elif any(p in name for p in ('gate_proj', 'up_proj')):
            key = 'mlp'
        else:
            continue  # o_proj, down_proj: skip
        if key not in groups:
            groups[key] = nn.Parameter(torch.ones(module.in_features, dtype=module.weight.dtype))
        module._channel_scale = groups[key]
    for key, param in groups.items():
        layer.register_parameter(f'channel_scale_{key}', param)



@torch.no_grad()
def bake_channel_scales_into_norms(layer):
    """Fuse learned channel scales into LayerNorm/RMSNorm weights, then remove params."""
    if hasattr(layer, 'channel_scale_attn'):
        layer.input_layernorm.weight.data.div_(layer.channel_scale_attn.data)
        delattr(layer, 'channel_scale_attn')
    if hasattr(layer, 'channel_scale_mlp'):
        layer.post_attention_layernorm.weight.data.div_(layer.channel_scale_mlp.data)
        delattr(layer, 'channel_scale_mlp')
