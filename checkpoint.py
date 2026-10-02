"""Saving checkpoints, resumable per-layer checkpoints, and checkpoint qparams restore."""
import os
import shutil

import torch

from qlinear import QLinear


def save_checkpoint(model, tokenizer_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer_files = [
        "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "added_tokens.json", "merges.txt", "vocab.json", "vocab.txt",
        "spiece.model", "chat_template.jinja",
    ]
    for fname in tokenizer_files:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    print(f"Checkpoint saved to {output_dir}")


# ── Checkpoint qparams restore (--init_scale_mode checkpoint) ────────────
def restore_checkpoint_qparams(layer_idx, quant_layer, checkpoint_qparams, logger=None):
    """Restore LWC alpha, w_absmax, and channel scales from a checkpoint quant_params.pt.

    Scales/zeros are already loaded by wrap_layer_with_qlinear(init_qparams=...).
    This restores the remaining learnable parameters (LWC, channel_scale).
    """
    n_restored = 0
    for name, module in quant_layer.named_modules():
        if not isinstance(module, QLinear):
            continue
        full_name = f"model.layers.{layer_idx}.{name}"
        if full_name not in checkpoint_qparams:
            continue
        entry = checkpoint_qparams[full_name]

        if module.lwc_alpha is not None and "lwc_alpha" in entry:
            module.lwc_alpha.data.copy_(entry["lwc_alpha"])
        if hasattr(module, 'lwc_w_absmax') and "lwc_w_absmax" in entry:
            module.lwc_w_absmax.copy_(entry["lwc_w_absmax"])
        if module._channel_scale is not None and "channel_scale" in entry:
            module._channel_scale.data.copy_(entry["channel_scale"])
        n_restored += 1

    if logger:
        logger.write(f"  Restored checkpoint qparams: {n_restored} modules\n")


# ── Resumable per-layer checkpointing ────────────────────────────────────
# Single-pass forward only. Embedding, final norm, and LM head are frozen, so
# a completed layer is fully described by its baked (fake-quant) weights, its
# pre-baked continuous weights, and its learnable quant params. The next
# layer's input activations are NOT stored (they're huge); instead they are
# rebuilt on resume by re-propagating the embedding cache through the already
# completed layers (see repropagate_hiddens_upto).

def layer_ckpt_path(ckpt_dir, layer_idx):
    return os.path.join(ckpt_dir, f"layer_{layer_idx:03d}.pt")


def resume_config_dict(args, num_layers):
    """Critical config that must match for a resume to be valid."""
    return {
        "num_layers": num_layers,
        "student_model": os.path.realpath(args.student_model),
        "teacher_model": os.path.realpath(args.teacher_model),
        "w_bit": args.w_bit,
        "w_group_size": args.w_group_size,
        "enable_lwc": bool(args.enable_lwc),
        "enable_channel_scale": bool(args.enable_channel_scale),
        "save_pre_baked": bool(args.save_pre_baked),
        "max_samples": args.max_samples,
        "max_seq_length": args.max_seq_length,
    }


def scan_completed_layers(ckpt_dir, num_layers):
    """Return the contiguous prefix [0..M] of layers that have a checkpoint file."""
    completed = []
    for i in range(num_layers):
        if os.path.exists(layer_ckpt_path(ckpt_dir, i)):
            completed.append(i)
        else:
            break
    return completed


def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_progress(ckpt_dir, completed, config_meta):
    _atomic_torch_save({"completed": list(completed), "config": config_meta},
                       os.path.join(ckpt_dir, "progress.pt"))


def save_layer_checkpoint(ckpt_dir, layer_idx, baked_layer,
                           pre_baked_weights, quant_params, layer_time):
    """Persist one finished layer (rank 0 only). Written atomically."""
    baked_sd = {k: v.detach().to("cpu", copy=True)
                for k, v in baked_layer.state_dict().items()}
    _atomic_torch_save({
        "layer_idx": layer_idx,
        "pre_baked_weights": pre_baked_weights,
        "quant_params": quant_params,
        "baked_state_dict": baked_sd,
        "layer_time": layer_time,
    }, layer_ckpt_path(ckpt_dir, layer_idx))
