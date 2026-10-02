#!/usr/bin/env python3
"""
DDP layer-wise QAT with sliding-window reconstruction.

Each transformer layer is quantized (weight-only, per-group asymmetric INT) and
trained in turn to match the BF16 teacher. With --window_size W > 1, the
student/teacher outputs are propagated through the next W-1 frozen teacher
layers before computing the reconstruction loss.

Learnable parameters per layer: quantization scales/zeros, optional LWC
(learnable weight clipping), optional per-channel scaling, and (unless
--freeze_weight) the layer weights themselves.

Distributed Data Parallel across GPUs:
  - Training: DDP all-reduce, loss reweighted by local/global token ratio so
    gradient averaging matches full-batch normalization.
  - Forward passes: each rank processes its shard independently.

Launch: torchrun --nproc_per_node=8 train.py [args]
Falls back to single GPU if not launched with torchrun.
"""
import argparse
import os
import psutil
import shutil
import sys
import time

import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Suppress CosineAnnealingLR warnings (we intentionally step scheduler before optimizer
# at the start of each epoch for cosine scheduling)
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

from .qlinear import (
    QLinear,
    wrap_layer_with_qlinear,
    bake_layer_weights,
    convert_qlinear_to_linear,
    get_quant_params,
    get_weight_params,
    set_quant_params_trainable,
    set_weight_params_trainable,
    setup_channel_scales,
    bake_channel_scales_into_norms,
)
from .data import (
    DiskBackedHiddens,
    get_hidden_length,
    load_calib_dataset,
    tokenize_dataset,
    make_batches,
    pad_and_batch,
    batch_position_ids,
    cache_layer_activations,
    forward_layer_inplace,
    repropagate_hiddens_upto,
)
from .checkpoint import (
    save_checkpoint,
    restore_checkpoint_qparams,
    layer_ckpt_path,
    resume_config_dict,
    scan_completed_layers,
    write_progress,
    save_layer_checkpoint,
)
from .utils import (
    release_memory,
    setup_distributed,
    cleanup_distributed,
    Logger,
    format_time,
    masked_mse_loss,
    masked_smooth_l1_loss,
)


# ── Args ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DDP layer-wise QAT with sliding-window reconstruction")

    # Models
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)

    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Local calibration set directory saved with save_to_disk "
                             "(e.g. ./data/Openr1-dataset-sampled-laquant, see README)")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--token_mode", type=str, default="prompt",
                        choices=["prompt", "response"],
                        help="prompt: first max_seq_length tokens of each sample. "
                             "response: tokens from the assistant response (see --response_pos).")
    parser.add_argument("--response_pos", type=str, default="early",
                        choices=["early", "middle", "end"],
                        help="Truncation position within the response (token_mode=response only). "
                             "early: from response start. "
                             "middle: random window in [L, N-2L) of the response. "
                             "end: last L tokens of the response.")
    parser.add_argument("--shuffle_samples", action="store_true",
                        help="Shuffle dataset indices with --shuffle_seed, then take the first "
                             "max_samples from the shuffled order. "
                             "Default (off) keeps the first max_samples sequentially.")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Seed for --shuffle_samples (uses an isolated RNG, does not "
                             "perturb token-window randomness).")

    # Quantization
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--w_group_size", type=int, default=128)
    parser.add_argument("--init_scale_mode", type=str, default="minmax",
                        choices=["minmax", "checkpoint"],
                        help="minmax: per-group min/max init. "
                             "checkpoint: resume scales/zeros/LWC/channel scales from "
                             "--checkpoint_qparams_file (e.g. W=1 -> W=2).")
    parser.add_argument("--checkpoint_qparams_file", type=str, default=None,
                        help="Path to quant_params.pt from a pre-baked checkpoint. "
                             "Used with --init_scale_mode checkpoint to resume QAT.")
    parser.add_argument("--enable_lwc", action="store_true",
                        help="Enable Learnable Weight Clipping (OmniQuant sigmoid reparameterization).")
    parser.add_argument("--enable_channel_scale", action="store_true",
                        help="Enable learnable per-channel scaling (AWQ-style equivalent transform). "
                             "Shared scales for Q/K/V and gate/up, initialized at identity.")

    # Training
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--quant_lr", type=float, default=1e-4)
    parser.add_argument("--weight_lr", type=float, default=0.0)
    parser.add_argument("--min_lr_factor", type=float, default=10.0,
                        help="Cosine schedule decays each LR to lr / min_lr_factor")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="Stop a layer after this many epochs without loss improvement (0=off)")
    parser.add_argument("--adaptive_grad_norm", action="store_true", default=True)
    parser.add_argument("--no_adaptive_grad_norm", action="store_true",
                        help="Use a fixed max grad norm of 1.0 instead of calibrating it "
                             "to the epoch-0 median gradient norm")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Effective batch size (split across GPUs in DDP)")
    parser.add_argument("--freeze_weight", action="store_true", default=True)
    parser.add_argument("--no_freeze_weight", action="store_true")

    # Loss
    parser.add_argument("--window_size", type=int, default=1,
                        help="Sliding window: propagate student/teacher output through W-1 "
                             "additional teacher layers before computing the loss")
    parser.add_argument("--loss_mode", type=str, default="mse",
                        choices=["mse", "smooth_l1"])
    parser.add_argument("--smooth_l1_beta", type=float, default=1.0,
                        help="Beta for SmoothL1Loss (transition point between L1 and L2)")

    # Logit probe (diagnostic only, no gradient)
    parser.add_argument("--logit_probe", action="store_true",
                        help="Periodically log the full-model logit KL of the current layer")
    parser.add_argument("--probe_interval", type=int, default=5,
                        help="Run logit probe every N epochs")
    parser.add_argument("--probe_samples", type=int, default=16,
                        help="Number of samples for logit probe")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/layerwise_qat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_pre_baked", action="store_true",
                        help="Also save the continuous (pre-quantization) weights to "
                             "output_dir/pre-baked/. Needed only to resume QAT from this run "
                             "(e.g. W=1 -> W=2 with --init_scale_mode checkpoint).")
    parser.add_argument("--disk_offload", action="store_true",
                        help="Offload hidden state caches to /tmp disk instead of RAM")
    parser.add_argument("--layer_checkpoint", action="store_true",
                        help="Save each layer's trained state to output_dir/layer_ckpt/ after it "
                             "finishes. On launch, auto-resume from the "
                             "last completed layer if checkpoints exist. Embedding/norm/LM head are "
                             "frozen so only per-layer weights + quant params are stored; the next "
                             "layer's input activations are rebuilt by re-propagation at resume.")
    parser.add_argument("--no_resume", action="store_true",
                        help="With --layer_checkpoint, ignore any existing layer checkpoints and "
                             "start fresh from layer 0 (clears output_dir/layer_ckpt/).")
    parser.add_argument("--layer_checkpoint_cleanup", action="store_true",
                        help="With --layer_checkpoint, delete output_dir/layer_ckpt/ after the final "
                             "checkpoints are saved successfully (default: keep for safety).")

    args = parser.parse_args()

    if args.no_freeze_weight:
        args.freeze_weight = False
    if args.no_adaptive_grad_norm:
        args.adaptive_grad_norm = False
    if args.freeze_weight:
        args.weight_lr = 0.0

    assert args.quant_lr > 0 or args.weight_lr > 0
    if args.init_scale_mode == "checkpoint":
        assert args.checkpoint_qparams_file, \
            "--init_scale_mode checkpoint requires --checkpoint_qparams_file"

    return args


# ── Logit probe (diagnostic) ─────────────────────────────────────────────
@torch.no_grad()
def logit_probe_kl(layer_idx, teacher_model, current_layer,
                   fp_hiddens, quant_hiddens, position_ids_list,
                   probe_indices, dev):
    """Compute KL divergence as a diagnostic metric (no gradient).

    Uses fp_hiddens (teacher output AFTER current layer) as teacher starting point.
    Processes one teacher layer at a time to minimize CPU memory churn.

    Teacher path: fp_hiddens → remaining layers → norm → lm_head  (no grad)
    Student path: quant_hiddens → current_layer → remaining layers → norm → lm_head  (no grad)

    Returns KL divergence value (float).
    """
    import torch.nn.functional as F

    num_layers = len(teacher_model.model.layers)
    rotary_emb = teacher_model.model.rotary_emb if hasattr(teacher_model.model, 'rotary_emb') else None

    # Prepare batches
    local_pos = [position_ids_list[i] for i in probe_indices]
    teacher_batch_list = [fp_hiddens[i].clone() for i in probe_indices]
    teacher_batch, lengths = pad_and_batch(teacher_batch_list, list(range(len(teacher_batch_list))))
    teacher_batch = teacher_batch.to(dev)

    student_batch, _ = pad_and_batch(quant_hiddens, probe_indices)
    student_batch = student_batch.to(dev)

    max_len = teacher_batch.shape[1]
    pos_batch = batch_position_ids(local_pos, list(range(len(local_pos))), max_len).to(dev)

    if rotary_emb is not None:
        rotary_emb.to(dev)

    # Student: forward through current layer
    with torch.amp.autocast('cuda'):
        pos_embed = rotary_emb(student_batch, pos_batch) if rotary_emb is not None else None
        out = current_layer(student_batch, position_embeddings=pos_embed, position_ids=pos_batch)
        if isinstance(out, tuple):
            out = out[0]
    student_batch = out

    # Forward both through remaining teacher layers (one at a time on GPU)
    for li in range(layer_idx + 1, num_layers):
        layer = teacher_model.model.layers[li]
        if layer is None:
            continue
        layer.to(dev)
        with torch.amp.autocast('cuda'):
            pos_embed = rotary_emb(teacher_batch, pos_batch) if rotary_emb is not None else None
            t_out = layer(teacher_batch, position_embeddings=pos_embed, position_ids=pos_batch)
            if isinstance(t_out, tuple):
                t_out = t_out[0]
            s_out = layer(student_batch, position_embeddings=pos_embed, position_ids=pos_batch)
            if isinstance(s_out, tuple):
                s_out = s_out[0]
        teacher_batch = t_out
        student_batch = s_out
        layer.cpu()

    # Compute KL per sample through norm + lm_head
    norm = teacher_model.model.norm.to(dev)
    lm_head = teacher_model.lm_head.to(dev)

    total_kl = 0.0
    total_tokens = 0

    for b in range(teacher_batch.shape[0]):
        seq_len_b = lengths[b]
        with torch.amp.autocast('cuda'):
            t_logits = lm_head(norm(teacher_batch[b:b+1, :seq_len_b])).float()
            s_logits = lm_head(norm(student_batch[b:b+1, :seq_len_b])).float()
            teacher_lp = F.log_softmax(t_logits, dim=-1)
            student_lp = F.log_softmax(s_logits, dim=-1)
            teacher_p = teacher_lp.exp()
            kl = (teacher_p * (teacher_lp - student_lp)).sum(dim=-1)
            total_kl += kl.sum().item()
            total_tokens += seq_len_b
        del t_logits, s_logits, teacher_lp, student_lp, teacher_p, kl

    norm.cpu()
    lm_head.cpu()
    if rotary_emb is not None:
        rotary_emb.cpu()

    del teacher_batch, student_batch
    release_memory()

    return total_kl / total_tokens if total_tokens > 0 else 0.0


# ── Sliding window ───────────────────────────────────────────────────────
def _get_window_layers(teacher, layer_idx, num_layers, window_size):
    """Frozen BF16 teacher layers idx+1 .. idx+W-1 used as lookahead (None if W=1)."""
    layers = []
    for w in range(1, window_size):
        j = layer_idx + w
        if j >= num_layers:
            break
        src = teacher.model.layers[j]
        if src is None:
            continue
        layers.append(src)
    return layers or None


# ── Per-layer training ───────────────────────────────────────────────────
def train_one_layer(
    layer_idx, num_layers, fp_layer, quant_layer, fp_hiddens, quant_hiddens,
    position_ids_list, rotary_emb, args, dev,
    rank, world_size, logger, layer_times,
    teacher_model=None,
    window_layers=None,
):
    """Train a single layer's quantization parameters (and weights) with DDP.

    Each rank processes its own shard of fp_hiddens/quant_hiddens.
    DDP handles gradient synchronization during training.
    """
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    logger.write(f"Layer {layer_idx}: RSS at entry = {rss_gb:.1f}GB\n")

    num_samples = len(fp_hiddens)
    per_rank_batch = args.batch_size // world_size

    # Sort samples by length for efficient batching
    sample_lengths = [get_hidden_length(fp_hiddens, i) for i in range(num_samples)]
    sorted_indices = sorted(range(num_samples), key=lambda i: sample_lengths[i])
    batches = make_batches(sorted_indices, per_rank_batch)
    num_batches = len(batches)

    # Step 1: Compute FP targets — overwrite fp_hiddens in-place
    logger.write(f"\n{'='*60}\n")
    logger.write(f"Layer {layer_idx}: Computing FP targets...\n")
    fp_layer.eval()
    for p in fp_layer.parameters():
        p.requires_grad_(False)

    forward_layer_inplace(
        fp_layer, rotary_emb, fp_hiddens, position_ids_list, batches, dev, logger,
    )

    # Step 2: Setup optimizer + training
    num_epochs = args.num_epochs
    if num_epochs > 0:
        quant_layer = quant_layer.to(dev).float()
        if rotary_emb is not None:
            rotary_emb = rotary_emb.to(dev)
        # Sliding window: move frozen lookahead teacher layers to GPU
        if window_layers:
            for wl in window_layers:
                wl.to(dev)
                wl.eval()
                for p in wl.parameters():
                    p.requires_grad_(False)

        param_groups = []
        pg_labels = []
        total_iters = num_epochs * num_batches

        quant_scheduler = None
        weight_scheduler = None
        quant_pg_idx = -1
        weight_pg_idx = -1

        if args.quant_lr > 0:
            qp = get_quant_params(quant_layer)
            set_quant_params_trainable(quant_layer, True)
            param_groups.append({"params": qp, "lr": args.quant_lr})
            dummy_opt = torch.optim.AdamW([torch.tensor(0.0)], lr=args.quant_lr)
            quant_scheduler = CosineAnnealingLR(
                dummy_opt, T_max=total_iters,
                eta_min=args.quant_lr / args.min_lr_factor
            )
            quant_pg_idx = len(param_groups) - 1
            pg_labels.append("qparams")
        else:
            set_quant_params_trainable(quant_layer, False)

        if args.weight_lr > 0:
            wp = get_weight_params(quant_layer)
            set_weight_params_trainable(quant_layer, True)
            param_groups.append({"params": wp, "lr": args.weight_lr})
            dummy_opt2 = torch.optim.AdamW([torch.tensor(0.0)], lr=args.weight_lr)
            weight_scheduler = CosineAnnealingLR(
                dummy_opt2, T_max=total_iters,
                eta_min=args.weight_lr / args.min_lr_factor
            )
            weight_pg_idx = len(param_groups) - 1
            pg_labels.append("weights")
        else:
            set_weight_params_trainable(quant_layer, False)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        scaler = torch.amp.GradScaler('cuda')

        n_qparams = sum(p.numel() for p in get_quant_params(quant_layer))
        n_wparams = sum(p.numel() for p in get_weight_params(quant_layer) if p.requires_grad)

        ddp_info = f"{world_size} GPUs, per_rank_bs={per_rank_batch}" if world_size > 1 else "1 GPU"
        logger.write(f"Layer {layer_idx}: trainable qparams={n_qparams:,}, wparams={n_wparams:,}\n")
        logger.write(f"  {num_samples} samples/rank, {num_batches} batches (eff_bs={args.batch_size}, {ddp_info})\n")

        # Wrap with DDP
        if world_size > 1:
            quant_layer_ddp = DDP(quant_layer, device_ids=[dev.index])
        else:
            quant_layer_ddp = quant_layer

        if args.loss_mode == "smooth_l1":
            loss_fn = lambda p, t, l: masked_smooth_l1_loss(p, t, l, beta=args.smooth_l1_beta)
        else:
            loss_fn = masked_mse_loss

        # Training loop
        best_loss = float('inf')
        early_stop_count = 0
        pg_max_norms = [1.0] * len(param_groups)

        # ETA info
        avg_layer_time = sum(layer_times) / len(layer_times) if layer_times else None
        remaining_layers = num_layers - layer_idx
        if avg_layer_time and rank == 0:
            eta_str = format_time(avg_layer_time * remaining_layers)
            print(f"  ETA remaining: ~{eta_str} ({remaining_layers} layers × {format_time(avg_layer_time)}/layer)")

        # Progress bar (rank 0 only)
        total_steps = num_epochs * num_batches
        pbar = None
        if rank == 0:
            pbar = tqdm(
                total=total_steps,
                desc=f"Layer {layer_idx}/{num_layers-1}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stderr,
            )

        layer_t0 = time.time()

        _disk_q = isinstance(quant_hiddens, DiskBackedHiddens)
        _disk_f = isinstance(fp_hiddens, DiskBackedHiddens)

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_gnorms_per_pg = [[] for _ in param_groups]
            t0 = time.time()

            # Prefetch first batch
            if _disk_q and len(batches) > 0:
                quant_hiddens.prefetch(batches[0])
            if _disk_f and len(batches) > 0:
                fp_hiddens.prefetch(batches[0])

            for bi, batch_idx in enumerate(batches):
                if _disk_q:
                    quant_hiddens.wait_prefetch()
                if _disk_f:
                    fp_hiddens.wait_prefetch()
                inp_batch, lengths = pad_and_batch(quant_hiddens, batch_idx)
                tgt_batch, _ = pad_and_batch(fp_hiddens, batch_idx)
                # Prefetch next batch while GPU processes current one
                if bi + 1 < len(batches):
                    if _disk_q:
                        quant_hiddens.prefetch(batches[bi + 1])
                    if _disk_f:
                        fp_hiddens.prefetch(batches[bi + 1])
                inp_batch = inp_batch.to(dev)
                tgt_batch = tgt_batch.to(dev)
                max_len = inp_batch.shape[1]
                pos_batch = batch_position_ids(position_ids_list, batch_idx, max_len).to(dev)

                with torch.amp.autocast('cuda'):
                    if rotary_emb is not None:
                        pos_embed = rotary_emb(inp_batch, pos_batch)
                    else:
                        pos_embed = None

                    out = quant_layer_ddp(
                        inp_batch,
                        position_embeddings=pos_embed,
                        position_ids=pos_batch,
                    )
                    if isinstance(out, tuple):
                        out = out[0]

                    # Sliding window: propagate through additional layers
                    # Use BF16 precision to avoid FP16 overflow in multi-layer composition
                    if window_layers:
                        window_tgt = tgt_batch
                        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            for wl in window_layers:
                                window_tgt = wl(window_tgt, position_embeddings=pos_embed,
                                                position_ids=pos_batch)
                                if isinstance(window_tgt, tuple):
                                    window_tgt = window_tgt[0]
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            for wl in window_layers:
                                out = wl(out, position_embeddings=pos_embed,
                                         position_ids=pos_batch)
                                if isinstance(out, tuple):
                                    out = out[0]
                        tgt_batch = window_tgt

                    loss = loss_fn(out, tgt_batch, lengths)

                    # Reweight loss so DDP gradient averaging matches global normalization.
                    if world_size > 1:
                        local_n_tokens = torch.tensor(float(sum(lengths)), device=dev)
                        global_n_tokens = local_n_tokens.clone()
                        dist.all_reduce(global_n_tokens, op=dist.ReduceOp.SUM)
                        loss = loss * (local_n_tokens * world_size / global_n_tokens)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Sync found_inf: if ANY rank has inf grads, ALL ranks must skip
                if world_size > 1:
                    opt_state = scaler._per_optimizer_states[id(optimizer)]
                    found_infs = opt_state["found_inf_per_device"]
                    local_found_inf = sum(t.item() for t in found_infs.values())
                    found_inf_tensor = torch.tensor([local_found_inf], dtype=torch.float32, device=dev)
                    dist.all_reduce(found_inf_tensor, op=dist.ReduceOp.MAX)
                    if found_inf_tensor.item() > 0.0:
                        for t in found_infs.values():
                            t.fill_(1.0)

                for pg_i, pg in enumerate(param_groups):
                    pg_params = [p for p in pg["params"] if p.grad is not None]
                    if pg_params:
                        gn = torch.nn.utils.clip_grad_norm_(pg_params, pg_max_norms[pg_i])
                        gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
                        epoch_gnorms_per_pg[pg_i].append(gn_val)

                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())

                if quant_scheduler is not None:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_pg_idx]['lr'] = quant_scheduler.get_lr()[0]
                if weight_scheduler is not None:
                    weight_scheduler.step()
                    optimizer.param_groups[weight_pg_idx]['lr'] = weight_scheduler.get_lr()[0]

                # Update progress bar
                if pbar is not None:
                    postfix = {"loss": f"{loss.item():.4f}"}
                    if quant_pg_idx >= 0:
                        postfix["qlr"] = f"{optimizer.param_groups[quant_pg_idx]['lr']:.1e}"
                    if weight_pg_idx >= 0:
                        postfix["wlr"] = f"{optimizer.param_groups[weight_pg_idx]['lr']:.1e}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

            # Epoch summary
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - t0

            # Adaptive grad norm: calibrate per-group max norm to the epoch-0 median
            if epoch == 0:
                for pg_i in range(len(param_groups)):
                    gnorms = epoch_gnorms_per_pg[pg_i]
                    valid = [g for g in gnorms if g == g]
                    if valid and args.adaptive_grad_norm:
                        median_gn = sorted(valid)[len(valid) // 2]
                        pg_max_norms[pg_i] = max(1.0, median_gn)
                mode_str = "adaptive" if args.adaptive_grad_norm else "fixed"
                norm_strs = ", ".join(
                    f"{pg_labels[i]}={pg_max_norms[i]:.2f}" for i in range(len(param_groups))
                )
                logger.log(f"  -> {mode_str} max_norm: {norm_strs} (init_loss={mean_loss:.4f})\n")

            # Logit probe
            if (args.logit_probe and teacher_model is not None
                    and epoch > 0 and epoch % args.probe_interval == 0):
                # Select probe samples (first N from sorted indices)
                n_probe = min(args.probe_samples, num_samples)
                probe_indices = sorted_indices[:n_probe]
                base_layer = quant_layer_ddp.module if world_size > 1 else quant_layer_ddp
                probe_kl_val = logit_probe_kl(
                    layer_idx, teacher_model, base_layer,
                    fp_hiddens, quant_hiddens, position_ids_list,
                    probe_indices, dev,
                )
                # Probe moves teacher layers + rotary_emb to CPU; restore to GPU
                if rotary_emb is not None:
                    rotary_emb.to(dev)
                if window_layers:
                    for wl in window_layers:
                        wl.to(dev)
                logger.log(f"  -> probe KL={probe_kl_val:.6f} at epoch {epoch}\n")

            # Build gnorm string
            pg_gnorm_strs = []
            for pg_i in range(len(param_groups)):
                gnorms = epoch_gnorms_per_pg[pg_i]
                valid = [g for g in gnorms if g == g]
                mg = sum(valid) / len(valid) if valid else float('nan')
                pg_gnorm_strs.append(f"{pg_labels[pg_i]}={mg:.4f}")
            gnorm_str = "  ".join(pg_gnorm_strs)

            # Build lr string
            lr_parts = []
            if quant_pg_idx >= 0:
                lr_parts.append(f"qlr={optimizer.param_groups[quant_pg_idx]['lr']:.2e}")
            if weight_pg_idx >= 0:
                lr_parts.append(f"wlr={optimizer.param_groups[weight_pg_idx]['lr']:.2e}")
            lr_str = "  ".join(lr_parts)

            max_mem = torch.cuda.max_memory_allocated(dev) / 1024**3

            # Write to log file only (tqdm handles stdout)
            logger.log(
                f"  epoch {epoch:3d}  loss={mean_loss:.6f}  gnorm=[{gnorm_str}]  "
                f"{lr_str}  time={elapsed:.1f}s  "
                f"peak_mem={max_mem:.2f}GB  {ddp_info}\n"
            )
            logger.flush()

            if args.early_stop > 0:
                # Synchronize loss across ranks for consistent early stopping
                if world_size > 1:
                    loss_tensor = torch.tensor(mean_loss, device=dev)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    mean_loss_sync = loss_tensor.item()
                else:
                    mean_loss_sync = mean_loss

                if mean_loss_sync < best_loss:
                    best_loss = mean_loss_sync
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= args.early_stop:
                        logger.log(f"  Early stopping at epoch {epoch}\n")
                        if pbar is not None:
                            pbar.set_description(f"Layer {layer_idx}/{num_layers-1} (early stop ep {epoch})")
                            pbar.close()
                            pbar = None
                        break

        if pbar is not None:
            pbar.close()

        layer_elapsed = time.time() - layer_t0
        layer_times.append(layer_elapsed)
        logger.write(f"  Layer {layer_idx} training done: {format_time(layer_elapsed)}\n")

        optimizer.zero_grad()
        if window_layers:
            for wl in window_layers:
                wl.cpu()
            release_memory()

        # Unwrap DDP before baking
        if world_size > 1:
            del quant_layer_ddp
        del optimizer, scaler
        release_memory()

    # Step 3: Extract learned params and pre-baked weights before baking
    quant_layer.half()

    pre_baked_weights = {}
    quant_params = {}
    for name, module in quant_layer.named_modules():
        if isinstance(module, QLinear):
            full_name = f"model.layers.{layer_idx}.{name}"
            pre_baked_weights[full_name + ".weight"] = module.weight.data.cpu().clone()
            entry = {
                "scale": module.weight_quantizer.scales.data.cpu().clone(),
                "zero": module.weight_quantizer.get_int_zeros().cpu().clone(),
            }
            if module.lwc_alpha is not None:
                entry["lwc_alpha"] = module.lwc_alpha.data.cpu().clone()
                entry["lwc_w_absmax"] = module.lwc_w_absmax.cpu().clone()
            if module._channel_scale is not None:
                entry["channel_scale"] = module._channel_scale.data.cpu().clone()
            quant_params[full_name] = entry

    # Fuse channel scales into norms before baking quantized weights
    if args.enable_channel_scale:
        bake_channel_scales_into_norms(quant_layer)

    bake_layer_weights(quant_layer)
    convert_qlinear_to_linear(quant_layer)

    # Step 4: Compute quant outputs — overwrite quant_hiddens in-place
    forward_layer_inplace(
        quant_layer, rotary_emb, quant_hiddens, position_ids_list, batches, dev,
    )

    release_memory()

    rss_gb = psutil.Process().memory_info().rss / 1024**3
    logger.write(f"Layer {layer_idx}: RSS at exit = {rss_gb:.1f}GB\n")
    logger.flush()

    return pre_baked_weights, quant_params


def main():
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    dev = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    torch.manual_seed(args.seed)  # Same seed across ranks for consistent optimizer init

    # Validate batch_size divisibility
    per_rank_batch = args.batch_size // world_size
    if per_rank_batch * world_size != args.batch_size:
        if is_main:
            print(f"WARNING: batch_size {args.batch_size} not divisible by world_size {world_size}, "
                  f"rounding down to {per_rank_batch * world_size}")
        args.batch_size = per_rank_batch * world_size
    assert per_rank_batch >= 1, f"batch_size {args.batch_size} too small for {world_size} GPUs"

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "loss_log.txt")
    log_file = Logger(log_path, rank=rank)
    log_file.write(f"Config: {vars(args)}\n\n")
    log_file.write(f"DDP: world_size={world_size}, per_rank_batch={per_rank_batch}\n")

    # ── Load dataset (all ranks tokenize, then shard) ────────────────────
    if is_main:
        print("Loading dataset...")
    # Rank 0 downloads/caches first so the other ranks read from the local cache
    if world_size > 1 and not is_main:
        dist.barrier()
    dataset = load_calib_dataset(args.dataset_path)
    if world_size > 1 and is_main:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    all_samples = tokenize_dataset(
        dataset, tokenizer, args.max_seq_length, args.max_samples,
        token_mode=args.token_mode, response_pos=args.response_pos,
        shuffle_samples=args.shuffle_samples, shuffle_seed=args.shuffle_seed,
    )

    # Truncate to multiple of world_size for equal sharding (last batch can be smaller)
    n_usable = (len(all_samples) // world_size) * world_size
    if n_usable < len(all_samples) and is_main:
        print(f"  Truncating {len(all_samples)} → {n_usable} samples (multiple of world_size={world_size})")
    all_samples = all_samples[:n_usable]

    # Sort globally by length, then shard interleaved for balanced length distribution
    all_samples.sort(key=lambda s: s.shape[1])
    my_samples = [all_samples[i] for i in range(rank, len(all_samples), world_size)]

    if is_main:
        print(f"Dataset: {n_usable} total, {len(my_samples)} per rank")
        token_lengths = [s.shape[1] for s in all_samples]
        log_file.write(f"Dataset: {n_usable} samples ({len(my_samples)}/rank)\n")
        log_file.write(f"Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
                       f"mean={sum(token_lengths)/len(token_lengths):.0f}\n")
    del all_samples

    # ── Load checkpoint quantization params ─────────────────────────────
    checkpoint_qparams = None
    if args.init_scale_mode == "checkpoint":
        checkpoint_qparams = torch.load(args.checkpoint_qparams_file, map_location="cpu")
        if is_main:
            sample_keys = list(checkpoint_qparams[list(checkpoint_qparams.keys())[0]].keys())
            print(f"Loaded checkpoint qparams: {len(checkpoint_qparams)} modules, "
                  f"fields: {sample_keys}")

    # ── Load models (all ranks) ─────────────────────────────────────────
    if is_main:
        print("Loading teacher model...")
    _model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **_model_kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False

    if is_main:
        print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **_model_kwargs)
    student.config.use_cache = False

    num_layers = len(teacher.model.layers)
    if is_main:
        print(f"Model: {num_layers} layers, hidden_size={teacher.config.hidden_size}")

    # ── Cache embeddings (each rank caches its own shard) ────────────────
    same_embeddings = (os.path.realpath(args.student_model) == os.path.realpath(args.teacher_model))

    if is_main:
        print("Caching embedding outputs...")
    fp_hiddens_list, position_ids_list, _ = cache_layer_activations(teacher, my_samples, dev)

    if args.disk_offload:
        disk_cache_dir = f"/tmp/layerwise_qat_{os.getpid()}"
        if is_main:
            cache_bytes = sum(t.numel() * t.element_size() for t in fp_hiddens_list)
            print(f"Disk offload: flushing {cache_bytes / 1024**3:.2f} GB to {disk_cache_dir}")
        fp_hiddens = DiskBackedHiddens.from_list(fp_hiddens_list, disk_cache_dir, "fp")
        if same_embeddings:
            quant_hiddens = DiskBackedHiddens.from_list(fp_hiddens_list, disk_cache_dir, "quant")
        del fp_hiddens_list
        release_memory()
        if not same_embeddings:
            quant_list, _, _ = cache_layer_activations(student, my_samples, dev)
            quant_hiddens = DiskBackedHiddens.from_list(quant_list, disk_cache_dir, "quant")
            del quant_list
            release_memory()
        log_file.write(f"Disk offload enabled: {disk_cache_dir}\n")
    else:
        fp_hiddens = fp_hiddens_list
        del fp_hiddens_list
        if same_embeddings:
            if is_main:
                print("Sharing embedding cache (embeddings identical)")
            quant_hiddens = list(fp_hiddens)
            log_file.write("Shared embedding cache\n")
        else:
            if is_main:
                print("Caching student embedding outputs...")
            quant_hiddens, _, _ = cache_layer_activations(student, my_samples, dev)

    rotary_emb = teacher.model.rotary_emb if hasattr(teacher.model, 'rotary_emb') else None

    if args.logit_probe:
        if is_main:
            print("  Logit probe enabled (uses fp_hiddens as teacher starting point)")
        log_file.write(f"Logit probe enabled (interval={args.probe_interval}, "
                       f"samples={args.probe_samples})\n")

    # Free teacher's embed_tokens and lm_head — only layers are needed for training
    # Keep lm_head if logit_probe is enabled (needed for probe forward)
    if hasattr(teacher.model, 'embed_tokens'):
        del teacher.model.embed_tokens
    if hasattr(teacher, 'lm_head') and not args.logit_probe:
        del teacher.lm_head

    del my_samples
    release_memory()

    log_file.write(f"peak_vram_after_cache={torch.cuda.max_memory_allocated(dev)/1024**3:.2f}GB\n")
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    log_file.write(f"RSS after cache = {rss_gb:.1f}GB\n")

    # Pre-baked weight accumulator
    all_pre_baked_weights = {}
    all_quant_params = {}
    layer_times = []  # Track per-layer time for ETA

    # ── Layer-checkpoint setup ───────────────────────────────────────────
    layer_ckpt_active = args.layer_checkpoint
    layer_ckpt_dir = os.path.join(args.output_dir, "layer_ckpt")
    if layer_ckpt_active:
        # Decide resume vs. fresh, and clean stale files on a fresh start (rank 0).
        if is_main:
            os.makedirs(layer_ckpt_dir, exist_ok=True)
            completed_probe = scan_completed_layers(layer_ckpt_dir, num_layers)
            fresh = args.no_resume or len(completed_probe) == 0
            if fresh and os.listdir(layer_ckpt_dir):
                shutil.rmtree(layer_ckpt_dir)
                os.makedirs(layer_ckpt_dir, exist_ok=True)
                print(f"Layer checkpoint: starting fresh, cleared {layer_ckpt_dir}")
                log_file.write(f"Layer checkpoint: fresh start, cleared {layer_ckpt_dir}\n")
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.barrier()

    # ── Layer-checkpoint resume ──────────────────────────────────────────
    # At this point fp_hiddens / quant_hiddens still hold the raw embedding
    # outputs. Load the completed layers, then re-propagate the embedding cache
    # through them to rebuild the input activations for the first un-trained layer.
    resume_from_layer = 0
    if layer_ckpt_active:
        completed = scan_completed_layers(layer_ckpt_dir, num_layers)
        if completed and not args.no_resume:
            M = completed[-1]
            # Validate config against what was stored (rank 0 warns; non-fatal).
            prog_path = os.path.join(layer_ckpt_dir, "progress.pt")
            if is_main and os.path.exists(prog_path):
                try:
                    saved_cfg = torch.load(prog_path, weights_only=False).get("config", {})
                    cur_cfg = resume_config_dict(args, num_layers)
                    mismatch = {k: (saved_cfg.get(k), cur_cfg[k])
                                for k in cur_cfg if saved_cfg.get(k) != cur_cfg[k]}
                    if mismatch:
                        print(f"WARNING: resume config mismatch (saved vs current): {mismatch}")
                        log_file.write(f"Resume config mismatch: {mismatch}\n")
                except Exception as e:
                    log_file.write(f"Resume config check skipped: {e}\n")

            if is_main:
                print(f"\n{'#'*60}")
                print(f"# Layer-checkpoint resume: layers 0..{M} done, "
                      f"continuing at {M + 1}/{num_layers - 1}")
                print(f"{'#'*60}")
            log_file.write(f"Layer-checkpoint resume: loading layers 0..{M}\n")

            # Load each completed layer: baked weights into the student (all ranks,
            # needed for re-propagation), accumulators on rank 0 (for final save).
            for i in completed:
                data = torch.load(layer_ckpt_path(layer_ckpt_dir, i), weights_only=False)
                student.model.layers[i].load_state_dict(data["baked_state_dict"])
                student.model.layers[i].half()
                if is_main:
                    all_pre_baked_weights.update(data["pre_baked_weights"])
                    all_quant_params.update(data["quant_params"])
                    if data.get("layer_time"):
                        layer_times.append(data["layer_time"])
                del data
            release_memory()

            # Rebuild input activations for layer M+1 by re-propagation (each rank
            # processes its own sample shard). quant_hiddens through baked student
            # layers, fp_hiddens through the (still-loaded) teacher layers.
            per_rank_batch = args.batch_size // world_size
            repropagate_hiddens_upto(student, quant_hiddens, position_ids_list,
                                     rotary_emb, per_rank_batch, dev, M,
                                     log_file, tag="student")
            repropagate_hiddens_upto(teacher, fp_hiddens, position_ids_list,
                                     rotary_emb, per_rank_batch, dev, M,
                                     log_file, tag="teacher")

            # Free the teacher layers we just consumed (matches steady-state flow).
            for i in completed:
                teacher.model.layers[i] = None
            release_memory()

            resume_from_layer = M + 1
            if world_size > 1:
                dist.barrier()

    # ── Layer-wise training ──────────────────────────────────────────────
    total_t0 = time.time()
    layer_order = list(range(resume_from_layer, num_layers))

    if layer_order:
        if is_main:
            print(f"\n{'#'*60}")
            print(f"# Layer-wise training (layers {layer_order[0]}→{layer_order[-1]})")
            print(f"{'#'*60}")
        log_file.write(f"\n{'='*60}\n=== Layer-wise training "
                       f"(layers {layer_order[0]}→{layer_order[-1]}) ===\n{'='*60}\n")

    for layer_idx in layer_order:
        if is_main:
            print(f"\n{'='*60}")
            print(f"Processing layer {layer_idx}/{num_layers-1}")

        fp_layer = teacher.model.layers[layer_idx]
        quant_layer = student.model.layers[layer_idx]
        quant_layer.cpu()

        wrap_layer_with_qlinear(
            quant_layer,
            layer_name_prefix=f"model.layers.{layer_idx}",
            w_bit=args.w_bit,
            w_group_size=args.w_group_size,
            enable_lwc=args.enable_lwc,
            init_qparams=checkpoint_qparams,
        )
        if args.enable_channel_scale:
            setup_channel_scales(quant_layer)

        # Restore LWC / channel_scale from checkpoint qparams
        if checkpoint_qparams is not None:
            restore_checkpoint_qparams(
                layer_idx, quant_layer, checkpoint_qparams, logger=log_file,
            )

        pre_baked_weights, quant_params = train_one_layer(
            layer_idx=layer_idx,
            num_layers=num_layers,
            fp_layer=fp_layer,
            quant_layer=quant_layer,
            fp_hiddens=fp_hiddens,
            quant_hiddens=quant_hiddens,
            position_ids_list=position_ids_list,
            rotary_emb=rotary_emb,
            args=args,
            dev=dev,
            rank=rank,
            world_size=world_size,
            logger=log_file,
            layer_times=layer_times,
            teacher_model=teacher,
            window_layers=_get_window_layers(teacher, layer_idx, num_layers, args.window_size),
        )

        if not args.save_pre_baked:
            pre_baked_weights = {}
        all_pre_baked_weights.update(pre_baked_weights)
        all_quant_params.update(quant_params)

        # Resumable per-layer checkpoint. quant_layer is now the baked
        # (fake-quant) nn.Linear layer inside the student.
        if layer_ckpt_active:
            if is_main:
                save_layer_checkpoint(
                    layer_ckpt_dir, layer_idx, student.model.layers[layer_idx],
                    pre_baked_weights, quant_params,
                    layer_times[-1] if layer_times else None,
                )
                write_progress(
                    layer_ckpt_dir, list(range(layer_idx + 1)),
                    resume_config_dict(args, num_layers),
                )
                log_file.write(f"  Layer {layer_idx}: checkpoint saved\n")
            if world_size > 1:
                dist.barrier()

        # Teacher layer no longer needed (window layers of later steps are ahead of it)
        teacher.model.layers[layer_idx] = None

        del fp_layer, pre_baked_weights, quant_params
        release_memory()

        if world_size > 1:
            dist.barrier()

    total_elapsed = time.time() - total_t0
    log_file.write(f"\nTotal training time: {format_time(total_elapsed)}\n")

    # ── Save (rank 0 only) ───────────────────────────────────────────────
    if is_main:
        print("\nSaving fake-quantized checkpoint...")
        save_checkpoint(student, args.teacher_model, args.output_dir)
        log_file.write(f"Fake-quantized checkpoint saved to {args.output_dir}\n")

        if args.save_pre_baked:
            pre_baked_dir = os.path.join(args.output_dir, "pre-baked")
            print("Saving pre-baked (continuous) checkpoint...")
            sd = student.state_dict()
            for key, tensor in all_pre_baked_weights.items():
                sd[key] = tensor
            student.load_state_dict(sd)
            del sd
            save_checkpoint(student, args.teacher_model, pre_baked_dir)
            log_file.write(f"Pre-baked checkpoint saved to {pre_baked_dir}\n")
        del all_pre_baked_weights

        qp_path = os.path.join(args.output_dir, "quant_params.pt")
        torch.save(all_quant_params, qp_path)
        log_file.write(f"Learned scales/zeros saved to {qp_path} ({len(all_quant_params)} modules)\n")

        # Final checkpoints are written — the per-layer resume cache is now
        # redundant. Remove it only if explicitly requested.
        if layer_ckpt_active and args.layer_checkpoint_cleanup and os.path.isdir(layer_ckpt_dir):
            shutil.rmtree(layer_ckpt_dir)
            print(f"Removed layer checkpoint cache: {layer_ckpt_dir}")
            log_file.write(f"Removed layer checkpoint cache: {layer_ckpt_dir}\n")

    log_file.write("\nDone.\n")
    log_file.close()

    # Clean up disk-backed caches
    if isinstance(fp_hiddens, DiskBackedHiddens):
        fp_hiddens.cleanup()
    if isinstance(quant_hiddens, DiskBackedHiddens):
        quant_hiddens.cleanup()

    cleanup_distributed()
    if is_main:
        print("Done!")


if __name__ == "__main__":
    main()
