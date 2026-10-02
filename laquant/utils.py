"""Distributed setup, memory management, logging and reconstruction losses."""
import ctypes
import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F




# ── Memory management ────────────────────────────────────────────────────
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def _malloc_trim():
        pass


def release_memory():
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()


# ── Distributed setup ────────────────────────────────────────────────────
def setup_distributed():
    """Initialize DDP. Falls back to single GPU if not launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Logging ──────────────────────────────────────────────────────────────
class Logger:
    """File + stdout logger. Rank-aware: only rank 0 writes."""
    def __init__(self, path, rank=0):
        self._rank = rank
        self._file = open(path, "w") if rank == 0 else None

    def write(self, msg):
        if self._rank == 0:
            print(msg, end="")
            self._file.write(msg)

    def flush(self):
        if self._rank == 0 and self._file:
            self._file.flush()

    def log(self, msg):
        """Write to log file only (no stdout). For epoch lines when tqdm is active."""
        if self._rank == 0 and self._file:
            self._file.write(msg)

    def close(self):
        if self._file:
            self._file.close()


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ── Losses ───────────────────────────────────────────────────────────────
def masked_mse_loss(pred, target, lengths):
    batch_size, max_len, hidden = pred.shape
    mask = torch.zeros(batch_size, max_len, 1, device=pred.device, dtype=pred.dtype)
    for b, l in enumerate(lengths):
        mask[b, :l, :] = 1.0
    diff = (pred - target) ** 2 * mask
    n_elements = sum(lengths) * hidden
    return diff.sum() / n_elements


def masked_smooth_l1_loss(pred, target, lengths, beta=1.0):
    batch_size, max_len, hidden = pred.shape
    mask = torch.zeros(batch_size, max_len, 1, device=pred.device, dtype=pred.dtype)
    for b, l in enumerate(lengths):
        mask[b, :l, :] = 1.0
    diff = F.smooth_l1_loss(pred, target, reduction='none', beta=beta) * mask
    n_elements = sum(lengths) * hidden
    return diff.sum() / n_elements
