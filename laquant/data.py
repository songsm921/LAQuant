"""Calibration data (tokenization, batching) and cached per-layer hidden states."""
import os
import random
import shutil

import torch
from datasets import load_from_disk

from .utils import release_memory



# ── Disk-backed hidden state cache ──────────────────────────────────────

class DiskBackedHiddens:
    """Drop-in replacement for list[Tensor] that stores tensors on local disk.

    Each sample is saved as a raw binary file.  An async prefetch thread
    loads the next batch while the GPU processes the current one, hiding
    nearly all I/O latency when GPU time >= disk-read time.
    """

    def __init__(self, cache_dir, name, n=0, lengths=None, hidden_size=0,
                 dtype=torch.bfloat16):
        self._dir = os.path.join(cache_dir, name)
        os.makedirs(self._dir, exist_ok=True)
        self._n = n
        self._lengths = list(lengths) if lengths else []
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._np_dtype = 'bfloat16'  # numpy dtype string
        # Prefetch state
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix="diskio")
        self._prefetch_cache = {}  # idx -> Tensor (CPU)
        self._pending = None       # Future

    @classmethod
    def from_list(cls, tensors, cache_dir, name):
        """Flush a list[Tensor] to disk and return a DiskBackedHiddens."""
        n = len(tensors)
        lengths = [t.shape[1] for t in tensors]
        hidden_size = tensors[0].shape[2]
        dtype = tensors[0].dtype
        obj = cls(cache_dir, name, n=n, lengths=lengths,
                  hidden_size=hidden_size, dtype=dtype)
        for i, t in enumerate(tensors):
            obj._write(i, t)
        return obj

    # ── low-level I/O ──────────────────────────────────────────────────

    def _path(self, idx):
        return os.path.join(self._dir, f"{idx}.bin")

    def _write(self, idx, tensor):
        """Write tensor to disk as raw bytes."""
        # Ensure consistent dtype (autocast may produce FP16 instead of BF16)
        t = tensor.to(dtype=self._dtype).contiguous()
        raw = t.view(torch.uint8).numpy().tobytes()
        with open(self._path(idx), 'wb') as f:
            f.write(raw)

    def _read(self, idx):
        """Read tensor from disk."""
        seq_len = self._lengths[idx]
        nbytes = 1 * seq_len * self._hidden_size * 2  # bfloat16 = 2 bytes
        with open(self._path(idx), 'rb') as f:
            raw = f.read(nbytes)
        t = torch.frombuffer(bytearray(raw), dtype=self._dtype)
        return t.view(1, seq_len, self._hidden_size)

    # ── list-like interface ────────────────────────────────────────────

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if idx in self._prefetch_cache:
            return self._prefetch_cache.pop(idx)
        return self._read(idx)

    def __setitem__(self, idx, tensor):
        self._lengths[idx] = tensor.shape[1]
        self._write(idx, tensor)
        self._prefetch_cache.pop(idx, None)

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    def get_length(self, idx):
        """Get seq_len without loading from disk."""
        return self._lengths[idx]

    # ── async prefetch ─────────────────────────────────────────────────

    def prefetch(self, indices):
        """Start async loading of a batch of samples."""
        self.wait_prefetch()
        # Only prefetch indices not already cached
        to_load = [i for i in indices if i not in self._prefetch_cache]
        if not to_load:
            return
        def _load(idxs):
            result = {}
            for i in idxs:
                result[i] = self._read(i)
            return result
        self._pending = self._executor.submit(_load, to_load)

    def wait_prefetch(self):
        """Block until pending prefetch completes, merge into cache."""
        if self._pending is not None:
            self._prefetch_cache.update(self._pending.result())
            self._pending = None

    def clear_cache(self):
        """Drop all prefetched tensors."""
        self._prefetch_cache.clear()

    # ── cleanup ────────────────────────────────────────────────────────

    def cleanup(self):
        """Remove all temp files."""
        self._executor.shutdown(wait=False)
        if os.path.exists(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def get_hidden_length(hiddens, idx):
    """Get seq_len from hiddens[idx] without necessarily loading from disk."""
    if isinstance(hiddens, DiskBackedHiddens):
        return hiddens.get_length(idx)
    return hiddens[idx].shape[1]


# ── Dataset ──────────────────────────────────────────────────────────────
def load_calib_dataset(path):
    """Load the calibration set from a local save_to_disk directory (see README, Data)."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"--dataset_path {path} is not a local directory. Download the calibration "
            "set first (see README, Data).")
    dataset = load_from_disk(path)
    if "train" in dataset:
        dataset = dataset["train"]
    return dataset


def _find_response_start(token_ids, assistant_marker_ids):
    marker_len = len(assistant_marker_ids)
    for i in range(len(token_ids) - marker_len + 1):
        if token_ids[i:i + marker_len] == assistant_marker_ids:
            return i + marker_len
    return 0


def tokenize_dataset(dataset, tokenizer, max_seq_length, max_samples=0,
                     token_mode="prompt", response_pos="early",
                     shuffle_samples=False, shuffle_seed=42):
    if token_mode == "response":
        assistant_marker_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    samples = []
    skipped = 0
    end = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    if shuffle_samples:
        rng = random.Random(shuffle_seed)
        order = list(range(len(dataset)))
        rng.shuffle(order)
        indices = order[:end]
    else:
        indices = range(end)
    n = len(indices)
    L = max_seq_length
    for k, i in enumerate(indices):
        text = dataset[i]["text"]
        full_ids = tokenizer.encode(text, add_special_tokens=False)

        if token_mode == "prompt":
            ids = full_ids[:L]
        elif token_mode == "response":
            resp_start = _find_response_start(full_ids, assistant_marker_ids)
            resp_ids = full_ids[resp_start:]
            N = len(resp_ids)
            if response_pos == "early":
                ids = resp_ids[:L]
            elif response_pos == "end":
                ids = resp_ids[-L:] if N > L else resp_ids
            elif response_pos == "middle":
                if N >= 3 * L:
                    start = random.randint(L, N - 2 * L - 1)
                    ids = resp_ids[start:start + L]
                elif N > L:
                    start = random.randint(0, N - L)
                    ids = resp_ids[start:start + L]
                else:
                    ids = resp_ids

        if len(ids) == 0:
            skipped += 1
            continue
        samples.append(torch.tensor([ids], dtype=torch.long))
        if (k + 1) % 2000 == 0:
            print(f"  Tokenized {k+1}/{n} samples...")

    if skipped > 0:
        print(f"  Skipped {skipped} empty samples")
    return samples


def make_batches(indices, batch_size):
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def pad_and_batch(tensors, batch_indices):
    selected = [tensors[i] for i in batch_indices]
    lengths = [t.shape[1] for t in selected]
    max_len = max(lengths)
    hidden = selected[0].shape[-1]

    padded = []
    for t in selected:
        if t.shape[1] < max_len:
            pad = torch.zeros(1, max_len - t.shape[1], hidden, dtype=t.dtype)
            padded.append(torch.cat([t, pad], dim=1))
        else:
            padded.append(t)

    return torch.cat(padded, dim=0), lengths


def batch_position_ids(position_ids_list, batch_indices, max_len):
    batch = []
    for i in batch_indices:
        pos = position_ids_list[i]
        seq_len = pos.shape[1]
        if seq_len < max_len:
            pad = torch.zeros(1, max_len - seq_len, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=1)
        batch.append(pos)
    return torch.cat(batch, dim=0)


# ── Layer-by-layer propagation of cached hidden states ───────────────────
@torch.no_grad()
def cache_layer_activations(model, samples, dev):
    embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        rotary_emb = model.model.rotary_emb.to(dev)
    else:
        rotary_emb = None

    all_hidden = []
    for i, sample in enumerate(samples):
        input_ids = sample.to(dev)
        hidden = embed_tokens(input_ids)
        all_hidden.append(hidden.cpu())
        if (i + 1) % 2000 == 0:
            print(f"  Embedded {i+1}/{len(samples)} samples...")

    all_position_ids = []
    for sample in samples:
        pos_ids = torch.arange(sample.shape[1], device="cpu").unsqueeze(0)
        all_position_ids.append(pos_ids)

    embed_tokens.cpu()
    if rotary_emb is not None:
        rotary_emb.cpu()
    release_memory()

    print(f"Cached {len(all_hidden)} samples from embedding layer")
    return all_hidden, all_position_ids, rotary_emb


# ── Single-GPU forward (in-place) ─────────────────────────────────────────
@torch.no_grad()
def forward_layer_inplace(layer, rotary_emb, hiddens, position_ids_list,
                          batches, dev, logger=None):
    """Forward all samples through a layer, overwriting hiddens[] in-place."""
    layer.to(dev)
    if rotary_emb is not None:
        rotary_emb.to(dev)

    _disk = isinstance(hiddens, DiskBackedHiddens)
    if _disk and len(batches) > 0:
        hiddens.prefetch(batches[0])

    with torch.amp.autocast('cuda'):
        for bi, batch_idx in enumerate(batches):
            if _disk:
                hiddens.wait_prefetch()
            inp_batch, lengths = pad_and_batch(hiddens, batch_idx)
            # Prefetch next batch while GPU processes current one
            if _disk and bi + 1 < len(batches):
                hiddens.prefetch(batches[bi + 1])
            inp_batch = inp_batch.to(dev)
            max_len = inp_batch.shape[1]
            pos_batch = batch_position_ids(position_ids_list, batch_idx, max_len).to(dev)

            pos_embed = rotary_emb(inp_batch, pos_batch) if rotary_emb is not None else None
            out = layer(
                inp_batch,
                position_embeddings=pos_embed,
                position_ids=pos_batch,
            )
            if isinstance(out, tuple):
                out = out[0]
            for j, idx in enumerate(batch_idx):
                hiddens[idx] = out[j:j+1, :lengths[j], :].cpu()

    if logger:
        logger.write(f"  Forward: {len(batches)} batches\n")
        logger.flush()

    layer.cpu()
    if rotary_emb is not None:
        rotary_emb.cpu()
    release_memory()


def repropagate_hiddens_upto(model, hiddens, position_ids_list, rotary_emb,
                             batch_size, dev, upto_idx, logger=None, tag=""):
    """Forward all samples through model layers 0..upto_idx (inclusive), in place.

    Used on resume to rebuild the input activations for layer upto_idx+1 by
    replaying the embedding cache through the already-completed layers.
    """
    num_samples = len(hiddens)
    sorted_indices = sorted(range(num_samples),
                            key=lambda i: get_hidden_length(hiddens, i))
    batches = make_batches(sorted_indices, batch_size)
    for layer_idx in range(upto_idx + 1):
        layer = model.model.layers[layer_idx]
        if layer is None:
            raise RuntimeError(
                f"{tag} layer {layer_idx} is None during resume re-propagation")
        forward_layer_inplace(layer, rotary_emb, hiddens, position_ids_list,
                              batches, dev)
        if logger and (layer_idx + 1) % 7 == 0:
            logger.write(f"  [{tag}] re-propagated through layer "
                         f"{layer_idx}/{upto_idx}\n")
            logger.flush()
    if logger:
        logger.write(f"  [{tag}] re-propagated hiddens through layers 0..{upto_idx} "
                     f"({num_samples} samples)\n")
        logger.flush()
