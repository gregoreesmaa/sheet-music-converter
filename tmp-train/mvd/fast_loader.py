"""Width-bucketed deterministic batching and async background prefetching.

1. **Width-bucketed batching** (:func:`measure_pair_widths`,
   :func:`epoch_bucketed_batches`):
   Rectified system strips span ``W = 600..3500 px``. Random batching pads
   every strip in a batch out to ``max(W)``, wasting 35-45% of ConvNeXt 4x8
   and cross-attention FLOPs on blank white padding. Width-bucketing groups
   similar-width strips into micro-batches (with slight per-epoch width jitter
   so micro-batch membership varies across epochs), then shuffles the batch
   order deterministically per ``(seed, epoch)`` so ``--resume`` is exact.

2. **Async background prefetching** (:class:`AsyncBatchPrefetcher`):
   Overlaps disk I/O + PNG/NPZ decoding + CPU tensor stacking with GPU
   forward/backward using a bounded background worker pool + optional
   ``pin_memory()`` for ``non_blocking=True`` H2D transfers.
"""

from __future__ import annotations

import hashlib
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image


def _det_seed(base_seed: int, epoch: int, salt: str = 'wbucket') -> int:
    payload = f'{int(base_seed)}:{int(epoch)}:{salt}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')


def measure_pair_widths(
    pairs: list[tuple[str, str, str]],
    height: int = 256,
    hybrid_index: dict | None = None,
) -> list[int]:
    """Fast header-only width estimate (at target ``height``) for each pair.

    Uses ``hybrid_index`` metadata when available (zero disk opens), else reads
    only the image header via ``PIL.Image.open(p).size`` (no pixel decode).
    """
    def _one_width(pair):
        img_path = pair[0]
        if hybrid_index is not None:
            row = hybrid_index.get(Path(img_path).name)
            if isinstance(row, dict):
                rw = row.get('rect_w') or row.get('width')
                if isinstance(rw, (int, float)) and rw > 0:
                    return int(round(rw))
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
            return max(1, int(round(iw * height / max(ih, 1))))
        except Exception:
            return height * 4

    if len(pairs) >= 64:
        with ThreadPoolExecutor(max_workers=16) as pool:
            return list(pool.map(_one_width, pairs, chunksize=64))
    return [_one_width(p) for p in pairs]


def epoch_bucketed_batches(
    pairs: list[tuple[str, str, str]],
    widths: list[int],
    batch_size: int,
    base_seed: int,
    epoch: int,
    jitter_frac: float = 0.08,
) -> list[list[tuple[str, str, str]]]:
    """Build deterministic width-bucketed batches for ``epoch``.

    Adds small multiplicative jitter (``+/- jitter_frac``) to widths before
    sorting so strips near bucket boundaries mix across epochs, slices into
    batches of ``batch_size``, then shuffles batch order with ``(base_seed, epoch)``.
    """
    B = max(1, int(batch_size))
    rng = random.Random(_det_seed(base_seed, epoch, salt='wbucket'))
    order = list(range(len(pairs)))
    if jitter_frac > 0.0:
        keys = [widths[i] * (1.0 + rng.uniform(-jitter_frac, jitter_frac))
                for i in order]
        order.sort(key=lambda i: keys[i])
    else:
        order.sort(key=lambda i: widths[i])

    batches = [
        [pairs[i] for i in order[k:k + B]]
        for k in range(0, len(order), B)
    ]
    rng.shuffle(batches)
    return batches


def padding_overhead_ratio(
    batches: list[list[tuple[str, str, str]]],
    width_by_img: dict[str, int],
) -> float:
    """Ratio of padded pixel area to true pixel area across all batches (>= 1.0)."""
    padded = 0
    actual = 0
    for b in batches:
        ws = [width_by_img[p[0]] for p in b]
        if not ws:
            continue
        padded += max(ws) * len(ws)
        actual += sum(ws)
    return float(padded) / max(float(actual), 1.0)


def _pin_item(obj):
    """Recursively pin CPU tensors for ``non_blocking=True`` CUDA H2D."""
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory() if obj.device.type == 'cpu' else obj
    if isinstance(obj, tuple):
        return tuple(_pin_item(v) for v in obj)
    if isinstance(obj, list):
        return [_pin_item(v) for v in obj]
    return obj


class AsyncBatchPrefetcher:
    """Ordered background batch prefetcher with optional pinned memory.

    Submits ``collate_fn(step, epoch, batch)`` jobs up to ``prefetch`` steps
    ahead on a ``ThreadPoolExecutor`` and yields ``(step, epoch, batch, collated)``
    in exact deterministic step order.
    """

    def __init__(
        self,
        step_iter,
        collate_fn,
        num_workers: int = 2,
        prefetch: int = 4,
        pin_memory: bool = False,
    ):
        self._step_iter = iter(step_iter)
        self._collate_fn = collate_fn
        self._num_workers = max(1, int(num_workers))
        self._prefetch = max(1, int(prefetch))
        self._pin = bool(pin_memory)
        self._pool: ThreadPoolExecutor | None = None
        self._q: queue.Queue = queue.Queue(maxsize=self._prefetch)
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None

    def _run_one(self, item):
        step, epoch, batch = item
        out = self._collate_fn(step, epoch, batch)
        if self._pin:
            out = _pin_item(out)
        return (step, epoch, batch, out)

    def _produce(self):
        assert self._pool is not None
        try:
            for item in self._step_iter:
                if self._stop.is_set():
                    break
                fut = self._pool.submit(self._run_one, item)
                while not self._stop.is_set():
                    try:
                        self._q.put(fut, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self._q.put(exc)
        finally:
            self._q.put(None)

    def __enter__(self):
        self._pool = ThreadPoolExecutor(max_workers=self._num_workers)
        self._producer = threading.Thread(target=self._produce, daemon=True)
        self._producer.start()
        return self

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item.result()

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._producer is not None:
            self._producer.join(timeout=2.0)
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        return False
