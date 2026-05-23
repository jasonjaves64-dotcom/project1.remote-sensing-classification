"""
Async batch preloader for DataLoader integration.

Runs a background thread that prefetches the next batch while the GPU
processes the current one. Uses a double-buffer pattern with CUDA
streams for overlap when available.
"""
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Iterator, Dict, Any, List

import numpy as np
import torch


@dataclass
class PreloadStats:
    """Preloader performance statistics."""
    batches_served: int = 0
    batches_prefetched: int = 0
    prefetch_hits: int = 0        # batch was ready before it was needed
    prefetch_misses: int = 0      # batch wasn't ready, consumer waited
    total_wait_time_ms: float = 0.0
    total_prefetch_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0


class AsyncPreloader:
    """Background prefetcher that loads the next batch while GPU computes.

    Uses a queue-based double-buffer: while the training loop consumes
    batch N, the preloader thread is already loading batch N+1.

    Usage:
        preloader = AsyncPreloader(dataloader, prefetch_depth=2)
        preloader.start()
        for batch in preloader:
            # batch is already on device (if pin_memory used)
            ...
        preloader.stop()
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        prefetch_depth: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.dataloader = dataloader
        self.prefetch_depth = max(1, prefetch_depth)
        self.device = device

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._iterator: Optional[Iterator] = None
        self.stats = PreloadStats()

    def start(self):
        """Spin up the background prefetch thread."""
        self._stop_event.clear()
        self._iterator = iter(self.dataloader)
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to exit and join."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        # Drain remaining items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop_event.is_set() and self._queue.empty():
            raise StopIteration

        t0 = time.perf_counter()
        try:
            item = self._queue.get(timeout=30.0)
        except queue.Empty:
            raise StopIteration

        wait_ms = (time.perf_counter() - t0) * 1000
        self.stats.total_wait_time_ms += wait_ms

        if isinstance(item, Exception):
            raise item

        if item is None:  # sentinel for end of data
            raise StopIteration

        if wait_ms < 1.0:
            self.stats.prefetch_hits += 1
        else:
            self.stats.prefetch_misses += 1

        self.stats.batches_served += 1

        if self.device is not None:
            item = self._move_to_device(item)

        return item

    def __len__(self):
        return len(self.dataloader)

    def _prefetch_loop(self):
        """Background thread: iterate the dataloader and push into queue."""
        try:
            for batch in self._iterator:
                if self._stop_event.is_set():
                    break
                t0 = time.perf_counter()
                # Pin to CPU memory if not already done by DataLoader
                batch = self._pin_batch(batch)
                self.stats.total_prefetch_time_ms += (time.perf_counter() - t0) * 1000
                self.stats.batches_prefetched += 1
                self._queue.put(batch)
        except Exception as e:
            self._queue.put(e)
        finally:
            self._queue.put(None)  # sentinel

    def _pin_batch(self, batch):
        """Ensure tensors are in pinned memory for fast GPU transfer."""
        if not torch.cuda.is_available():
            return batch
        if isinstance(batch, dict):
            return {
                k: v.pin_memory() if isinstance(v, torch.Tensor) and not v.is_pinned() else v
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.pin_memory() if isinstance(v, torch.Tensor) and not v.is_pinned() else v
                for v in batch
            )
        return batch

    def _move_to_device(self, batch):
        """Non-blocking transfer of tensors to device."""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for v in batch
            )
        return batch


class StreamingPrefetchLoader:
    """Drop-in replacement for DataLoader with built-in async prefetching.

    Wraps a standard DataLoader and adds background prefetch without
    changing existing training loop structure significantly.

    Usage:
        loader = StreamingPrefetchLoader(dataloader, device=device)
        for batch in loader:
            # batch is already on the target device
            train_step(batch)
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        prefetch_depth: int = 2,
    ):
        self._preloader = AsyncPreloader(dataloader, prefetch_depth, device)
        self._preloader.start()

    def __iter__(self):
        return self._preloader.__iter__()

    def __len__(self):
        return len(self._preloader)

    def close(self):
        self._preloader.stop()

    @property
    def stats(self) -> PreloadStats:
        return self._preloader.stats

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
