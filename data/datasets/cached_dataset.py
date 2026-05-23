"""
LRU-cached Dataset wrapper with tiered storage.

Wraps any PyTorch Dataset and caches __getitem__ results in a three-tier
cache (hot memory / warm SSD / cold source). Supports:

- Transparent cache lookup on every __getitem__ call
- Warm-tier persistence across runs (SSD-backed)
- Cache invalidation on manifest change (dataset version mismatch)
- Cache hit/miss statistics
- Auto-fallback to the underlying dataset on cache miss
"""
import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..cache.lru_cache import TieredLRUCache, CacheStats
from ..cache.manifest import DatasetManifest


_CACHE_DIR = Path(__file__).parent.parent / "cache" / ".warm_cache"


class CachedDataset(Dataset):
    """Wraps a Dataset with tiered LRU caching of per-sample results.

    The cold loader reads from the underlying dataset on cache miss.
    Hot tier is in-process memory, warm tier is SSD-backed for persistence.

    Usage:
        base_ds = FusionCropDatasetEDL(...)
        cached_ds = CachedDataset(
            base_ds,
            hot_capacity=512,
            warm_cache_dir="data/cache/.warm_cache",
            dataset_name="fusion_2023",
        )
        loader = DataLoader(cached_ds, batch_size=8, num_workers=4)
    """

    def __init__(
        self,
        dataset: Dataset,
        hot_capacity: int = 512,
        warm_cache_dir: Optional[str] = None,
        max_warm_gb: int = 4,
        dataset_name: str = "default",
        manifest_path: Optional[str] = None,
        track_stats: bool = True,
    ):
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._track_stats = track_stats

        # Set up warm cache directory
        warm_dir = warm_cache_dir or str(_CACHE_DIR / dataset_name)
        os.makedirs(warm_dir, exist_ok=True)

        # Build or load manifest for cache invalidation
        self._manifest = None
        if manifest_path and os.path.exists(manifest_path):
            mgr = DatasetManifest(os.path.dirname(manifest_path))
            try:
                mgr.load(manifest_path)
                self._manifest = mgr.manifest
            except Exception:
                pass

        # Cold loader: fetch from underlying dataset
        def _cold_loader(key: str) -> Optional[Dict[str, np.ndarray]]:
            try:
                idx = int(key.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                return None
            sample = self._dataset[idx]
            return _sample_to_numpy(sample)

        self._cache = TieredLRUCache(
            hot_capacity=hot_capacity,
            warm_dir=warm_dir,
            max_warm_bytes=max_warm_gb * 1024 * 1024 * 1024,
            cold_loader=_cold_loader,
        )

        # Cache invalidation on manifest change
        if self._manifest:
            self._check_and_invalidate()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)

        cache_key = self._make_key(idx)
        cached = self._cache.get(cache_key)

        if cached is not None:
            return _numpy_to_sample(cached)

        # Cache miss: read from source, then cache
        sample = self._dataset[idx]
        np_sample = _sample_to_numpy(sample)
        self._cache.put(cache_key, np_sample)

        # Also persist to warm tier every N samples
        if idx % 100 == 0:
            self._cache.promote_to_warm(cache_key, np_sample)

        return sample

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def warm_fill(self, indices: Optional[list] = None, max_items: int = 1000):
        """Pre-populate cache with specified indices. Useful before training."""
        if indices is None:
            n = len(self)
            step = max(1, n // max_items)
            indices = list(range(0, n, step))[:max_items]

        for i in indices:
            self[i]  # triggers cache put

    def get_cache_stats(self) -> CacheStats:
        return self._cache.get_stats()

    def reset_cache_stats(self):
        self._cache.reset_stats()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def invalidate(self, idx: int):
        """Invalidate a specific sample's cache entry."""
        self._cache.invalidate(self._make_key(idx))

    def get_dataset_info(self) -> Dict[str, Any]:
        stats = self._cache.get_stats() if self._track_stats else None
        return {
            "dataset_name": self._dataset_name,
            "num_samples": len(self),
            "hot_capacity": self._cache._hot_capacity,
            "cache_stats": stats.to_dict() if stats else {},
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _make_key(self, idx: int) -> str:
        if self._manifest:
            return f"{self._manifest.dataset_uuid}_{self._dataset_name}_{idx}"
        return f"{self._dataset_name}_{idx}"

    def _check_and_invalidate(self):
        """Invalidate warm cache if manifest is stale."""
        mgr = DatasetManifest(os.path.dirname(self._manifest) if hasattr(self._manifest, 'path') else ".")
        if mgr.is_stale():
            self._cache.clear()


def _sample_to_numpy(sample) -> Dict[str, np.ndarray]:
    """Convert a sample (dict of tensors/arrays) to numpy dict for caching."""
    result = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            result[k] = v.copy()
        else:
            result[k] = np.array(v)
    return result


def _numpy_to_sample(np_dict: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Convert cached numpy dict back to sample format with tensors."""
    result = {}
    for k, v in np_dict.items():
        if k in ("opt", "sar", "dem"):
            result[k] = torch.from_numpy(v).float()
        elif k in ("y", "label"):
            result[k] = torch.from_numpy(v).long()
        else:
            result[k] = torch.from_numpy(v).float() if v.dtype.kind == "f" else v
    return result


class DatasetBenchmark:
    """Benchmark helper that measures DataLoader throughput."""

    def __init__(self, loader: torch.utils.data.DataLoader, warmup_batches: int = 5):
        self.loader = loader
        self.warmup_batches = warmup_batches

    def measure(self, max_batches: int = 100) -> Dict[str, Any]:
        """Measure throughput in samples/sec by iterating the dataloader."""
        import time

        # Warmup
        for i, _ in enumerate(self.loader):
            if i >= self.warmup_batches:
                break

        # Timed run
        total_samples = 0
        total_batches = 0
        t0 = time.perf_counter()

        for batch in self.loader:
            # Determine batch size from first tensor
            if isinstance(batch, dict):
                first_val = next(iter(batch.values()))
                bs = first_val.shape[0] if hasattr(first_val, "shape") else 1
            elif isinstance(batch, (list, tuple)):
                bs = batch[0].shape[0] if hasattr(batch[0], "shape") else 1
            else:
                bs = 1

            total_samples += bs
            total_batches += 1

            if total_batches >= max_batches:
                break

        elapsed = time.perf_counter() - t0
        throughput = total_samples / elapsed if elapsed > 0 else 0.0

        return {
            "total_samples": total_samples,
            "total_batches": total_batches,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "avg_batch_time_ms": round((elapsed / total_batches) * 1000, 2) if total_batches > 0 else 0,
        }
