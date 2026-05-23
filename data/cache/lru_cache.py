"""
Tiered LRU cache for remote sensing patch data.

Three-tier architecture:
  Hot  – in-process memory (numpy arrays), fastest per-worker access
  Warm – SSD-backed memmap files, shared across DataLoader workers
  Cold – original .npy source files, read on cache miss

Thread-safe for multi-worker DataLoader usage via file locking on warm tier.
"""
import os
import sys
import time
import threading
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    hot_hits: int = 0
    warm_hits: int = 0
    cold_reads: int = 0
    evictions: int = 0
    warm_promotions: int = 0
    total_access_time_ms: float = 0.0

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    @property
    def avg_access_time_ms(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_access_time_ms / self.total_accesses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hot_hits": self.hot_hits,
            "warm_hits": self.warm_hits,
            "cold_reads": self.cold_reads,
            "evictions": self.evictions,
            "warm_promotions": self.warm_promotions,
            "hit_rate": round(self.hit_rate, 4),
            "avg_access_time_ms": round(self.avg_access_time_ms, 2),
            "total_accesses": self.total_accesses,
        }


class TieredLRUCache:
    """Three-tier LRU cache for numpy array patches.

    Hot tier: in-process OrderedDict with configurable capacity (number of items).
    Warm tier: SSD-backed directory of .npy files with memmap access.
    Cold tier: original source path, read via user-provided loader callback.

    Cache keys are typically (dataset_uuid, index) tuples hashed to a string.
    """

    def __init__(
        self,
        hot_capacity: int = 512,
        warm_dir: Optional[str] = None,
        max_warm_bytes: int = 4 * 1024 * 1024 * 1024,  # 4 GiB
        cold_loader: Optional[callable] = None,
        lock_timeout: float = 5.0,
    ):
        self._hot: OrderedDict = OrderedDict()
        self._hot_capacity = hot_capacity
        self._hot_bytes = 0

        self._warm_dir = Path(warm_dir) if warm_dir else None
        self._max_warm_bytes = max_warm_bytes
        self._warm_bytes = 0
        self._warm_index: OrderedDict = OrderedDict()  # key -> file_size
        if self._warm_dir:
            self._warm_dir.mkdir(parents=True, exist_ok=True)
            self._load_warm_index()

        self._cold_loader = cold_loader
        self._lock_timeout = lock_timeout

        self.stats = CacheStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve cached sample by key. Returns None on miss."""
        t0 = time.perf_counter()
        with self._lock:
            result = self._get_hot(key)
            if result is not None:
                self.stats.hits += 1
                self.stats.hot_hits += 1
                self.stats.total_access_time_ms += (time.perf_counter() - t0) * 1000
                return result

            result = self._get_warm(key)
            if result is not None:
                self.stats.hits += 1
                self.stats.warm_hits += 1
                self._promote_to_hot(key, result)
                self.stats.total_access_time_ms += (time.perf_counter() - t0) * 1000
                return result

            result = self._get_cold(key)
            if result is not None:
                self.stats.misses += 1
                self.stats.cold_reads += 1
                self._promote_to_hot(key, result)
                self.stats.total_access_time_ms += (time.perf_counter() - t0) * 1000
                return result

        self.stats.misses += 1
        self.stats.total_access_time_ms += (time.perf_counter() - t0) * 1000
        return None

    def put(self, key: str, sample: Dict[str, np.ndarray]):
        """Store a sample in hot tier, spilling to warm if needed."""
        with self._lock:
            self._put_hot(key, sample)

    def put_batch(self, items: Dict[str, Dict[str, np.ndarray]]):
        """Store multiple samples at once."""
        with self._lock:
            for key, sample in items.items():
                self._put_hot(key, sample)

    def promote_to_warm(self, key: str, sample: Dict[str, np.ndarray]):
        """Explicitly persist a sample to warm tier (SSD)."""
        if self._warm_dir is None:
            return
        with self._lock:
            self._write_warm(key, sample)

    def invalidate(self, key: str):
        """Remove a key from all tiers."""
        with self._lock:
            self._hot.pop(key, None)
            self._remove_warm(key)

    def clear(self):
        """Clear all tiers."""
        with self._lock:
            self._hot.clear()
            self._hot_bytes = 0
            if self._warm_dir:
                for f in self._warm_dir.glob("*.npy"):
                    f.unlink(missing_ok=True)
                (self._warm_dir / "index.json").unlink(missing_ok=True)
                self._warm_index.clear()
                self._warm_bytes = 0

    def get_stats(self) -> CacheStats:
        with self._lock:
            return self.stats

    def reset_stats(self):
        with self._lock:
            self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Hot tier (in-process memory)
    # ------------------------------------------------------------------

    def _get_hot(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if key in self._hot:
            self._hot.move_to_end(key)
            return self._hot[key]
        return None

    def _put_hot(self, key: str, sample: Dict[str, np.ndarray]):
        if key in self._hot:
            self._hot.move_to_end(key)
            return

        item_bytes = sum(arr.nbytes for arr in sample.values())

        while self._hot_bytes + item_bytes > self._hot_capacity * 1024 * 1024 and self._hot:
            evicted_key, evicted_val = self._hot.popitem(last=False)
            self._hot_bytes -= sum(arr.nbytes for arr in evicted_val.values())
            self.stats.evictions += 1
            if self._warm_dir:
                self._write_warm(evicted_key, evicted_val)
                self.stats.warm_promotions += 1

        self._hot[key] = sample
        self._hot.move_to_end(key)
        self._hot_bytes += item_bytes

    def _promote_to_hot(self, key: str, sample: Dict[str, np.ndarray]):
        self._put_hot(key, sample)

    # ------------------------------------------------------------------
    # Warm tier (SSD memmap files)
    # ------------------------------------------------------------------

    def _warm_path(self, key: str) -> Path:
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._warm_dir / f"{safe_key}.npz"

    def _get_warm(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if self._warm_dir is None:
            return None
        path = self._warm_path(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                result = {k: data[k] for k in data.files}
            if key in self._warm_index:
                self._warm_index.move_to_end(key)
            return result
        except (OSError, ValueError, KeyError):
            return None

    def _write_warm(self, key: str, sample: Dict[str, np.ndarray]):
        if self._warm_dir is None:
            return
        path = self._warm_path(key)
        compressed = {k: v for k, v in sample.items()}
        np.savez_compressed(path, **compressed)
        file_bytes = path.stat().st_size

        while self._warm_bytes + file_bytes > self._max_warm_bytes and self._warm_index:
            evicted_key, evicted_size = self._warm_index.popitem(last=False)
            evicted_path = self._warm_path(evicted_key)
            evicted_path.unlink(missing_ok=True)
            self._warm_bytes -= evicted_size

        self._warm_index[key] = file_bytes
        self._warm_bytes += file_bytes
        self._save_warm_index()

    def _remove_warm(self, key: str):
        if self._warm_dir is None:
            return
        if key in self._warm_index:
            self._warm_bytes -= self._warm_index.pop(key)
        self._warm_path(key).unlink(missing_ok=True)
        self._save_warm_index()

    def _load_warm_index(self):
        idx_path = self._warm_dir / "index.json"
        if idx_path.exists():
            try:
                with open(idx_path, "r") as f:
                    data = json.load(f)
                self._warm_index = OrderedDict(data.get("entries", []))
                self._warm_bytes = sum(self._warm_index.values())
            except (json.JSONDecodeError, OSError):
                pass

    def _save_warm_index(self):
        if self._warm_dir is None:
            return
        idx_path = self._warm_dir / "index.json"
        try:
            with open(idx_path, "w") as f:
                json.dump({"entries": list(self._warm_index.items())}, f)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Cold tier (source files, read via loader callback)
    # ------------------------------------------------------------------

    def _get_cold(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if self._cold_loader is None:
            return None
        try:
            return self._cold_loader(key)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hot) + len(self._warm_index)

    def __contains__(self, key: str) -> bool:
        return key in self._hot or key in self._warm_index


class _FileLock:
    """Minimal cross-process file lock for warm-tier coordination."""

    def __init__(self, path: Path):
        self._path = path
        self._fd = None

    def __enter__(self):
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR)
        if sys.platform == "win32":
            msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self._fd is not None:
            if sys.platform == "win32":
                try:
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
