"""
Dataset manifest and version management.

Produces a manifest.json that records:
- SHA256 hashes of every data file
- File shapes, dtypes, sizes
- Preprocessing parameters used to produce each file
- Dataset-level fingerprint for cache invalidation

When the manifest changes (files modified, new preprocessing),
caches are automatically invalidated and rebuilt.
"""
import os
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np


def compute_file_hash(path: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute a hash for a file, streaming in chunks for large files.

    For .npy files, uses the header + a sample of the array content
    for faster hashing of multi-GB files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.new(algorithm)

    if path.suffix == ".npy":
        _hash_npy_fast(path, h)
    else:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)

    return h.hexdigest()


def _hash_npy_fast(path: Path, h) -> None:
    """Fast hash for large .npy files: header + first/last 1MB + middle samples."""
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        # Always hash the full header (first 64KB covers all realistic headers)
        header_chunk = f.read(64 * 1024)
        h.update(header_chunk)
        h.update(str(file_size).encode())

    if file_size <= 128 * 1024:
        # Small file: hash entire content
        with open(path, "rb") as f:
            h.update(f.read())
        return

    # Large file: sample head, middle, tail
    with open(path, "rb") as f:
        f.seek(64 * 1024)  # skip header
        h.update(f.read(1024 * 1024))  # first 1MB of data

        mid = file_size // 2
        f.seek(mid)
        h.update(f.read(1024 * 1024))  # middle 1MB

        f.seek(max(0, file_size - 1024 * 1024))
        h.update(f.read(1024 * 1024))  # last 1MB


@dataclass
class FileEntry:
    """Metadata for a single data file."""
    path: str           # relative to data root
    sha256: str
    shape: List[int] = field(default_factory=list)
    dtype: str = ""
    size_bytes: int = 0
    num_samples: int = 0  # for sequence files, number of time steps

    @classmethod
    def from_file(cls, file_path: str, data_root: str = "") -> "FileEntry":
        abs_path = Path(file_path)
        rel_path = str(abs_path.relative_to(data_root)) if data_root else abs_path.name

        try:
            arr = np.load(abs_path, mmap_mode="r")
            shape = list(arr.shape)
            dtype = str(arr.dtype)
            num_samples = shape[0] if shape else 0
        except (ValueError, OSError):
            shape = []
            dtype = "unknown"
            num_samples = 0

        return cls(
            path=rel_path,
            sha256=compute_file_hash(str(abs_path)),
            shape=shape,
            dtype=dtype,
            size_bytes=abs_path.stat().st_size,
            num_samples=num_samples,
        )


@dataclass
class Manifest:
    """Complete dataset manifest with versioning info."""
    version: str = "1.0"
    created_at: str = ""
    dataset_name: str = ""
    dataset_uuid: str = ""          # canonical hash for cache keys
    files: Dict[str, FileEntry] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    total_size_bytes: int = 0
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["files"] = {k: asdict(v) for k, v in self.files.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        files = {
            k: FileEntry(**v) for k, v in data.pop("files", {}).items()
        }
        return cls(files=files, **data)


class DatasetManifest:
    """Manages dataset versioning: build, save, load, and compare manifests.

    Usage:
        mgr = DatasetManifest("data/processed/")
        mgr.build(name="crop_2023", preprocess_params={...})
        mgr.save("data/processed/manifest.json")

        # Later: check if cache is still valid
        if mgr.is_stale():
            mgr.rebuild_cache()
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.manifest: Optional[Manifest] = None

    def build(
        self,
        name: str = "",
        preprocess_params: Optional[Dict[str, Any]] = None,
        glob_pattern: str = "*.npy",
    ) -> Manifest:
        """Scan data_root for .npy files and build a manifest."""
        files: Dict[str, FileEntry] = {}
        total_bytes = 0
        total_samples = 0

        npy_files = sorted(self.data_root.glob(glob_pattern))
        for npy_path in npy_files:
            entry = FileEntry.from_file(str(npy_path), str(self.data_root))
            files[entry.path] = entry
            total_bytes += entry.size_bytes
            total_samples += entry.num_samples

        dataset_uuid = self._compute_dataset_uuid(files, preprocess_params or {})

        self.manifest = Manifest(
            version="1.0",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset_name=name,
            dataset_uuid=dataset_uuid,
            files=files,
            preprocessing=preprocess_params or {},
            total_size_bytes=total_bytes,
            total_samples=total_samples,
        )
        return self.manifest

    def save(self, output_path: Optional[str] = None):
        """Persist manifest to JSON."""
        if self.manifest is None:
            raise ValueError("No manifest built. Call build() first.")
        path = Path(output_path) if output_path else self.data_root / "manifest.json"
        data = self.manifest.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, manifest_path: str) -> Manifest:
        """Load a manifest from JSON."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.manifest = Manifest.from_dict(data)
        return self.manifest

    def is_stale(self, manifest_path: Optional[str] = None) -> bool:
        """Check if existing manifest is stale (files changed on disk)."""
        if self.manifest is None:
            if manifest_path:
                try:
                    self.load(manifest_path)
                except (FileNotFoundError, json.JSONDecodeError):
                    return True
            else:
                return True

        for rel_path, entry in self.manifest.files.items():
            abs_path = self.data_root / rel_path
            if not abs_path.exists():
                return True
            current_hash = compute_file_hash(str(abs_path))
            if current_hash != entry.sha256:
                return True

        return False

    def get_dataset_uuid(self) -> Optional[str]:
        """Get the canonical dataset UUID for cache keying."""
        return self.manifest.dataset_uuid if self.manifest else None

    def _compute_dataset_uuid(
        self, files: Dict[str, FileEntry], params: Dict[str, Any]
    ) -> str:
        """Generate a stable UUID from file hashes + preprocessing params."""
        h = hashlib.sha256()
        for name in sorted(files.keys()):
            h.update(name.encode())
            h.update(files[name].sha256.encode())
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        h.update(params_str.encode())
        return h.hexdigest()[:16]  # 64-bit unique ID
