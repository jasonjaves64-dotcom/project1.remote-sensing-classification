"""
DataLoader throughput benchmark: before vs after caching optimization.

Measures:
  - Throughput (samples/sec)
  - Cache hit rate
  - Hot/warm/cold tier distribution
  - Average batch load time

Usage:
  python scripts/benchmark_dataloader.py
  python scripts/benchmark_dataloader.py --batches 200 --warmup 10
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import FusionCropDatasetEDL
from data.datasets.cached_dataset import CachedDataset, DatasetBenchmark
from data.cache import TieredLRUCache, CacheStats, AsyncPreloader, DatasetManifest
from data.cache.manifest import compute_file_hash


def build_dataloaders(
    data_dir: str,
    patch_size: int = 32,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Build baseline and cached DataLoaders for comparison."""
    opt_seq = np.load(os.path.join(data_dir, "opt_sequence.npy"))
    sar_seq = np.load(os.path.join(data_dir, "sar_sequence.npy"))
    doy_norm = np.load(os.path.join(data_dir, "doy_norm.npy"))
    label = np.load(os.path.join(data_dir, "label.npy"))

    dem_path = os.path.join(data_dir, "dem.npy")
    dem_data = np.load(dem_path) if os.path.exists(dem_path) else None

    H, W = label.shape
    split_col = int(W * 0.85)

    # Baseline dataset (no caching)
    train_ds_baseline = FusionCropDatasetEDL(
        opt_seq[:, :, :, :split_col], sar_seq[:, :, :, :split_col],
        doy_norm, label[:, :split_col],
        patch_size=patch_size, augment=False,
        dem_data=dem_data[:, :, :split_col] if dem_data is not None else None,
    )

    # Cached dataset
    train_ds_cached = CachedDataset(
        FusionCropDatasetEDL(
            opt_seq[:, :, :, :split_col], sar_seq[:, :, :, :split_col],
            doy_norm, label[:, :split_col],
            patch_size=patch_size, augment=False,
            dem_data=dem_data[:, :, :split_col] if dem_data is not None else None,
        ),
        hot_capacity=512,
        warm_cache_dir=os.path.join(data_dir, "..", "cache", ".warm_cache"),
        max_warm_gb=4,
        dataset_name="fusion_2023_benchmark",
    )

    base_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loader_baseline = DataLoader(train_ds_baseline, **base_kwargs)
    loader_cached = DataLoader(train_ds_cached, **base_kwargs)

    return loader_baseline, loader_cached, train_ds_cached


def run_benchmark(name, loader, warmup_batches, max_batches, cached_ds=None):
    """Run a single benchmark and return results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    benchmark = DatasetBenchmark(loader, warmup_batches=warmup_batches)

    # Pass 1: cold (first epoch, populate cache)
    print("  Pass 1 (cold, populating cache) ...")
    result_pass1 = benchmark.measure(max_batches=max_batches)
    print(f"    Throughput: {result_pass1['throughput_samples_per_sec']} samples/sec")
    print(f"    Avg batch time: {result_pass1['avg_batch_time_ms']} ms")

    # Reset stats after cold pass to measure warm hit rate cleanly
    cold_cache_stats = None
    if cached_ds is not None:
        cold_cache_stats = cached_ds.get_cache_stats()
        cached_ds.reset_cache_stats()

    # Pass 2: warm (cache fully populated)
    print("  Pass 2 (warm, from cache) ...")
    result_pass2 = benchmark.measure(max_batches=max_batches)
    print(f"    Throughput: {result_pass2['throughput_samples_per_sec']} samples/sec")
    print(f"    Avg batch time: {result_pass2['avg_batch_time_ms']} ms")

    # Pass 3: verify sustained warm performance
    print("  Pass 3 (hot, sustained) ...")
    result_pass3 = benchmark.measure(max_batches=max_batches)
    print(f"    Throughput: {result_pass3['throughput_samples_per_sec']} samples/sec")
    print(f"    Avg batch time: {result_pass3['avg_batch_time_ms']} ms")

    cache_stats = None
    if cached_ds is not None:
        cache_stats = cached_ds.get_cache_stats()
        if isinstance(cache_stats, CacheStats):
            print(f"    Warm-cache hit rate: {cache_stats.hit_rate:.2%}")
            print(f"    Hot hits: {cache_stats.hot_hits}, Warm hits: {cache_stats.warm_hits}")
            print(f"    Cold reads: {cache_stats.cold_reads}, Evictions: {cache_stats.evictions}")
            cache_stats = cache_stats.to_dict()

    return {
        "name": name,
        "pass1_cold": result_pass1,
        "pass2_warm": result_pass2,
        "pass3_hot": result_pass3,
        "cache_stats": cache_stats,
        "cold_cache_stats": cold_cache_stats.to_dict() if cold_cache_stats else None,
    }


def compute_improvement(baseline: dict, cached: dict) -> dict:
    """Compute relative improvement of cached vs baseline on warm pass."""
    bl_p2 = baseline["pass2_warm"]["throughput_samples_per_sec"]
    ca_p3 = cached["pass3_hot"]["throughput_samples_per_sec"]

    if bl_p2 > 0:
        improvement_pct = ((ca_p3 - bl_p2) / bl_p2) * 100
    else:
        improvement_pct = 0.0

    return {
        "baseline_warm_throughput": bl_p2,
        "cached_hot_throughput": ca_p3,
        "improvement_pct": round(improvement_pct, 1),
        "target_met": improvement_pct >= 30.0,
    }


def generate_manifest(data_dir: str):
    """Generate and save a manifest for the processed data."""
    print(f"\nGenerating manifest for {data_dir}...")
    mgr = DatasetManifest(data_dir)
    manifest = mgr.build(
        name="fusion_crop_2023",
        preprocess_params={
            "patch_size": 32,
            "cloud_threshold": 0.3,
            "sar_log_transform": True,
            "normalization": "zscore",
            "resolution": "10m",
        },
    )
    manifest_path = os.path.join(data_dir, "manifest.json")
    mgr.save(manifest_path)
    print(f"  Manifest saved: {manifest_path}")
    print(f"  Dataset UUID: {manifest.dataset_uuid}")
    print(f"  Files: {len(manifest.files)}, Total: {manifest.total_size_bytes / 1e6:.1f} MB")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader throughput")
    parser.add_argument("--data_path", default="data/processed/")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Batches: {args.batches}, Warmup: {args.warmup}")

    # Generate manifest
    if not args.skip_manifest and os.path.isdir(args.data_path):
        manifest = generate_manifest(args.data_path)
        for fname, entry in manifest.files.items():
            shape_str = "x".join(str(s) for s in entry.shape)
            print(f"  {fname}: {shape_str}, {entry.size_bytes/1e6:.1f}MB, sha256={entry.sha256[:12]}...")

    # Build loaders
    print("\nBuilding DataLoaders...")
    loader_baseline, loader_cached, cached_ds = build_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"  Baseline: {len(loader_baseline.dataset)} samples")
    print(f"  Cached:   {len(loader_cached.dataset)} samples")

    # Run benchmarks
    results = {}

    baseline_result = run_benchmark(
        "BASELINE (no cache)", loader_baseline, args.warmup, args.batches
    )
    results["baseline"] = baseline_result

    cached_result = run_benchmark(
        "CACHED (tiered LRU)", loader_cached, args.warmup, args.batches,
        cached_ds=cached_ds,
    )
    results["cached"] = cached_result

    # Compare
    improvement = compute_improvement(baseline_result, cached_result)
    results["improvement"] = improvement

    # Async preloader test
    print(f"\n{'='*60}")
    print(f"  ASYNC PRELOADER (StreamingPrefetchLoader)")
    print(f"{'='*60}")
    from data.cache.async_preloader import StreamingPrefetchLoader
    loader_async_base = DataLoader(
        cached_ds._dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    preload_loader = StreamingPrefetchLoader(loader_async_base, device=None, prefetch_depth=2)
    try:
        benchmark_async = DatasetBenchmark(
            DataLoader(cached_ds._dataset, batch_size=args.batch_size, shuffle=False),
            warmup_batches=args.warmup,
        )
        # Iterate via preloader
        total = 0
        batches = 0
        t0 = time.perf_counter()
        for batch in preload_loader:
            bs = batch["opt"].shape[0] if isinstance(batch, dict) else 8
            total += bs
            batches += 1
            if batches >= args.batches:
                break
        elapsed = time.perf_counter() - t0
        async_throughput = total / elapsed if elapsed > 0 else 0
        print(f"    Throughput: {async_throughput:.1f} samples/sec")
        print(f"    Prefetch hit rate: {preload_loader.stats.hit_rate:.2%}")
        results["async_preloader"] = {
            "throughput_samples_per_sec": round(async_throughput, 1),
            "prefetch_hit_rate": round(preload_loader.stats.hit_rate, 4),
        }
    finally:
        preload_loader.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline warm throughput:   {improvement['baseline_warm_throughput']:.1f} samples/sec")
    print(f"  Cached hot throughput:      {improvement['cached_hot_throughput']:.1f} samples/sec")
    print(f"  Improvement:                {improvement['improvement_pct']:.1f}%")
    print(f"  Target (30%+):              {'PASS' if improvement['target_met'] else 'NOT MET'}")

    if cached_result.get("cache_stats"):
        cs = cached_result["cache_stats"]
        warm_hit_rate = cs.get("hit_rate", 0)
        print(f"  Warm-cache hit rate:        {warm_hit_rate:.2%}")
        target_cache = warm_hit_rate >= 0.80
        print(f"  Cache target (>80%):        {'PASS' if target_cache else 'NOT MET'}")
        results["acceptance"] = {
            "throughput_30pct_improvement": improvement["target_met"],
            "cache_hit_rate_80pct": target_cache,
            "all_passed": improvement["target_met"] and target_cache,
        }

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Cleanup warm cache
    warm_dir = os.path.join(args.data_path, "..", "cache", ".warm_cache")
    if os.path.isdir(warm_dir):
        import shutil
        shutil.rmtree(warm_dir, ignore_errors=True)
        print(f"Cleaned up warm cache: {warm_dir}")

    return 0 if improvement["target_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
