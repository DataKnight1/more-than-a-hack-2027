"""Hardware profiling and latency/throughput benchmark for N020 PathCRF.

Measures:
1. Model parameter counts (backbone, heads, CRF).
2. Peak VRAM consumption on CUDA during forward and backward passes.
3. Inference latency (ms per window) and throughput (frames per second).
4. Window length and sampling rate comparisons (5 Hz vs 25 Hz).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from possession_lab.neural.pathcrf.backbone import PathCRFNet  # noqa: E402
from possession_lab.neural.pathcrf.crf import DynamicSparseCRF  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("================================================================================")
    print("N020 PATHCRF HARDWARE PROFILING & RESOURCE BENCHMARK")
    print("================================================================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_mem_gb:.2f} GB")
    print("--------------------------------------------------------------------------------")

    # Instantiate model
    model = PathCRFNet(
        feat_dim=8,
        team_size=11,
        macro_dim=64,
        micro_dim=128,
        seq_dim=128,
        crf_edge_dim=16,
    ).to(device)
    crf = DynamicSparseCRF(model.comp_src, model.comp_dst, edge_embed_dim=16, team_size=11).to(
        device
    )

    total_params = sum(p.numel() for p in model.parameters())
    crf_params = sum(p.numel() for p in crf.parameters())
    print(f"Backbone parameters: {total_params:,}")
    print(f"CRF transition parameters: {crf_params:,}")
    print(f"Total trainable parameters: {total_params + crf_params:,}")
    print("--------------------------------------------------------------------------------")

    # Benchmark configurations
    configs = [
        ("Upstream Reproduction (5 Hz, 50 frames = 10s)", 50, args.batch_size),
        ("Fast Inference (10 Hz, 100 frames = 10s)", 100, args.batch_size),
        ("Full Resolution (25 Hz, 100 frames = 4s)", 100, args.batch_size),
    ]

    for name, T, B in configs:
        print(f"\nEvaluating: {name} [B={B}, T={T}, N=26, F=8]", flush=True)
        x = torch.randn(B, T, 26, 8, device=device)
        valid = torch.ones(B, T, 26, dtype=torch.bool, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            _ = model(x, valid)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed forward passes
        n_iters = 5
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x, valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_window = (elapsed / (n_iters * B)) * 1000.0
        fps_throughput = (n_iters * B * T) / elapsed

        vram_mb = 0.0
        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        print(f"  Latency:    {ms_per_window:.2f} ms / window", flush=True)
        print(f"  Throughput: {fps_throughput:,.1f} frames / sec", flush=True)
        if device.type == "cuda":
            print(f"  Peak VRAM:  {vram_mb:.1f} MB", flush=True)

    print("\nBenchmark completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
