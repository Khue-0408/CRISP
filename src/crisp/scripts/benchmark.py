"""
CLI entry point for lightweight compute benchmarking.

This script is intended to support:
- inference fps measurement,
- peak memory measurement,
- train-time throughput diagnostics.

It is useful when reproducing the compute profile section.
"""

from __future__ import annotations

import argparse
import time

import torch

from crisp.registry import build_model, build_projector, get_model_decoder_channels
from crisp.utils.config import load_config
from crisp.modules.calibration import calibrate_logits_with_alpha


def main() -> None:
    """
    Main benchmark entry point.
    """
    parser = argparse.ArgumentParser(description="CRISP inference benchmark")
    parser.add_argument("--config", type=str, default=None, help="Experiment config path")
    parser.add_argument("--image-size", type=int, default=352, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup-iters", type=int, default=10, help="GPU warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--no-projector", action="store_true", help="Benchmark without projector")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model.
    if args.config:
        config = load_config(args.config)
    else:
        config = {"model": {"name": "pranet"}}

    model = build_model(config).to(device).eval()
    projector = None
    if not args.no_projector:
        decoder_ch = get_model_decoder_channels(model)
        projector = build_projector(
            config.get("crisp", {"projection": {}, "projector_head": {}}),
            in_channels=decoder_ch,
        ).to(device).eval()

    # Dummy input.
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    # Warmup.
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            out = model(x)
            if projector is not None:
                alpha = projector(out.features, out.logits)
                _ = calibrate_logits_with_alpha(out.logits, alpha)

    # Benchmark.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.bench_iters):
            out = model(x)
            if projector is not None:
                alpha = projector(out.features, out.logits)
                _ = calibrate_logits_with_alpha(out.logits, alpha)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = args.bench_iters * args.batch_size / elapsed
    ms_per_image = elapsed / (args.bench_iters * args.batch_size) * 1000

    # Peak memory.
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = float("nan")

    mode = "backbone_only" if args.no_projector else "backbone+projector"
    print(f"Mode:             {mode}")
    print(f"Image size:       {args.image_size}x{args.image_size}")
    print(f"Batch size:       {args.batch_size}")
    print(f"FPS:              {fps:.1f}")
    print(f"ms/image:         {ms_per_image:.2f}")
    print(f"Peak GPU memory:  {peak_mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
