#!/usr/bin/env python3
"""
Transformer Attention - Main Entry Point

Automatically detects available GPU and runs the appropriate version.
Usage:
    python main.py              # Auto-detect device
    python main.py --device mps   # Force Apple Silicon
    python main.py --device cuda  # Force NVIDIA
    python main.py --device cpu   # Force CPU
"""

import argparse
import sys
import torch


def detect_best_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_system_info():
    """Print system and PyTorch information."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print()

    # CUDA info
    print("CUDA:")
    print(f"  Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name()}")

    # MPS info
    print("\nMPS (Apple Silicon):")
    print(f"  Available: {torch.backends.mps.is_available()}")
    print(f"  Built: {torch.backends.mps.is_built()}")

    # Recommended device
    best = detect_best_device()
    print(f"\nRecommended device: {best}")
    print()


def run_on_device(device: str):
    """Run attention on specified device."""
    if device == "mps":
        from run_mps import run_attention_on_mps, benchmark_different_sizes
        run_attention_on_mps()
        print("\n" + "=" * 60)
        print("Sequence Length Benchmark")
        print("=" * 60)
        benchmark_different_sizes()

    elif device == "cuda":
        from run_cuda import run_attention_on_cuda, benchmark_different_sizes, compare_with_flash_attention
        run_attention_on_cuda()
        print("\n" + "=" * 60)
        print("Sequence Length Benchmark")
        print("=" * 60)
        benchmark_different_sizes()
        compare_with_flash_attention()

    else:  # CPU
        run_on_cpu()


def run_on_cpu():
    """Run attention on CPU (fallback)."""
    import time
    from attention import SelfAttention

    print("=" * 60)
    print("Transformer Attention - CPU Runner")
    print("=" * 60)

    device = torch.device("cpu")

    batch_size = 4
    seq_len = 128
    d_model = 512
    num_heads = 8

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print("-" * 50)

    model = SelfAttention(d_model=d_model, num_heads=num_heads)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    print("\nRunning forward pass...")

    with torch.no_grad():
        try:
            # Warmup
            for _ in range(3):
                _ = model(x)

            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                output, attention_weights = model(x)
            end = time.perf_counter()

            avg_time = (end - start) / 20 * 1000
            print(f"\nResults:")
            print(f"  Output shape: {output.shape}")
            print(f"  Attention weights shape: {attention_weights.shape}")
            print(f"  Average time per forward pass: {avg_time:.3f} ms")

        except NotImplementedError as e:
            print(f"\n[Expected] {e}")
            print("Implement the attention mechanism in attention.py to see it run!")


def main():
    parser = argparse.ArgumentParser(
        description="Run Transformer Attention on GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Auto-detect best device
  python main.py --device cuda      # Run on NVIDIA GPU
  python main.py --device mps       # Run on Apple Silicon GPU
  python main.py --device cpu       # Run on CPU
  python main.py --info             # Show system info only
        """
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information and exit"
    )

    args = parser.parse_args()

    print_system_info()

    if args.info:
        return

    # Determine device
    if args.device == "auto":
        device = detect_best_device()
        print(f"Auto-detected device: {device}")
    else:
        device = args.device

    # Validate device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but not available!")
        sys.exit(1)
    if device == "mps" and not torch.backends.mps.is_available():
        print("Error: MPS requested but not available!")
        sys.exit(1)

    print(f"\nRunning on: {device.upper()}")
    print("=" * 60)

    run_on_device(device)


if __name__ == "__main__":
    main()
