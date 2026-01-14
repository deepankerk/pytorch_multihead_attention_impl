"""
Apple Silicon MPS Runner for Transformer Attention

This script runs the attention implementation on Apple Silicon GPU using MPS backend.
"""

import torch
import time
from attention import MultiHeadAttention, SelfAttention, create_causal_mask


def check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current macOS version is not 12.3+ "
                  "or you don't have an MPS-enabled device")
        return False
    return True


def run_attention_on_mps():
    """Run transformer attention on Apple Silicon MPS."""

    if not check_mps_available():
        print("Falling back to CPU...")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    seq_len = 128
    d_model = 512
    num_heads = 8
    dropout = 0.1

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print("-" * 50)

    # Create model and move to MPS
    model = SelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
    model = model.to(device)
    model.eval()

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create causal mask (optional)
    # mask = create_causal_mask(seq_len, device)
    mask = None

    print("\nRunning forward pass...")

    # Warmup (important for accurate timing on MPS)
    with torch.no_grad():
        for _ in range(3):
            try:
                _ = model(x, mask)
                torch.mps.synchronize()  # Wait for MPS operations to complete
            except NotImplementedError as e:
                print(f"\n[Expected] {e}")
                print("Implement the attention mechanism in attention.py to see it run!")
                return

    # Benchmark
    num_iterations = 100
    torch.mps.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            output, attention_weights = model(x, mask)
            torch.mps.synchronize()

    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    print(f"\nResults:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Average time per forward pass: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {num_iterations / (end_time - start_time):.1f} forward passes/sec")

    # Memory info (MPS specific)
    print(f"\nMPS Memory:")
    print(f"  Current allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    print(f"  Driver allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")


def benchmark_different_sizes():
    """Benchmark attention with different sequence lengths on MPS."""

    if not check_mps_available():
        print("MPS not available. Exiting benchmark.")
        return

    device = torch.device("mps")
    d_model = 512
    num_heads = 8
    batch_size = 4

    seq_lengths = [64, 128, 256, 512, 1024]

    print("\nBenchmarking different sequence lengths on MPS:")
    print("-" * 60)
    print(f"{'Seq Length':<12} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 60)

    model = SelfAttention(d_model=d_model, num_heads=num_heads).to(device)
    model.eval()

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Warmup
        with torch.no_grad():
            try:
                for _ in range(3):
                    _ = model(x)
                    torch.mps.synchronize()
            except NotImplementedError:
                print("Implement attention first to run benchmark!")
                return

        # Benchmark
        torch.mps.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(50):
                _ = model(x)
                torch.mps.synchronize()

        end = time.perf_counter()
        avg_time = (end - start) / 50 * 1000
        memory = torch.mps.current_allocated_memory() / 1024**2

        print(f"{seq_len:<12} {avg_time:<15.3f} {memory:<15.2f}")

        # Clear cache between runs
        torch.mps.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Attention - Apple Silicon MPS Runner")
    print("=" * 60)

    run_attention_on_mps()

    print("\n" + "=" * 60)
    print("Sequence Length Benchmark")
    print("=" * 60)
    benchmark_different_sizes()
