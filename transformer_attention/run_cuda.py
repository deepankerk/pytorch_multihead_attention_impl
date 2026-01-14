"""
NVIDIA CUDA Runner for Transformer Attention

This script runs the attention implementation on NVIDIA GPU using CUDA backend.
"""

import torch
import time
from attention import MultiHeadAttention, SelfAttention, create_causal_mask


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        print("CUDA not available. Please ensure:")
        print("  1. You have an NVIDIA GPU")
        print("  2. CUDA drivers are installed")
        print("  3. PyTorch is installed with CUDA support")
        return False
    return True


def print_cuda_info():
    """Print CUDA device information."""
    print(f"\nCUDA Device Information:")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")

    # Memory info
    props = torch.cuda.get_device_properties(0)
    print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-processor count: {props.multi_processor_count}")


def run_attention_on_cuda():
    """Run transformer attention on NVIDIA CUDA."""

    if not check_cuda_available():
        print("Falling back to CPU...")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print_cuda_info()

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

    # Create model and move to CUDA
    model = SelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
    model = model.to(device)
    model.eval()

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create causal mask (optional)
    # mask = create_causal_mask(seq_len, device)
    mask = None

    print("\nRunning forward pass...")

    # Warmup (important for accurate CUDA timing)
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(x, mask)
                torch.cuda.synchronize()
            except NotImplementedError as e:
                print(f"\n[Expected] {e}")
                print("Implement the attention mechanism in attention.py to see it run!")
                return

    # Benchmark using CUDA events for precise timing
    num_iterations = 100

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(num_iterations):
            output, attention_weights = model(x, mask)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iterations

    print(f"\nResults:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Average time per forward pass: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {1000 / avg_time_ms:.1f} forward passes/sec")

    # Memory info
    print(f"\nCUDA Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


def benchmark_different_sizes():
    """Benchmark attention with different sequence lengths on CUDA."""

    if not check_cuda_available():
        print("CUDA not available. Exiting benchmark.")
        return

    device = torch.device("cuda")
    d_model = 512
    num_heads = 8
    batch_size = 4

    seq_lengths = [64, 128, 256, 512, 1024, 2048]

    print("\nBenchmarking different sequence lengths on CUDA:")
    print("-" * 70)
    print(f"{'Seq Length':<12} {'Time (ms)':<15} {'Memory (MB)':<15} {'TFLOPS':<15}")
    print("-" * 70)

    model = SelfAttention(d_model=d_model, num_heads=num_heads).to(device)
    model.eval()

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            try:
                for _ in range(5):
                    _ = model(x)
                    torch.cuda.synchronize()
            except NotImplementedError:
                print("Implement attention first to run benchmark!")
                return

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(50):
                _ = model(x)
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / 50
        memory = torch.cuda.max_memory_allocated() / 1024**2

        # Approximate FLOPs for attention: 4 * batch * seq^2 * d_model
        flops = 4 * batch_size * (seq_len ** 2) * d_model
        tflops = flops / (avg_time / 1000) / 1e12

        print(f"{seq_len:<12} {avg_time:<15.3f} {memory:<15.2f} {tflops:<15.4f}")


def compare_with_flash_attention():
    """Compare standard attention with Flash Attention if available."""

    if not check_cuda_available():
        return

    device = torch.device("cuda")

    # Check if Flash Attention is available (PyTorch 2.0+)
    has_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    if not has_flash:
        print("\nFlash Attention not available (requires PyTorch 2.0+)")
        return

    print("\n" + "=" * 60)
    print("Comparing with PyTorch's scaled_dot_product_attention (Flash)")
    print("=" * 60)

    batch_size = 4
    seq_len = 512
    d_model = 512
    num_heads = 8
    d_k = d_model // num_heads

    # Create test tensors in the format expected by SDPA
    q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)

    # Your implementation would go here for comparison
    print("\nOnce you implement attention, you can compare with:")
    print("  torch.nn.functional.scaled_dot_product_attention(q, k, v)")
    print("\nThis uses Flash Attention or Memory-Efficient Attention automatically!")


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Attention - NVIDIA CUDA Runner")
    print("=" * 60)

    run_attention_on_cuda()

    print("\n" + "=" * 60)
    print("Sequence Length Benchmark")
    print("=" * 60)
    benchmark_different_sizes()

    compare_with_flash_attention()
