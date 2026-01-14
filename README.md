# Transformer Attention from Scratch

A scaffold for implementing Transformer attention mechanism in PyTorch, with runners for both Apple Silicon (MPS) and NVIDIA (CUDA) GPUs.

## Files

| File | Description |
|------|-------------|
| `attention.py` | Core attention modules to implement |
| `verify.py` | Verification suite to check your implementation |
| `main.py` | Auto-detects GPU and runs benchmarks |
| `run_mps.py` | Apple Silicon MPS-specific runner |
| `run_cuda.py` | NVIDIA CUDA-specific runner |

## Setup

```bash
pip install torch
```

## Usage

```bash
# Run with auto-detected device
python main.py

# Force specific device
python main.py --device mps    # Apple Silicon
python main.py --device cuda   # NVIDIA
python main.py --device cpu    # CPU fallback

# Verify implementation correctness
python verify.py
```

## What to Implement

In `attention.py`, implement:

1. **ScaledDotProductAttention.forward()** - Core attention formula:
   ```
   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
   ```

2. **MultiHeadAttention.forward()** - Multi-head wrapper:
   - Project Q, K, V through learned weights
   - Split into multiple heads
   - Apply attention per head
   - Combine heads and project output

3. **_split_heads() / _combine_heads()** - Reshape tensors for multi-head processing

4. **create_causal_mask()** - Autoregressive mask for decoder self-attention

## Verification

Run `python verify.py` to check your implementation against PyTorch's built-in attention:

```
============================================================
SUMMARY
============================================================
  Scaled Dot-Product Attention: PASSED
  Attention Weights Properties: PASSED
  Multi-Head Attention: PASSED
  Causal Mask: PASSED
  Gradient Flow: PASSED

Passed: 5, Failed: 0, Skipped: 0
All tests passed!
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
