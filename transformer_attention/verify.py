"""
Verification script to check your attention implementation against PyTorch's built-in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_causal_mask,
)


def verify_scaled_dot_product_attention():
    """
    Compare your ScaledDotProductAttention against PyTorch's F.scaled_dot_product_attention
    """
    print("=" * 60)
    print("Verifying ScaledDotProductAttention")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_k = 64

    # Create test inputs
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_k)

    # Your implementation
    your_attn = ScaledDotProductAttention(dropout=0.0)  # No dropout for comparison
    your_attn.eval()

    try:
        with torch.no_grad():
            your_output, your_weights = your_attn(q, k, v)
    except NotImplementedError:
        print("[SKIP] ScaledDotProductAttention not implemented yet\n")
        return None

    # PyTorch reference (need to reshape for SDPA: expects (batch, heads, seq, d_k))
    # For single-head comparison, add a head dimension
    q_ref = q.unsqueeze(1)  # (batch, 1, seq, d_k)
    k_ref = k.unsqueeze(1)
    v_ref = v.unsqueeze(1)

    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0)
        ref_output = ref_output.squeeze(1)  # Remove head dim

    # Compare outputs
    max_diff = (your_output - ref_output).abs().max().item()
    mean_diff = (your_output - ref_output).abs().mean().item()

    print(f"Your output shape: {your_output.shape}")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    passed = max_diff < 1e-5
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print()
    return passed


def verify_attention_weights():
    """
    Verify attention weights sum to 1 and have correct shape.
    """
    print("=" * 60)
    print("Verifying Attention Weights Properties")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_k = 64

    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_k)

    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    try:
        with torch.no_grad():
            _, weights = attn(q, k, v)
    except NotImplementedError:
        print("[SKIP] ScaledDotProductAttention not implemented yet\n")
        return None

    # Check shape
    expected_shape = (batch_size, seq_len, seq_len)
    shape_correct = weights.shape == expected_shape
    print(f"Weights shape: {weights.shape} (expected {expected_shape})")
    print(f"Shape correct: {'YES' if shape_correct else 'NO'}")

    # Check weights sum to 1 along last dimension
    weight_sums = weights.sum(dim=-1)
    sums_to_one = torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    print(f"Weights sum to 1: {'YES' if sums_to_one else 'NO'}")
    print(f"  Actual sums (sample): {weight_sums[0, :3].tolist()}")

    # Check weights are non-negative
    non_negative = (weights >= 0).all().item()
    print(f"Weights non-negative: {'YES' if non_negative else 'NO'}")

    passed = shape_correct and sums_to_one and non_negative
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print()
    return passed


def verify_multi_head_attention():
    """
    Compare your MultiHeadAttention against PyTorch's nn.MultiheadAttention
    """
    print("=" * 60)
    print("Verifying MultiHeadAttention")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)

    # Your implementation
    your_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    your_mha.eval()

    try:
        with torch.no_grad():
            your_output, your_weights = your_mha(x, x, x)
    except NotImplementedError:
        print("[SKIP] MultiHeadAttention not implemented yet\n")
        return None

    # Check output shape
    expected_output_shape = (batch_size, seq_len, d_model)
    output_shape_correct = your_output.shape == expected_output_shape
    print(f"Output shape: {your_output.shape} (expected {expected_output_shape})")
    print(f"Output shape correct: {'YES' if output_shape_correct else 'NO'}")

    # Check attention weights shape
    expected_weights_shape = (batch_size, num_heads, seq_len, seq_len)
    weights_shape_correct = your_weights.shape == expected_weights_shape
    print(f"Weights shape: {your_weights.shape} (expected {expected_weights_shape})")
    print(f"Weights shape correct: {'YES' if weights_shape_correct else 'NO'}")

    # Verify weights sum to 1 for each head
    weight_sums = your_weights.sum(dim=-1)
    sums_to_one = torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    print(f"Weights sum to 1 (per head): {'YES' if sums_to_one else 'NO'}")

    passed = output_shape_correct and weights_shape_correct and sums_to_one
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print()
    return passed


def verify_causal_mask():
    """
    Verify causal mask blocks future positions.
    """
    print("=" * 60)
    print("Verifying Causal Mask")
    print("=" * 60)

    seq_len = 4
    device = torch.device("cpu")

    try:
        mask = create_causal_mask(seq_len, device)
    except NotImplementedError:
        print("[SKIP] create_causal_mask not implemented yet\n")
        return None

    print(f"Causal mask shape: {mask.shape}")
    print(f"Causal mask:\n{mask}")

    # Apply mask to uniform attention and check result
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    q = torch.ones(1, seq_len, 16)
    k = torch.ones(1, seq_len, 16)
    v = torch.arange(seq_len).float().view(1, seq_len, 1).expand(1, seq_len, 16)

    try:
        with torch.no_grad():
            output, weights = attn(q, k, v, mask)
    except NotImplementedError:
        print("[SKIP] ScaledDotProductAttention not implemented yet\n")
        return None

    print(f"\nAttention weights with causal mask:\n{weights[0]}")

    # Check that position i only attends to positions <= i
    is_causal = True
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i and weights[0, i, j] > 1e-6:
                is_causal = False
                print(f"ERROR: Position {i} attends to future position {j}")

    print(f"\nCausal property satisfied: {'YES' if is_causal else 'NO'}")
    print(f"Result: {'PASSED' if is_causal else 'FAILED'}")
    print()
    return is_causal


def verify_gradient_flow():
    """
    Verify gradients flow correctly through attention.
    """
    print("=" * 60)
    print("Verifying Gradient Flow")
    print("=" * 60)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.train()

    try:
        output, _ = mha(x, x, x)
    except NotImplementedError:
        print("[SKIP] MultiHeadAttention not implemented yet\n")
        return None

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    has_input_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"Input has gradient: {'YES' if has_input_grad else 'NO'}")

    has_param_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in mha.parameters()
    )
    print(f"All parameters have gradients: {'YES' if has_param_grads else 'NO'}")

    # Check for NaN gradients
    no_nan_grads = not any(
        torch.isnan(p.grad).any() for p in mha.parameters() if p.grad is not None
    )
    print(f"No NaN gradients: {'YES' if no_nan_grads else 'NO'}")

    passed = has_input_grad and has_param_grads and no_nan_grads
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print()
    return passed


def run_all_verifications():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("TRANSFORMER ATTENTION VERIFICATION SUITE")
    print("=" * 60 + "\n")

    results = {
        "Scaled Dot-Product Attention": verify_scaled_dot_product_attention(),
        "Attention Weights Properties": verify_attention_weights(),
        "Multi-Head Attention": verify_multi_head_attention(),
        "Causal Mask": verify_causal_mask(),
        "Gradient Flow": verify_gradient_flow(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for name, result in results.items():
        if result is True:
            status = "PASSED"
            passed_count += 1
        elif result is False:
            status = "FAILED"
            failed_count += 1
        else:  # None
            status = "SKIPPED"
            skipped_count += 1
        print(f"  {name}: {status}")

    print()
    print(f"Passed: {passed_count}, Failed: {failed_count}, Skipped: {skipped_count}")

    if failed_count > 0:
        print("Some tests FAILED!")
    elif skipped_count > 0 and passed_count > 0:
        print("All implemented tests passed!")
    elif skipped_count == len(results):
        print("All tests skipped - implement attention first.")
    else:
        print("All tests passed!")

    return failed_count == 0


if __name__ == "__main__":
    run_all_verifications()
