"""
Transformer Attention Scaffold
Implement the core attention mechanism yourself in the marked TODO sections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    TODO: Implement the forward pass yourself
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_k)
            key: (batch, seq_len, d_k)
            value: (batch, seq_len, d_v)
            mask: Optional attention mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        d_k = query.size(-1)

        # Step 1: Compute attention scores: Q @ K^T
        attention_scores = query @ key.transpose(-2, -1)
        # Step 2: Scale by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(d_k)
        # Step 3: Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Step 5: Apply dropout
        attention_weights = self.dropout(attention_weights)
        # Step 6: Compute output: attention_weights @ V
        output = attention_weights @ value
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    TODO: Implement the forward pass yourself
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # TODO: Implement multi-head attention
        # Step 1: Linear projections Q, K, V through W_q, W_k, W_v
        # Size of projected_query, projected_key, projected_value is (batch, seq_len, d_model)
        projected_query = self.W_q(query)
        projected_key = self.W_k(key)
        projected_value = self.W_v(value)
        # Step 2: Reshape to (batch, num_heads, seq_len, d_k) for parallel attention
        projected_query = self._split_heads(projected_query)
        projected_key = self._split_heads(projected_key)
        projected_value = self._split_heads(projected_value)
        # Step 3: Apply scaled dot-product attention
        output, attention_weights = self.attention(projected_query, projected_key, projected_value, mask)
        # Step 4: Reshape back to (batch, seq_len, d_model)
        output = self._combine_heads(output)
        # Step 5: Final linear projection through W_o
        output = self.W_o(output)
        return output, attention_weights

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k)

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        # View essentially splits the last dimension into (num_heads, d_k)
        # Pytorch view reads the values in left , right, top, bottom order to fill the new tensor
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back to original shape

        Args:
            x: (batch, num_heads, seq_len, d_k)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


class SelfAttention(nn.Module):
    """
    Self-Attention wrapper where Q=K=V come from the same input
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        return self.mha(x, x, x, mask)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) mask for decoder self-attention.

    Returns a mask where position i can only attend to positions <= i.
    """
    # TODO: Implement causal mask creation
    # Hint: Use torch.triu to create upper triangular matrix of -inf
    raise NotImplementedError("Implement causal mask creation")


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask based on sequence lengths.

    Args:
        lengths: (batch,) actual lengths of each sequence
        max_len: maximum sequence length

    Returns:
        mask: (batch, 1, 1, max_len) where True indicates padding positions
    """
    # TODO: Implement padding mask creation
    raise NotImplementedError("Implement padding mask creation")
