"""
Multi-Head Attention module with Rotary Position Embeddings (RoPE).

Implements scaled dot-product attention with multiple heads, causal masking,
and RoPE applied to queries and keys (not values).

Architecture details:
- Input: (batch, seq_len, n_embd)
- Q, K, V projections from same input (no bias)
- RoPE applied to Q and K before attention
- Attention scores scaled by √(d_head)
- Causal mask prevents attention to future tokens
- Output projection maps back to n_embd
"""

import numpy as np
from typing import Optional

from rope.rope_numpy import get_rope_cache, apply_rope
from utils.activation import softmax
from utils.initializers import init_linear


class MultiHeadAttention:
    """
    Multi-head self-attention with RoPE.

    This implements the attention sub-layer of a Transformer decoder block.
    """

    def __init__(self, config):
        """
        Initialize multi-head attention.

        Args:
            config: GPT2Config with n_embd, n_head, initializer_range, etc.
        """
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_head = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "Embedding dim must be divisible by number of heads"

        # Combined QKV projection: (n_embd, 3 * n_embd)
        self.wqkv = init_linear((self.n_embd, 3 * self.n_embd), config.initializer_range)
        # Output projection: (n_embd, n_embd)
        self.wo = init_linear((self.n_embd, self.n_embd), config.initializer_range)

        # Optional dropout masks (not implemented in pure NumPy for now)
        self.attn_dropout_p = config.attn_pdrop
        self.resid_dropout_p = config.resid_pdrop

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into (n_head, d_head) and transpose.

        Args:
            x: (batch, seq_len, n_embd)

        Returns:
            (batch, n_head, seq_len, d_head)
        """
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_head, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Merge heads: inverse of split_heads.

        Args:
            x: (batch, n_head, seq_len, d_head)

        Returns:
            (batch, seq_len, n_embd)
        """
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, n_head, d_head)
        batch, seq_len, _, _ = x.shape
        return x.reshape(batch, seq_len, self.n_embd)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, n_embd)
            mask: Optional causal mask, shape (seq_len, seq_len).
                  Should have -inf (or large negative) on masked positions.

        Returns:
            Output tensor, shape (batch, seq_len, n_embd)
        """
        batch, seq_len, _ = x.shape

        # 1. Project to Q, K, V using combined matrix
        qkv = np.dot(x, self.wqkv)  # (batch, seq_len, 3*n_embd)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_head, self.d_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, n_head, seq_len, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Apply RoPE to Q and K
        # Compute RoPE cache for head dimension
        cos, sin = get_rope_cache(seq_len, self.d_head)  # (seq_len, d_head)
        # Expand for broadcasting: (1, 1, seq_len, d_head)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q = apply_rope(q, cos, sin)   # (batch, n_head, seq_len, d_head)
        k = apply_rope(k, cos, sin)

        # 3. Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_head)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        # scores shape: (batch, n_head, seq_len, seq_len)

        # 4. Apply causal mask (if provided)
        if mask is not None:
            # mask shape: (seq_len, seq_len) broadcast to all heads and batch
            scores = scores + mask

        # 5. Softmax to get attention weights
        attn_weights = softmax(scores, axis=-1)  # (batch, n_head, seq_len, seq_len)

        # Optional: dropout on attention weights (skip for now)

        # 6. Weighted sum of values
        out = np.matmul(attn_weights, v)  # (batch, n_head, seq_len, d_head)

        # 7. Merge heads
        out = self.merge_heads(out)  # (batch, seq_len, n_embd)

        # 8. Output projection
        out = np.dot(out, self.wo)

        # Optional: scaling by 1/√(n_head) after output? GPT-2 config has scale_attn_weights,
        # but typically scaling is before softmax. We'll skip additional output scaling.

        return out
