"""
Pure NumPy implementation of Rotary Position Embeddings (RoPE).

This module implements RoPE from scratch without any deep learning libraries,
following the mathematical formulation from Su et al. (2021) "RoFormer: Enhanced
Transformer with Rotary Position Embedding".

Key concepts:
- RoPE encodes position by rotating query/key vectors in complex 2D planes
- For each dimension pair (x_{2i}, x_{2i+1}), apply rotation matrix:
    [cos(mθ_i)  -sin(mθ_i)]
    [sin(mθ_i)   cos(mθ_i)]
- Frequencies: θ_i = base^(-2i/d) where base=10000 (standard)
- Result: dot product encodes relative position: (R(m)q)·(R(n)k) = q·k rotated by (m-n)
"""

import numpy as np


def compute_inv_freq(d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Compute inverse frequencies for RoPE.

    θ_i = 1 / base^(2i/d) for i = 0, 1, ..., d/2-1

    Args:
        d_model: Embedding dimension
        base: Base frequency (default 10000)

    Returns:
        inv_freq: shape (d_model/2,)
    """
    dim = np.arange(0, d_model, 2, dtype=np.float32) / d_model
    inv_freq = 1.0 / (base ** dim)
    return inv_freq


def get_rope_cache(seq_len: int, d_model: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute RoPE rotation matrices (cosine and sine) for all positions.

    This is the efficient implementation that precomputes sin/cos values to avoid
    recomputing them during every forward pass.

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension
        base: Base frequency for rotations

    Returns:
        cos_cache: shape (seq_len, d_model)
        sin_cache: shape (seq_len, d_model)

    Note:
        The cache is arranged so that for dimension pair (i, i + d/2),
        both dimensions share the same cosine/sine values. This is achieved
        by concatenating the frequency values with themselves, not repeating.
        i.e., cos[pos] = [cos_0, cos_1, ..., cos_{d/2-1}, cos_0, cos_1, ..., cos_{d/2-1}]
    """
    inv_freq = compute_inv_freq(d_model, base)  # (d_model/2,)
    positions = np.arange(seq_len, dtype=np.float32)[:, None]  # (seq_len, 1)
    freqs = positions * inv_freq[None, :]  # (seq_len, d_model/2)

    cos_vals = np.cos(freqs)  # (seq_len, d_model/2)
    sin_vals = np.sin(freqs)

    # Tile to full dimension: concatenate along feature axis
    # This ensures dim i and i+d/2 share the same frequency
    cos_cache = np.concatenate([cos_vals, cos_vals], axis=1)  # (seq_len, d_model)
    sin_cache = np.concatenate([sin_vals, sin_vals], axis=1)

    return cos_cache.astype(np.float32), sin_cache.astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Rotate half the dimensions according to RoPE pairing (x_i with x_{i + d/2}).

    For input vector x = [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}],
    the output is: [-x_{d/2}, -x_{d/2+1}, ..., -x_{d-1}, x_0, x_1, ..., x_{d/2-1}]

    This corresponds to the "neg_half" operation used in HuggingFace's RoPE
    implementation, where dimensions are paired as (i, i + d/2).

    Combined with cos/sin: result = x * cos + rotate_half(x) * sin
    gives the 2D rotation for each pair (x_i, x_{i+d/2}).

    Args:
        x: shape (..., d_model)

    Returns:
        rotated: same shape as x
    """
    d = x.shape[-1]
    mid = d // 2
    x1 = x[..., :mid]   # First half
    x2 = x[..., mid:]  # Second half
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embeddings to input tensor.

    The rotation formula for each 2D pair (x_{2i}, x_{2i+1}) at position m:
        x'_{2i}   = x_{2i} * cos(mθ_i) - x_{2i+1} * sin(mθ_i)
        x'_{2i+1} = x_{2i+1} * cos(mθ_i) + x_{2i} * sin(mθ_i)

    Vectorized form: x_rotated = x * cos + rotate_half(x) * sin

    Args:
        x: input tensor, shape (..., seq_len, d_model) or (..., d_model)
        cos: cosine cache, shape (seq_len, d_model) or broadcastable
        sin: sine cache, shape (seq_len, d_model) or broadcastable

    Returns:
        rotated tensor of same shape as x
    """
    return (x * cos) + (rotate_half(x) * sin)


class RoPE:
    """
    Rotary Position Embedding module with caching.

    This class manages RoPE cache and applies rotations to query/key tensors.
    Typically used in multi-head attention modules.
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        """
        Args:
            d_model: Embedding dimension (must be even)
            base: Base frequency for rotations
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.base = base
        self.cos_cache = None
        self.sin_cache = None

    def build_cache(self, seq_len: int):
        """
        Build and cache sin/cos values for given sequence length.

        Cache is built lazily and reused for subsequent forward passes
        as long as seq_len doesn't exceed cached length.

        Args:
            seq_len: Maximum sequence length to cache
        """
        self.cos_cache, self.sin_cache = get_rope_cache(seq_len, self.d_model, self.base)

    def forward(self, x: np.ndarray, position_ids: np.ndarray = None) -> np.ndarray:
        """
        Apply RoPE to input tensor.

        Args:
            x: shape (..., seq_len, d_model) or (..., d_model)
            position_ids: optional explicit position indices, shape (..., seq_len)
                         If None, uses sequential positions [0, 1, ..., seq_len-1]

        Returns:
            rotated tensor of same shape as x
        """
        if position_ids is None:
            seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[-1] // 2
            position_ids = np.arange(seq_len, dtype=np.int32)

        # Build cache if needed
        max_pos = np.max(position_ids) + 1
        if self.cos_cache is None or max_pos > self.cos_cache.shape[0]:
            self.build_cache(int(max_pos))

        # Gather cos/sin for each position
        cos = self.cos_cache[position_ids]  # (..., seq_len, d_model)
        sin = self.sin_cache[position_ids]

        return apply_rope(x, cos, sin)

    def reset_cache(self):
        """Clear cached sin/cos values."""
        self.cos_cache = None
        self.sin_cache = None
