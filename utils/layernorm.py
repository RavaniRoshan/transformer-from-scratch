"""
Layer Normalization implementation.

Normalizes activations across the feature dimension (last axis).
Stabilizes training by reducing internal covariate shift.
"""

import numpy as np


class LayerNorm:
    """
    Layer normalization layer.

    Applies per-sample normalization over the last dimension:
        y = γ * (x - μ) / √(σ² + ε) + β

    where μ and σ² are computed over the last axis.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize LayerNorm.

        Args:
            d_model: Feature dimension (size of last axis)
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        # Gain (γ) and bias (β) parameters
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.

        Args:
            x: input array of shape (..., d_model) or (..., seq_len, d_model)
               Normalization is applied over the last axis.

        Returns:
            Normalized array of same shape as x
        """
        # Compute mean and variance over last axis
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def load_weights(self, gamma: np.ndarray, beta: np.ndarray):
        """Load pretrained weights (for compatibility)."""
        assert gamma.shape == (self.d_model,)
        assert beta.shape == (self.d_model,)
        self.gamma = gamma.astype(np.float32)
        self.beta = beta.astype(np.float32)
