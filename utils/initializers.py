"""
Weight initialization utilities for Transformer components.

Follows GPT-2 initialization practices:
- Linear layers: truncated normal (approximated by normal with small std)
- LayerNorm: γ=1, β=0 (default already)
- Embeddings: normal(0, initializer_range)
"""

import numpy as np


def init_linear(shape: tuple[int, ...], initializer_range: float = 0.02) -> np.ndarray:
    """
    Initialize linear layer weights with normal distribution.

    GPT-2 uses a modified initialization accounting for residual accumulation:
    weights of residual layers are scaled by 1/√N at initialization.
    This function provides the base initialization; scaling should be applied separately.

    Args:
        shape: weight shape (fan_in, fan_out) or (out, in)
        initializer_range: standard deviation of normal distribution

    Returns:
        Weight matrix with shape
    """
    return np.random.normal(0.0, initializer_range, size=shape).astype(np.float32)


def init_embedding(shape: tuple[int, ...], initializer_range: float = 0.02) -> np.ndarray:
    """
    Initialize embedding table with normal distribution.

    Args:
        shape: (vocab_size, d_model)
        initializer_range: standard deviation

    Returns:
        Embedding matrix
    """
    return np.random.normal(0.0, initializer_range, size=shape).astype(np.float32)


def xavier_uniform(shape: tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.

    Useful for linear layers when using tanh or ReLU and no residual.

    Args:
        shape: (fan_in, fan_out)
        gain: scaling factor (√2 for ReLU)

    Returns:
        Initialized weight matrix
    """
    fan_in, fan_out = shape[-2], shape[-1]
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def kaiming_normal(shape: tuple[int, ...], mode: str = 'fan_in') -> np.ndarray:
    """
    Kaiming/He normal initialization for ReLU activations.

    Args:
        shape: weight shape
        mode: 'fan_in' or 'fan_out'

    Returns:
        Initialized weight matrix
    """
    fan = shape[-1] if mode == 'fan_out' else shape[-2]
    std = np.sqrt(2.0 / fan)
    return np.random.normal(0.0, std, size=shape).astype(np.float32)


def scaled_init(weights: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Scale residual weights by 1/√N as done in GPT-2.

    Applied to projection matrices that sit on residual paths.
    Formula: w = w_original / √num_layers

    Args:
        weights: original weight matrix
        num_layers: total number of residual layers (N in the paper)

    Returns:
        Scaled weight matrix
    """
    scale = 1.0 / np.sqrt(num_layers)
    return weights * scale
