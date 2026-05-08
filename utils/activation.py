"""
Activation functions and utility operations.

Implements GELU variants and numerically stable softmax.
"""

import numpy as np
# from scipy.special import erf  # optional for exact GELU


def gelu_new(x: np.ndarray) -> np.ndarray:
    """
    GELU activation used in GPT-2 (approximation).

    Formula: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    This is the approximate GELU from the original paper, used in GPT-2.
    It's faster than exact GELU and numerically stable.

    Args:
        x: input array

    Returns:
        GELU-activated array
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """
    Exact GELU using error function.

    Formula: x * 0.5 * (1 + erf(x / √2))

    Requires scipy. Slightly slower but more accurate.

    Args:
        x: input array

    Returns:
        GELU-activated array
    """
    from scipy.special import erf
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def relu(x: np.ndarray) -> np.ndarray:
    """Simple ReLU."""
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Subtracts max before exponentiating to avoid overflow.

    Args:
        x: input array (any shape)
        axis: dimension along which to compute softmax

    Returns:
        Softmax probabilities (same shape as x)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True) + 1e-10)
    return x - x_max - log_sum_exp
