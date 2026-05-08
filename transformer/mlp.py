"""
Feed-Forward Network (MLP) for Transformer blocks.

GPT-2 uses a two-layer MLP with GELU activation and a 4x expansion factor:
    FFN(x) = gelu(x W1 + b1) W2 + b2

Where:
- W1: (n_embd, n_inner) typically n_inner = 4 * n_embd
- W2: (n_inner, n_embd)
- No bias on W2 in GPT-2? Actually GPT-2 uses bias on both linear layers, but many implementations omit. We'll include bias optionally.
"""

import numpy as np

from utils.activation import gelu_new
from utils.initializers import init_linear


class MLP:
    """
    Position-wise feed-forward network with GELU activation.

    Each position is processed independently with the same two linear layers.
    """

    def __init__(self, config):
        """
        Initialize MLP.

        Args:
            config: GPT2Config with n_embd, n_inner, activation_function, initializer_range
        """
        self.n_embd = config.n_embd
        self.n_inner = config.n_inner

        # First linear layer: expands to n_inner
        self.w1 = init_linear((self.n_embd, self.n_inner), config.initializer_range)
        self.b1 = np.zeros((self.n_inner,), dtype=np.float32)  # bias

        # Second linear layer: projects back to n_embd
        self.w2 = init_linear((self.n_inner, self.n_embd), config.initializer_range)
        self.b2 = np.zeros((self.n_embd,), dtype=np.float32)

        # Activation
        if config.activation_function == "gelu_new":
            self.activation = gelu_new
        elif config.activation_function == "gelu":
            from utils.activation import gelu_exact
            self.activation = gelu_exact
        elif config.activation_function == "relu":
            self.activation = relu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through MLP.

        Args:
            x: input of shape (batch, seq_len, n_embd)

        Returns:
            output of shape (batch, seq_len, n_embd)
        """
        # First linear + activation
        h = np.dot(x, self.w1) + self.b1  # (batch, seq_len, n_inner)
        h = self.activation(h)

        # Second linear
        out = np.dot(h, self.w2) + self.b2  # (batch, seq_len, n_embd)
        return out
