"""
GPT-2 configuration hyperparameters.

Based on the original GPT-2 paper and HuggingFace implementation.
Default is GPT-2 Small (124M parameter) configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPT2Config:
    """
    Configuration class for GPT-2 model.

    Default values correspond to GPT-2 Small (124M params):
        n_layer=12, n_embd=768, n_head=12, vocab_size=50257
    """
    vocab_size: int = 50257
    n_positions: int = 1024           # Maximum sequence length
    n_embd: int = 768                 # Embedding dimension
    n_layer: int = 12                 # Number of transformer blocks
    n_head: int = 12                 # Number of attention heads
    n_inner: Optional[int] = None    # FFN inner dimension (4*n_embd if None)
    activation_function: str = "gelu_new"  # "gelu", "gelu_new", "relu"
    resid_pdrop: float = 0.1         # Dropout residual
    embd_pdrop: float = 0.1          # Dropout embeddings
    attn_pdrop: float = 0.1          # Dropout attention
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02  # Std dev for weight init
    scale_attn_weights: bool = True  # Scale attention weights by 1/√(n_head) after output projection?
    # Additional GPT-2 specific flags
    scale_attn_by_inverse_layer_idx: bool = False  # Scale by 1/(layer_idx+1)
    reorder_and_upcast_attn: bool = False          # Mixed precision attention trick

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

        # Validate
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.activation_function in ["gelu", "gelu_new", "relu"], "Invalid activation"

    def get_num_params(self, exact: bool = False) -> int:
        """
        Estimate total number of parameters.

        Args:
            exact: If True, compute exact count based on config; if False, approximate known GPT-2 sizes.

        Returns:
            Total parameter count
        """
        if not exact:
            # Return approximate known counts
            sizes = {
                (117, 12, 768): 124_000_000,
                (345, 24, 1024): 345_000_000,
                (762, 36, 1280): 762_000_000,
                (1542, 48, 1600): 1_542_000_000,
            }
            key = (self.n_layer, self.n_head, self.n_embd)
            return sizes.get(key, -1)

        # Compute roughly
        # Token embeddings: vocab_size * n_embd
        # Position embeddings: not used with RoPE (0)
        # Each block:
        #   LayerNorms: 2 * 2 * n_embd (gain + bias)
        #   Attention: QKV projection: n_embd * 3*n_embd (bias usually absent in GPT2) = 3*n_embd^2
        #              Output projection: n_embd * n_embd = n_embd^2
        #   MLP: n_embd * 4*n_embd + 4*n_embd * n_embd = 4*n_embd^2 + 4*n_embd^2 = 8*n_embd^2 (approx)
        # Final LayerNorm: 2*n_embd
        # LM head tied with embeddings (no extra)
        total = self.vocab_size * self.n_embd  # token embeddings
        total += 2 * self.n_embd  # final layernorm (gain+bias)

        for _ in range(self.n_layer):
            # Two LayerNorms per block (weight + bias each)
            total += 4 * self.n_embd
            # Attention QKV + output
            total += 4 * (self.n_embd * self.n_embd)
            # MLP: two linear layers
            total += 2 * (self.n_embd * self.n_inner) + self.n_inner + self.n_embd

        return total
