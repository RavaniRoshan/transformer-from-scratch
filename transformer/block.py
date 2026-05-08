"""
Single Transformer Block (decoder layer without cross-attention).

Each block consists of:
1. Multi-head self-attention with pre-layer normalization and residual connection
2. Position-wise MLP with pre-layer normalization and residual connection

Architecture (pre-norm):
    x -> LayerNorm -> Attention -> x + (residual)
      -> LayerNorm -> MLP        -> x + (residual)
"""

import numpy as np

from transformer.attention import MultiHeadAttention
from transformer.mlp import MLP
from utils.layernorm import LayerNorm


class TransformerBlock:
    """
    A single Transformer decoder block with self-attention and feed-forward network.

    Uses pre-normalization: LayerNorm applied before each sub-layer.
    Residual connections surround each sub-layer.
    """

    def __init__(self, config, layer_idx: int):
        """
        Initialize Transformer block.

        Args:
            config: GPT2Config
            layer_idx: Index of this block in the full model (used for potential scaling)
        """
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd

        # Layer norms (pre-attention and pre-ffn)
        self.ln1 = LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.n_embd, config.layer_norm_epsilon)

        # Sub-modules
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through the block.

        Args:
            x: input tensor, shape (batch, seq_len, n_embd)
            mask: optional causal mask, shape (seq_len, seq_len)

        Returns:
            output of same shape as x
        """
        # 1. Attention sub-layer with residual
        # Pre-norm: apply LayerNorm before attention
        attn_input = self.ln1(x)
        attn_output = self.attn.forward(attn_input, mask=mask)
        # Residual connection
        x = x + attn_output

        # 2. MLP sub-layer with residual
        mlp_input = self.ln2(x)
        mlp_output = self.mlp.forward(mlp_input)
        x = x + mlp_output

        return x
