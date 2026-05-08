"""
Full GPT-2 model implementation.

Architecture:
1. Token embeddings (vocab_size x n_embd)
2. N Transformer blocks with:
   - Multi-head self-attention (with RoPE)
   - Position-wise feed-forward network
3. Final layer normalization
4. Language modeling head (projection to vocab_size)

Weight tying: The input token embedding weights are tied to the output LM head.
"""

import numpy as np

from transformer.config import GPT2Config
from transformer.block import TransformerBlock
from utils.layernorm import LayerNorm
from utils.initializers import init_embedding


class GPT2:
    """
    GPT-2 language model.

    This is a decoder-only Transformer that predicts the next token given previous context.
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize GPT-2 model.

        Args:
            config: GPT2Config specifying architecture hyperparameters.
        """
        self.config = config

        # Token embedding table
        self.wte = init_embedding((config.vocab_size, config.n_embd), config.initializer_range)

        # Dropout placeholders (not implemented in pure NumPy; would require mask)
        self.embd_dropout = config.embd_pdrop
        self.resid_dropout = config.resid_pdrop
        self.attn_dropout = config.attn_pdrop

        # Transformer blocks
        self.blocks = [TransformerBlock(config, i) for i in range(config.n_layer)]

        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, config.layer_norm_epsilon)

        # Language modeling head (tied to wte)
        # For weight tying, we simply use wte.T as the projection matrix.
        self.lm_head = self.wte.T  # shape (n_embd, vocab_size)

    def forward(self, input_ids: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass returning logits.

        Args:
            input_ids: token IDs of shape (batch, seq_len)
            mask: optional causal mask of shape (seq_len, seq_len).
                  If None, a causal mask (upper triangular -inf) is created.

        Returns:
            logits: unnormalized predictions, shape (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape

        # 1. Token embeddings
        # wte: (vocab_size, n_embd), gather by input_ids
        x = self.wte[input_ids]  # (batch, seq_len, n_embd)

        # 2. Build causal mask if not provided
        if mask is None:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32) * (-1e10)
            # mask shape: (seq_len, seq_len)

        # 3. Pass through Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # 4. Final layer norm
        x = self.ln_f(x)

        # 5. Project to vocabulary
        logits = np.dot(x, self.lm_head)  # (batch, seq_len, vocab_size)

        return logits

    def get_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create an upper-triangular causal mask.

        Positions (i, j) where j > i are masked to -inf.
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32) * (-1e10)
        return mask

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Integer parameter count
        """
        total = 0
        # Token embeddings
        total += self.wte.size
        # Final LayerNorm
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        # Each block:
        for block in self.blocks:
            # Attention: wqkv, wo, ln1 gamma/beta
            total += block.attn.wqkv.size + block.attn.wo.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            # MLP: w1, b1, w2, b2, ln2 gamma/beta
            total += block.mlp.w1.size + block.mlp.b1.size + block.mlp.w2.size + block.mlp.b2.size
            total += block.ln2.gamma.size + block.ln2.beta.size

        # LM head is tied, not counted extra
        return total
