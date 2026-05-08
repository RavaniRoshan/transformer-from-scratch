"""
Text generation utilities for GPT-2 model.

Implements:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Autoregressive generation with optional caching (KV cache not implemented in pure NumPy yet)

Generation strategy: For each step, compute logits for next token, apply sampling
filters, sample a token, append to context, repeat.
"""

import numpy as np
from typing import Optional, Tuple

from utils.activation import softmax


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k logits, set the rest to -inf.

    Args:
        logits: (vocab_size,)
        k: number of top tokens to keep (k > 0)

    Returns:
        filtered logits (same shape)
    """
    if k <= 0:
        return logits  # no filtering

    # Get indices of top-k logits
    top_k_indices = np.argsort(logits)[-k:]
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_indices] = logits[top_k_indices]
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep the smallest set of tokens with cumulative probability >= p.

    Args:
        logits: (vocab_size,)
        p: cumulative probability threshold (0 < p <= 1)

    Returns:
        filtered logits
    """
    if p >= 1.0:
        return logits

    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]  # descending order
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff_idx = np.searchsorted(cumulative, p) + 1
    cutoff_idx = min(cutoff_idx, len(logits))

    kept_indices = sorted_indices[:cutoff_idx]
    filtered = np.full_like(logits, -np.inf)
    filtered[kept_indices] = logits[kept_indices]
    return filtered


def sample_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[np.random.Generator] = None
) -> int:
    """
    Sample a single token from logits with optional filtering.

    Args:
        logits: (vocab_size,) unnormalized logits
        temperature: softmax temperature. 1.0 = no change, <1 = sharper, >1 = flatter, 0 = greedy
        top_k: keep top-k tokens only (0 or None = disabled)
        top_p: cumulative probability cutoff for nucleus sampling (None = disabled)
        rng: optional NumPy random generator

    Returns:
        token index (int)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Temperature scaling
    if temperature > 0:
        scaled_logits = logits / temperature
    else:
        # Greedy decode
        return int(np.argmax(logits))

    # Top-k
    if top_k is not None and top_k > 0:
        scaled_logits = top_k_filter(scaled_logits, top_k)

    # Top-p
    if top_p is not None and top_p < 1.0:
        scaled_logits = top_p_filter(scaled_logits, top_p)

    # Convert to probabilities
    probs = softmax(scaled_logits)

    # Sample
    token_id = int(rng.choice(len(probs), p=probs))
    return token_id


def generate(
    model,
    prompt_ids: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_token: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Autoregressive text generation.

    Args:
        model: GPT2 instance
        prompt_ids: initial token IDs, shape (batch, seq_len) or (seq_len,)
        max_new_tokens: maximum number of new tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (0/None to disable)
        top_p: nucleus sampling threshold (None to disable)
        stop_token: optional token ID that halts generation (e.g., EOS)
        rng: optional random generator

    Returns:
        Generated token IDs, shape (batch, seq_len + generated_len)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure 2D
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]  # (1, seq_len)

    tokens = prompt_ids.copy()
    batch_size, prompt_len = tokens.shape
    seq_len = model.config.n_positions

    for _ in range(max_new_tokens):
        # Use only the most recent `seq_len` tokens to avoid index errors
        context = tokens[:, -seq_len:]

        # Forward pass
        logits = model.forward(context)  # (batch, seq_len, vocab_size)

        # Get logits for the last token only
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        # For now assume batch=1; extend later
        token_id = sample_logits(next_logits[0], temperature, top_k, top_p, rng)

        # Append token
        tokens = np.concatenate([tokens, [[token_id]]], axis=1)

        # Stop if EOS
        if stop_token is not None and token_id == stop_token:
            break

    return tokens
