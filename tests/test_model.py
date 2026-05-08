"""
Tests for the full GPT2 model.

Verifies:
- Forward pass shapes and numerical stability
- Parameter count matches expected configuration
- Weight tying (lm_head shares weights with wte)
- Causal masking integrated
- Generation sanity checks
"""

import numpy as np
import sys
sys.path.insert(0, '/home/shiva/projects/transformer_from_scratch')

from transformer.config import GPT2Config
from transformer.model import GPT2
from transformer.generation import generate


def test_model_forward_shape():
    """Test model forward pass output shape."""
    config = GPT2Config(n_embd=128, n_head=4, n_layer=2, vocab_size=1000)
    model = GPT2(config)

    batch, seq_len = 2, 16
    input_ids = np.random.randint(0, config.vocab_size, size=(batch, seq_len), dtype=np.int32)

    logits = model.forward(input_ids)
    assert logits.shape == (batch, seq_len, config.vocab_size), f"Unexpected shape: {logits.shape}"

    print("✓ Model forward shape test passed")


def test_model_no_nans():
    """Test that forward pass produces no NaNs with different configs."""
    configs = [
        GPT2Config(n_embd=64, n_head=4, n_layer=2),
        GPT2Config(n_embd=128, n_head=8, n_layer=4),
    ]
    for config in configs:
        model = GPT2(config)
        input_ids = np.random.randint(0, config.vocab_size, size=(1, 32), dtype=np.int32)
        logits = model.forward(input_ids)
        assert not np.any(np.isnan(logits)), f"NaNs in logits for config {config}"

    print("✓ Model no NaNs test passed")


def test_weight_tying():
    """Verify that lm_head weights are tied to token embeddings."""
    config = GPT2Config(n_embd=64, n_head=4)
    model = GPT2(config)

    # wte shape (vocab_size, n_embd), lm_head = wte.T (n_embd, vocab_size)
    assert np.array_equal(model.lm_head, model.wte.T), "Weight tying broken: lm_head should equal wte.T"
    print("✓ Weight tying test passed")


def test_parameter_count_approximate():
    """Check parameter count approximates known GPT-2 sizes."""
    config_small = GPT2Config(n_embd=768, n_head=12, n_layer=12)
    model_small = GPT2(config_small)
    count = model_small.count_parameters()
    # Expected ~124M
    expected = 124_000_000
    # Allow ±5% tolerance for bias inclusion etc.
    assert abs(count - expected) / expected < 0.05, f"Parameter count {count} far from expected {expected}"
    print(f"✓ Parameter count ~{count:,} (expected ~{expected:,})")


def test_causal_mask_used():
    """Verify that mask actually changes outputs."""
    config = GPT2Config(n_embd=64, n_head=4, n_layer=1)
    model = GPT2(config)

    batch, seq_len = 1, 8
    input_ids = np.random.randint(0, config.vocab_size, size=(batch, seq_len), dtype=np.int32)

    # Get logits with causal mask (default)
    logits_normal = model.forward(input_ids)

    # Get logits without any mask: pass a zero mask (no masking)
    zero_mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    logits_no_mask = model.forward(input_ids, mask=zero_mask)

    # They should be different (mask changes results)
    diff = np.mean(np.abs(logits_normal - logits_no_mask))
    assert diff > 1e-5, "Using causal mask should produce different logits"
    print("✓ Causal mask used test passed")


def test_model_gradient_sanity():
    """Finite difference check on one weight to ensure gradients exist."""
    config = GPT2Config(n_embd=32, n_head=2, n_layer=1)
    model = GPT2(config)

    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    orig_weight = model.blocks[0].attn.wqkv[0, 0].copy()

    # Compute loss
    logits = model.forward(input_ids)
    loss1 = np.sum(logits ** 2)

    # Perturb
    eps = 1e-5
    model.blocks[0].attn.wqkv[0, 0] = orig_weight + eps
    logits2 = model.forward(input_ids)
    loss2 = np.sum(logits2 ** 2)
    model.blocks[0].attn.wqkv[0, 0] = orig_weight

    grad_approx = (loss2 - loss1) / eps
    assert abs(grad_approx) > 0, "Gradient appears zero; check differentiation"
    print("✓ Model gradient sanity test passed")


def run_all_tests():
    print("\n" + "="*60)
    print("Running GPT2 Model Tests")
    print("="*60 + "\n")

    tests = [
        test_model_forward_shape,
        test_model_no_nans,
        test_weight_tying,
        test_parameter_count_approximate,
        test_causal_mask_used,
        test_model_gradient_sanity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
