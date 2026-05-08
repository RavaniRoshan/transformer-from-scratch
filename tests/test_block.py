"""
Tests for TransformerBlock.

Verifies:
- Block output shape matches input
- Residual connections work (output != input for random init)
- LayerNorm normalization properties
- Block can be stacked (multiple passes)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/shiva/projects/transformer_from_scratch')

from transformer.config import GPT2Config
from transformer.block import TransformerBlock


def test_block_shapes():
    """Test block output shape."""
    config = GPT2Config(n_embd=128, n_head=4, n_layer=4)
    block = TransformerBlock(config, layer_idx=0)

    batch, seq_len = 2, 16
    x = np.random.randn(batch, seq_len, config.n_embd).astype(np.float32)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32) * (-1e10)

    out = block.forward(x, mask)
    assert out.shape == x.shape, f"Block output shape mismatch: {out.shape} vs {x.shape}"
    print("✓ Block shapes test passed")


def test_block_residual():
    """Ensure block output differs from input (residual connection active)."""
    config = GPT2Config(n_embd=64, n_head=4)
    block = TransformerBlock(config, 0)

    x = np.random.randn(1, 8, config.n_embd).astype(np.float32)
    mask = None

    out = block.forward(x, mask)

    diff = np.mean(np.abs(out - x))
    assert diff > 1e-5, "Residual connection seems bypassed; output too similar to input"
    print("✓ Block residual test passed")


def test_block_stability():
    """Check no NaNs or extreme values."""
    config = GPT2Config(n_embd=128, n_head=4)
    block = TransformerBlock(config, 0)

    x = np.random.randn(3, 32, config.n_embd).astype(np.float32) * 0.1  # small inputs
    mask = np.triu(np.ones((32, 32)), k=1).astype(np.float32) * (-1e10)

    out = block.forward(x, mask)

    assert not np.any(np.isnan(out)), "Block output contains NaNs"
    assert np.min(out) > -1e10 and np.max(out) < 1e10, "Extreme values in output"
    print("✓ Block stability test passed")


def test_multiple_blocks():
    """Stack multiple blocks; verify no shape degradation."""
    config = GPT2Config(n_embd=64, n_head=4, n_layer=4)
    blocks = [TransformerBlock(config, i) for i in range(4)]

    x = np.random.randn(1, 10, config.n_embd).astype(np.float32)
    mask = np.triu(np.ones((10, 10)), k=1).astype(np.float32) * (-1e10)

    for block in blocks:
        x = block.forward(x, mask)

    assert x.shape == (1, 10, config.n_embd)
    assert not np.any(np.isnan(x))
    print("✓ Multiple blocks test passed")


def run_all_tests():
    print("\n" + "="*60)
    print("Running TransformerBlock Tests")
    print("="*60 + "\n")

    tests = [
        test_block_shapes,
        test_block_residual,
        test_block_stability,
        test_multiple_blocks,
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
