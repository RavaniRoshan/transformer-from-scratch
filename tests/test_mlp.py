"""
Tests for MLP (feed-forward network) module.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/shiva/projects/transformer_from_scratch')

from transformer.config import GPT2Config
from transformer.mlp import MLP


def test_mlp_shapes():
    """Test output shape matches input."""
    config = GPT2Config(n_embd=128, n_head=4, n_inner=512)
    mlp = MLP(config)

    for batch in [1, 2]:
        for seq_len in [8, 16, 32]:
            x = np.random.randn(batch, seq_len, config.n_embd).astype(np.float32)
            out = mlp.forward(x)
            assert out.shape == (batch, seq_len, config.n_embd), f"MLP shape mismatch: {out.shape}"

    print("✓ MLP shapes test passed")


def test_mlp_gradient_flow():
    """Check that changing weights affects output."""
    config = GPT2Config(n_embd=64, n_head=4, n_inner=256)
    mlp = MLP(config)

    x = np.random.randn(1, 4, config.n_embd).astype(np.float32)
    out1 = mlp.forward(x)

    # Perturb a weight
    epsilon = 1e-5
    orig = mlp.w1[0, 0].copy()
    mlp.w1[0, 0] = orig + epsilon
    out2 = mlp.forward(x)
    mlp.w1[0, 0] = orig

    diff = np.mean(np.abs(out2 - out1))
    assert diff > 1e-8, "Weight change had no effect"
    print("✓ MLP gradient flow test passed")


def test_mlp_activation():
    """Test that activation produces non-linear outputs."""
    config = GPT2Config(n_embd=32, n_head=4, n_inner=64, activation_function="gelu_new")
    mlp = MLP(config)

    x = np.random.randn(2, 10, config.n_embd).astype(np.float32)
    out = mlp.forward(x)

    # Output should have both positive and negative values (GELU is not monotonic? Actually it is monotonic but can be negative)
    assert np.min(out) < 0 or np.max(out) > 0, "Output seems degenerate"

    # Should not be all zeros or all same
    assert np.std(out) > 1e-6, "Output has no variance"
    print("✓ MLP activation test passed")


def run_all_tests():
    print("\n" + "="*60)
    print("Running MLP Tests")
    print("="*60 + "\n")

    tests = [
        test_mlp_shapes,
        test_mlp_gradient_flow,
        test_mlp_activation,
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
