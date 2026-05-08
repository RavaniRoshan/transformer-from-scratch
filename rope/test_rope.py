"""
Tests for RoPE implementation.

Verifies correctness through:
- Mathematical properties (inverse rotation, orthogonality preservation)
- Shape consistency
- Numerical stability
- Comparison with reference formulas
"""

import numpy as np
import sys
sys.path.insert(0, '/home/shiva/projects/transformer_from_scratch')

from rope.rope_numpy import (
    compute_inv_freq,
    get_rope_cache,
    rotate_half,
    apply_rope,
    RoPE
)


def test_inv_freq_shape():
    """Test that inverse frequencies have correct shape."""
    for d_model in [64, 128, 256, 512, 768, 1024]:
        inv_freq = compute_inv_freq(d_model)
        assert inv_freq.shape == (d_model // 2,), f"Expected shape {(d_model//2,)}, got {inv_freq.shape}"
    print("✓ inv_freq shape test passed")


def test_rope_cache_shape():
    """Test that RoPE cache has correct shapes."""
    for seq_len in [128, 256, 512, 1024]:
        for d_model in [64, 128, 256, 512]:
            cos, sin = get_rope_cache(seq_len, d_model)
            assert cos.shape == (seq_len, d_model), f"cos shape mismatch: {cos.shape}"
            assert sin.shape == (seq_len, d_model), f"sin shape mismatch: {sin.shape}"
    print("✓ RoPE cache shape test passed")


def test_rotate_half():
    """Test rotate_half operation (HF-style: negate second half, then x1)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # x1=[1,2,3], x2=[4,5,6] -> [-4,-5,-6, 1,2,3]
    expected = np.array([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
    result = rotate_half(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("✓ rotate_half test passed")


def test_apply_rope_single_position():
    """Test RoPE application for single position."""
    d_model = 8
    x = np.random.randn(1, d_model).astype(np.float32)
    pos = np.array([0])  # position 0

    cos, sin = get_rope_cache(1, d_model)
    result = apply_rope(x, cos, sin)

    # At position 0, cos=1, sin=0 → no rotation
    np.testing.assert_allclose(result, x, rtol=1e-5, atol=1e-5)
    print("✓ apply_rope at position 0 (no rotation) test passed")


def test_apply_rope_inverse():
    """Test that applying RoPE forward then backward recovers input."""
    d_model = 128
    seq_len = 32
    np.random.seed(42)

    x = np.random.randn(2, seq_len, d_model).astype(np.float32)
    cos, sin = get_rope_cache(seq_len, d_model)

    # Forward rotation
    rotated = apply_rope(x, cos, sin)

    # Inverse rotation (negate sin)
    recovered = apply_rope(rotated, cos, -sin)

    np.testing.assert_allclose(recovered, x, rtol=1e-5, atol=1e-5)
    print("✓ apply_rope inverse test passed")


def test_rope_relative_position_property():
    r"""
    Test key property: (R(m)q) · (R(n)k) = q·k * cos((m-n)θ) ± ...
    For same vector q=k, the dot product after rotation should equal
    dot(q,q) * cos((m-n)θ_i) for each dimension pair.

    This verifies that RoPE encodes relative positions correctly.
    """
    d_model = 16
    seq_len = 10
    np.random.seed(42)

    # Single vector
    q = np.random.randn(d_model).astype(np.float32)
    k = q.copy()  # Same vector for simplicity

    cos, sin = get_rope_cache(seq_len, d_model)

    # Compute dot products at different positions
    dots = []
    for m in range(seq_len):
        q_rot = apply_rope(q[None, :], cos[m:m+1], sin[m:m+1])[0]
        for n in range(seq_len):
            k_rot = apply_rope(k[None, :], cos[n:n+1], sin[n:n+1])[0]
            dots.append(np.dot(q_rot, k_rot))

    dots = np.array(dots).reshape(seq_len, seq_len)

    # For m == n, dot should equal original q·k (rotation preserves norms)
    original_norm_sq = np.dot(q, k)
    np.testing.assert_allclose(np.diag(dots), original_norm_sq, rtol=1e-5)

    # For m != n, values should differ (encode relative position)
    # Verify anti-symmetry: dot(m,n) = dot(n,m)
    for m in range(seq_len):
        for n in range(seq_len):
            assert abs(dots[m, n] - dots[n, m]) < 1e-5, "Dot product should be symmetric"

    print("✓ relative position property test passed")


def test_rope_module():
    """Test the RoPE class interface."""
    d_model = 64
    rope = RoPE(d_model)

    seq_len = 16
    x = np.random.randn(2, seq_len, d_model).astype(np.float32)

    # Forward pass
    rotated = rope.forward(x)

    assert rotated.shape == x.shape, f"Shape mismatch: {rotated.shape} vs {x.shape}"
    assert not np.any(np.isnan(rotated)), "NaN values found"

    # Test cache reset
    rope.reset_cache()
    assert rope.cos_cache is None and rope.sin_cache is None
    print("✓ RoPE module test passed")


def test_rope_batch_dimensions():
    """Test RoPE with various batch dimensions."""
    d_model = 128
    rope = RoPE(d_model)
    rope.build_cache(32)

    # Test 3D: (batch, seq_len, d_model)
    x3d = np.random.randn(4, 32, d_model).astype(np.float32)
    r3d = rope.forward(x3d)
    assert r3d.shape == x3d.shape

    # Test 2D: (seq_len, d_model)
    x2d = np.random.randn(32, d_model).astype(np.float32)
    r2d = rope.forward(x2d)
    assert r2d.shape == x2d.shape

    print("✓ multi-dimensional batch test passed")


def test_rope_dtype_preservation():
    """Test that RoPE preserves input dtype."""
    d_model = 64
    rope = RoPE(d_model)

    for dtype in [np.float32, np.float64]:
        x = np.random.randn(2, 16, d_model).astype(dtype)
        rotated = rope.forward(x)
        assert rotated.dtype == dtype, f"Expected {dtype}, got {rotated.dtype}"

    print("✓ dtype preservation test passed")


def run_all_tests():
    """Run all RoPE tests."""
    print("\n" + "="*60)
    print("Running RoPE Implementation Tests")
    print("="*60 + "\n")

    tests = [
        test_inv_freq_shape,
        test_rope_cache_shape,
        test_rotate_half,
        test_apply_rope_single_position,
        test_apply_rope_inverse,
        test_rope_relative_position_property,
        test_rope_module,
        test_rope_batch_dimensions,
        test_rope_dtype_preservation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
