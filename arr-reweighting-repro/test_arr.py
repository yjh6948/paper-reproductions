import numpy as np
from arr import min_norm_simplex, eq10_weights, gains, C, PRIOR


def test_min_norm_two_opposite_unit_vectors_is_half():
    G = np.array([[1.0, 0], [-1.0, 0]])
    assert np.allclose(min_norm_simplex(G), [0.5, 0.5], atol=1e-6)


def test_min_norm_matches_brute_force_grid():
    rng = np.random.default_rng(1)
    G = rng.standard_normal((3, 4))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    fw = np.linalg.norm(min_norm_simplex(G, iters=2000) @ G)
    grid = min(np.linalg.norm(np.array([a, b, 1 - a - b]) @ G)
               for a in np.linspace(0, 1, 201) for b in np.linspace(0, 1, 201) if a + b <= 1)
    assert fw <= grid + 1e-3


def test_eq10_prior_before_warmup_and_floor_after():
    gb = np.array([1.0, 0, 0, 0, 0])
    assert np.allclose(eq10_weights(gb, 10, 50), PRIOR)
    w = eq10_weights(gb, 60, 50)
    assert np.isclose(w.sum(), PRIOR.sum())
    assert np.all(w >= 0.5 * PRIOR - 1e-12)              # prior floor (1-kappa)*prior


def test_gains_zero_at_origin_one_at_own_optimum():
    assert np.allclose(gains(np.zeros(8)), 0)
    assert np.isclose(gains(C[2])[2], 1)
