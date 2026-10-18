import numpy as np
from clm_repro import common_prefix, prefix_reuse_flops, eff_advantage, N_PARAMS


def test_append_only_has_no_reprefill():
    calls = [([("sys", 100)], 10), ([("sys", 100), (("out", 0), 10), ("obs", 50)], 10)]
    # call 2 re-uses sys+out (110 tokens), prefills only obs (50)
    assert prefix_reuse_flops(calls) == 2 * N_PARAMS * (100 + 10 + 50 + 10)


def test_front_edit_costs_more_than_tail_edit():
    base = [("sys", 100), ("a", 1000), ("b", 1000), ("c", 1000)]
    first = [(base, 0)]
    front = first + [([("sys", 100), ("a2", 10), ("b", 1000), ("c", 1000)], 0)]
    tail = first + [([("sys", 100), ("a", 1000), ("b", 1000), ("c2", 10)], 0)]
    assert prefix_reuse_flops(front) > prefix_reuse_flops(tail)


def test_common_prefix():
    assert common_prefix([("x", 5), ("y", 3)], [("x", 5), ("z", 3)]) == 5


def test_eq6_fewer_than_two_successes_is_zero():
    assert np.all(eff_advantage(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 0])) == 0)


def test_eq6_failures_get_zero_and_clip():
    c = np.array([1.0, 100.0, 0.5, 9.0])
    s = np.array([1, 1, 0, 0])
    A = eff_advantage(c, s)
    assert A[2] == 0 and A[3] == 0
    assert np.isclose(A[0], (50.5 - 1) / 50.5)
    assert np.isclose(A[1], (50.5 - 100) / 50.5)         # -0.98 -> not clipped
    A2 = eff_advantage(np.array([1.0, 1.0, 1000.0]), np.array([1, 1, 1]))
    assert A2[2] == -1.0                                 # (334-1000)/334 = -1.99 -> clipped


def test_eq6_average_cost_gets_zero():
    A = eff_advantage(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert np.isclose(A[1], 0)


def test_logpi_gradient_finite_difference():
    theta, a, eps = np.array([0.3, -0.2, 0.5, 0.1]), 2, 1e-6
    logpi = lambda th: th[a] - np.log(np.exp(th).sum())
    pi = np.exp(theta) / np.exp(theta).sum()
    analytic = np.eye(4)[a] - pi
    numeric = np.array([(logpi(theta + eps * e) - logpi(theta - eps * e)) / (2 * eps) for e in np.eye(4)])
    assert np.allclose(analytic, numeric, atol=1e-6)
