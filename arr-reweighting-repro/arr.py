"""Toy reproduction of Adaptive Reward Routing (arXiv:2609.37200) Eq.10 preference-preserving reweighting."""
import numpy as np

D, K = 8, 5
NAMES = ["v1", "v2", "a", "c1", "c2"]
C = np.zeros((K, D))
C[0, :2] = [1, .3]
C[1, :2] = [1, -.3]
C[2, [0, 2]] = [-.5, 1]
C[3, [0, 2, 3]] = [.3, .5, 1]
C[4, [2, 4]] = [.5, 1]
SIGMA = np.array([.3, .3, 1.5, .3, .3])
PRIOR = np.full(K, 0.2)


def rewards(theta):
    return -0.5 * ((theta - C) ** 2).sum(1)


def gains(theta):
    r0 = rewards(np.zeros(D))
    return (rewards(theta) - r0) / (0 - r0)


def min_norm_simplex(G, iters=100):
    """MGDA / MARBLE: argmin over simplex of ||sum_k gamma_k G_k||, Frank-Wolfe."""
    gamma = np.full(len(G), 1 / len(G))
    M = G @ G.T
    for _ in range(iters):
        grad = M @ gamma
        j = np.argmin(grad)
        d = np.eye(len(G))[j] - gamma
        denom = d @ M @ d
        if denom <= 1e-12:
            break
        step = np.clip(-(gamma @ M @ d) / denom, 0, 1)
        gamma = gamma + step * d
    return gamma


def eq10_weights(gamma_bar, e, e_warm, kappa=0.5):
    if e < e_warm:
        return PRIOR.copy()
    return (1 - kappa) * PRIOR + kappa * PRIOR.sum() * gamma_bar   # C_m keeps total prior weight


def run(method, seed, rounds=300, lr=0.05, rho=0.9, e_warm=50):
    rng = np.random.default_rng(seed)
    theta, gamma_bar, gamma_log = np.zeros(D), np.full(K, 1 / K), []
    for e in range(rounds):
        G = (C - theta) + SIGMA[:, None] * rng.standard_normal((K, D))
        if method == "GDPO":
            w = PRIOR
        else:
            gamma = min_norm_simplex(G / np.linalg.norm(G, axis=1, keepdims=True))
            gamma_log.append(gamma)
            if method == "MARBLE":
                w = gamma
            else:
                gamma_bar = rho * gamma_bar + (1 - rho) * gamma
                w = eq10_weights(gamma_bar, e, e_warm if method == "Residual+Warmup" else 0)
        theta = theta + lr * (w[:, None] * G).sum(0)
    mean_gamma = np.mean(gamma_log, 0) if gamma_log else None
    return gains(theta), mean_gamma
