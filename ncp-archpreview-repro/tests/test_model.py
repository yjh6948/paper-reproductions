import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import CHUNK_LEN, generate_sequence
from model import MiniNCP, TokenOnlyBaseline


class TestShapes(unittest.TestCase):
    def test_baseline_step_runs_and_returns_scalar_loss(self):
        tok = generate_sequence(n_chunks=8, seed=0)
        model = TokenOnlyBaseline(seed=0)
        loss = model.train_step(tok)
        self.assertIsInstance(float(loss), float)

    def test_mini_ncp_step_runs_and_losses_are_finite(self):
        tok = generate_sequence(n_chunks=8, seed=0)
        model = MiniNCP(seed=0)
        total, ntp, ncp, vq = model.train_step(tok)
        for v in (total, ntp, ncp, vq):
            self.assertTrue(np.isfinite(v))


class TestGradientCorrectness(unittest.TestCase):
    """Finite-difference check of the hand-derived backward pass against
    numerical gradients, on a handful of parameters across every module
    (embedding, decoder, concept-module weights, codebook)."""

    def setUp(self):
        self.tok = generate_sequence(n_chunks=6, seed=1)
        self.model = MiniNCP(seed=2)

    def _check_param(self, param_name, get_array, indices, eps=1e-4):
        _, grads = self.model._forward_backward(self.tok, alpha=1.0, beta=1.0)
        arr = get_array()
        analytic = grads[param_name] if not isinstance(grads[param_name], list) else None
        for idx in indices:
            if analytic is not None:
                g_analytic = analytic[idx]
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_p = self.model.compute_loss(self.tok)
            arr[idx] = orig - eps
            loss_m = self.model.compute_loss(self.tok)
            arr[idx] = orig
            g_numeric = (loss_p - loss_m) / (2 * eps)
            if analytic is not None:
                self.assertAlmostEqual(g_analytic, g_numeric, delta=1e-3)

    def test_decoder_weight_gradient(self):
        self._check_param("Wd", lambda: self.model.Wd, [(0, 0), (3, 5)])

    def test_embedding_gradient(self):
        # alpha=0, beta=0 isolates the NTP-only path (direct + through the
        # concept-module INPUT, which carries a real gradient). The VQ/NCP
        # *targets* use stop-gradient on c -- a value-level no-op that naive
        # whole-scalar finite differencing cannot distinguish from a real
        # dependency, so those terms must be excluded here to test fairly.
        self._check_param_isolated("E", lambda: self.model.E, [(0, 0), (5, 3)], alpha=0.0, beta=0.0)

    def test_concept_module_weight_gradient(self):
        self._check_param("Wc1", lambda: self.model.Wc1, [(0, 0), (2, 4)])

    def test_codebook_gradient(self):
        # Codebook entries also feed the injection path (pred_seg -> b -> NTP),
        # which the model deliberately does NOT backprop into the codebook
        # (documented simplification in README). Isolate the VQ loss alone
        # -- the only term the analytic codebook gradient is meant to cover.
        arr = self.model.codebooks[0]
        beta = 1.0
        _, grads = self.model._forward_backward(self.tok, alpha=1.0, beta=beta)
        for idx in [(0, 0), (1, 2)]:
            g_analytic = grads["codebooks"][0][idx]
            orig = arr[idx]
            eps = 1e-4
            arr[idx] = orig + eps
            loss_p = self._vq_loss_only(beta)
            arr[idx] = orig - eps
            loss_m = self._vq_loss_only(beta)
            arr[idx] = orig
            g_numeric = (loss_p - loss_m) / (2 * eps)
            self.assertAlmostEqual(g_analytic, g_numeric, delta=1e-3)

    def _vq_loss_only(self, beta):
        M = len(self.tok) // CHUNK_LEN
        h = self.model.E[self.tok]
        c = h[: M * CHUNK_LEN].reshape(M, CHUNK_LEN, -1).mean(axis=1)
        assign, c_seg = self.model._quantize(c)
        terms = []
        for s in range(len(self.model.codebooks)):
            diff = c_seg[:, s, :] - self.model.codebooks[s][assign[:, s]]
            terms.append((diff ** 2).sum(axis=1))
        return beta * np.mean(np.concatenate(terms))

    def _check_param_isolated(self, param_name, get_array, indices, alpha, beta, eps=1e-4):
        _, grads = self.model._forward_backward(self.tok, alpha=alpha, beta=beta)
        arr = get_array()
        analytic = grads[param_name]
        for idx in indices:
            g_analytic = analytic[idx]
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_p = self.model.compute_loss(self.tok, alpha=alpha, beta=beta)
            arr[idx] = orig - eps
            loss_m = self.model.compute_loss(self.tok, alpha=alpha, beta=beta)
            arr[idx] = orig
            g_numeric = (loss_p - loss_m) / (2 * eps)
            self.assertAlmostEqual(g_analytic, g_numeric, delta=1e-3)


class TestParamCount(unittest.TestCase):
    def test_mini_ncp_has_more_params_than_baseline(self):
        baseline = TokenOnlyBaseline(seed=0)
        ncp = MiniNCP(seed=0)
        self.assertGreater(ncp.n_params(), baseline.n_params())


if __name__ == "__main__":
    unittest.main()
