import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import TinyVisionCNN


class TestArchitectureMatchesPaper(unittest.TestCase):
    """Table 2 (Ellis, 2026): Conv1 108+4, Conv2 288+8, Dense 20184+3, total 20,595."""

    def setUp(self):
        self.model = TinyVisionCNN(num_classes=3, seed=0)

    def test_param_count_matches_paper_table_2(self):
        self.assertEqual(self.model.conv1.n_params(), 112)
        self.assertEqual(self.model.conv2.n_params(), 296)
        self.assertEqual(self.model.dense.n_params(), 20187)
        self.assertEqual(self.model.n_params(), 20595)

    def test_layer_output_shapes_match_paper_table_2(self):
        x = np.random.default_rng(0).uniform(0, 1, size=(2, 64, 64, 3))
        a1 = self.model.conv1.forward(x)
        self.assertEqual(a1.shape, (2, 62, 62, 4))
        p1 = self.model.pool.forward(a1)
        self.assertEqual(p1.shape, (2, 31, 31, 4))
        a2 = self.model.conv2.forward(p1)
        self.assertEqual(a2.shape, (2, 29, 29, 8))
        flat = a2.reshape(a2.shape[0], -1)
        self.assertEqual(flat.shape[1], 6728)


class TestGradientCorrectness(unittest.TestCase):
    """Numerical gradient check against analytic backward() on a tiny input."""

    def test_dense_layer_gradient(self):
        from model import Dense
        rng = np.random.default_rng(1)
        layer = Dense(5, 3, rng)
        x = rng.uniform(-1, 1, size=(2, 5))
        out = layer.forward(x)
        dout = rng.uniform(-1, 1, size=out.shape)
        dx_analytic = layer.backward(dout)

        eps = 1e-5
        dx_numeric = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                xp = x.copy(); xp[i, j] += eps
                xm = x.copy(); xm[i, j] -= eps
                out_p = xp @ layer.w + layer.b
                out_m = xm @ layer.w + layer.b
                dx_numeric[i, j] = ((out_p - out_m) / (2 * eps) * dout).sum()

        np.testing.assert_allclose(dx_analytic, dx_numeric, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
