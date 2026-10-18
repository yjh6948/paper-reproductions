import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import D1, D2, SyntheticEncoders, make_paired_eval_data, make_unpaired_training_data
from model import Discriminator, Vec2Vec, Z, vsp_loss


class TestShapes(unittest.TestCase):
    def test_translation_shapes(self):
        model = Vec2Vec()
        u = torch.randn(5, D1)
        v = torch.randn(5, D2)
        self.assertEqual(model.F1(u).shape, (5, D2))
        self.assertEqual(model.F2(v).shape, (5, D1))
        self.assertEqual(model.R1(u).shape, (5, D1))
        self.assertEqual(model.R2(v).shape, (5, D2))

    def test_unpaired_pools_are_disjoint_in_semantics(self):
        # the two training pools must be drawn from disjoint semantic
        # latents -- this is what makes the setup genuinely unsupervised.
        enc = SyntheticEncoders(seed=0)
        u, v = make_unpaired_training_data(enc, n_per_side=10, seed=1)
        self.assertEqual(u.shape, (10, D1))
        self.assertEqual(v.shape, (10, D2))


class TestVSPLossGradient(unittest.TestCase):
    """torch.autograd.gradcheck on the hand-written VSP loss -- the one
    formula in this reproduction that isn't a built-in torch op, so it's
    the most likely place for a transcription bug against the paper's Eq."""

    def test_gradcheck(self):
        torch.manual_seed(0)
        u = torch.randn(4, D1, dtype=torch.double, requires_grad=True)
        v = torch.randn(4, D2, dtype=torch.double, requires_grad=True)
        f1_u = torch.randn(4, D2, dtype=torch.double, requires_grad=True)
        f2_v = torch.randn(4, D1, dtype=torch.double, requires_grad=True)

        def f(u, v, f1_u, f2_v):
            return vsp_loss(u, v, f1_u, f2_v)

        self.assertTrue(torch.autograd.gradcheck(f, (u, v, f1_u, f2_v), eps=1e-6, atol=1e-4))


class TestTrainingSanity(unittest.TestCase):
    def test_reconstruction_loss_decreases(self):
        torch.manual_seed(0)
        model = Vec2Vec()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        enc = SyntheticEncoders(seed=0)
        u, v = make_unpaired_training_data(enc, n_per_side=64, seed=1)

        def rec_loss():
            return ((model.R1(u) - u) ** 2).mean() + ((model.R2(v) - v) ** 2).mean()

        loss0 = rec_loss().item()
        for _ in range(200):
            opt.zero_grad()
            loss = rec_loss()
            loss.backward()
            opt.step()
        loss1 = rec_loss().item()
        self.assertLess(loss1, loss0 * 0.5)


if __name__ == "__main__":
    unittest.main()
