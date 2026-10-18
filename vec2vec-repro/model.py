import torch
import torch.nn as nn

from data import D1, D2

Z = 8
HIDDEN = 32


class MLP2(nn.Module):
    """Linear -> SiLU -> Linear. Paper uses deeper MLPs with residuals and
    LayerNorm; this reproduction uses a single hidden layer per module as a
    documented simplification (see README)."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


class Vec2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.A1 = MLP2(D1, HIDDEN, Z)
        self.A2 = MLP2(D2, HIDDEN, Z)
        self.T = MLP2(Z, HIDDEN, Z)
        self.B1 = MLP2(Z, HIDDEN, D1)
        self.B2 = MLP2(Z, HIDDEN, D2)

    def F1(self, u):  # M1-space -> M2-space
        return self.B2(self.T(self.A1(u)))

    def F2(self, v):  # M2-space -> M1-space
        return self.B1(self.T(self.A2(v)))

    def R1(self, u):  # M1-space -> M1-space (autoencode)
        return self.B1(self.T(self.A1(u)))

    def R2(self, v):  # M2-space -> M2-space (autoencode)
        return self.B2(self.T(self.A2(v)))


def vsp_loss(u, v, f1_u, f2_v):
    """Vector space preservation (paper Eq. in Sec 3.2): the pairwise Gram
    matrix of a real batch should match the Gram matrix of the OTHER pool's
    batch after being translated into this space. u, v are unpaired batches
    of the same size B."""
    gram_u = u @ u.T
    gram_v = v @ v.T
    gram_f2v = f2_v @ f2_v.T  # F2(v) lives in M1-space, compare to real u's gram
    gram_f1u = f1_u @ f1_u.T  # F1(u) lives in M2-space, compare to real v's gram
    return ((gram_u - gram_f2v) ** 2).mean() + ((gram_v - gram_f1u) ** 2).mean()


class Discriminator(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
