import torch

# Two synthetic "embedding encoders" M1, M2 that map a shared underlying
# semantic latent s to genuinely different, incompatible vector spaces
# (different dims, different random weights, different nonlinear mixing) --
# mirroring the paper's setup where GTR/GTE/CLIP/etc. all encode the same
# semantics into incompatible geometries.

SEM_DIM = 4
D1 = 16
D2 = 12


class SyntheticEncoders:
    """M1, M2 are DIFFERENT linear-ish random projections of the same shared
    low-dim semantic latent s (SEM_DIM=4) into higher, incompatible raw
    coordinate systems (D1=16, D2=12 dims). Because both derive from the same
    low-dim source, their RELATIVE geometry (which pairs of embeddings are
    close) is strongly correlated across the two spaces even though the raw
    coordinates are not directly comparable -- this is exactly the "shared
    relational structure despite incompatible coordinates" premise the paper
    calls the (Strong) Platonic Representation Hypothesis. A light nonlinear
    warp is added per space so the two encoders are not literally the same
    map up to rotation (which would make translation trivial)."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(SEM_DIM, D1, generator=g)
        self.b1 = torch.randn(D1, generator=g) * 0.1
        self.W2 = torch.randn(SEM_DIM, D2, generator=g)
        self.b2 = torch.randn(D2, generator=g) * 0.1

    def m1(self, s, noise=0.02, generator=None):
        out = s @ self.W1 + self.b1
        out = out + 0.15 * torch.tanh(out)  # light per-space nonlinear warp
        out = out / out.norm(dim=-1, keepdim=True)
        if noise > 0:
            out = out + torch.randn(out.shape, generator=generator) * noise
        return out

    def m2(self, s, noise=0.02, generator=None):
        out = s @ self.W2 + self.b2
        out = out + 0.15 * torch.tanh(out)
        out = out / out.norm(dim=-1, keepdim=True)
        if noise > 0:
            out = out + torch.randn(out.shape, generator=generator) * noise
        return out


def make_unpaired_training_data(enc, n_per_side=1500, seed=1):
    """Two DISJOINT pools of semantic latents -> genuinely unpaired embedding
    sets, exactly matching the paper's unsupervised setting: no example is
    shared between the M1-space training set and the M2-space training set."""
    g = torch.Generator().manual_seed(seed)
    s_pool_a = torch.randn(n_per_side, SEM_DIM, generator=g)
    s_pool_b = torch.randn(n_per_side, SEM_DIM, generator=g)
    u = enc.m1(s_pool_a, generator=g)
    v = enc.m2(s_pool_b, generator=g)
    return u, v


def make_paired_eval_data(enc, n=500, seed=999):
    """Held-out PAIRED latents -- ground truth correspondence used only for
    computing evaluation metrics, never seen during training."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, SEM_DIM, generator=g)
    u = enc.m1(s, generator=g)
    v = enc.m2(s, generator=g)
    return u, v
