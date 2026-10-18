"""Toy reproduction of CLM (arXiv:2609.37725) Eq.3 prefix-reuse FLOPs and Eq.6 success-gated efficiency advantage."""
import numpy as np

N_PARAMS = 9e9          # Qwen3.5-9B scale, FLOPs/token ~= 2N (attention term ignored)
SYS, OUT, OBS, NOTE, SUMMARY = 2000, 300, 3000, 200, 2000
LIMIT, TURNS = 28672, 40


def common_prefix(a, b):
    """Length in tokens of the shared prefix of two block lists [(id, ntok), ...]."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += x[1]
    return n


def prefix_reuse_flops(calls):
    """Eq.3. calls = [(input_blocks, n_generated)]; cache of a call = its input + its output block."""
    total, cache = 0.0, []
    for i, (ctx, gen) in enumerate(calls):
        ctx_len = sum(t for _, t in ctx)
        unmatched = ctx_len - common_prefix(ctx, cache)
        total += 2 * N_PARAMS * (unmatched + gen)
        cache = ctx + [(("out", i), gen)]
    return total


def simulate(strategy):
    """Run one long-horizon trajectory and return its call list."""
    ctx, calls = [("sys", SYS)], []
    for t in range(TURNS):
        calls.append((list(ctx), OUT))
        ctx.append((("out", len(calls) - 1), OUT))
        ctx.append((("obs", t), OBS))
        size = sum(n for _, n in ctx)
        if strategy == "summary" and size > LIMIT:
            calls.append((list(ctx), SUMMARY))           # model writes the summary
            ctx = [("sys", SYS), (("sum", t), SUMMARY)]
        elif strategy == "surgical" and t >= 1:
            # replace the previous turn's raw observation with a short note (edit near the tail)
            i = ctx.index((("obs", t - 1), OBS))
            calls.append((list(ctx), NOTE))              # model writes the note
            ctx[i] = (("note", t - 1), NOTE)
        elif strategy == "aggressive":
            ctx = [ctx[0]] + ctx[-2:]                    # keep only the latest step, no notes
        elif strategy == "truncate":
            while sum(n for _, n in ctx) > LIMIT:
                del ctx[1]                               # drop oldest block (front edit)
    return calls


STRATEGIES = ["summary", "surgical", "aggressive", "truncate"]
BASE_COST = np.array([prefix_reuse_flops(simulate(s)) for s in STRATEGIES]) / 1e15  # PFLOPs


def eff_advantage(costs, success, gated=True):
    """Eq.6. gated=False is the naive variant: every trajectory gets the clipped cost bonus."""
    mask = success.astype(bool) if gated else np.ones_like(success, dtype=bool)
    A = np.zeros_like(costs)
    if mask.sum() < 2:
        return A
    cbar = costs[mask].mean()
    A[mask] = np.clip((cbar - costs[mask]) / cbar, -1, 1)
    return A


def train(p_success, variant, seed, steps=300, G=8, lr=0.1, w_eff=0.25, noise=0.15):
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(STRATEGIES))
    for _ in range(steps):
        pi = np.exp(theta - theta.max()); pi /= pi.sum()
        a = rng.choice(len(pi), size=G, p=pi)
        r = (rng.random(G) < p_success[a]).astype(float)
        c = BASE_COST[a] * rng.lognormal(0, noise, G)
        A = (r - r.mean()) / (r.std() + 1e-6)
        if variant == "G":
            A = A + w_eff * eff_advantage(c, r, gated=True)
        elif variant == "U":
            A = A + w_eff * eff_advantage(c, r, gated=False)
        grad = np.zeros_like(theta)
        for ai, Ai in zip(a, A):
            grad += Ai * (np.eye(len(pi))[ai] - pi)      # d log pi(a) / d theta
        theta += lr * grad / G
    pi = np.exp(theta - theta.max()); pi /= pi.sum()
    return pi
