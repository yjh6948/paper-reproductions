"""Attempt 2: attention-aware Eq.3 + on-demand editing strategy. See plan-attempt2.md."""
import json
import numpy as np
import clm_repro as base
from clm_repro import common_prefix, SYS, OUT, OBS, NOTE, SUMMARY, LIMIT, TURNS

N, L, D = 9e9, 36, 4096
ONDEMAND_AT = 20000


def span_flops(start, n):
    """FLOPs to process n tokens at positions start..start+n-1: sum of 2N + 2*L*D*p."""
    if n <= 0:
        return 0.0
    return 2 * N * n + 2 * L * D * (n * start + n * (n - 1) / 2)


def prefix_reuse_flops_attn(calls):
    total, cache = 0.0, []
    for i, (ctx, gen) in enumerate(calls):
        ctx_len = sum(t for _, t in ctx)
        m = common_prefix(ctx, cache)
        total += span_flops(m, ctx_len - m) + span_flops(ctx_len, gen)
        cache = ctx + [(("out", i), gen)]
    return total


def simulate(strategy):
    if strategy != "ondemand":
        return base.simulate(strategy)
    ctx, calls = [("sys", SYS)], []
    for t in range(TURNS):
        calls.append((list(ctx), OUT))
        ctx.append((("out", len(calls) - 1), OUT))
        ctx.append((("obs", t), OBS))
        if sum(n for _, n in ctx) > ONDEMAND_AT:
            old = [i for i, b in enumerate(ctx) if b[0][0] == "obs" and b[0][1] != t]
            calls.append((list(ctx), NOTE * len(old)))   # model writes all notes in one edit
            for i in old:
                ctx[i] = (("note", ctx[i][0][1]), NOTE)
    return calls


ALL = ["summary", "surgical", "ondemand", "aggressive", "truncate"]
COST = {s: prefix_reuse_flops_attn(simulate(s)) / 1e15 for s in ALL}

if __name__ == "__main__":
    print("attention-aware PFLOPs:", {k: round(v, 3) for k, v in COST.items()})
    P1 = COST["ondemand"] < COST["summary"]
    P2 = COST["aggressive"] <= 0.5 * COST["summary"]
    print("P1 ondemand < summary:", P1, "| P2 aggressive <= 0.5*summary:", P2)

    arms = ["summary", "ondemand", "aggressive", "truncate"]
    base.STRATEGIES = arms
    base.BASE_COST = np.array([COST[s] for s in arms])
    worlds = {"W1": np.array([0.60, 0.60, 0.25, 0.30]), "W2": np.array([0.60, 0.55, 0.25, 0.30])}
    results = {}
    for w, p in worlds.items():
        for v in ["O", "G", "U"]:
            pis = np.array([base.train(p, v, s) for s in range(20)])
            acc, cost = pis @ p, pis @ base.BASE_COST
            results[f"{w}-{v}"] = {"acc": round(float(acc.mean()), 4), "cost": round(float(cost.mean()), 4),
                                   "pi": dict(zip(arms, pis.mean(0).round(3).tolist()))}
            print(f"{w}-{v}", results[f"{w}-{v}"])
    json.dump({"cost_pflops": COST, "P1": bool(P1), "P2": bool(P2), "results": results},
              open("results-attempt2.json", "w"), indent=1)
