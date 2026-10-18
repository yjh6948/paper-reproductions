import json
import numpy as np
from clm_repro import train, STRATEGIES, BASE_COST

WORLDS = {"W1": np.array([0.60, 0.60, 0.25, 0.30]), "W2": np.array([0.60, 0.55, 0.25, 0.30])}
SEEDS = range(20)

results = {}
for w, p in WORLDS.items():
    for v in ["O", "G", "U"]:
        pis = np.array([train(p, v, s) for s in SEEDS])
        acc, cost = pis @ p, pis @ BASE_COST
        results[f"{w}-{v}"] = {
            "acc": round(acc.mean(), 4), "acc_sd": round(acc.std(), 4),
            "cost": round(cost.mean(), 4), "cost_sd": round(cost.std(), 4),
            "pi": dict(zip(STRATEGIES, pis.mean(0).round(3).tolist())),
        }
        print(f"{w}-{v}", results[f"{w}-{v}"])
json.dump({"base_cost_pflops": dict(zip(STRATEGIES, BASE_COST.round(3).tolist())), "results": results},
          open("results.json", "w"), indent=1)
