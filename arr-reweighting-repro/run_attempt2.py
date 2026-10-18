import json
import numpy as np
from arr import run, NAMES

res = {}
for m in ["GDPO", "MARBLE", "Residual", "Residual+Warmup"]:
    g = np.array([run(m, s)[0] for s in range(20)])
    res[m] = {"gains": dict(zip(NAMES, g.mean(0).round(3).tolist())),
              "g_v": round(float(g[:, :2].mean()), 3), "g_v_sd": round(float(g[:, :2].mean(1).std()), 3),
              "g_a": round(float(g[:, 2].mean()), 3), "mean": round(float(g.mean()), 3)}
    print(m, res[m])
M, R = res["MARBLE"], res["Residual+Warmup"]
c1 = R["g_v"] >= M["g_v"] + 0.10
c2 = R["mean"] >= M["mean"] - 0.02
print("P1'", M["g_v"] < res["GDPO"]["g_v"], "| crit video +0.10:", c1, "| crit mean -0.02:", c2,
      "| order:", M["g_v"] < res["Residual"]["g_v"] < R["g_v"])
json.dump(res, open("results-attempt2.json", "w"), indent=1)
