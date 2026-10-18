"""
AutoSizer의 "Two-Loop Optimization Framework"(Algorithm 1)만 재현.

논문 전체는 Gemini-2.5-Flash 기반 circuit understanding/algorithm orchestration
+ ngspice/SKY130 실제 회로 시뮬레이션(24개 벤치마크)으로 구성돼 있어 이 세션에서
그대로 재현 불가능. 대신 알고리즘적 novelty의 핵심 — **outer loop의 search-space
reflection 규칙**(경계 클러스터링 감지 → 해당 방향 확장, 정체 감지 → 확장/재시도)
만 결정론적 규칙 엔진으로 재현한다. LLM 호출은 전혀 쓰지 않는다.

FoM 정의는 논문 Eq.(1)-(2) 그대로:
  FoM(y) = prod_{i in D} y~_i / prod_{j in M} y~_j,  y~ = y / y_spec
  D: maximize 대상, M: minimize 대상. feasible <=> y~_i>=1 (D), y~_j<=1 (M)
"""
import random
import numpy as np

random.seed(0)
np.random.seed(0)

VAR_NAMES = ["Wdiff", "Wcasc", "Wtail", "Wload"]
GLOBAL_BOUNDS = {  # SKY130류 물리적으로 가능한 전체 범위 (논문 Figure 3 수치대)
    "Wdiff": (0.5, 3.5),
    "Wcasc": (0.5, 3.5),
    "Wtail": (0.5, 3.5),
    "Wload": (0.5, 3.5),
}
# 논문 Figure 3의 "보수적인 1차 탐색공간"을 흉내 - 일부러 진짜 최적점보다 좁게 잡는다
INITIAL_RANGES = {
    "Wdiff": (0.84, 2.10),
    "Wcasc": (0.84, 2.10),
    "Wtail": (0.84, 2.10),
    "Wload": (0.84, 2.10),
}

GAIN_SPEC = 60.0    # dB, maximize
UGBW_SPEC = 40.0    # MHz, maximize
POWER_SPEC = 3.0    # mW, minimize


def simulate_circuit(x):
    """ngspice 대신 쓰는 해석적 surrogate. 진짜 최적점(약 Wdiff=2.7)이
    초기 보수적 범위(<=2.10) 밖에 있도록 설계 - outer loop expansion이
    실제로 필요한 상황을 만든다."""
    wdiff, wcasc, wtail, wload = x["Wdiff"], x["Wcasc"], x["Wtail"], x["Wload"]
    gain = 40 + 30 * np.tanh((wdiff - 2.7) / 1.0) + 15 * np.tanh((wcasc - 2.0) / 1.0)
    ugbw = 20 + 25 * np.tanh((wdiff - 2.5) / 1.2) + 8 * np.tanh((wtail - 1.5) / 1.0)
    power = 0.6 * (wtail + wload)
    return {"gain": gain, "ugbw": ugbw, "power": power}


def fom(y):
    """Eq.(1)-(2): D={gain, ugbw}(maximize), M={power}(minimize)."""
    g = y["gain"] / GAIN_SPEC
    u = y["ugbw"] / UGBW_SPEC
    p = y["power"] / POWER_SPEC
    feasible = (g >= 1.0) and (u >= 1.0) and (p <= 1.0)
    value = (g * u) / max(p, 1e-6)
    return value, feasible


# ---------- Inner loop: 단순화한 최적화 엔진 (LHS + 국소 탐색) ----------
# 논문의 novelty는 BO/GA 자체가 아니라 orchestration이므로, 알고리즘 풀은
# LHS(탐색)와 국소 섭동(활용) 두 개로 단순화한다.

def lhs_sample(ranges, n):
    samples = []
    keys = list(ranges.keys())
    grids = {k: np.linspace(*ranges[k], n) for k in keys}
    for k in keys:
        np.random.shuffle(grids[k])
    for i in range(n):
        samples.append({k: float(grids[k][i]) for k in keys})
    return samples


def local_search_sample(ranges, center, n, scale=0.15):
    samples = []
    for _ in range(n):
        cand = {}
        for k, (lo, hi) in ranges.items():
            span = hi - lo
            v = center[k] + np.random.normal(0, scale * span)
            cand[k] = float(np.clip(v, lo, hi))
        samples.append(cand)
    return samples


def inner_loop(ranges, history, budget=60):
    """LHS로 절반, 최고점 주변 국소탐색으로 절반. history에 (x, y, fom, feasible) 누적."""
    n_lhs = budget // 2
    candidates = lhs_sample(ranges, n_lhs)

    best_so_far = max(history, key=lambda r: r["fom"])["x"] if history else None
    if best_so_far is None:
        # history 없으면 범위 중심에서 시작
        best_so_far = {k: (lo + hi) / 2 for k, (lo, hi) in ranges.items()}
    candidates += local_search_sample(ranges, best_so_far, budget - n_lhs)

    for x in candidates:
        y = simulate_circuit(x)
        f, feasible = fom(y)
        history.append({"x": x, "y": y, "fom": f, "feasible": feasible})
    return history


# ---------- Outer loop: search-space reflection 규칙 엔진 ----------
# 논문 Figure 4 텍스트에서 그대로 가져온 두 가지 트리거:
#  (1) top-K design의 다수가 경계에 몰림 -> 해당 방향으로 확장
#  (2) best FoM이 최근 N iteration 동안 2% 미만 개선 -> 정체(stagnation)

def detect_boundary_clustering(history, ranges, top_k=10, boundary_tol=0.06, cluster_frac=0.6):
    top = sorted(history, key=lambda r: r["fom"], reverse=True)[:top_k]
    issues = []
    for var, (lo, hi) in ranges.items():
        span = hi - lo
        at_lower = sum(1 for r in top if r["x"][var] <= lo + boundary_tol * span)
        at_upper = sum(1 for r in top if r["x"][var] >= hi - boundary_tol * span)
        if at_lower / len(top) >= cluster_frac:
            issues.append((var, "lower"))
        if at_upper / len(top) >= cluster_frac:
            issues.append((var, "upper"))
    return issues


def detect_stagnation(fom_trace, window=3, threshold=0.02):
    if len(fom_trace) < window + 1:
        return False
    recent = fom_trace[-(window + 1):]
    improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-6)
    return improvement < threshold


def expand_ranges(ranges, issues, global_bounds, expand_factor=0.5):
    new_ranges = dict(ranges)
    for var, side in issues:
        lo, hi = new_ranges[var]
        span = hi - lo
        glo, ghi = global_bounds[var]
        if side == "lower":
            lo = max(glo, lo - expand_factor * span)
        else:
            hi = min(ghi, hi + expand_factor * span)
        new_ranges[var] = (lo, hi)
    return new_ranges


def run_autosizer(use_srl=True, outer_budget=6, inner_budget=80):
    """Algorithm 1 재현. use_srl=False면 outer loop의 reflection(확장)을 꺼서
    Table 3의 'w/o SRL' ablation을 흉내낸다 - 초기 좁은 범위에 고정된 채로
    inner loop만 반복."""
    ranges = dict(INITIAL_RANGES)
    history = []
    fom_trace = []
    best_fom_per_cycle = []

    for cycle in range(outer_budget):
        history = inner_loop(ranges, history, budget=inner_budget)
        best = max(history, key=lambda r: r["fom"])
        fom_trace.append(best["fom"])
        best_fom_per_cycle.append(best["fom"])

        if any(r["feasible"] for r in history):
            break  # Algorithm 1 line 12-14: feasible solution found -> break

        if not use_srl:
            continue  # SRL 없이 같은 좁은 범위로 그냥 반복(=w/o SRL ablation)

        issues = detect_boundary_clustering(history, ranges)
        stagnant = detect_stagnation(fom_trace)
        if issues or stagnant:
            ranges = expand_ranges(ranges, issues, GLOBAL_BOUNDS)

    best = max(history, key=lambda r: r["fom"])
    return {
        "best_fom": best["fom"],
        "feasible": any(r["feasible"] for r in history),
        "n_evals": len(history),
        "final_ranges": ranges,
        "fom_trace": fom_trace,
    }


# ---------- 유닛테스트: reflection 규칙이 의도대로 트리거되는지 결정론적으로 검증 ----------

def unit_tests():
    ranges = {"Wdiff": (0.84, 2.10), "Wcasc": (0.84, 2.10)}

    # top-10 전부 Wdiff 하한 근처 -> lower 경계 클러스터링 감지돼야 함
    fake_history = [
        {"x": {"Wdiff": 0.85, "Wcasc": 1.5}, "fom": 10 - i}
        for i in range(10)
    ]
    issues = detect_boundary_clustering(fake_history, ranges)
    assert ("Wdiff", "lower") in issues, "경계 클러스터링(하한) 감지 실패"

    # top-10이 범위 중앙에 흩어짐 -> 경계 이슈 없어야 함
    fake_history2 = [
        {"x": {"Wdiff": 1.2 + 0.05 * i, "Wcasc": 1.5}, "fom": 10 - i}
        for i in range(10)
    ]
    issues2 = detect_boundary_clustering(fake_history2, ranges)
    assert issues2 == [], f"경계 이슈 오탐: {issues2}"

    # FoM이 3 iteration 정체 -> stagnation True
    assert detect_stagnation([0.099, 0.0991, 0.0995, 0.0990]) is True

    # FoM이 계속 오름 -> stagnation False
    assert detect_stagnation([0.05, 0.08, 0.12, 0.20]) is False

    # expand_ranges: lower 확장은 하한을 줄이고 상한은 그대로
    expanded = expand_ranges(ranges, [("Wdiff", "lower")], GLOBAL_BOUNDS)
    assert expanded["Wdiff"][0] < ranges["Wdiff"][0]
    assert expanded["Wdiff"][1] == ranges["Wdiff"][1]

    print("모든 유닛테스트 통과: reflection 규칙(경계 클러스터링/정체 감지/범위 확장) 정상 동작")


def ablation(n_trials=20):
    """논문 Table 3의 'Full vs w/o SRL' 방향을 합성 벤치마크에서 재현."""
    results = {"with_srl": [], "without_srl": []}
    for trial in range(n_trials):
        random.seed(trial)
        np.random.seed(trial)
        r_on = run_autosizer(use_srl=True)
        results["with_srl"].append(r_on)

        random.seed(trial)
        np.random.seed(trial)
        r_off = run_autosizer(use_srl=False)
        results["without_srl"].append(r_off)

    def summarize(rs):
        foms = [r["best_fom"] for r in rs]
        sr = sum(1 for r in rs if r["feasible"]) / len(rs)
        evals = [r["n_evals"] for r in rs]
        return {
            "mean_fom": float(np.mean(foms)),
            "std_fom": float(np.std(foms)),
            "success_rate": sr,
            "mean_evals": float(np.mean(evals)),
        }

    return {k: summarize(v) for k, v in results.items()}


if __name__ == "__main__":
    print("=== 유닛테스트 ===")
    unit_tests()

    print("\n=== SRL on/off ablation (합성 벤치마크, 20 trial) ===")
    summary = ablation(n_trials=20)
    for name, s in summary.items():
        print(
            f"{name:14s} | FoM {s['mean_fom']:.3f}±{s['std_fom']:.3f} | "
            f"success_rate {s['success_rate']*100:.0f}% | mean_evals {s['mean_evals']:.1f}"
        )

    with_srl, without_srl = summary["with_srl"], summary["without_srl"]
    fom_drop = (with_srl["mean_fom"] - without_srl["mean_fom"]) / with_srl["mean_fom"] * 100
    print(f"\nw/o SRL로 인한 FoM 하락: {fom_drop:.1f}% (논문 Table 3: -28.6%, 35.47->25.32)")
    print(
        f"성공률: with_srl {with_srl['success_rate']*100:.0f}% vs "
        f"without_srl {without_srl['success_rate']*100:.0f}% "
        f"(논문 Table 3: 100% -> 50%)"
    )
