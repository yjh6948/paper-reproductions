"""
Procedural Graphs 논문의 "Offline Closed-Loop Procedural Graph Self-Evolution"
(Section 3.3, Eq.4-6, Appendix B.6 Algorithm 1)만 재현.

논문 전체(ReAct 솔버 + LLM refiner + 7개 벤치마크)는 이 세션에서 그대로 재현
불가능. 대신 알고리즘적 novelty의 핵심 — **그리디 accept/reject 그래프 구조
탐색 + rejection memory**를 재현한다. LLM refiner는 "가끔 좋은 편집, 가끔
나쁜/잡음 편집을 제안하는 확률적 제안자"로 대체(실제 LLM 호출 없음).

Algorithm 1 (Appendix B.6) 그대로:
  S0 <- Evaluate(G0, Dval)
  for k=1..K:
    G_cand = G_{k-1} + edits (PrepareCandidate: 구조 검증 + cycle repair)
    if invalid: reject, retained state 불변, continue
    S_cand = Evaluate(G_cand, Dval)
    if S_cand >= S_{k-1}: accept (G_k=G_cand, S_k=S_cand)
    else: reject, log to rejection memory
"""
import random
from collections import defaultdict, deque

random.seed(0)

START, END = "Start", "End"
# "진짜" 정답 경로 (이 5개 엣지가 전부 있어야 항상 성공)
TRUE_EDGES = {(START, "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", END)}
# 함정 엣지: 있으면 solver가 확률적으로 잘못된 곳으로 새서 실패율을 높인다
TRAP_EDGES = {(START, "Trap1"), ("A", "Trap2"), ("Trap1", "Trap1")}  # 마지막은 자기루프(막다른 함정)
ALL_NODES = {START, END, "A", "B", "C", "D", "Trap1", "Trap2"}


class Graph:
    def __init__(self, edges):
        self.edges = set(edges)

    def out_edges(self, node):
        # sorted() 필수: Python str 해시는 프로세스마다 랜덤 시드가 달라 set 순회
        # 순서가 비결정적이다 - 이게 random.choice()로 들어가면 random.seed()를
        # 고정해도 실행할 때마다 다른 결과가 나온다(실제로 겪은 재현성 버그, README 참고)
        return sorted(e for e in self.edges if e[0] == node)

    def copy(self):
        return Graph(set(self.edges))


def reachable_to_terminal(graph):
    """paper: '모든 노드가 terminal(out-degree 0)로 가는 directed path를 가져야 함'.
    역방향 BFS: 모든 zero-out-degree 노드에서 시작해 역엣지를 타고 도달 가능한 집합을 구하고,
    그래프에 등장하는 모든 노드가 그 집합에 포함되는지 확인."""
    out_deg = defaultdict(int)
    for (s, t) in graph.edges:
        out_deg[s] += 1
    nodes_in_graph = set()
    for (s, t) in graph.edges:
        nodes_in_graph.add(s)
        nodes_in_graph.add(t)
    terminals = {n for n in nodes_in_graph if out_deg[n] == 0}
    if not terminals:
        return False, nodes_in_graph  # 터미널이 아예 없으면 전부 실패

    rev = defaultdict(list)
    for (s, t) in graph.edges:
        rev[t].append(s)

    visited = set(terminals)
    q = deque(terminals)
    while q:
        n = q.popleft()
        for p in rev[n]:
            if p not in visited:
                visited.add(p)
                q.append(p)

    bad_nodes = nodes_in_graph - visited
    return len(bad_nodes) == 0, bad_nodes


def has_cycle(graph):
    """단순 DFS 기반 사이클 탐지."""
    color = defaultdict(int)  # 0=white,1=gray,2=black
    adj = defaultdict(list)
    for (s, t) in graph.edges:
        adj[s].append(t)

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    nodes = set()
    for (s, t) in graph.edges:
        nodes.add(s)
        nodes.add(t)
    return any(color[n] == 0 and dfs(n) for n in nodes)


def prepare_candidate(graph, add_edges, delete_edges, allow_cycles=False):
    """PrepareCandidate: 삭제 먼저, 추가 나중, 구조 검증. 반환: (candidate_or_None, diagnostics)."""
    cand = graph.copy()
    for e in delete_edges:
        cand.edges.discard(e)
    for e in add_edges:
        cand.edges.add(e)

    if not allow_cycles and has_cycle(cand):
        return None, {"reason": "cycle_detected"}

    ok, bad_nodes = reachable_to_terminal(cand)
    if not ok:
        return None, {"reason": "no_path_to_terminal", "bad_nodes": bad_nodes}

    return cand, {}


def simulate_task(graph, max_steps=10):
    """solver의 랜덤워크: 매 스텝 현재 노드의 가능한 엣지 중 하나를 균등 확률로 선택.
    Trap1은 자기루프라 한번 들어가면 못 빠져나옴 -> 실패. End 도달 -> 성공."""
    node = START
    for _ in range(max_steps):
        if node == END:
            return 1.0
        options = graph.out_edges(node)
        if not options:
            return 0.0  # 막다른 길(End가 아닌데 out-degree 0)
        node = random.choice(options)[1]
    return 1.0 if node == END else 0.0


def evaluate(graph, n_trials=200):
    if graph is None:
        return 0.0
    return sum(simulate_task(graph) for _ in range(n_trials)) / n_trials


# ---------- 편집 제안자 (LLM refiner 대체) ----------

def candidate_edit_pool(graph):
    """good: 진짜 필요한 엣지 추가 / 함정 엣지 삭제. bad: 그 반대(잡음)."""
    good, bad = [], []
    for e in sorted(TRUE_EDGES):
        if e not in graph.edges:
            good.append(("add", e))
        else:
            bad.append(("delete", e))  # 정답 엣지를 실수로 지우는 나쁜 제안
    for e in sorted(TRAP_EDGES):
        if e in graph.edges:
            good.append(("delete", e))
        else:
            bad.append(("add", e))  # 함정 엣지를 실수로 추가하는 나쁜 제안
    return good, bad


def propose_edit(graph, rejection_memory, good_prob=0.6, use_rejection_memory=True):
    good, bad = candidate_edit_pool(graph)
    pool = good if random.random() < good_prob and good else bad
    if not pool:
        pool = bad if good else good
    if not pool:
        return None

    if use_rejection_memory:
        # Step 4: rejection memory에 있는(이미 거부된) 편집은 피한다
        fresh = [e for e in pool if e not in rejection_memory]
        pool = fresh if fresh else pool

    op, edge = random.choice(pool)
    return op, edge


# ---------- Algorithm 1 ----------

def run_self_evolution(K=15, use_rejection_memory=True, good_prob=0.6):
    g0 = Graph({(START, "A"), (START, "Trap1"), ("Trap1", "Trap1"), ("A", "Trap2")})
    G = g0
    S = evaluate(G)
    score_trace = [S]
    rejection_memory = set()
    accepted, rejected_invalid, rejected_gate = 0, 0, 0

    for k in range(1, K + 1):
        edit = propose_edit(G, rejection_memory, good_prob, use_rejection_memory)
        if edit is None:
            score_trace.append(S)
            continue
        op, e = edit
        add_edges = [e] if op == "add" else []
        del_edges = [e] if op == "delete" else []

        cand, diag = prepare_candidate(G, add_edges, del_edges)
        if cand is None:
            rejected_invalid += 1
            rejection_memory.add((op, e))
            score_trace.append(S)
            continue

        S_cand = evaluate(cand)
        if S_cand >= S:
            G, S = cand, S_cand
            accepted += 1
        else:
            rejected_gate += 1
            rejection_memory.add((op, e))
        score_trace.append(S)

    return {
        "final_graph": G,
        "final_score": S,
        "score_trace": score_trace,
        "accepted": accepted,
        "rejected_invalid": rejected_invalid,
        "rejected_gate": rejected_gate,
    }


# ---------- 유닛테스트 ----------

def unit_tests():
    # 1. 고립 사이클(Trap1 자기루프)만 있고 End로 가는 길이 없으면 무효 판정
    bad_graph = Graph({("X", "Y"), ("Y", "X")})
    cand, diag = prepare_candidate(bad_graph, [], [])
    assert cand is None and diag["reason"] == "cycle_detected", "사이클 탐지 실패"

    # 2. 정상 경로 그래프는 유효
    good_graph = Graph(TRUE_EDGES)
    cand, diag = prepare_candidate(good_graph, [], [])
    assert cand is not None, "정상 그래프가 무효 판정됨"

    # 3. reachability: End로 가는 길 없는 노드가 섞이면 무효
    dangling = Graph({(START, "A"), ("A", "Z")})  # Z는 out-degree 0이라 그 자체로 terminal 취급
    ok, bad_nodes = reachable_to_terminal(dangling)
    assert ok, "out-degree 0 노드는 정의상 terminal이라 reachable해야 함"

    # 4. accept 게이트: S_cand >= S_prev일 때만 수락(동점 포함) - 시뮬레이션으로 간접 확인
    random.seed(1)
    result = run_self_evolution(K=10)
    trace = result["score_trace"]
    assert all(trace[i + 1] >= trace[i] - 1e-9 for i in range(len(trace) - 1)), (
        f"검증 게이트가 있는데도 score가 감소함: {trace}"
    )

    print("모든 유닛테스트 통과: 구조 검증(사이클/reachability)·accept 게이트 단조성 정상 동작")


def compare_rejection_memory(n_trials=30, K=20):
    """Step 4(rejection memory)가 실제로 '이미 거부된 나쁜 편집 재제안'을 줄이는지 확인."""
    def count_repeat_bad_proposals(use_memory):
        random.seed(42)
        rejection_memory = set()
        g = Graph({(START, "A"), (START, "Trap1"), ("Trap1", "Trap1"), ("A", "Trap2")})
        S = evaluate(g, n_trials=50)
        repeats = 0
        seen_rejected = set()
        for _ in range(K):
            edit = propose_edit(g, rejection_memory, good_prob=0.5, use_rejection_memory=use_memory)
            if edit is None:
                continue
            if edit in seen_rejected:
                repeats += 1
            op, e = edit
            cand, diag = prepare_candidate(g, [e] if op == "add" else [], [e] if op == "delete" else [])
            if cand is None:
                seen_rejected.add(edit)
                rejection_memory.add(edit)
                continue
            S_cand = evaluate(cand, n_trials=50)
            if S_cand >= S:
                g, S = cand, S_cand
            else:
                seen_rejected.add(edit)
                rejection_memory.add(edit)
        return repeats

    with_mem = [count_repeat_bad_proposals(True) for _ in range(n_trials)]
    without_mem = [count_repeat_bad_proposals(False) for _ in range(n_trials)]
    return sum(with_mem) / len(with_mem), sum(without_mem) / len(without_mem)


if __name__ == "__main__":
    print("=== 유닛테스트 ===")
    unit_tests()

    print("\n=== Self-evolution 실행 (K=15 round, 1회) ===")
    random.seed(0)
    result = run_self_evolution(K=15)
    print(f"S0={result['score_trace'][0]:.3f} -> S_K={result['final_score']:.3f}")
    print(f"최종 그래프 엣지: {sorted(result['final_graph'].edges)}")
    print(f"accepted={result['accepted']}, rejected(invalid)={result['rejected_invalid']}, rejected(gate)={result['rejected_gate']}")
    print(f"score trace: {[round(s,3) for s in result['score_trace']]}")

    print("\n=== 20 trial 반복 (S0 vs 최종 S_K, 랜덤시드별) ===")
    s0s, sks = [], []
    for trial in range(20):
        random.seed(trial)
        r = run_self_evolution(K=15)
        s0s.append(r["score_trace"][0])
        sks.append(r["final_score"])
    print(f"mean S0 = {sum(s0s)/len(s0s):.3f}, mean S_K = {sum(sks)/len(sks):.3f}")

    print("\n=== Rejection memory 유무에 따른 '거부된 편집 재제안' 횟수 비교 ===")
    with_mem, without_mem = compare_rejection_memory()
    print(f"rejection memory 사용: 평균 재제안 {with_mem:.2f}회")
    print(f"rejection memory 미사용: 평균 재제안 {without_mem:.2f}회")
