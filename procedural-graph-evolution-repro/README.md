# Procedural Graphs — Self-Evolution 알고리즘 경량 재현

원논문: *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents* (Lu, Chen, Wu, Arık — Google, arXiv:2609.09153). 2026-09-14 기준 지난 7일 HN arXiv 화제글 중 재현 가능한 Algorithm 박스를 가진 논문(57pts, 1위였던 "Will there be a 7G?"는 정책 서베이라 제외).

## 재현 대상

논문은 (1) 절차 지식을 그래프로 구성해 추론 시 지역 서브그래프를 자연어 가이던스로 변환하는 부분(Eq.2-3)과 (2) 오프라인에서 그래프 구조 자체를 진화시키는 self-evolution 루프(Eq.4-6, Appendix B.6 Algorithm 1)로 구성된다. **(2)의 self-evolution 루프만** 재현했다 — LLM refiner가 편집을 제안하고 validation score로 accept/reject하는 그리디 탐색 + rejection memory가 이 논문의 진짜 알고리즘적 기여다.

## 논문 대비 단순화한 것

| 항목 | 논문 | 이 재현 |
|---|---|---|
| LLM refiner (편집 제안) | Gemini/Claude/Grok 등이 실패 궤적을 분석해 add/delete 편집을 JSON으로 생성 | 확률적 제안자로 대체 — 60% 확률로 "좋은" 편집(누락된 정답 엣지 추가/함정 엣지 삭제), 40% 확률로 "나쁜" 편집(잡음) 중 하나를 샘플링. 실제 LLM 호출 없음 |
| 솔버(태스크 실행) | ReAct 루프의 실제 LLM 에이전트, 7개 벤치마크(HotpotQA 등) | 5노드 합성 절차 그래프(Start→A→B→C→D→End) + 3개 함정 엣지. 솔버는 현재 노드의 가능한 엣지 중 균등 확률로 랜덤워크 |
| Evaluate(G, D_val) | 독립 validation set에서 실제 태스크 채점 | 같은 합성 그래프에서 200회 랜덤워크 시뮬레이션의 성공률 평균 |
| 구조 검증(PrepareCandidate) | 사이클 정책 준수 + 모든 노드의 terminal 도달성 | 논문 정의(Appendix B.6) 그대로 구현 — DFS 사이클 탐지, 역방향 BFS로 zero-out-degree(terminal) 도달성 확인 |
| Accept 게이트(Eq.5) | S_val(G_cand) ≥ S_val(G_{k-1}) | 동일 |
| Rejection memory(Step 4) | 거부된 편집+궤적을 refiner 프롬프트에 negative evidence로 제공 | 거부된 (op, edge) 튜플 집합을 유지, 제안자가 이를 피하도록 필터링(실제 LLM에 텍스트로 주입하는 건 재현 안 함 — 편집 자체를 회피하는 효과만 재현) |

## 검증 절차

1. **유닛테스트**: (a) 고립된 사이클(탈출구 없음)이 있는 그래프는 `prepare_candidate`가 무효 판정, (b) 정상 경로 그래프는 유효 판정, (c) out-degree 0 노드는 정의상 terminal이라 reachable해야 함(논문 정의 그대로), (d) accept 게이트가 있는 한 score trace가 절대 감소하지 않음(Eq.5의 단조성 보장) — 전부 통과.
2. **재현성 버그를 실제로 발견·수정**: 최초 구현은 `Graph.edges`를 Python `set`으로 관리했는데, 문자열 키를 가진 `set`의 순회 순서는 프로세스마다 해시 시드가 달라 **`random.seed()`를 고정해도 실행할 때마다 다른 결과**가 나왔다(같은 시드로 두 번 실행해서 평균 성공률이 0.26 vs 0.14로 다르게 나온 걸 보고 발견). `out_edges()`와 `candidate_edit_pool()`에서 set을 `random.choice`에 넘기기 전에 `sorted()`로 고정해서 해결 — 고친 뒤 동일 시드 반복 실행이 정확히 일치함을 확인.

## 결과 (정직하게)

라운드 예산 K(Algorithm 1의 `round budget`)에 따른 20-trial 평균 최종 성공률:

```
K= 15: mean S_K = 0.100 (완전 수렴 2/20)
K= 30: mean S_K = 0.700 (완전 수렴 14/20)
K= 50: mean S_K = 0.950 (완전 수렴 19/20)
K= 80: mean S_K = 0.950 (완전 수렴 19/20, K=50 이후 정체)
```

- **그리디 accept/reject 탐색은 라운드 예산이 충분해야 신뢰성 있게 수렴한다** — K=15에서는 20번 중 2번만 완벽한 그래프(S=1.0)에 도달, K=50 이상에서는 대부분(19/20) 도달. 단일 trial(seed=0, K=15)만 보면 S0=0.0→S_K=0.0으로 전혀 개선이 안 된 것처럼 보이지만, 이건 그 시드에서 우연히 좋은 편집이 늦게 나온 경우 — 20-trial 평균과 K-sweep으로 봐야 알고리즘의 실제 거동이 보인다.
- Rejection memory 사용 시 "이미 거부된 나쁜 편집 재제안" 평균 0.00회, 미사용 시 2.00회(20-round 실험) — 논문 Step 4의 의도(거부된 편집을 다시 제안하지 않도록 막음)대로 작동.
- 논문 자체의 절대 수치(Table 1의 벤치마크 정확도)와는 비교 대상이 아니다 — 이 재현은 "그리디 accept/reject + rejection memory 메커니즘이 설계대로 작동하는가"만 검증한다.

## 실행

```
python pg_evolution.py
```
