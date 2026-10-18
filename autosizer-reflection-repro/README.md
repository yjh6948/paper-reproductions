# AutoSizer — Two-Loop Optimization Framework 경량 재현

원논문: *AutoSizer: Automatic Sizing of Analog and Mixed-Signal Circuits via Large Language Model (LLM) Agents* (Yu et al., Brookhaven National Laboratory, ICML 2026, arXiv:2602.02849)

## 재현 대상

AutoSizer는 circuit understanding(LLM) + adaptive search-space construction(LLM) + two-loop 최적화(Algorithm 1)로 구성된다. 이 중 **two-loop 최적화 프레임워크, 특히 outer loop의 search-space reflection 규칙**(Figure 4에 텍스트로 명시된 두 트리거 — 경계 클러스터링 감지·정체 감지)만 재현했다. Ablation(Table 3)에서 SRL(Self-Reflection Loop) 제거가 가장 큰 성능 하락을 낳는다(FOM -28.6%, 성공률 100%→50%) — 이게 프레임워크의 핵심 기여다.

## 논문 대비 단순화한 것

| 항목 | 논문 | 이 재현 |
|---|---|---|
| LLM 호출 | Gemini-2.5-Flash가 circuit understanding·search-space decision·algorithm orchestration 전부 수행 | LLM 없음 — Figure 4 텍스트의 규칙을 그대로 결정론적 코드로 옮김 |
| 회로 시뮬레이션 | Ngspice + SKY130 PDK, 24개 실제 회로 | 4변수(Wdiff/Wcasc/Wtail/Wload) analytic surrogate FoM 함수 하나(tanh 기반 gain/UGBW/power 근사) |
| Inner loop 옵티마이저 | LHS/GA/BO/SA 알고리즘 풀 중 LLM이 매 iteration 선택 | LHS + 국소 섭동(local search) 2개로 단순화. 알고리즘 선택 로직(오케스트레이션) 자체는 재현 안 함 |
| Outer loop reflection | LLM이 최적화 히스토리를 자연어로 분석 후 판단 | Figure 4에 명시된 두 규칙(top-10 중 X% 이상이 경계에 몰림 → 확장, FoM 개선 2%p 미만 N iteration → 정체)을 그대로 코드화 |
| 벤치마크 | AMS-SIZINGBENCH 24개(Easy/Medium/Hard) | 합성 FoM 1개, "진짜 feasible점이 초기 보수적 범위 밖 모서리에 있도록" 의도적으로 설계 |

**이 재현은 "AutoSizer가 24개 회로에서 몇 % 성공하는가"를 재현한 게 아니라 "outer loop의 경계-클러스터링/정체 감지 규칙이 실제로 탐색공간을 필요한 방향으로 넓히고, 그게 없으면 최적화가 초기 범위에 갇혀버린다"는 메커니즘을 검증한 것**이다.

## 검증 절차

1. **유닛테스트** (`unit_tests()`): 경계 클러스터링 감지(top-10이 전부 하한 근처 → 감지돼야 함 / 중앙에 퍼져있음 → 감지 안 돼야 함), 정체 감지(FoM 3-iteration 정체 → True / 계속 개선 → False), 범위 확장(lower 확장 시 하한만 줄고 상한은 그대로) — 결정론적 입력으로 전부 통과.
2. **SRL on/off ablation** (20 trial, 서로 다른 시드): 초기 탐색범위(0.84~2.10)에 진짜 feasible 해가 없고, GLOBAL_BOUNDS 끝단(대략 3.5)에만 존재하도록 surrogate를 설계했다 — SRL이 실제로 작동해야만 도달 가능한 상황.

## 결과 (정직하게)

```
with_srl       | FoM 2.814±0.625 | success_rate 35% | mean_evals 464.0
without_srl    | FoM 0.292±0.000 | success_rate  0% | mean_evals 480.0
```

- **SRL 없이는 20 trial 전부 feasible 해를 못 찾음(0%)** — 초기 범위에 영원히 갇힘. SRL 있으면 35%가 feasible에 도달. 방향은 논문(100%→50%, SRL 있어야 압도적으로 유리)과 일치.
- **절대 성공률은 논문(100%)보다 훨씬 낮다(35%)** — 단순화한 LHS+국소탐색 inner-loop 옵티마이저가 논문의 실제 BO/GA/SA보다 훨씬 약하고, 합성 FoM의 feasible 영역이 4개 변수가 동시에 극단값을 가져야 하는 좁은 모서리라 실제 BO보다 찾기 어렵다. 포장하지 않고 그대로 남긴다 — 이 재현의 목적은 절대 수치 매칭이 아니라 reflection 메커니즘의 존재 증명.

## 실행

```
python autosizer_repro.py
```
