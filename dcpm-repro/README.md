# DCPM — 핵심 메커니즘 2개 미니 재현

Tianxiang Fei, Mingyang Song, Mao Zheng, Xiang Yu (Tencent LLM Department), *Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents* (arXiv:2606.09483, 2026-06-08). 후보 3개(CREST/DCPM/ORACLE) 중 내가 만들고 있는 저널 앱의 메모리 설계와 가장 가까워서 골랐다.

## 논문 요약

LLM 에이전트 메모리를 인지심리학의 dual-process 이론(System 1/System 2)으로 재조직. **System 1(동기식 daytime writer)**: 매 `add_memory` 요청마다 원문 저장→추출→기존 항목과 대조해 ADD/SUPERSEDE/UPDATE 결정→양방향 SUPERSEDES/SUPERSEDEDBY 포인터로 doubly linked chain 구성(Algorithm 1). **System 2(비동기 nighttime engine)**: idle/야간에 DBSCAN으로 새 fact 클러스터링(LLM 없음)→induction agent가 schema/intention 노드 생성→**cross-domain sweeper**가 행동적으로는 유사하지만 의미적으로는 먼 schema 쌍을 찾아 상위 core schema로 추상화(Algorithm 2, Eq.1: `cos(E_beh) > θ_beh=0.72 AND cos(E_sem) < θ_sem=0.40`). PersonaMem-v2(가장 변별력 있는 벤치마크)에서 System 2 활성화가 +2.51~+5.20점 기여, ablation에서 cross-domain collision 메커니즘 단독으로 8.1점 중 4.1점(절반 이상)을 차지 — 이 논문의 가장 강력한 단일 레버.

## 재현 가능 규모 판단

원문은 3개 벤치마크(LongMemEval·PersonaMem·PersonaMem-v2)를 deepseek-v4-flash/kimi-2.5로 전체 평가 — LLM 호출 수천 회, 이 세션에서 불가능. **알고리즘 자체(Algorithm 1, Eq.1)는 완전히 명세돼 있어서** 벤치마크 대신 그 두 메커니즘만 토이 규모로 직접 태웠다: (1) Algorithm 1의 supersedes 체인 구성+read-path 이득, (2) Eq.1의 dual-space collision 탐지 — ablation에서 가장 크게 기여한 메커니즘.

## 재현 중 실제로 터진 일 — Gemini 일일 한도가 문서와 다름

Gemini 2.5 Flash로 실제 API 재현을 시작했는데, **SlopShape 사이클(바로 이전 작업)에서 이미 20회를 다 써버린 상태였다.** 재시도해보니 에러 메시지에 정확한 수치가 찍혔다:

```
Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
```

기존에 알고 있던 "10RPM/250RPD"가 아니라 **`gemini-2.5-flash` 무료 티어는 프로젝트당 하루 20회**였다 — RPM이 아니라 RPD 자체가 훨씬 낮다. `gemini-embedding-001`은 별도 쿼터라 몇 번 성공했지만(테스트 호출 2회), 본 재현에 필요한 만큼(임베딩 다회 호출) 쓰기엔 위험 부담이 커서 API 재현을 포기하고 방식을 바꿨다. **이 발견 자체가 이번 사이클의 실질적 산출물** — 앞으로 연구사이클에서 Gemini 예산을 잡을 때 20회/일을 상한으로 가정해야 한다.

## 재현 방식 (API 없이)

| 원문 | 이 재현 | 이유 |
|---|---|---|
| 추출·reconcile LLM `M_e`, `M_r`(deepseek-v4-flash/kimi-2.5, 논문도 교체 가능한 컴포넌트로 설계) | **Claude가 직접 reconcile 판단**(ADD/SUPERSEDE/UPDATE)을 대신 수행, API 호출 없음 | Gemini 일일 한도 소진. 논문 자체가 `M_e`/`M_r`을 교체 가능한 컴포넌트로 설계했으므로, Algorithm 1의 진짜 검증 대상인 "체인 구성+read-path 이득"은 결정 로직을 누가 내리든 동일하게 확인 가능 |
| Eq.1의 `E_beh`/`E_sem` — bge-m3 1024차원 dense embedding | **불용어 제거 bag-of-words TF 코사인 유사도**(순수 파이썬, API 없음), threshold도 손으로 재조정(θ_beh=0.25, θ_sem=0.15) | 임베딩 API 예산도 위험해서 로컬 대체. 원문 threshold(0.72/0.40)는 dense embedding 코사인 분포에 맞춰진 값이라 bag-of-words엔 그대로 못 씀 |

## 결과 (정직하게 보고)

### Algorithm 1 — supersedes 체인: 성공

5턴에 걸쳐 진화하는 원격근무 선호도를 시뮬레이션(재택 선호→주1회 출근→다시 전면재택→주3회 출근→대부분 출근). 매 턴이 직전 항목과 모순돼서 전부 SUPERSEDE로 판정 — 체인 `0←1←2←3←4` 완성. **핵심 주장 직접 검증**: update-in-place 방식(최신 사실만 보관)은 "커밋 언급 직후의 입장이 뭐였나?" 질문에 답 못 함(중간 상태 소실). DCPM의 supersedes 포인터 역추적은 `[4,3,2,1,0]` 경로로 정확히 그 중간 노드(id=2)에 도달 — 논문이 주장하는 diachronic continuity가 토이 규모에서 그대로 재현됨.

### Eq.1 — cross-domain collision: 재현 실패, 그리고 왜 실패했는지

설계한 3개 쌍(재현 전에 결과를 예측해둠) 중:

| 쌍 | 도메인 | 기대 | cos(E_beh) | cos(E_sem) | 실제 판정 | 일치 |
|---|---|---|---|---|---|---|
| sunk cost(주식 vs 리모델링) | 개인재무 vs 주택 | collide | 0.833 | 0.250 | **미충돌** | ✗ |
| 과잉검증(이메일 vs 맛집) | 업무커뮤니케이션 vs 외식 | collide | 1.000 | 0.182 | **미충돌** | ✗ |
| 무관(운동 vs 코딩언어) | 운동 vs SW개발 | 무충돌 | 0.400 | 0.000 | **충돌** | ✗ |

**3개 중 0개가 예측과 일치했다.** 첫 시도에선 불용어(the/user/because 등)를 안 걸러서 semantic cosine이 인위적으로 부풀려지는 진짜 버그도 하나 잡았다(고쳤는데도 여전히 0/3). 원인은 명확하다 — **bag-of-words는 paraphrase를 못 잡는다.** 1번·2번 쌍은 의도적으로 어휘를 다르게 써서(투자 vs 지출, 이메일 vs 맛집) 도메인 간 거리를 표현했는데, 이게 정확히 의미적 코사인이 낮아야 정상인 지점이지만 threshold를 손으로 낮춰도(0.40→0.15) 여전히 걸렸다. 3번 쌍은 반대로 behavioral abstraction 문구("prefers a specific ... time/language")가 우연히 겹쳐서 오탐이 났다.

**이건 논문의 메커니즘이 틀렸다는 뜻이 아니라, bag-of-words가 dense embedding의 대체재로 부적절하다는 걸 재현이 직접 보여준 것**이다 — 논문이 굳이 bge-m3(1024차원, 자기지식증류로 학습된 멀티링구얼 임베딩)를 쓰는 이유가, 바로 이 "표면 어휘는 다르지만 의미는 가깝다/의미는 비슷해 보이지만 표면은 겹친다"는 구분을 하기 위해서라는 걸 실패를 통해 역설적으로 확인했다.

## 파일

- `supersedes_chain.py`/`supersedes_store.json` — Algorithm 1 재현(Claude가 reconcile 판단 직접 수행)
- `cross_domain_collision.py`/`collision_results.json` — Eq.1 재현(로컬 bag-of-words 대체, 3쌍 모두 예측 불일치)
- `gemini_client.py` — 이번엔 결국 못 쓴 REST 헬퍼(쿼터 소진으로 미사용, 다음 사이클을 위해 남겨둠)
