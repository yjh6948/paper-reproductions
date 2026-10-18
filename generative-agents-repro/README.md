# Generative Agents — 핵심 메커니즘 미니 재현

Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023, arXiv:2304.03442)의 memory / retrieval / reflection 아키텍처를 토이 규모로 재현한 것.

## 실행

표준 라이브러리만 사용 — 별도 설치 불필요.

```
python -m unittest tests.test_retrieval -v   # 결정론적 부분 — API 키 불필요
python demo.py                                # 실제 Gemini 호출 포함 전체 파이프라인
```

`demo.py`는 `GEMINI_API_KEY` 환경변수가 필요하다. 호출 사이 3초 페이싱을 넣어뒀다(무료 티어 429 방지).

## 구현한 것

- **Retrieval function** (`retrieval.py`) — `score = normalize(recency) + normalize(importance) + normalize(relevance)`. recency는 논문 그대로 감쇠율 0.99 지수 감쇠, importance는 Gemini에 1~10점 직접 질의, relevance는 논문의 임베딩 코사인 유사도 대신 **단어 빈도 기반 코사인 유사도(bag-of-words)**로 대체.
- **Reflection 트리거** (`reflection.py`) — 미반영 관찰들의 importance 합이 임계값을 넘으면 트리거. 논문의 임계값 150은 대규모 시뮬레이션(하루 수십~수백 개 관찰)에 맞춰진 값이라 그대로 못 쓰고, 토이 데이터셋(관찰 12개) 규모에 맞춰 **40**으로 재보정.
- **Reflection 질문 생성·답변** (`llm.py`, `demo.py`) — 논문 프롬프트("가장 중요한 상위 N개 질문은?")를 그대로 써서 실제 Gemini 호출로 질문 생성 → 그 질문으로 다시 retrieval → 인출된 근거로 답변 생성까지 논문의 트리 구조(leaf=관찰, non-leaf=추상화된 생각)를 1레벨로 실행.

## 논문 대비 단순화한 것 (오버클레임 방지)

| 논문 | 이 재현 | 이유 |
|---|---|---|
| 임베딩 모델 코사인 유사도 | 단어 빈도 bag-of-words 코사인 | Claude/Gemini 무료 티어엔 임베딩 API가 마땅치 않음. 토이 데이터셋 규모에선 의미 차이가 크지 않음 |
| 25개 에이전트, 게임 시간 단위 시뮬레이션 월드 | 단일 가상 인물의 하루, 관찰 12개 고정 시나리오 | 멀티에이전트 상호작용·행성 시계는 이 재현의 목적(retrieval/reflection 메커니즘 검증)과 무관 |
| reflection 임계값 150, 최근 100개 기억 참조 | 임계값 40, 전체 12개 참조 | 데이터 규모 비례 재보정 (관찰 총량이 논문의 1/10 이하) |
| Planning/reacting 레이어(하루 일정 하향 분해, 재계획) | 미구현 | 이번 재현 범위 밖 — retrieval/reflection이 핵심 검증 대상 |
| TrueSkill μ ablation으로 believability 정량 평가 | 미실시 (실험 참가자·비교 조건 필요) | n=1 토이 재현에서 통계적 평가는 스코프 밖. 논문의 ablation 수치는 인용값으로만 참고 |

## 검증

`tests/test_retrieval.py` 6개 테스트(결정론적 recency/relevance/top-k 정렬) 통과. `demo.py` 실행 결과 importance 채점이 상식적 순위(아파트 매각 9점 > CSS 디버깅 3점 > 커피 1점)를 냈고, reflection이 "everything happened at once"라는 원 관찰의 뉘앙스를 상위 통찰로 정확히 재구성했음을 실행 로그로 확인(2026-09-13).
