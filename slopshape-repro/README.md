# SlopShape — 휴머나이저 전/후 구조 비교로 재설계한 미니 재현

Jochen Madler (Sitefire), *SlopShape: Identifying AI-Generated Commercial Web Content* (arXiv:2609.15369, 2026-09-17). HN 이번 주 화제 스캔(지난 7일 arXiv 링크 포인트순) 3위로 선정 — 1위(DeepSeek Elastic Compute, 232pt)는 확인 결과 알고리즘/수식 없는 순수 인프라 논문이라 스킵, 2위는 arXiv 자체 블로그 공지(논문 아님)라 스킵.

## 논문 요약

AI가 쓴 커머셜 블로그 글을 **단어가 아니라 구조**로 탐지한다. StoryScope(AI 소설 탐지 연구)의 방법론을 커머셜 블로그 도메인으로 복제 — 사람 글 2,250개(pre-ChatGPT 아카이브) vs 5개 AI 모델이 쓴 미러 11,250개를 11차원 스키마(목적/청중/구조와흐름/설명/증거/화자/실행가능성/커머셜통합/시의성/페이지형식/문체)로 LLM이 214개 특성마다 점수 매김. 구조 특성만으로 human-vs-AI 탐지 **macro-F1 98.0**, 리워딩(AI가 자기 글을 다시 쓴 버전)에도 정확도 불변(98.1) — 즉 단어 수준 공격에 구조 신호는 견고하다는 게 핵심 주장. SHAP으로 뽑은 10개 core feature가 "정갈하고 스스로 알리는(tidy, self-announcing)" AI 글의 형태를 구성: 제목에 payoff 선(先)약속, 첫 섹션 전에 thesis 선언, 요약/재진술로 마무리, editorial-explainer 화자.

## 왜 원문 그대로는 불가능한가, 어떻게 재설계했나

원문은 13,500개 글 × 11차원 = 148,500회 LLM 채점 호출. 이 세션에서 그대로는 불가능 — 그리고 애초에 "진짜 인간 글 코퍼스"를 이 세션이 새로 수집할 방법도 마땅치 않다(Claude는 AI처럼 못 쓰는 게 아니라 AI처럼만 쓸 수 있음). 그래서 **논문의 도구(구조 특성 인스트루먼트)를 그대로 가져와서, 내 실제 작업(`ai-smell-checklist`/`humanizer`/`휴머나이저-한글` 스킬)에 직결되는 질문으로 재설계**했다: *"휴머나이저 스킬이 실제로 SlopShape가 측정하는 'AI 형태'를 줄이는가?"*

| 원문 | 이 재현 | 이유 |
|---|---|---|
| Human 2,250개 vs AI 5모델 미러 11,250개 | raw AI 글 10개 vs 같은 글을 `humanizer` 스킬(영문판, Wikipedia Signs of AI writing 기반)로 고친 humanized 버전 10개 — **paired 비교** | 진짜 인간 코퍼스 확보가 이 세션 스코프 밖. 대신 "AI 글 → AI 냄새 제거 편집"이라는, 이 세션이 실제로 통제 가능하고 내 작업과 직결되는 축으로 질문을 바꿈 |
| 214개 특성, XGBoost+SHAP으로 core 10개 도출 | 논문 Table 6에 이미 명시된 **core 10개 중 9개**(순서형이라 방향이 불명확한 Problem Placement 1개 제외)를 그대로 사용 — 별도 특성 발굴/분류기 학습 없음 | XGBoost 학습에 필요한 표본 수가 이 재현 규모(20개)로는 애초에 부족. 논문이 이미 검증한 10개 core feature를 재사용하는 게 더 정직함 |
| 특성당 LLM 1콜(dimension별) × 13,500글 | **글 1개당 1콜**에 9개 특성 전부 forced-choice로 동시 채점(Gemini 2.5 Flash 실제 API 호출, 20콜) | 논문도 "한 번에 전체 vs 차원별" 차이가 99.97% vs 99.75%로 작다고 스스로 보고(Sec 4.4) — 배치 채점이 문서화된 단순화로 정당화됨 |
| XGBoost 분류기 macro-F1 | 표본 20개로 분류기 학습 불가 → **AI-leaning 특성 개수(0~9) paired 비교**로 대체 | 통계 검정력 있는 분류기 학습에는 표본이 턱없이 부족. 대신 "몇 개 특성이 AI 쪽으로 기울어 있나"를 직접 셈 |

## 결과 (정직하게 보고)

`score_features.py`로 raw 10개 + humanized 10개를 각각 실제 Gemini 2.5 Flash 호출로 9개 core feature에 forced-choice 채점(`scores.json`), `analyze.py`로 집계.

**AI-leaning 특성 개수 (9개 중), post별:**

| # | raw | humanized | delta |
|---|-----|-----------|-------|
| 1 | 8 | 3 | -5 |
| 2 | 7 | 6 | -1 |
| 3 | 6 | 4 | -2 |
| 4 | 7 | 4 | -3 |
| 5 | 7 | 4 | -3 |
| 6 | 8 | 4 | -4 |
| 7 | 5 | 6 | +1 |
| 8 | 7 | 4 | -3 |
| 9 | 7 | 7 | 0 |
| 10 | 7 | 4 | -3 |

평균 6.90/9 → 4.60/9. **10개 중 8개에서 감소, 1개 증가, 1개 불변** — 표본이 작아 정식 유의성 검정은 안 돌렸지만(n=10, sign test로도 엄밀한 결론엔 부족), 방향은 뚜렷하다.

**특성별로 뜯어보면 (raw_ai_count/10 → humanized_ai_count/10):**

| 특성 | raw | humanized | 해석 |
|---|---|---|---|
| Payoff First-Promise Location (제목에 payoff) | 8 | **0** | 완전히 제거됨 — 제목을 "How to X" 형식에서 벗어나게 다시 쓴 효과가 그대로 반영 |
| Functional Stage Inventory (요약/종합 단락 존재) | 10 | 2 | 대부분 제거 — `humanizer` §25(Generic Positive Conclusions)·§28(Signposting)·§29(Fragmented Headers) 규칙이 정확히 이 특성을 겨냥 |
| Conclusion Behavior (결론이 thesis 재진술) | 10 | 6 | 절반 정도 제거 |
| Stakes Escalation | 7 | 4 | 절반 정도 제거 |
| Thesis Before First Unit | 9 | 7 | 약간만 제거 |
| Legacy-vs-Modern Contrast | 6 | 7 | **오히려 증가**(우연한 노이즈로 보임, n=10에서 1개 차이) |
| External Participation Pathway | 9 | 10 | 거의 변화 없음(원래도 거의 다 "absent"였음 — 양쪽 다 AI-leaning 값) |
| **Primary Voice Identity (editorial-explainer 화자)** | **10** | **10** | **전혀 안 바뀜** |
| Article Length Class | 0 | 0 | 변화 없음(둘 다 800단어 미만으로 설계했으니 당연 — 이 특성은 이 토이 코퍼스에선 애초에 구분력이 없는 통제 변수) |

## 재현 과정에서 나온 진짜 발견

`humanizer` 스킬을 그대로 적용했더니 SlopShape가 측정하는 "AI 형태" 9개 특성 중 **제목·요약단락·thesis 선언·stakes 강조 같은 "정보 배치" 관련 특성은 크게 줄었는데, "화자 정체성"(editorial-explainer vs 1인칭/대화체)만은 10개 전부 그대로**였다. `humanizer` 규칙 33개를 다시 훑어보면, em dash·rule of three·signposting·filler 등 대부분은 **문장·구조 레벨** 패턴이고, "authority가 독자에게 설명하는 어조 자체를 1인칭 경험담이나 잡담체로 바꾸라"는 규칙은 명시적으로 없다(§ PERSONALITY AND SOUL이 근접하지만 "content와 저자 목소리가 요구할 때만"이라 조건부다). 즉 **휴머나이저 스킬은 "AI가 정보를 배열하는 방식"은 잘 지우지만 "AI가 화자로서 서는 자리"는 놔둔다** — SlopShape의 틀로 보면 이게 정확히 남은 격차다.

이건 우연한 관찰이 아니라 재현이 실제로 드러낸 구조적 공백이라, `humanizer`/`휴머나이저-한글` 두 스킬 모두에 도움이 될 개선 지점으로 보인다(아래 "적용" 참고).

## 파일

- `posts_raw.json` — Claude가 직접 생성한 "raw AI" 스타일 커머셜 블로그 글 10개(정갈하고 스스로 알리는 형태로 의도적으로 작성)
- `posts_humanized.json` — 위 10개를 `humanizer` 스킬(embedded mode) 규칙으로 고친 버전
- `score_features.py` — Gemini 2.5 Flash REST 호출로 9개 core feature forced-choice 채점
- `scores.json` — 채점 원본 결과
- `analyze.py` — raw vs humanized AI-leaning 개수 집계
