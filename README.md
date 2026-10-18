# 논문 재현 기록 (2026.09 ~ 10)

AI·하드웨어 논문을 읽고 핵심 주장 하나를 골라 작게 직접 돌려본 기록 17건입니다. 논문 수치를 그대로 맞추는 게 목표가 아니라 "이 메커니즘이 주장대로 움직이는가"를 확인하는 게 목표였고 기대와 다르게 나온 결과도 고치지 않고 그대로 남겼습니다.

각 사이클은 같은 순서로 진행했습니다.

1. 논문 선정. Algorithm 박스나 수식이 있어 재현할 대상이 분명한지 먼저 확인
2. 재현 규모 판단. 원문 규모를 그대로 못 돌리면 무엇을 줄였는지 표로 남김
3. 사전 등록. 15번째 사이클부터는 판정 기준을 `plan.md`에 먼저 적고 실행 후에는 고치지 않음
4. 경량 구현과 실행
5. 판정. 재현 · 부분 재현 · 재현 안 됨 · 미해결 중 하나로 적고 이유를 붙임

구현과 실행에는 Claude Code를 연구 보조로 썼습니다. LLM 호출이 필요한 실험은 Gemini 2.5 Flash 무료 티어로 돌려서 호출 수가 작습니다.

## 하드웨어와 가까운 것

| 논문 | 확인한 것 | 판정 |
|---|---|---|
| [BITCOS](bitcos-ternary-repro) · Intel · ternary LLM 저장 포맷 | 인코딩/디코딩 45개 조건 전부 비트 단위로 일치. 1.485 bits/weight와 roofline 6.90 bytes가 논문 값과 같음 | 재현 (수식·산술 수준) |
| [On-Device Vision MCU](mcu-vision-repro) · 엄지 크기 MCU에서 CNN 학습 | numpy로 CNN과 학습 루프를 다시 짬. 검증 정확도 73.6%로 논문 보고 범위(67~78%) 안. 논문이 지적한 배치 버그는 이 작은 데이터에서 오히려 정확도가 더 높았음(90.3%) | 부분 재현 · 예상과 다른 발견 |
| [AutoSizer](autosizer-reflection-repro) · LLM 에이전트로 아날로그 회로 소자값 결정 | 탐색 범위를 다시 잡는 reflection 규칙이 있을 때 20번 중 35%가 해를 찾음 · 없을 때 0%. 방향은 논문과 같음 | 메커니즘 확인 (경량) |

## 모델 내부와 학습

| 논문 | 확인한 것 | 판정 |
|---|---|---|
| [Superposition Linearity](superposition-linearity-repro) · 실제 Pythia 모델 | Table 1의 비율 두 개가 논문과 0.01 차이로 일치. Fig.2는 추세만 확인 | 재현 · 부분 재현 |
| [vec2vec](vec2vec-repro) · 짝 없는 임베딩 공간 번역 | 토이 공간에서 cosine 0.778 · top-1 3.0%(우연의 15배). 손실 항을 하나씩 빼면 성능이 무너짐 | 부분 재현 |
| [NCP-ArchPreview](ncp-archpreview-repro) · 개념 단위 예측 | 300 epoch에서는 개선처럼 보였지만 800 epoch까지 학습하자 효과가 사라지고 역전 | 재현 안 됨 |
| [LaDiR](ladir-diversity-guidance-repro) · 잠재 diffusion 추론의 다양성 유도 | 반발력 세기를 키울수록 샘플 사이 거리는 줄고 품질은 나빠짐. 토이 과제에 애초에 다양성 부족이 없었던 것이 원인으로 보임 | 재현 안 됨 |
| [Adaptive Reward Routing](arr-reweighting-repro) · 다중 보상 가중치 | 1차 시도는 전제가 성립하지 않아 중단. 2차에서 약한 목표가 회복되는 것은 확인(−0.010 → 0.255) | 부분 재현 |
| [CLM](clm-efficiency-advantage-repro) · 문맥 관리 효율 보상 | 방향은 논문과 같지만 사전 기준(비용 0.8배 이하)을 못 넘음(0.856배) | 미해결 |

## 에이전트와 메모리

| 논문 | 확인한 것 | 판정 |
|---|---|---|
| [Generative Agents](generative-agents-repro) | recency · importance · relevance 검색과 reflection 트리거를 구현. 결정적 테스트 6개 통과 | 메커니즘 구현 |
| [RMM](rmm-retrospective-reflection-repro) · 인용 신호로 reranker 학습 | gradcheck 통과. 유용 기억 비율이 우연 수준 0.25에서 0.57로 상승. 처음 구현의 advantage 계산 실수도 남겨 둠 | 메커니즘 확인 |
| [Procedural Graphs](procedural-graph-evolution-repro) · 그래프 자기 진화 | 라운드 예산 50 이상에서 20번 중 19번 완전 수렴 · 15에서는 2번 | 메커니즘 확인 |
| [AdaMem](adamem-policy-repro) · 기억 쓰기 정책 | 정책 갱신 뒤 잡음 거르기 비율 1.00. API 한도로 4세션 중 3세션만 완료 | 부분 재현 |
| [DCPM](dcpm-repro) · 이중 과정 메모리 | 사실이 바뀌어 온 이력을 포인터 체인으로 되짚는 부분은 재현. 임베딩 API 없이는 교차 도메인 판별이 안 됨 | 부분 재현 |
| [CodeMidas](codemidas-pipeline-repro) · 코드에서 RL 환경 생성 | 실행 일관성 검사가 일부러 심은 버그를 실제로 잡음. 같은 프로세스 반복 실행으로는 안 드러나는 해시 순서 문제도 발견 | 메커니즘 확인 |
| [Paper2Agent](paper2agent-repro) · 논문을 MCP 도구로 변환 | 테스트 루프를 실제로 돌리자 stdout 오염 등 버그 2개가 나옴 | 방법 적용 |
| [SlopShape](slopshape-repro) · 구조로 AI 글 탐지 | 휴머나이저 편집 전후로 AI 쪽 구조 특성이 평균 6.90 → 4.60(9개 중). 화자 정체성 특성만 안 움직임 | 방법 적용 (n=10) |

## 이 기록에서 얻은 것

- 짧게 돌리면 맞아 보이는 결과가 길게 돌리면 사라지는 경우가 있었습니다(NCP). 그 뒤로는 판정 기준을 실행 전에 적어 두었습니다.
- 버그가 결과를 나쁘게 만들 거라는 예상이 틀린 경우도 있었습니다(MCU). 정확도보다 의도한 동작이 지켜지는지를 먼저 봐야 한다는 걸 확인했습니다.
- LLM API 무료 티어의 실제 일일 한도가 문서에 적힌 값보다 훨씬 낮다는 것도 직접 부딪혀 알았습니다(DCPM · AdaMem).

## 실행

각 폴더 README에 실행 방법이 있습니다. LLM을 부르는 실험은 `GEMINI_API_KEY` 환경변수가 필요합니다. `superposition-linearity-repro/fineweb500.json`은 FineWeb 데이터셋(ODC-By)에서 뽑은 500개 샘플입니다.
