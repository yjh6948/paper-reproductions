# Superposition Linearity — 실제 Pythia 모델 재현

논문: Your Transformer Can Hold Two Thoughts at Once (Tikhonov et al., arXiv:2609.29845). 사전 등록: [plan.md](plan.md).

**판정: C1 (Table 1) 재현 · C2 (Fig.2 추세) 부분 재현.** 이번 파이프라인에서 토이가 아니라 논문과 같은 공개 모델로 돌린 첫 사이클이다.

## 실행

```
python -m venv venv && venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu && venv/Scripts/pip install transformers numpy scipy
venv/Scripts/python superpose.py table1     # Pythia-160M, 200쌍, 연산 ~5분
venv/Scripts/python superpose.py fig2       # Pythia-70m 체크포인트 5개, 50쌍
```
데이터: `fineweb500.json` (HuggingFaceFW/fineweb sample-10BT 앞 500행, datasets-server API).

## 기록 항목

| 항목 | 값 |
|---|---|
| 시드 | 쌍 구성 numpy default_rng(0). 모델 추론은 결정론적 |
| 환경 | Python 3.13.14, torch 2.14.1+cpu, transformers 5.18.0, numpy 2.5.3, Intel64 Family 6 Model 140 (4 threads), Windows 11 |
| 모델 | EleutherAI/pythia-160m (main), EleutherAI/pythia-70m revision step128/1000/8000/32000/143000 |
| LLM API | 0회 |
| 지표 구현 | 분포는 softmax(logits/1.5), float64. KL(P_target‖P_mix), JS 자연로그. 32개 위치 평균 후 쌍 평균. R = 분자 평균 / KL(P_A‖P_B) 평균 (Eq.3) |
| Eq.8 구현 | 블록 출력 1..L층(임베딩 층 제외), 각 forward 종류별 층 평균으로 중심화(쌍·위치 전체), ‖·‖² |

## 전제

| 전제 | 결과 |
|---|---|
| P1 같은 문서끼리 섞으면 P_mix = P_A | 최대 KL 0.0 ✓ |
| P2 분모 KL(P_A‖P_B) > 0.5 | 3.67 ✓ |
| P3 32토큰 이상 문서 400개 | 400 ✓ |

## C1 — Table 1 (Pythia-160M, L=32, 200쌍)

| 지표 | 이 재현 | 논문 | |
|---|---|---|---|
| **R_KL** | **0.361** | 0.35 | 범위 [0.28, 0.42] ✓ |
| **R_JS** | **0.410** | 0.40 | 범위 [0.32, 0.48] ✓ |
| KL 절대값 | 1.325 | 0.91 | 다름 |
| JS 절대값 | 0.167 | 0.23 | 다름 |
| A의 top-1이 혼합 분포 top-10 안 | 53.4% | (Fig.1, 큰 모델 30–40%) | 참고 |

**판정: 재현.** 비율 두 개 모두 논문과 0.01 차이로 일치한다.

절대값은 다르다. KL은 크고 JS는 작다. 분자와 분모가 같이 움직여서 비율에서는 상쇄된다. 원인 추정: temperature를 어디에 적용했는지(로짓 vs 확률 재정규화), 데이터 샘플, 위치 평균 방식 중 하나 이상이 논문과 다를 가능성. 원문에 그 세부가 없어서 확정 못 한다. 논문이 비율을 주 지표로 쓴 이유가 이런 구현 차이에 강하기 때문이라는 걸 재현이 직접 보여준 셈이다.

## C2 — Fig.2 (Pythia-70m 체크포인트, 50쌍)

| step | 128 | 1000 | 8000 | 32000 | 143000 |
|---|---|---|---|---|---|
| Ē | 0.175 | 0.329 | 0.517 | 0.573 | 0.624 |

Spearman(step, Ē) = 1.0, Ē(143000) > Ē(128). **판정: 부분 재현** (그림에서 정확한 수치를 못 읽으므로 사전 등록대로 추세만).

층별 Ē (블록 1→6):

| step | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|
| 128 | .128 | .178 | .187 | .186 | .182 | .190 |
| 143000 | .385 | .606 | .458 | .519 | .810 | .964 |

관찰(판정 밖): 학습 끝 모델에서 마지막 블록의 오차가 가장 크다(0.964). 논문 Appendix F는 "깊은 층이 거의 선형으로 남는 U자 프로필"을 말하는데, 그건 층 간 전이의 선형성(Razzhigaev 지표)이라 이 Eq.8 오차와 같은 양이 아니다. 모순이라고 단정하지 않고 다른 지표라는 점만 적는다.

## 사전 계획 대비 변경 / 실패한 시도

- 실행 전 plan.md에 "임베딩 층 제외" 해석 한 줄 추가 (결과 보기 전).
- Fig.2 첫 실행이 백그라운드 30분 제한에 걸려 체크포인트 2개만 끝나고 중단. 원인 측정: 모델 로드 1.6s, forward 0.22s/회 → 체크포인트당 연산 ~35초. 지연은 **HF 체크포인트 다운로드**였다. 사전 등록한 "연산 20분"은 넘지 않았고(전체 연산 약 8분) 다운로드는 그 한도에 넣지 않았던 항목이라 같은 설정으로 재실행했다. 재실행 결과 step128·1000 값은 첫 실행과 같았다.

## 논문 대비 단순화

| 논문 | 이 재현 |
|---|---|
| Pythia 여러 크기 + Qwen·Llama·OLMo·Gemma | Pythia-160M(C1), Pythia-70m(C2)만 |
| FineWeb·TinyStories, 쌍 수 미기재 | FineWeb 앞 500행 중 200쌍(C1)·50쌍(C2) |
| L=32와 L=512 | L=32만 |
| KL·JS·Wasserstein | Wasserstein 생략 |
| Fig.2 체크포인트 다수·6개 모델 | 5개 체크포인트·1개 모델 |
| 파인튜닝(Eq.4–5)·Joint Contrastive 디코딩 | 미재현 (A100 수십~수백 시간) |
