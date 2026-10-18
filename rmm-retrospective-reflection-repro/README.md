# RMM Retrospective Reflection — 경량 재현

원논문: *In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents* (Tan et al., Google Cloud AI Research, arXiv:2503.08026)

## 재현 대상

논문은 Prospective Reflection(토픽 기반 메모리 조직)과 Retrospective Reflection(인용 신호 기반 reranker 온라인 refinement) 두 메커니즘을 결합한 RMM을 제안한다. 이 중 **Retrospective Reflection만** 재현했다 — Algorithm 1, Eq.(1)-(3)에 수식으로 명시된 부분이고, Ablation(Table 2)에서 reranker가 빠지면 RR 단독이 오히려 RAG보다 나빠지는(31.0% vs 58.8% acc) 걸 보면 이게 프레임워크의 핵심 기여다.

## 논문 대비 단순화한 것

| 항목 | 논문 | 이 재현 |
|---|---|---|
| 생성기 LLM | Gemini-1.5-Flash/-Pro | 없음 — citation을 합성 오라클로 대체 |
| 리트리버 | Contriever / Stella-1.5B / GTE-Qwen2-7B (사전학습 임베딩 모델) | 16차원 랜덤 가우시안 임베딩 + 고정 회전 행렬(도메인 불일치 시뮬레이션) |
| 벤치마크 | MSC, LongMemEval (실제 대화 데이터) | 매 스텝 새로 샘플링하는 합성 쿼리·메모리 20개 |
| Prospective Reflection | LLM 프롬프트 기반 토픽 추출·병합 | 재현 안 함 (알고리즘적 novelty가 적어 스코프 제외) |
| 나머지(reranker 구조, Gumbel-top-k 샘플링, REINFORCE 업데이트) | Eq.(1)-(3) 그대로 | 동일 수식, 동일 하이퍼파라미터(batch=4, τ=0.5, b=0.5, η=1e-3) |

**따라서 이 재현은 "논문의 벤치마크 수치를 재현"한 게 아니라 "reranker+REINFORCE 메커니즘이 수학적으로 의도대로 작동하는지"를 검증한 것이다.** 절대 수치(0.57 usefulness 등)는 논문 수치(69.8% recall 등)와 비교 대상이 아니다 — 합성 태스크 난이도가 다르다.

## 검증 절차

1. **Gradient check** (`gradcheck()`): Gumbel 노이즈를 제거한 결정론적 Plackett-Luce log-prob에 대해 `torch.autograd.gradcheck`로 W_q, W_m에 대한 gradient를 finite-difference와 대조 → **통과**.
2. **REINFORCE 학습**: 500 step(내부 batch 4)에서 reranker의 citation-usefulness(선택한 M=5개 중 "진짜 관련" 메모리 비율)가 chance(0.25) 근처에서 0.57까지 상승. 회전으로 왜곡된 retriever-only baseline은 계속 chance 이하(~0.17~0.19)에 머무름 — reranker가 citation 신호만으로 misalignment를 학습해 교정한다는 것을 보여준다.

## 실수했던 부분 (정직하게 남겨둠)

첫 구현에서는 선택된 M개 메모리의 advantage를 평균 낸 뒤 전체 log-prob 합에 곱했다(`-mean(advantage) * sum(logp)`). 이러면 +1/-1 보상이 서로 상쇄되어 reranker가 거의 학습되지 않았다(usefulness가 chance 근처에 고정). Eq.(3)을 항목별로 다시 읽고 `-(advantage_i * logp_i).sum()`으로 고치자 곧바로 수렴이 나타났다. REINFORCE 계열에서 "보상을 어느 단위로 집계하느냐"가 학습 여부를 가르는 지점이라는 걸 실제로 겪었다.

## 실행

```
venv/Scripts/python.exe reranker.py
```
