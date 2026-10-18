# NCP-ArchPreview — 핵심 메커니즘 미니 재현

The Intern-NCP Team(Shanghai AI Lab / SJTU), *NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction* (arXiv:2609.10715, 2026-09).이번 사이클은 "조회수(Hugging Face Daily Papers 투표수) 1위 AI 논문으로 사이클을 돌려보자"는 요청으로 선정됨 — 265 votes, 8.9B 파라미터·5.73T 토큰 사전학습 논문.

## 왜 이 규모로 재현하지 못하는가 (경계 명시)

8.9B 파라미터, 5.73T 토큰 사전학습, 수십 개 GPU가 필요한 연구다. 이 세션에서 원 논문 그대로 재현하는 건 원천적으로 불가능하다. 대신 **논문의 진짜 새로운 기여만 추출해서 정확히 재현**했다 — 트랜스포머 백본 자체(32-head attention, RoPE 등)는 이미 알려진 표준 구성요소라 논문의 기여가 아니다. 논문이 실제로 새로 제안하는 것은:

1. Product-quantized concept vocabulary (S개 세그먼트, 세그먼트당 N개 코드워드)
2. Concept Module — 이전 concept으로부터 다음 concept을 코드북 가중결합으로 예측
3. Causal shift + injection — 예측된 concept을 토큰 디코더에 residual로 주입
4. Joint loss `L = L_NTP + α·L_NCP + β·L_VQ`

이 4가지를 **트랜스포머 백본 없이(embedding lookup만 쓰는 window=1 "토큰 인코더")** 정확히 구현했다. 백본을 걷어낸 이유는 self-attention의 순수 재현이 이 사이클의 목적이 아니고(이미 잘 알려진 메커니즘), 논문의 novel한 부분에 재현 예산을 집중하기 위해서다.

## 실행

```
python -m unittest tests.test_model -v   # gradient check(finite-difference) + shape 테스트
python train.py                           # baseline vs mini-NCP 학습 비교
```

## 실험 설계

논문의 핵심 주장("concept-level objective가 token-level만으로는 못 잡는 구조를 잡아낸다")을 시험하기 위해, **의도적으로 분리 가능한** 합성 데이터를 만들었다: 6종의 길이-4 토큰 청크를 노이즈 섞인 마르코프 체인(80% 확률로 "다음 청크")으로 이어붙인 시퀀스. 청크 *내부*의 다음 토큰은 현재 토큰만으로 완전히 결정되지만(window=1 모델이 풀 수 있음), 청크 *경계*에서의 다음 토큰은 "어느 청크가 다음에 오는가"라는 장거리 정보가 필요해서 window=1 토큰 모델은 원리적으로 풀 수 없다 — 정확히 이 지점에서 concept pathway가 도움이 되는지가 재현의 핵심 질문.

- `TokenOnlyBaseline`: `logits_t = Dense(embed(tok_t))`, concept pathway 없음
- `MiniNCP`: 위에 VQ+Concept Module+injection을 얹은 버전, 두 모델 파라미터 수는 `n_params()`로 비교 가능(MiniNCP가 더 큼 — 유닛테스트로 확인)

## 결과 — 정직하게 보고 (기대와 다르게 나온 부분 포함)

**300 epoch**: 청크 경계 NLL이 baseline 1.634 → mini-NCP 1.602로 소폭 개선(방향은 논문 가설과 일치).
**800 epoch까지 더 학습**: baseline 1.689 vs mini-NCP 1.697로 **효과가 사라지고 오히려 역전**됐다(표준편차 ~0.19~0.22 안에서 통계적으로 유의미하지 않음).

**결론: 이 토이 규모 재현에서는 concept pathway가 청크 경계 예측을 안정적으로 개선한다고 주장할 수 없다.** 300 epoch 결과만 보고했다면 논문 가설을 "확인"한 것처럼 보였겠지만, 더 학습시키자 사라졌다 — 이 불안정성 자체가 재현의 결과다. 가능한 이유:
- Concept Module 입력을 직전 concept 1개(window=1)로만 제한한 단순화가 너무 약함(논문은 전체 concept 히스토리에 대한 causal transformer를 씀)
- 코드북 크기(세그먼트 2개×코드워드 6개)가 6종 청크 정체성을 표현하기엔 충분하지만 학습 신호로 안정적으로 수렴시키기엔 너무 작을 수 있음
- 합성 데이터가 800 epoch씩이나 필요 없을 만큼 단순해서, 과학습 구간에서 두 모델 다 노이즈에 가까운 거동을 보였을 가능성

## 논문 대비 단순화한 것

| 논문 | 이 재현 | 이유 |
|---|---|---|
| 32-head self-attention Transformer 백본(16+8+16 레이어, 4096 hidden) | embedding lookup만 쓰는 window=1 "인코더"(hidden dim=8) | 백본은 논문의 novel한 기여가 아님, 재현 예산을 VQ+NCP+injection에 집중 |
| 8.9B 파라미터, 5.73T 토큰 | 수백~수천 파라미터, 수백 청크(수천 토큰) | 세션 내 실행 가능한 규모로 축소 |
| PQ 세그먼트 32개×코드워드 128개(코드북 표현력 128³²) | 세그먼트 2개×코드워드 6개 | 데이터의 실제 구조(청크 6종)에 맞춘 최소 표현력 |
| Concept Module: 전체 concept 히스토리에 대한 8-layer causal Transformer | 직전 concept 1개만 보는 2-layer MLP | 시퀀셜 BPTT 없이 전부 벡터화 가능하게 단순화 — 실제 정보 손실 있음(위 결과의 원인 후보) |
| NCP 손실이 코드북 자체도 업데이트(Eq. 9/22 — 예측 경로가 differentiable하게 코드워드까지 통과) | 코드북은 VQ 손실로만 업데이트, NCP 예측 경로에서는 코드북을 상수로 취급 | 역전파 구현을 단순화하기 위한 의도적 근사 — README와 테스트에 명시(`test_codebook_gradient`가 이 경계를 정확히 검증) |
| Muon 옵티마이저(matrix-aware) | Adam | Muon 재구현은 이번 사이클 스코프 밖 |

## 검증

`tests/test_model.py`: 결정론적 shape/param-count 테스트 + **finite-difference gradient check 4건**(임베딩·디코더·Concept Module·코드북) 전부 통과. Stop-gradient가 있는 파라미터(임베딩, 코드북)는 naive finite-diff가 stop-gradient를 구분 못 해서 처음에 2건 실패했는데, 원인이 진짜 backward 버그가 아니라 테스트 설계 문제임을 확인하고(코드북/NCP 경로를 분리한 손실로 재구성) 고쳤다 — 이 과정 자체가 stop-gradient 구현을 제대로 검증하는 방법의 기록.
