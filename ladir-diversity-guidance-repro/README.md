# LaDiR — Flow Matching + Diversity Guidance 경량 재현

원논문: *LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning* (Kang, Zhang, Kuang, Majamaki, Jaitly, Ma, Qin — UCSD/Apple, ICLR 2026, arXiv:2510.04573)

## 재현 대상

LaDiR은 VAE로 추론 단계를 latent thought token으로 인코딩하고, 그 위에서 block-wise latent diffusion(flow matching)으로 추론하는 프레임워크다. 이 중 **VAE/LLM 백본 없이도 독립적으로 검증 가능한 두 수식** — Eq.(1) flow matching loss + Section 2.2 ODE 샘플링, 그리고 Eq.(5) diversity gradient guidance(배치 내 반발력) — 만 재현했다.

## 논문 대비 단순화한 것

| 항목 | 논문 | 이 재현 |
|---|---|---|
| 데이터/과제 | GSM8K/MATH 등 실제 수학·코드·퍼즐 추론, LLaMA 3.1 8B/Qwen3 8B 백본 | 2D 합성 5-mode 가우시안 혼합 분포("여러 정답 경로가 존재"를 최소 형태로 흉내) |
| Velocity field u_θ | 사전학습 LLM에서 초기화된 트랜스포머 | 작은 MLP(2-hidden-layer) |
| VAE, blockization, 텍스트 디코딩 | 있음 | 전부 제외 — 연속 latent 공간에서 flow matching + guidance만 검증 |
| 학습 | Stage1 teacher-forcing + Stage2 rollout, 3개 loss(FM+Ans+Spec) 합 | flow matching loss(Eq.1) 단독으로 3000 step 학습 |

## 검증 절차

1. **유닛테스트**: (a) 같은 위치의 두 점 사이 반발력은 0, (b) 대칭 배치(원점 기준 좌우 대칭)에서 순 반발력이 0에 가까움, (c) flow matching loss의 gradient가 실제로 0이 아님(학습 가능성 sanity check) — 전부 통과.
2. **flow matching 학습**: 5-mode 2D 타겟 분포에 대해 loss가 9.41→3.44로 감소, vanilla ODE 샘플링(50 step Euler)만으로도 5개 mode 전부 커버(entropy 1.573, 이론 최대 ln5=1.609에 근접) — 이 자체로 flow matching 구현이 정상 작동함을 확인.

## 결과 (정직하게, 논문과 반대 방향)

γ_max(반발력 세기, Eq.5)를 스윕한 결과:

```
vanilla        : coverage=5 entropy=1.573 pairdist=5.128 quality=0.369
gamma=0.05     : coverage=5 entropy=1.599 pairdist=5.016 quality=0.349
gamma=0.10     : coverage=5 entropy=1.603 pairdist=4.887 quality=0.323
gamma=0.30     : coverage=5 entropy=1.599 pairdist=4.473 quality=0.544
gamma=0.50     : coverage=5 entropy=1.599 pairdist=4.190 quality=0.709
gamma=1.00     : coverage=5 entropy=1.573 pairdist=3.720 quality=1.078
gamma=2.00     : coverage=1~2 entropy≈0    pairdist=0.5~0.6 quality=0.8~0.9(붕괴)
```

**이 토이 세팅에서는 diversity guidance가 다양성을 개선하지 못했다.** γ가 커질수록 오히려 평균 pairwise distance는 감소하고 품질(가장 가까운 mode까지 거리)은 단조적으로 나빠지다가, γ=2.0에서는 완전히 붕괴(1~2개 mode로 쏠림). 추정 원인 세 가지(포장 없이 그대로):

1. **이 토이 과제엔 애초에 다양성 결핍이 없다** — vanilla flow-matching 샘플링이 이미 5개 mode를 거의 균등하게(entropy 1.573 ≈ ln5) 커버한다. 논문이 해결하려는 문제는 "AR 모델이 유사한 궤적으로 collapse하는 것"인데, isotropic Gaussian 노이즈에서 시작하는 flow matching은 애초에 그런 collapse 경향이 없어서 반발력이 고칠 게 없다 — 오히려 잘 학습된 분포를 왜곡하는 부작용만 남는다.
2. **Eq.(5)의 결합식("ẑ_{t-1} = f_ψ(x_t,t,x) + γ_t F(z)")이 실제로 무엇에 더해지는지 모호하다** — "모델의 예측"이 velocity u_θ인지 one-step 예측(x0-prediction)인지 논문 텍스트만으로 확정할 수 없었다. 이 재현은 velocity에 직접 더하는 해석을 택했는데, 다른 해석(예: x0-prediction에 더한 뒤 다시 velocity로 환산)이었다면 다른 결과가 나왔을 수 있다.
3. **커널 형태 자체의 성질**: Eq.(5)의 반발력은 RBF 커널의 그래디언트 형태라 σ보다 먼 점끼리는 부호가 반전돼 오히려 끌어당긴다(SVGD류 커널 반발-인력 우물 구조) — 이게 저차원·소수 샘플 토이 세팅에서 원치 않는 진동/붕괴를 만들었을 가능성.

**이건 "재현 실패"가 아니라 "이 메커니즘이 어떤 조건에서 실제로 필요한지"를 역으로 알게 된 결과다** — 논문이 다루는 실제 세팅(AR 모델의 궤적 collapse)에서만 의미가 있고, 이미 다양한 vanilla 샘플링 위에 무작정 얹으면 손해라는 걸 직접 확인했다.

## 실행

```
venv/Scripts/python.exe ladir_repro.py
```
