# 사전 등록 — Adaptive Reward Routing: Preference-Preserving Reweighting (Eq.10)

작성: 2026-10-02, 실험 실행 전. 결과를 본 뒤 이 파일은 고치지 않는다.

## 논문
Adaptive Reward Routing (Yang et al., HKUST MMLab/Tencent Video/HKU, arXiv:2609.37200, 2026-09-29).
HF Daily Papers 투표순 1위(64표, 2026-10-02 조회). Algorithm 1 + Eq.1–14 있음.

## 재현할 주장 (1개)
Table 2 회색 행(weighting chain, LTX-2): 순수 MARBLE에서 시작해 사전 가중치를 남기는 residual mixing(Eq.10)과 warm-up을 더하면
**MARBLE이 눌러버린 약한 목표가 회복된다.** 오디오 품질 AQ 기준 MARBLE 5.100 → +Branch-Aware 5.290 → +Residual 5.526 → +Warm-up 5.791.
비교 기준선: Table 1에서 MARBLE AQ 5.100 < Base 5.201 < GDPO 5.450 (MARBLE이 오디오를 base보다 떨어뜨림).

## 축소 설계
- 19B/22B 확산 모델·DiffusionNFT·라우팅(Sec 4.2)·branch-aware(브랜치 없음)는 전부 제외. Eq.10의 가중치 규칙만.
- 파라미터 θ∈R^8, 보상 5개 R_k(θ) = −½‖θ − c_k‖² (최적점이 달라 서로 충돌).
  c_v1=[1,.3,0,0,0,0,0,0], c_v2=[1,−.3,0,...], c_a=[−.5,0,1,0,...], c_c1=[.3,0,.5,1,0,...], c_c2=[0,0,.5,0,1,0,...]
  v1·v2는 정렬(영상 보상 둘), a는 영상과 충돌하는 약한 목표(오디오), c1·c2는 교차 보상.
- 그래디언트 추정 노이즈: g̃_k = ∇R_k + σ_k·ε, σ = 0.3 (a만 1.5). "초기 그래디언트가 노이즈가 크거나 충돌한다"(Sec 4.3)를 노이즈로 표현.
- 업데이트: θ ← θ + η Σ_k ω_k g̃_k, η=0.05, 300 라운드, θ0 = 0.
- 방법 4개:
  - GDPO: ω = ω_prior = 0.2 고정
  - MARBLE: ω = γ*, γ* = argmin_{γ∈Δ} ‖Σγ_k ĝ_k‖ (ĝ = g̃/‖g̃‖, Frank–Wolfe 100회), 매 라운드
  - +Residual: Eq.10, κ=0.5, C=1 (사전 가중치 합 1 = simplex 합 1), γ̄ ← ργ̄ + (1−ρ)γ*, ρ=0.9, e_warm=0
  - +Residual+Warm-up: 같은 것, e_warm=50
  - κ·ρ·e_warm·ω_prior는 논문에 값이 없어 **가정값**.
- 지표: 보상별 정규화 이득 g_k = (R_k(θ) − R_k(0)) / (0 − R_k(0)) (1 = 자기 최적점 도달). 20 시드(0..19) 평균.

## 전제 (본 실험 전에 GDPO·MARBLE만 먼저 돌려 확인)
- P1: MARBLE의 오디오 계수 평균 γ̄_a < 0.2 (실제로 약한 목표를 낮게 가중함)
- P2: MARBLE 최종 오디오 이득 < GDPO 최종 오디오 이득 (Table 1의 MARBLE < GDPO 관계)
- 하나라도 깨지면 Residual/Warm-up 실험은 돌리지 않고 "검증 불가"로 보고

## 판정 기준
- **부분 재현**: P1·P2 성립 AND g_a(+Residual+Warm-up) ≥ g_a(MARBLE) + 0.10 AND 평균 이득(+Residual+Warm-up) ≥ 평균 이득(MARBLE) − 0.02
- 보조(판정에 안 넣음, 기록만): g_a가 MARBLE < +Residual < +Residual+Warm-up 순서인지 (Table 2 추세)
- 그 외 **미해결**. 원문 수치 비교는 지표가 달라 불가 → `재현` 라벨은 후보에서 제외

## 중단 조건
연산 10분, LLM 호출 0회. 결과를 본 뒤 σ·κ·ρ·e_warm·η·c_k를 바꾸지 않는다.
