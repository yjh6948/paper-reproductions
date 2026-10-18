# 사전 등록 — 시도 2 (2026-10-02, 실행 전 작성, 이후 수정 안 함)

시도 1(plan.md) 판정은 미해결이었다. 시도 2는 별도 시도로 진행한다. 시도 1 결과는 그대로 둔다.

## 바꾸는 것 (2개, 둘 다 시도 1 README의 원인 추정에서 나옴)
1. **Eq.3에 attention 항 포함.** 위치 p 토큰 1개의 FLOPs = 2N + 2·L·d·p (Kaplan식 forward 근사).
   상수는 가정값: N=9e9, L=36, d=4096. Qwen3.5-9B 실제 상수(하이브리드 linear attention 여부 포함)는 확인 못 함.
   prefill·decode 모두 적용 (decode 토큰도 앞 컨텍스트 전체에 attend).
2. **on-demand 전략 추가** (CLM 본래 성격에 가까운 것): 컨텍스트가 20,000 토큰을 넘을 때만 한 번에 편집해서
   최신 관찰을 뺀 모든 원시 관찰을 200토큰 노트로 바꾼다. 노트 작성 decode 비용 = 200 × 바꾼 관찰 수.
   RL 팔(arm)에서 surgical 자리를 on-demand가 대신한다. 나머지(summary·aggressive·truncate)는 그대로.

## 바꾸지 않는 것
성공 확률(W1: 0.60/0.60/0.25/0.30, W2: 0.60/0.55/0.25/0.30 — 두 번째 자리가 on-demand), 시드 0..19, 300 스텝, G=8, lr=0.1, w_eff=0.25, 비용 노이즈 lognormal(0, 0.15).

## 전제 확인 (RL 전에 먼저, 실패하면 해당 하위 테스트는 "검증 불가"로 보고하고 RL 결과에서 해석하지 않음)
- P1: on-demand 비용 < summary 비용 (W1 비용 기준 테스트와 W2 trade-off 테스트의 전제)
- P2: aggressive 비용 ≤ 0.5 × summary 비용 (reward hacking 테스트의 전제 — 싸야 ungated 보너스가 hack을 민다)
- 참고 기록(판정 없음): attention 포함 시 매 턴 surgical과 summary의 순서

## 판정 기준 (시도 1과 같은 문턱)
- **부분 재현**: P1 성립 AND W1에서 cost_G ≤ 0.8 × cost_O AND |acc_G − acc_O| ≤ 0.03
- W2: acc_G가 O 대비 0.05 이내면 "뚜렷한 손실 없음"과 일치
- 보조(P2 성립 시만): acc_U ≤ acc_G − 0.10이면 reward hacking 확인
- 그 외 **미해결**

## 중단 조건
연산 10분, LLM 호출 0회. 결과를 본 뒤 상수·확률·스텝을 바꾸지 않는다.
