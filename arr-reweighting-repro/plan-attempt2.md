# 사전 등록 — 시도 2 (2026-10-02, 실행 전 작성, 이후 수정 안 함)

시도 1(plan.md)은 전제 P1·P2가 깨져 미해결. 별도 시도로 진행한다.

## 명시: 결과를 보고 정한 대상
시도 1의 GDPO·MARBLE 결과를 **본 뒤** 주장 대상을 바꾼다. MARBLE이 실제로 누른 목표는 오디오가 아니라 정렬된 영상 보상 v1·v2였다(이득 −0.033·0.014 vs GDPO 0.284·0.318). 대상 선정이 결과에 의존했으므로 이 시도의 판정은 시도 1보다 약한 증거로 취급한다.

## 재현할 주장
Table 2 weighting chain에서 MARBLE이 떨어뜨린 VQ가 회복된다: 2.384 → +Branch-Aware 2.612 → +Residual 2.891 → +Warm-up 2.999.
토이 대응: Eq.10(residual + warm-up)이 MARBLE이 누른 영상 목표 이득을 회복시키는가.

## 바꾸지 않는 것
plan.md의 토이 설정 전부(c_k, σ, η=0.05, 300 라운드, κ=0.5, ρ=0.9, e_warm=50, prior 0.2, 시드 0..19). 코드 `arr.py` 수정 없음.

## 전제
- P1': MARBLE의 영상 평균 이득 g_v = mean(g_v1, g_v2)가 GDPO보다 낮다 — 시도 1에서 이미 관측(−0.010 vs 0.301). 재확인만.

## 판정 기준
- **부분 재현**: g_v(Residual+Warmup) ≥ g_v(MARBLE) + 0.10 AND 평균 이득(Residual+Warmup) ≥ 평균 이득(MARBLE) − 0.02
- 보조(기록만): g_v 순서 MARBLE < Residual < Residual+Warmup (Table 2 추세), 오디오 g_a가 MARBLE 대비 얼마나 유지되는지 (논문은 AQ도 같이 오름)
- 그 외 **미해결**

## 중단 조건
연산 10분, LLM 호출 0회. 결과 후 하이퍼파라미터 변경 금지.
