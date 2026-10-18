# 사전 등록 — CLM success-gated efficiency advantage

작성: 2026-10-02, 실험 실행 전. 결과를 본 뒤 이 파일은 고치지 않는다.

## 논문
Context Language Models (Shao et al., UW/Meta, arXiv:2609.37725, 2026-09-29). HN 지난 7일 arXiv 포인트순 3위(130pt).
1위 DSec는 13번째 사이클에서 알고리즘 없는 인프라 논문으로 이미 스킵, 2위는 2021년 입장문이라 스킵.

## 재현할 주장 (1개)
Sec 4.2 + Table 2 / Fig 21: stepwise GRPO에 success-gated efficiency advantage (Eq.6, w_eff=0.25)를 더하면
**정확도의 뚜렷한 손실 없이 추론 비용(prefix-reuse FLOPs, Eq.3)이 줄어든다.**
보조 주장 (Sec 4.2 서술, 논문에 ablation 수치 없음): 효율 보상을 성공 여부로 게이트하지 않으면 실패한 싼 궤적에 보너스가 가서 reward hacking이 생긴다.

## 축소 설계
- LLM·실제 컨텍스트 편집 없음. 정책 = 컨텍스트 관리 전략 4개에 대한 softmax (numpy, 손 유도 gradient — 손실 1개, 경로 단순).
- 전략별 비용은 Eq.3을 구현한 토큰 단위 시뮬레이터에서 계산 (prefix 캐시 일치 구간 재사용, 첫 불일치부터 재-prefill + decode).
- 전략별 성공 확률은 **가정값** (논문에서 도출 불가):
  - A 요약 압축(summary): 0.60
  - B 외과적 편집(surgical): 0.60 (월드 W1) / 0.55 (월드 W2)
  - C 공격적 삭제(aggressive): 0.25 ← 싸지만 정보 손실, reward hacking 대상
  - D 무관리 append + 한계 도달 시 truncate: 0.30
- GRPO: 그룹 G=8, A_out = (r - mean)/(std+1e-6), A_eff = Eq.6, A = A_out + w_eff·A_eff, w_eff=0.25 (논문 Appendix E 값).
- 변형 3개: (O) outcome only / (G) outcome + gated eff (Eq.6) / (U) outcome + ungated eff (모든 궤적에 같은 clip 공식, 성공 무관)
- 20 시드 × 300 스텝, lr=0.1, 균등 초기화. 지표 = 최종 정책의 기대 정확도·기대 비용 (시드 평균).

## 판정 기준
- **부분 재현** (토이라 최대치): W1에서 G의 기대 비용 ≤ 0.8 × O의 기대 비용 AND |acc_G − acc_O| ≤ 0.03
- W2 (효율 전략이 5%p 덜 정확): G 정확도가 O 대비 0.05 이내면 "뚜렷한 손실 없음"과 일치, 넘으면 불일치로 보고
- 보조 주장: U의 정확도가 G보다 0.10 이상 낮으면 reward hacking 확인, 아니면 "토이에서는 안 드러남"으로 보고
- 위 기준 미달 시 **미해결**. 재현(원문 수치 일치)은 이 설계로 불가능하므로 라벨 후보에서 제외.

## 사전 검증 (학습 전)
- Eq.6 유닛테스트: 성공 2개 미만 → 전부 0, 실패 궤적 → 0, clip [-1,1], 평균 비용 궤적 → 0
- Eq.3 유닛테스트: append-only면 재-prefill 0, 앞쪽 편집이 뒤쪽 편집보다 비싼지
- 정책 경사 gradient check (finite-difference, 기대 목적함수 기준)

## 중단 조건
- 연산 10분, Gemini API 0회 (오늘 예산 사용 안 함)
- 하이퍼파라미터(lr, 스텝, 성공 확률)는 결과 본 뒤 바꾸지 않음. 바꾸면 README에 별도 시도로 기록.
