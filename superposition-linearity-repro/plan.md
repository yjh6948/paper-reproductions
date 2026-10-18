# 사전 등록 — Superposition Linearity (실제 모델 재현)

작성: 2026-10-02, 실험 실행 전. 결과를 본 뒤 이 파일은 고치지 않는다.

## 논문
Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs (Tikhonov et al., arXiv:2609.29845, 2026-09-24).
후보 3개 중 선택. HF Daily Papers 100표.

## 이번엔 토이가 아님
논문이 쓴 공개 모델 Pythia(EleutherAI)를 그대로 쓴다. 바뀌는 건 표본 크기와 데이터 샘플뿐이다. 그래서 `재현` 라벨이 가능하다.

## 주장 C1 (주 판정) — Table 1, Pythia-160M, L=32, FineWeb
- 혼합 입력: 토큰 위치마다 임베딩 평균 e_t = ½(E(A_t)+E(B_t)) (Eq.1), 모델 가중치는 그대로
- P_target = ½(P_A + P_B) (Eq.2), D = KL(P_target ‖ P_mix), JS도 계산
- 비율 R_D = E[D(P_target‖P_mix)] / E[D(P_A‖P_B)] (Eq.3)
- 논문 수치: KL 0.91, R_KL 0.35, JS 0.23, R_JS 0.40
- 구현 해석(원문이 모호한 부분, 가정): 모든 분포에 temperature τ=1.5 적용(softmax(logits/1.5), 각주 1). 32개 위치 전부 평균. JS는 자연로그.
- 표본: FineWeb(HuggingFaceFW/fineweb, sample-10BT) 앞쪽 행에서 32토큰 이상인 문서 400개 → 무작위 200쌍, 시드 0. 논문의 쌍 개수는 원문에 없음.
- Wasserstein은 생략(지면상 top-256·임베딩 코사인 ground metric 세부가 부족).

### 판정
- **재현**: R_KL ∈ [0.28, 0.42] AND R_JS ∈ [0.32, 0.48]
- **부분 재현**: R_KL < 1 이지만 위 범위 밖
- **미해결**: R_KL ≥ 1

## 주장 C2 (보조 판정, 따로 라벨) — Fig.2, Pythia-70m 학습 체크포인트
- Appendix C Eq.8: 층별 평균 중심화한 hidden state로 ε = ‖ĥ_mix/‖ĥ_mix‖ − (ĥ_A+ĥ_B)/‖ĥ_A+ĥ_B‖‖², 층·위치·쌍 평균 Ē
- 체크포인트: step 128, 1000, 8000, 32000, 143000 (Fig.2 x축 10²~10⁵ 범위). 같은 50쌍, L=32
- **부분 재현**: Ē(143000) > Ē(128) AND 스텝 순위와 Ē 순위의 Spearman ≥ 0.8 (그림에서 정확한 수치를 못 읽으므로 추세만, `재현` 라벨은 후보에서 제외)
- 그 외 **미해결**
- 층 l은 transformer 블록 출력 1..L만 사용, 임베딩 층(index 0)은 제외 — 임베딩 층에선 혼합이 정의상 정확한 평균이라 ε≈0이 되어 평균을 끌어내림 (실행 전 추가한 해석, 결과 보기 전)
- "층별 평균"은 각 forward 종류(A/B/mix)별로 평가셋 전체 위치에 대해 구한다(원문 "µ^(·)_l"의 위첨자 해석)

## 전제 (본 측정 전에 확인)
- P1 정합성: A와 A를 섞으면(같은 문서) P_mix = P_A 이므로 KL(P_target‖P_mix) < 1e-5
- P2 분모 의미: E[KL(P_A‖P_B)]가 0보다 충분히 커야 비율이 의미 있음 (> 0.5 nats)
- P3 데이터: 32토큰 이상 문서 400개 확보
- 비정상성 점검: 이 논문은 학습 중 변화를 *관측*하는 주장(C2)이고 기법이 아님 → 체크포인트로 직접 관측하므로 해당 없음

## 중단 조건
- 연산 20분, LLM API 0회. 다운로드: pythia-160m 1개 + pythia-70m 체크포인트 5개
- 결과를 본 뒤 τ·L·쌍 수·체크포인트를 바꾸지 않는다
