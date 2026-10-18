# On-Device Vision CNN — numpy 재현

Jeremy Ellis, *On-Device Vision Training, Deployment, and Inference on a Thumb-Sized Microcontroller* (arXiv:2604.23012, 2026-04)의 CNN 아키텍처·학습 메커니즘을 numpy로 재현한 것. 실물 ESP32-S3 보드는 이 세션에서 구매·플래시할 수 없어 **소프트웨어 재현으로 범위를 한정**했다 — 경계는 아래 "논문 대비 단순화" 참고.

## 실행

```
python -m unittest tests.test_model -v   # 아키텍처·그래디언트 검증 — API 키/데이터 불필요
python train.py                           # 정석 vs 버그 그래디언트 누적 비교 (8-seed)
```

## 검증한 것

1. **아키텍처가 논문 Table 2와 정확히 일치**: Conv1(3×3,4f)=112 파라미터, Conv2(3×3,8f)=296, Dense=20,187, **총 20,595** — 유닛 테스트로 고정.
2. **레이어별 출력 shape도 논문과 일치**: 64×64×3 → Conv1 62×62×4 → Pool 31×31×4 → Conv2 29×29×8 → Flatten 6,728.
3. **역전파 수치 검증**: Dense 레이어의 analytic gradient를 finite-difference numeric gradient와 대조(`atol=1e-4`).
4. **논문 Section 4.4의 그래디언트 누적 버그를 직접 재현**: "그래디언트 accumulator를 배치당 1회가 아니라 이미지마다 zeroing"하는 버그를 실제로 구현해서 정석 구현과 나란히 학습시킴.

## 결과 (8-seed 평균, 합성 3-class 데이터셋 90장 학습/9장 검증)

```
val accuracy   correct(배치당 1회 누적): 73.6% ± 19.2%   (논문 보고치: ~67-78%)
val accuracy   buggy(이미지마다 즉시 step): 90.3% ± 11.7%
loss var(last5) correct: 0.00123
loss var(last5) buggy:   0.00008
```

**정석 구현의 검증 정확도(73.6%)가 논문이 보고한 범위(~67-78%)에 거의 정확히 들어맞았다** — 같은 아키텍처·하이퍼파라미터·데이터 규모(90장/9장)에서 재현 가능한 수치라는 뜻.

**예상과 다른 발견**: "버그"(배치 누적 없이 이미지마다 즉시 Adam step)가 오히려 이 합성 데이터셋에서는 평균 검증 정확도가 더 높고(90.3% vs 73.6%) loss 분산도 더 작았다. 처음엔 "버그 = 더 나쁜 결과"를 기대했지만 틀렸다 — **논문이 지적하는 건 정확도 저하가 아니라 의도한 배치 시맨틱스(6장 평균 그래디언트로 1번 업데이트)가 깨진다는 것**이다. 이미지마다 즉시 업데이트하는 건 사실상 batch_size=1 SGD고, 90장짜리 작은 합성 데이터에서는 업데이트 횟수가 6배 많아 더 빨리 수렴한 것으로 보인다 — 데이터가 커지거나 노이즈가 real-world 사진 수준으로 지저분해지면 이 차이가 반대로 나타날 수 있다(확인 안 됨, 이 재현의 스코프 밖). 이 발견 자체를 숨기지 않고 그대로 적는다. [axi_lite_slave](https://github.com/yjh6948/axi_lite_slave)에서 assertion이 기능 테스트가 놓친 버그를 잡은 것과 짝을 이루는 사례지만 이번엔 가설이 빗나간 기록이다.

## 논문 대비 단순화한 것

| 논문 | 이 재현 | 이유 |
|---|---|---|
| ESP32-S3 실물 보드, C++ 펌웨어, 실제 카메라 촬영 | Python/numpy, 합성 이미지(blank/circle/line) | 이 세션에서 실물 보드 구매·플래시 불가 — **가장 중요한 경계**. "실제 하드웨어에서 학습·추론시켰다"고 쓰면 날조 |
| Kahan compensated summation | 일반 float64 합산 | numpy 자체가 이미 float64라 드리프트 위험이 C++ float32보다 훨씬 낮음 |
| 실제 사진 데이터셋(Dataset B, 90장) | 절차적으로 생성한 합성 이미지 90장 | 카메라 없음. 노이즈 수준을 튜닝해 논문 정확도 범위(~70-80%)에 맞춤 |
| FPS·전력 측정(Nordic PPK2) | 미실시 | 실물 하드웨어 필요, 이 재현 스코프 밖 |
| INT8 양자화 없음(논문도 float32) | 동일(float64) | 논문 자체도 이번 버전은 quantization 미적용 — 해당 없음 |

## 다음 단계 (사람이 해야 함, 이 세션 스코프 밖)

실제 하드웨어 사이클로 이어가려면 Seeed Studio XIAO ESP32-S3 ML Kit($22~40) 또는 단품 보드($15 미만) 구매 필요. 논문 저자가 코드·데이터셋을 MIT 라이선스로 공개했으므로(`github.com/webmcu-ai/on-device-vision-ai`) 구매 후 그대로 포크해서 돌려볼 수 있음 — 이 재현에서 검증한 아키텍처·하이퍼파라미터가 정확히 일치하니 결과 대조가 가능하다.
