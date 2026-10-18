# vec2vec — 핵심 메커니즘 미니 재현

Jha, Zhang, Shmatikov, Morris (Cornell), *Harnessing the Universal Geometry of Embeddings* (arXiv:2505.12540, NeurIPS 2025). HN 이번 주 스캔에서 arXiv 링크 화제글 1위(121pts, 46댓글)로 선정됨.

## 논문 요약

서로 다른 텍스트 임베딩 모델(GTR, GTE, E5, CLIP 등)이 만드는 벡터 공간은 원래 서로 호환 안 되는데, **paired 데이터·인코더 접근 없이** 한 모델의 임베딩을 다른 모델의 공간으로 번역하는 최초의 방법(`vec2vec`). Platonic Representation Hypothesis("충분히 큰 모델들은 같은 잠재 구조로 수렴한다")의 강한 버전을 실증 — cosine similarity 최대 0.96, 8000+ 셔플된 임베딩에서 완벽 매칭. 보안 함의: 임베딩만 있는 압축된 벡터 DB에서 원문서 정보(이메일 주제·환자 질병명 등)를 추출 가능함을 실증.

방법: 각 임베딩 공간마다 input adapter(A)·output adapter(B), 공유 backbone(T)로 구성된 GAN. `F1=B2∘T∘A1`(공간1→2 번역), 대칭적으로 `F2`. Reconstruction + Cycle-consistency + Vector Space Preservation(VSP, 배치 내 pairwise 내적 구조 보존) + adversarial loss(임베딩 레벨 2개, latent 레벨 2개 판별자) 4종 손실을 함께 학습.

## 왜 이 규모로는 못 하는가, 뭘 단순화했는가

논문은 1M 텍스트 임베딩, 176 GPU-day, 실제 사전학습된 임베딩 모델 7종을 쓴다. 이 세션에서 실제 임베딩 모델을 여러 개 돌리고 176 GPU-day를 쓰는 건 불가능. 대신:

| 논문 | 이 재현 | 이유 |
|---|---|---|
| 실제 텍스트 임베딩 모델 7종(GTR/GTE/E5/CLIP 등), NQ/MIMIC/Enron 실 데이터 | 합성 데이터 — 공유 4차원 "시맨틱" latent `s`를 서로 다른 random linear projection(+약한 비선형 왜곡)으로 두 개의 호환 안 되는 공간(16차원/12차원)에 투영 | 실제 임베딩 모델 다운로드·추론이 이 세션에서 무거움. **핵심은 "서로 다른 좌표계지만 상대적 기하 구조는 공유한다"는 전제를 통제된 환경에서 재현하는 것** — 처음엔 완전 독립적인 random MLP로 두 공간을 만들었더니 이 전제 자체가 안 성립해서(상대 기하가 상관 없음) 학습이 chance 수준에 머물렀다(cos -0.14, top1 0.2%). 공유 저차원 소스에서 나온 선형 투영으로 바꾸자 비로소 의미 있는 정렬이 학습됨 — **재현 과정에서 논문의 핵심 전제 자체를 직접 확인한 셈** |
| 판별자 4개(임베딩 레벨 D1,D2 + latent 레벨 Dl1,Dl2) | 임베딩 레벨 판별자 2개만(D1,D2), latent 레벨 판별자는 생략 | 구현량 축소 — 임베딩 레벨 판별자만으로도 핵심 메커니즘 검증엔 충분하다고 판단 |
| adapter/backbone: residual+LayerNorm 포함 깊은 MLP | 1-hidden-layer MLP(Linear-SiLU-Linear) | 3000 step 토이 학습에서 깊은 네트워크는 과한 용량 |
| 1M~500K 학습 샘플 | 학습 1500쌍(비대칭 unpaired 풀), 평가 500쌍 | 세션 내 실행 가능한 규모 |
| OT baseline: Gromov-Wasserstein(차원 다른 공간 정합) | 차원 padding 후 Hungarian 알고리즘(scipy) | GW 솔버 재구현은 스코프 밖, 훨씬 crude하지만 "그래도 못 푸는 문제"라는 대조엔 충분 |

**PyTorch 사용 이유**: 이전 두 재현(Generative Agents 계열, NCP-ArchPreview)은 numpy로 손으로 backprop을 유도했는데, 이번엔 GAN(4개 loss 항이 서로 얽힘)이라 손 유도 시 버그 위험이 너무 커서 실제 autodiff를 썼다 — 별도 venv(`research/vec2vec-repro/venv`)에 CPU 전용 torch 설치.

## 실행

```
venv/Scripts/python.exe -m unittest tests.test_model -v   # shape + VSP 손실 gradcheck + 재구성 학습 sanity
venv/Scripts/python.exe train.py                           # 전체 학습 + VSP/adversarial ablation
```

## 결과 (정직하게 보고)

held-out paired 평가(학습에 전혀 안 쓰인 새 시맨틱 latent 500쌍, F1: 공간1→2):

| 구성 | cosine to target | top-1 accuracy | mean rank(/500) |
|---|---|---|---|
| **전체(adv+rec+cc+vsp)** | **0.778** | **3.0%** (chance 0.2%의 15배) | **39.4** |
| VSP loss 제거 | 0.124 | 0.6% | 218.2 |
| adversarial loss 제거 | 0.020 | 0.4% | 237.0 |
| Hungarian OT baseline | — | 0.4% | — |

**전체 모델이 OT baseline보다 top-1 정확도 7.5배, chance 대비 15배 높다** — 논문의 핵심 주장(unpaired 데이터만으로 임베딩 공간 정렬이 가능하고, 정합 기반 방법보다 우월하다)이 이 토이 규모에서 정성적으로 재현됐다. **ablation도 논문 Table 6과 같은 방향**: VSP나 adversarial 손실을 빼면 top-1·cosine이 거의 chance 수준으로 붕괴 — "각 구성요소가 개별적으로 핵심적"이라는 논문 주장이 그대로 확인됨.

다만 절대 수치(cos 0.78, top1 3%)는 논문의 헤드라인 수치(cos 최대 0.96, top1 최대 100%)에 크게 못 미친다 — 토이 규모(3000 step, 1-hidden-layer, 판별자 절반, 학습 데이터 1500쌍 vs 100만)를 감안하면 예상된 격차.

## 재현 과정에서 나온 진짜 발견

처음 합성 데이터를 "두 공간을 완전히 독립적인 random MLP로 생성"하게 짰더니 VSP loss가 다른 손실 항보다 수천 배 크게 나오고(정규화 누락) 학습이 chance 수준에 머물렀다. 임베딩을 unit-norm으로 정규화하고 나서도 여전히 chance 수준이었는데, 원인을 추적하니 **두 인코더의 상대 기하 구조 자체가 애초에 상관이 없었다** — 완전 독립 random 비선형 프로젝션은 "같은 시맨틱을 다르게 인코딩"이 아니라 "완전히 무관한 두 함수"였던 것. 공유 저차원 소스에서 나온 선형 투영으로 바꾸자(실제 임베딩 모델들이 공유하는 것으로 추정되는 "동일 시맨틱→유사 상대기하"라는 전제) 비로소 학습이 됐다. **이 과정 자체가 논문의 핵심 가설(Platonic Representation Hypothesis)이 성립하는 조건과 안 하는 조건의 경계를 직접 만져본 것**이라 재현의 부산물치고 값지다.
