# Paper2Agent — 미니 재현 (논문의 방법론을 실제로 적용)

Miao, Davis, Zhang, Pritchard, Zou (Stanford), *Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents* (arXiv:2509.06917v2, Nature 2026-09-16). 내가 직접 지정해서 선정.

## 논문 요약

연구 논문 + 그 코드베이스를 자동으로 MCP(Model Context Protocol) 서버로 변환해서, 논문을 "질문하면 답하고 실행해주는" AI 에이전트로 만드는 프레임워크. Claude Code를 orchestrator로 쓰고, 4개 서브에이전트(environment-manager, tutorial-scanner, tutorial-tool-extractor-implementor, test-verifier-improver)가 협업 — 코드베이스 확보→환경설정→튜토리얼 탐색→튜토리얼 실행/감사→도구 추출→MCP 서버 조립(+테스트 리파인 루프) 6단계. 생성된 도구는 원 코드베이스 결과와 대조 검증 후 "잠금"돼서 재현성을 보장한다는 게 핵심 주장.

**실측 수치(원문에서 직접 확인, WebFetch 소형 모델 요약본의 수치는 전부 가짜였음 — 아래 "교훈" 참고)**: AlphaGenome 논문에 적용해 22개 MCP 도구를 3시간 무인 생성. 벤치마크(튜토리얼 기반 15문항 + 신규 15문항)에서 AlphaGenome agent 100%/100% 정확도, Claude+Repo 60%/80%, Biomni 40%/60%. 실행시간 1.8~4.6배 단축. Scanpy는 7개 도구를 45분 만에 생성.

## 왜 이 논문은 "알고리즘 재현"이 아니라 "방법론 적용"인가

Algorithm 박스도 손실함수도 없는 순수 시스템/파이프라인 논문. 기존 11개 사이클(수식 유도→backprop/gradcheck 검증)과 재현 방식 자체가 다를 수밖에 없어서, 방식을 먼저 정했다 — "실제로 작게 돌려보기"로 결정.

## 재현 설계

Paper2Agent가 "논문 코드베이스 → MCP 서버"를 하는 시스템이므로, 이 재현은 **Paper2Agent의 방법론 자체를 이 세션에서 수동으로 수행**했다: 지난 사이클(`research/vec2vec-repro/`)의 검증된 재현 코드베이스를 "논문 코드베이스" 삼아 실제 MCP 서버로 wrapping하고, Paper2Agent가 주장하는 testing-refine 루프가 실제로 필요한지/뭘 잡아내는지 직접 확인했다.

| Paper2Agent 6단계 | 이 재현에서 한 일 |
|---|---|
| 1. 코드베이스 확보 | `research/vec2vec-repro/`(지난 사이클 결과물, 이미 존재) |
| 2. 환경 설정 | `venv/`에 torch(CPU)+numpy+scipy+**mcp SDK 2.2.0** 설치 |
| 3. 튜토리얼 탐색 | `vec2vec-repro/README.md`의 실행 커맨드 + 결과표를 "튜토리얼"로 사용 |
| 4. 튜토리얼 실행/감사 | `train.py` 재실행 → README 수치와 소수점 셋째 자리까지 일치 확인(결정론적, seed=0) |
| 5. 도구 추출/구현 | `train()`+`evaluate()`를 파라미터화된 단일 MCP 도구 `train_and_evaluate_vec2vec()`로 wrapping(`vec2vec-repro/vec2vec_mcp.py`) — seed/lambda_vsp/lambda_cc/use_adv 노출, AlphaGenome 논문의 `visualize_variant_effects()`처럼 "유연하고 잘 주석된 파라미터" 패턴을 따름 |
| 6. MCP 서버 조립 + 테스트 리파인 | resource(참조 결과표)·prompt(ablation 워크플로우) 추가. `tests/test_mcp_server.py`가 **실제 stdio 서브프로세스**를 띄워 진짜 MCP 프로토콜로 도구를 호출하고 참조 수치와 대조 |

## 결과: testing-refine 루프가 실제로 뭘 잡았나 (정직하게 보고)

논문의 핵심 주장 — "iterative testing으로 코드를 refine한다" — 을 실제로 태워보니 **1차 시도에서 진짜 버그 2개**가 나왔다:

1. **stdout 오염으로 MCP 프로토콜 붕괴**: `train.py`의 `train()`/`evaluate()`가 진행상황을 `print()`로 stdout에 찍는데, stdio 기반 MCP transport는 stdout을 JSON-RPC 프레이밍에 그대로 쓴다. 첫 실행에서 `pydantic_core.ValidationError: Invalid JSON`으로 즉시 깨짐 — Paper2Agent가 "튜토리얼 스크립트를 그대로 MCP 도구로 옮기면 안 되고 중간 산출물을 격리해야 한다"고 암묵적으로 요구하는 지점을 직접 만졌다. 원본 `train.py`는 건드리지 않고 MCP 서버 쪽에서 `contextlib.redirect_stdout`으로 격리해 해결.
2. **딕셔너리 키 불일치**: `evaluate()`가 반환하는 키는 `"F1(u)->v: cos"` 같은 사람이 읽기 좋은 문자열인데, MCP 도구 wrapper는 처음에 `"cos"`라고 가정하고 짜서 `KeyError`로 터졌다. 실제 소스를 다시 읽고 정확한 키로 수정.

두 버그 모두 **실제 MCP 서브프로세스를 통해 호출해봐야만** 드러났다 — 함수를 직접 파이썬에서 호출했다면 1번은 절대 안 드러났을 것(stdout이 그냥 터미널로 나갔을 테니). 이게 Paper2Agent가 "튜토리얼 실행"과 "MCP 도구화"를 별개 단계로 나누고 testing agent를 따로 둔 이유를 체감하게 해준 부분.

수정 후 최종 검증(`tests/test_mcp_server.py`, 실제 stdio 프로토콜):

| 설정 | cosine (got/want) | top1 (got/want) |
|---|---|---|
| full | 0.7785 / 0.778 | 0.030 / 0.030 |
| VSP 제거 | 0.1239 / 0.124 | 0.006 / 0.006 |
| adversarial 제거 | 0.0198 / 0.020 | 0.004 / 0.004 |

3개 설정 전부 tolerance(±0.02) 안에서 일치 — 결정론적 seed 덕에 사실상 정확히 일치.

## 논문 대비 단순화한 것

| 논문 | 이 재현 | 이유 |
|---|---|---|
| 4개 전문 서브에이전트(environment-manager 등)가 LLM 판단으로 코드 탐색·추출·테스트를 자동 수행 | 사람(Claude, 이 세션)이 직접 6단계를 수동 수행 | 서브에이전트를 실제로 오케스트레이션하려면 Claude Code CLI를 재귀적으로 스폰해야 하는데, 이 세션 안에서 별도 세션을 인스턴스화하는 건 스코프 밖 — 방법론의 각 단계가 "왜 필요한지"를 검증하는 데 집중 |
| GitHub 레포를 처음부터 클론하고 전체 코드베이스를 LLM이 훑어서 튜토리얼을 자동 식별 | 이미 존재하는 지난 사이클의 검증된 코드베이스를 사용 | 코드베이스 자동 발견은 이 재현의 관심사가 아님 — "추출된 함수를 MCP 도구로 만들고 검증하는" 핵심 메커니즘에 집중 |
| AlphaGenome처럼 대규모 실제 모델(22개 도구, 유전체 데이터) | vec2vec 토이 재현(도구 1개, 3개 설정) | 규모보다 **testing-refine 루프가 실제로 버그를 잡는가**라는 질문에 답하는 게 목적이었고, 도구 1개로도 충분히 답이 나왔다 |
| LLM-judge 없이 사람이 수동으로 정확도 채점(논문도 이렇게 함) | 수치 tolerance 비교(자동) | 이 재현은 정성적 텍스트 응답이 아니라 정량적 지표만 다뤄서 자동 비교가 가능했음 |

## 재현 과정에서 나온 진짜 발견

Paper2Agent 원문은 "testing agent가 실패를 진단하고 코드를 고친다"를 한 문장으로 요약하는데, 실제로 손으로 해보니 그 한 문장 안에 **두 가지 서로 다른 실패 모드**가 숨어 있었다 — (1) 프로토콜 레벨 문제(stdout 오염, 함수 자체는 정상)와 (2) 인터페이스 레벨 문제(반환값 계약 오해, 함수는 정상이지만 wrapper가 틀림). 논문은 이 둘을 구분하지 않고 뭉뚱그리지만, 실제 튜토리얼→도구 변환에서는 **원본 함수가 정확해도 "MCP 도구로 wrapping하는 코드" 자체가 별도의 버그 표면**이라는 게 이번 재현에서 확인됐다. AlphaGenome처럼 22개 도구를 무인 생성했다는 주장이 믿기려면, 이 두 실패 모드를 각각 잡아낼 자동화된 체크(프로토콜 무결성 테스트 + 스키마/키 대조)가 반드시 있어야 한다는 뜻이고, 논문 본문에는 이 디테일이 없다.

## 파일

- `research/vec2vec-repro/vec2vec_mcp.py` — Paper2Agent 스타일 MCP 서버(도구 1개 + resource + prompt)
- `research/vec2vec-repro/tests/test_mcp_server.py` — 실제 stdio 서브프로세스로 프로토콜 전체를 태우는 테스트
- `research/vec2vec-repro/venv/` — torch(CPU) + mcp SDK 2.2.0 추가 설치됨(기존 venv 없어서 재생성)
