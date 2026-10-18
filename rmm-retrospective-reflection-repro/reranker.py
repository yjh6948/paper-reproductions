"""
RMM Retrospective Reflection의 reranker + REINFORCE 업데이트만 재현.

논문 Algorithm 1 / Eq.(1)-(3):
  q' = q + W_q q,  m_i' = m_i + W_m m_i          (Eq. 1, embedding adaptation)
  s_i = q'^T m_i', gumbel noise 추가 후 softmax   (Eq. 2)
  Δφ = η (R - b) ∇φ log P(M_M | q, M_K; φ)        (Eq. 3, REINFORCE)

토이 세팅: 실제 LLM/리트리버 대신 합성 임베딩을 쓴다.
- q_true, m_true: 쿼리/메모리의 "진짜" 시맨틱 임베딩 (정규분포)
- true_relevance_i = q_true . m_true_i, 상위 5개가 "진짜 유용한" 메모리(LLM 인용 시뮬레이션의 정답)
- retriever는 메모리 임베딩에 고정 회전 R을 씌워 내놓는다 -> 리트리버 점수가 진짜 관련성과
  체계적으로 어긋남 (논문이 말하는 "off-the-shelf retriever가 도메인에 안 맞는" 상황의 최소 재현)
- reranker(W_q, W_m)는 인용 보상(+1/-1)만으로 이 misalignment를 얼마나 복구하는지 관찰
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

D = 16          # 임베딩 차원
K = 20          # Top-K (retriever가 넘기는 후보 수)
M = 5           # Top-M (reranker가 최종 선택하는 수)
TAU = 0.5       # Gumbel 온도 (논문 A.1 하이퍼파라미터)
BASELINE = 0.5  # REINFORCE baseline b (논문 A.1)
LR = 1e-3       # policy gradient 학습률 (논문 A.1)
STEPS = 1000


class Reranker(torch.nn.Module):
    """Eq.(1): 잔차 연결이 있는 선형 임베딩 적응."""

    def __init__(self, dim):
        super().__init__()
        self.W_q = torch.nn.Linear(dim, dim, bias=False)
        self.W_m = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.zeros_(self.W_q.weight)
        torch.nn.init.zeros_(self.W_m.weight)

    def forward(self, q, m):
        q_adapted = q + self.W_q(q)
        m_adapted = m + self.W_m(m)
        return q_adapted, m_adapted


def make_episode(rotation, rng):
    """한 세션(쿼리+메모리뱅크)을 합성 생성."""
    q_true = torch.randn(D, generator=rng)
    m_true = torch.randn(K, D, generator=rng)
    true_relevance = m_true @ q_true
    true_topk = set(true_relevance.topk(M).indices.tolist())

    # retriever: 쿼리는 그대로, 메모리는 고정 회전으로 왜곡된 임베딩을 돌려준다
    q_retriever = q_true.clone()
    m_retriever = m_true @ rotation.T
    return q_retriever, m_retriever, true_topk


def gumbel_topk_logprob(scores, tau, m):
    """Gumbel-top-k 샘플링(Eq.2)으로 M개를 뽑고, 뽑힌 순서의 개별 log-prob(Plackett-Luce)을 반환.

    (버그 수정) 이전 버전은 선택된 M개의 log-prob을 하나로 합쳐 놓고 평균 advantage를
    곱했다 - 이러면 citation이 +1인 항목과 -1인 항목의 보상 신호가 서로 상쇄되어
    reranker가 사실상 아무것도 배우지 못한다. 각 항목의 log-prob을 개별로 남겨서
    항목별 citation reward를 항목별 log-prob에 곱해야 한다(REINFORCE의 기본형).
    """
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
    perturbed = (scores + gumbel) / tau
    order = torch.argsort(perturbed, descending=True)
    selected = order[:m]

    remaining_mask = torch.ones_like(scores, dtype=torch.bool)
    logps = []
    for idx in selected:
        logits = scores / tau
        logits = logits.masked_fill(~remaining_mask, float("-inf"))
        logps.append(F.log_softmax(logits, dim=0)[idx])
        remaining_mask[idx] = False
    return selected, torch.stack(logps)


def gradient_check():
    """결정론적 log-prob 부분(Gumbel 노이즈 제외)의 gradient를 finite-difference로 검증.

    W_q, W_m이 gradcheck 입력의 leaf 텐서 자체가 되도록 F.linear로 직접 계산한다
    (nn.Module에 copy_로 주입하면 계산 그래프가 끊겨 analytical grad가 0으로 나온다).
    """
    torch.manual_seed(42)
    q = torch.randn(D, dtype=torch.double, requires_grad=False)
    m = torch.randn(K, D, dtype=torch.double, requires_grad=False)

    def logprob_fn(W_q, W_m):
        q_a = q + F.linear(q, W_q)
        m_a = m + F.linear(m, W_m)
        scores = m_a @ q_a
        # 결정론적 버전: top-M을 그냥 점수 순으로 고정(gumbel 제거), Plackett-Luce logprob
        order = torch.argsort(scores, descending=True)[:M]
        remaining = torch.ones(K, dtype=torch.bool)
        logp = torch.zeros((), dtype=torch.double)
        for idx in order:
            logits = scores.masked_fill(~remaining, float("-inf"))
            logp = logp + F.log_softmax(logits, dim=0)[idx]
            remaining[idx] = False
        return logp

    W_q0 = (0.1 * torch.randn(D, D, dtype=torch.double)).requires_grad_(True)
    W_m0 = (0.1 * torch.randn(D, D, dtype=torch.double)).requires_grad_(True)
    ok = torch.autograd.gradcheck(logprob_fn, (W_q0, W_m0), eps=1e-6, atol=1e-4)
    return ok


BATCH_SIZE = 4  # 논문 A.1 하이퍼파라미터(REINFORCE batch size)


def train():
    rng = torch.Generator().manual_seed(123)
    # 고정된 misalignment 회전 (retriever의 도메인 불일치를 흉내)
    raw = torch.randn(D, D, generator=rng)
    rotation, _ = torch.linalg.qr(raw)

    reranker = Reranker(D)
    opt = torch.optim.SGD(reranker.parameters(), lr=LR)

    usefulness_log = []
    baseline_usefulness_log = []

    for step in range(STEPS):
        batch_loss = 0.0
        batch_usefulness = []
        batch_baseline = []

        for _ in range(BATCH_SIZE):
            q_r, m_r, true_topk = make_episode(rotation, rng)

            q_a, m_a = reranker(q_r, m_r)
            scores = m_a @ q_a
            selected, logps = gumbel_topk_logprob(scores, TAU, M)

            rewards = torch.tensor(
                [1.0 if idx.item() in true_topk else -1.0 for idx in selected]
            )
            advantage = rewards - BASELINE
            batch_loss = batch_loss + -(advantage * logps).sum()  # Eq.(3), 항목별

            usefulness = sum(1 for idx in selected if idx.item() in true_topk) / M
            batch_usefulness.append(usefulness)

            with torch.no_grad():
                base_scores = m_r @ q_r
                base_topM = set(base_scores.topk(M).indices.tolist())
                batch_baseline.append(len(base_topM & true_topk) / M)

        loss = batch_loss / BATCH_SIZE
        opt.zero_grad()
        loss.backward()
        opt.step()

        usefulness_log.append(sum(batch_usefulness) / BATCH_SIZE)
        baseline_usefulness_log.append(sum(batch_baseline) / BATCH_SIZE)

    return usefulness_log, baseline_usefulness_log


def bucket_mean(xs, bucket=100):
    return [
        sum(xs[i : i + bucket]) / len(xs[i : i + bucket])
        for i in range(0, len(xs), bucket)
    ]


if __name__ == "__main__":
    print("=== Gradient check (finite-difference vs autograd) ===")
    ok = gradient_check()
    print(f"gradcheck passed: {ok}")

    print("\n=== REINFORCE training (reranker vs frozen retriever baseline) ===")
    usefulness_log, baseline_log = train()
    print("step-bucket(100) | reranker usefulness | retriever-only baseline")
    for i, (u, b) in enumerate(zip(bucket_mean(usefulness_log), bucket_mean(baseline_log))):
        print(f"{i*100:5d}-{i*100+99:5d} | {u:.3f} | {b:.3f}")

    print(f"\nfinal 100-step reranker usefulness: {bucket_mean(usefulness_log)[-1]:.3f}")
    print(f"final 100-step retriever-only baseline: {bucket_mean(baseline_log)[-1]:.3f}")
    print(f"chance level (M/K): {M/K:.3f}")
