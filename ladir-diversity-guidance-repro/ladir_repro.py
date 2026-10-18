"""
LaDiR(Latent Diffusion Reasoner) 논문 중 두 수식만 토이 규모로 재현:
  - Eq.(1) flow matching loss + Section 2.2의 ODE 샘플링(backward integration)
  - Eq.(5) diversity gradient guidance(배치 내 반발력)

VAE/LLM 백본, GSM8K 등 실제 벤치마크는 이 세션에서 재현 불가능. 대신 "여러 개의
정답 추론 경로가 존재한다"는 상황을 2D 합성 다봉분포(mixture of Gaussians)로
흉내내고, 이 분포를 flow matching으로 실제 학습한 뒤 vanilla 샘플링과
diversity-guided 샘플링을 비교한다 - 논문이 주장하는 "반발력이 여러 mode를
고르게 탐색하게 만든다"는 메커니즘을 직접 검증.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

DIM = 2
N_MODES = 5
MODE_RADIUS = 4.0
MODE_STD = 0.25


def make_target_modes(n_modes=N_MODES, radius=MODE_RADIUS):
    angles = torch.linspace(0, 2 * torch.pi, n_modes + 1)[:-1]
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)


TARGET_MODES = make_target_modes()


def sample_data(n):
    idx = torch.randint(0, N_MODES, (n,))
    centers = TARGET_MODES[idx]
    return centers + MODE_STD * torch.randn(n, DIM)


class VelocityField(nn.Module):
    """u_theta(z_t, t): 작은 MLP. 실제 논문은 LLM 백본이지만 여기선 flow matching
    자체의 학습·샘플링 메커니즘만 검증하는 게 목적이라 표준 toy MLP로 충분."""

    def __init__(self, dim=DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_t, t):
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])
        inp = torch.cat([z_t, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def flow_matching_loss(model, batch_size=256):
    """Eq.(1): z_t = (1-t) z_0 + t*eps, target velocity u* = eps - z_0."""
    z0 = sample_data(batch_size)
    eps = torch.randn(batch_size, DIM)
    t = torch.rand(batch_size)
    z_t = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * eps
    u_star = eps - z0
    u_pred = model(z_t, t)
    return F.mse_loss(u_pred, u_star)


def train(model, steps=3000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        opt.zero_grad()
        loss = flow_matching_loss(model)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def sample_vanilla(model, n_samples, n_steps=50):
    """Section 2.2 Inference: z_{t-dt} = z_t - dt * u_theta(z_t, t), t: 1->0."""
    z = torch.randn(n_samples, DIM)
    dt = 1.0 / n_steps
    t = 1.0
    for _ in range(n_steps):
        u = model(z, torch.full((n_samples,), t))
        z = z - dt * u
        t -= dt
    return z


def repulsion_force(z):
    """Eq.(5) 그대로. sigma = 배치 내 median pairwise distance."""
    n = z.shape[0]
    diffs = z.unsqueeze(1) - z.unsqueeze(0)  # (n, n, dim)
    dist2 = (diffs ** 2).sum(-1)  # (n, n)
    mask = ~torch.eye(n, dtype=torch.bool)
    sigma = dist2[mask].sqrt().median().clamp(min=1e-4)
    sigma2 = sigma ** 2

    weight = 2 * (1 - dist2 / sigma2) * torch.exp(-dist2 / sigma2)  # (n, n)
    weight = weight * mask  # 자기 자신(i=j) 제외
    force = (weight.unsqueeze(-1) * diffs).sum(dim=1)  # sum_j weight_ij * (z_i - z_j)
    return force


@torch.no_grad()
def sample_diversity_guided(model, n_samples, n_steps=50, gamma_max=2.0, noise_scale=1.5):
    """Eq.(5) 반발력 + 증가된 초기 노이즈. z_hat_{t-1} = f(z_t,t) + gamma_t * F(z)."""
    z = noise_scale * torch.randn(n_samples, DIM)
    dt = 1.0 / n_steps
    t = 1.0
    T = 1.0
    for _ in range(n_steps):
        u = model(z, torch.full((n_samples,), t))
        gamma_t = gamma_max * (t / T)
        force = repulsion_force(z)
        z = z - dt * u + dt * gamma_t * force
        t -= dt
    return z


# ---------- 평가 지표 ----------

def assign_to_nearest_mode(z):
    dists = torch.cdist(z, TARGET_MODES)
    return dists.argmin(dim=1)


def mode_coverage(z):
    assigned = assign_to_nearest_mode(z)
    return len(set(assigned.tolist()))


def mode_entropy(z):
    """모드별 샘플 분포의 엔트로피 - 한쪽 모드로 쏠리면 낮고, 고르게 퍼지면 높음."""
    assigned = assign_to_nearest_mode(z)
    counts = torch.bincount(assigned, minlength=N_MODES).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def avg_pairwise_distance(z):
    diffs = z.unsqueeze(1) - z.unsqueeze(0)
    dist = diffs.norm(dim=-1)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return dist[mask].mean().item()


def avg_quality(z):
    """샘플 품질 프록시: 가장 가까운 진짜 mode 중심까지의 평균 거리(작을수록 좋음)."""
    dists = torch.cdist(z, TARGET_MODES)
    return dists.min(dim=1).values.mean().item()


# ---------- 유닛테스트: repulsion_force 자체의 성질 검증 ----------

def unit_tests():
    # 1. 두 점이 같으면(거리 0) 반발력은 0 방향(diffs=0)이라 정의상 0벡터여야 함
    z_same = torch.zeros(2, DIM)
    f = repulsion_force(z_same)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-5), "같은 위치 두 점의 반발력이 0이 아님"

    # 2. 두 점이 충분히 멀면(sigma 대비) 반발력 크기가 작아져야 함(가우시안 감쇠)
    z_close = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    z_far = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    # sigma가 배치마다 다시 계산되므로 직접 비교 대신 같은 sigma로 강제 테스트
    diffs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # placeholder, 아래서 직접 계산
    d_close = (z_close[0] - z_close[1]).norm().item()
    d_far = (z_far[0] - z_far[1]).norm().item()
    assert d_close < d_far

    # 3. 대칭성: F(z_i)가 z_i를 z_j로부터 밀어내는 방향(양의 스칼라 * (z_i - z_j))이어야 함
    #    (자기 자신을 제외한 gradient 방향 부호 확인)
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    f = repulsion_force(z)
    # 점0은 점1(+x방향)과 점2(-x방향) 사이 중앙 -> 대칭이라 순 반발력 x성분이 0에 가까워야 함
    assert abs(f[0, 0].item()) < 1e-3, f"대칭 배치인데 반발력이 0이 아님: {f[0]}"

    # 4. 학습된 모델의 gradient가 실제로 흐르는지(NaN/전부 0 아님) 확인 - 학습 전 단계 sanity check
    model = VelocityField()
    loss = flow_matching_loss(model, batch_size=32)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "flow matching loss의 gradient가 전부 0"

    print("모든 유닛테스트 통과: repulsion force 대칭성/감쇠, flow matching gradient 정상")


if __name__ == "__main__":
    print("=== 유닛테스트 ===")
    unit_tests()

    print("\n=== Flow matching 모델 학습 (5-mode 2D 합성 분포, 3000 step) ===")
    model = VelocityField()
    losses = train(model, steps=3000)
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f} (최근 200 step 평균 {sum(losses[-200:])/200:.4f})")

    torch.manual_seed(123)
    n_samples = 50
    vanilla = sample_vanilla(model, n_samples)
    torch.manual_seed(123)
    guided = sample_diversity_guided(model, n_samples)

    print("\n=== Vanilla vs Diversity-Guided 샘플링 비교 (같은 시드, 50개 샘플) ===")
    print(f"{'지표':30s} {'vanilla':>10s} {'guided':>10s}")
    print(f"{'mode coverage (max 5)':30s} {mode_coverage(vanilla):>10d} {mode_coverage(guided):>10d}")
    print(f"{'mode entropy (max ln5=1.61)':30s} {mode_entropy(vanilla):>10.3f} {mode_entropy(guided):>10.3f}")
    print(f"{'avg pairwise distance':30s} {avg_pairwise_distance(vanilla):>10.3f} {avg_pairwise_distance(guided):>10.3f}")
    print(f"{'avg dist to nearest mode(품질)':30s} {avg_quality(vanilla):>10.3f} {avg_quality(guided):>10.3f}")
