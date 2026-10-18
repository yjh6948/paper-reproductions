import numpy as np

from data import boundary_positions, generate_sequence
from model import MiniNCP, TokenOnlyBaseline

EPOCHS = 800
LR = 0.05


def run(seed):
    tok = generate_sequence(n_chunks=300, seed=seed)
    val_tok = generate_sequence(n_chunks=100, seed=seed + 1000)
    bpos = boundary_positions(len(val_tok))
    non_b = np.setdiff1d(np.arange(len(val_tok) - 1), bpos)

    baseline = TokenOnlyBaseline(seed=seed)
    ncp = MiniNCP(seed=seed)

    for _ in range(EPOCHS):
        baseline.train_step(tok, lr=LR)
        ncp.train_step(tok, lr=LR, alpha=1.0, beta=1.0)

    return {
        "baseline_boundary": baseline.eval_loss_at(val_tok, bpos),
        "baseline_nonboundary": baseline.eval_loss_at(val_tok, non_b),
        "ncp_boundary": ncp.eval_loss_at(val_tok, bpos),
        "ncp_nonboundary": ncp.eval_loss_at(val_tok, non_b),
    }


def main():
    n_seeds = 5
    results = [run(s) for s in range(n_seeds)]
    print(f"=== TokenOnly(window=1) vs MiniNCP, {n_seeds}-seed average, held-out validation ===\n")
    for key in ["baseline_boundary", "ncp_boundary", "baseline_nonboundary", "ncp_nonboundary"]:
        vals = [r[key] for r in results]
        print(f"  {key:24s} NLL = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print("\n=== 해석 ===")
    print("  경계 위치(다음 청크 정체성이 필요, window=1로는 원리적으로 못 풂):")
    print(f"    baseline: {np.mean([r['baseline_boundary'] for r in results]):.4f}"
          f"  vs  mini-NCP: {np.mean([r['ncp_boundary'] for r in results]):.4f}")
    print("  청크 내부(직전 토큰만으로 결정됨, 둘 다 잘 풀어야 정상):")
    print(f"    baseline: {np.mean([r['baseline_nonboundary'] for r in results]):.4f}"
          f"  vs  mini-NCP: {np.mean([r['ncp_nonboundary'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
