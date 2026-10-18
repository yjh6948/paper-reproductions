import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from data import D1, D2, SyntheticEncoders, make_paired_eval_data, make_unpaired_training_data
from model import Discriminator, Vec2Vec, vsp_loss

BATCH = 64
STEPS = 3000
LAMBDA_REC = 15.0
LAMBDA_CC = 15.0
LAMBDA_VSP = 1.0
LR = 1e-3


def train(seed=0, lambda_vsp=LAMBDA_VSP, lambda_cc=LAMBDA_CC, use_adv=True):
    torch.manual_seed(seed)
    enc = SyntheticEncoders(seed=0)  # same encoders across seeds -- only the GAN init/training varies
    u_pool, v_pool = make_unpaired_training_data(enc, n_per_side=1500, seed=1)

    model = Vec2Vec()
    disc1 = Discriminator(D1)
    disc2 = Discriminator(D2)

    g_opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(list(disc1.parameters()) + list(disc2.parameters()), lr=LR, betas=(0.5, 0.999))

    n = u_pool.shape[0]
    for step in range(STEPS):
        idx_u = torch.randint(0, n, (BATCH,))
        idx_v = torch.randint(0, n, (BATCH,))
        u, v = u_pool[idx_u], v_pool[idx_v]

        # ---- Discriminator step ----
        with torch.no_grad():
            f1_u = model.F1(u)
            f2_v = model.F2(v)
        d_opt.zero_grad()
        d1_loss = F.binary_cross_entropy_with_logits(disc1(u), torch.ones(BATCH)) + \
            F.binary_cross_entropy_with_logits(disc1(f2_v), torch.zeros(BATCH))
        d2_loss = F.binary_cross_entropy_with_logits(disc2(v), torch.ones(BATCH)) + \
            F.binary_cross_entropy_with_logits(disc2(f1_u), torch.zeros(BATCH))
        (d1_loss + d2_loss).backward()
        d_opt.step()

        # ---- Generator step ----
        g_opt.zero_grad()
        f1_u = model.F1(u)
        f2_v = model.F2(v)
        r1_u = model.R1(u)
        r2_v = model.R2(v)
        cc_u = model.F2(f1_u)
        cc_v = model.F1(f2_v)

        l_rec = ((r1_u - u) ** 2).mean() + ((r2_v - v) ** 2).mean()
        l_cc = ((cc_u - u) ** 2).mean() + ((cc_v - v) ** 2).mean()
        l_vsp = vsp_loss(u, v, f1_u, f2_v)
        l_adv = F.binary_cross_entropy_with_logits(disc1(f2_v), torch.ones(BATCH)) + \
            F.binary_cross_entropy_with_logits(disc2(f1_u), torch.ones(BATCH))

        g_loss = LAMBDA_REC * l_rec + lambda_cc * l_cc + lambda_vsp * l_vsp
        if use_adv:
            g_loss = g_loss + l_adv
        g_loss.backward()
        g_opt.step()

        if step % 500 == 0:
            print(f"  step {step:4d}  rec={l_rec.item():.4f}  cc={l_cc.item():.4f}  "
                  f"vsp={l_vsp.item():.4f}  adv={l_adv.item():.4f}")

    return model, enc


def evaluate(model, enc, n_eval=500, seed=999):
    u_eval, v_eval = make_paired_eval_data(enc, n=n_eval, seed=seed)
    with torch.no_grad():
        f1_u = model.F1(u_eval)  # should land near v_eval
        f2_v = model.F2(v_eval)  # should land near u_eval

    def metrics(translated, target):
        t_norm = translated / translated.norm(dim=1, keepdim=True)
        tgt_norm = target / target.norm(dim=1, keepdim=True)
        sims = t_norm @ tgt_norm.T  # (n, n)
        ranks = []
        top1 = 0
        mean_cos = 0.0
        for i in range(sims.shape[0]):
            order = torch.argsort(sims[i], descending=True)
            rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            if rank == 1:
                top1 += 1
            mean_cos += sims[i, i].item()
        n = sims.shape[0]
        return mean_cos / n, top1 / n, sum(ranks) / n

    cos1, top1_1, rank1 = metrics(f1_u, v_eval)
    cos2, top1_2, rank2 = metrics(f2_v, u_eval)

    # OT (Hungarian) baseline: zero-pad the smaller-dim raw embeddings to
    # equal length and solve exact assignment on cosine cost. Crude vs the
    # paper's proper Gromov-Wasserstein solver -- documented simplification.
    def ot_baseline(src, tgt):
        d_src, d_tgt = src.shape[1], tgt.shape[1]
        d = max(d_src, d_tgt)
        src_p = torch.nn.functional.pad(src, (0, d - d_src))
        tgt_p = torch.nn.functional.pad(tgt, (0, d - d_tgt))
        src_n = src_p / src_p.norm(dim=1, keepdim=True).clamp_min(1e-8)
        tgt_n = tgt_p / tgt_p.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cost = (-(src_n @ tgt_n.T)).numpy()
        row, col = linear_sum_assignment(cost)
        top1 = (col == row).mean()
        return top1

    ot_top1_1 = ot_baseline(u_eval, v_eval)
    ot_top1_2 = ot_baseline(v_eval, u_eval)

    return {
        "F1(u)->v: cos": cos1, "F1(u)->v: top1": top1_1, "F1(u)->v: rank": rank1,
        "F2(v)->u: cos": cos2, "F2(v)->u: top1": top1_2, "F2(v)->u: rank": rank2,
        "OT baseline top1 (u->v)": ot_top1_1, "OT baseline top1 (v->u)": ot_top1_2,
        "n_eval": n_eval,
    }


def summarize(name, results):
    print(f"\n[{name}]")
    for k in ["F1(u)->v: cos", "F1(u)->v: top1", "F1(u)->v: rank"]:
        v = results[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main():
    print("=== vec2vec 미니 재현: unpaired 임베딩 공간 간 번역 학습 ===\n")
    print("--- full model ---")
    model, enc = train(seed=0)
    results_full = evaluate(model, enc)
    summarize("full (adv+rec+cc+vsp)", results_full)

    print("\n--- ablation: VSP loss 제거 ---")
    model_novsp, enc2 = train(seed=0, lambda_vsp=0.0)
    results_novsp = evaluate(model_novsp, enc2)
    summarize("no VSP", results_novsp)

    print("\n--- ablation: adversarial loss 제거 ---")
    model_noadv, enc3 = train(seed=0, use_adv=False)
    results_noadv = evaluate(model_noadv, enc3)
    summarize("no adversarial", results_noadv)

    print("\n=== 논문 Table 6과 같은 취지: 각 구성요소가 실제로 기여하는가 ===")
    print(f"  full           top1={results_full['F1(u)->v: top1']:.3f}  cos={results_full['F1(u)->v: cos']:.3f}")
    print(f"  - VSP          top1={results_novsp['F1(u)->v: top1']:.3f}  cos={results_novsp['F1(u)->v: cos']:.3f}")
    print(f"  - adversarial  top1={results_noadv['F1(u)->v: top1']:.3f}  cos={results_noadv['F1(u)->v: cos']:.3f}")


if __name__ == "__main__":
    main()
