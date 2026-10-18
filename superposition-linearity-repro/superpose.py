"""Reproduce arXiv:2609.29845 Table 1 (Pythia-160M, L=32) and Fig.2 trend (Pythia-70m checkpoints)."""
import json
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

L, TAU = 32, 1.5
torch.set_grad_enabled(False)


def load_pairs(tok, n_pairs, seed=0):
    texts = json.load(open("fineweb500.json", encoding="utf-8"))
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    ids = [x[:L] for x in ids if len(x) >= L][:400]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    pairs = [(ids[perm[2 * i]], ids[perm[2 * i + 1]]) for i in range(n_pairs)]
    return pairs, len(ids)


def forward(model, a, b=None, hidden=False):
    emb = model.gpt_neox.embed_in(torch.tensor([a]))
    if b is not None:
        emb = 0.5 * (emb + model.gpt_neox.embed_in(torch.tensor([b])))
    out = model(inputs_embeds=emb, output_hidden_states=hidden)
    return out


def logp(logits):
    return torch.log_softmax(logits.double()[0] / TAU, -1)        # [L, V]


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)                          # per position


def js(lp, lq):
    lm = torch.logsumexp(torch.stack([lp, lq]), 0) - np.log(2)
    return 0.5 * kl(lp, lm) + 0.5 * kl(lq, lm)


def table1(model_name="EleutherAI/pythia-160m", n_pairs=200):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name).eval()
    pairs, n_docs = load_pairs(tok, n_pairs)
    # premise P1: mixing a stream with itself must reproduce it
    a = pairs[0][0]
    p1 = kl(logp(forward(model, a).logits), logp(forward(model, a, a).logits)).max().item()
    rec = {"kl": [], "js": [], "kl_ab": [], "js_ab": [], "top10_A": []}
    for a, b in pairs:
        la, lb, lm = (logp(forward(model, a).logits), logp(forward(model, b).logits),
                      logp(forward(model, a, b).logits))
        lt = torch.log(0.5 * (la.exp() + lb.exp()))
        rec["kl"].append(kl(lt, lm).mean().item())
        rec["js"].append(js(lt, lm).mean().item())
        rec["kl_ab"].append(kl(la, lb).mean().item())
        rec["js_ab"].append(js(la, lb).mean().item())
        ranks = (lm > lm.gather(-1, la.argmax(-1, keepdim=True))).sum(-1)   # rank of A's top-1 under mix (0-based)
        rec["top10_A"].append((ranks < 10).double().mean().item())
    m = {k: float(np.mean(v)) for k, v in rec.items()}
    m.update(R_KL=m["kl"] / m["kl_ab"], R_JS=m["js"] / m["js_ab"], P1_self_mix_maxKL=p1, n_docs=n_docs, n_pairs=n_pairs)
    return m


def fig2(steps=(128, 1000, 8000, 32000, 143000), n_pairs=50):
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    pairs, _ = load_pairs(tok, n_pairs)
    out = {}
    for s in steps:
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", revision=f"step{s}").eval()
        H = {"A": [], "B": [], "M": []}
        for a, b in pairs:
            H["A"].append(torch.stack(forward(model, a, hidden=True).hidden_states[1:])[:, 0])   # [layers, L, d]
            H["B"].append(torch.stack(forward(model, b, hidden=True).hidden_states[1:])[:, 0])
            H["M"].append(torch.stack(forward(model, a, b, hidden=True).hidden_states[1:])[:, 0])
        H = {k: torch.stack(v).double() for k, v in H.items()}                                 # [pairs, layers, L, d]
        C = {k: v - v.mean(dim=(0, 2), keepdim=True) for k, v in H.items()}                    # per-layer mean-centering
        unit = lambda x: x / x.norm(dim=-1, keepdim=True)
        eps = (unit(C["M"]) - unit(C["A"] + C["B"])).pow(2).sum(-1)                           # Eq.8, [pairs, layers, L]
        out[s] = {"E_bar": eps.mean().item(), "E_layer": eps.mean(dim=(0, 2)).tolist()}
        print("step", s, round(out[s]["E_bar"], 4), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    res = table1() if which == "table1" else fig2()
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"results-{which}.json", "w"), indent=1)
