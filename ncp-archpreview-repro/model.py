import numpy as np

from data import CHUNK_LEN, NUM_CHUNK_TYPES, VOCAB_SIZE

# Mini reproduction of NCP-ArchPreview's *novel* mechanism (Liu et al. / Ellis-style
# paper report: product-quantized concept vocabulary + Concept Module + causal
# injection + joint NTP/NCP/VQ loss), with the Transformer backbone itself replaced
# by the simplest possible causal token representation (a bare embedding lookup,
# window=1). See README for why this substitution is a defensible simplification.

D = 8              # hidden / concept dim
S = 2              # product-quantization segments
SEG_DIM = D // S
N_CODES = NUM_CHUNK_TYPES  # codewords per segment
HIDDEN_C = 16       # Concept Module hidden width


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, labels):
    probs = softmax(logits)
    n = logits.shape[0]
    loss = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, dlogits


class AdamParam:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def step(self, w, g, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * (g * g)
        lr_t = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        w -= lr_t * self.m / (np.sqrt(self.v) + eps)
        return w


class TokenOnlyBaseline:
    """Window=1 token model: logits_t = Dense(embed(tok_t)). No concept pathway."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, size=(VOCAB_SIZE, D))
        self.Wd = rng.normal(0, 0.1, size=(D, VOCAB_SIZE))
        self.bd = np.zeros(VOCAB_SIZE)
        self.adam = {k: AdamParam(getattr(self, k).shape) for k in ["E", "Wd", "bd"]}
        self.t = 0

    def n_params(self):
        return self.E.size + self.Wd.size + self.bd.size

    def train_step(self, tok, lr=0.05):
        h = self.E[tok]
        logits = h[:-1] @ self.Wd + self.bd
        loss, dlogits = softmax_cross_entropy(logits, tok[1:])

        dWd = h[:-1].T @ dlogits
        dbd = dlogits.sum(0)
        dh = np.zeros_like(h)
        dh[:-1] = dlogits @ self.Wd.T

        dE = np.zeros_like(self.E)
        np.add.at(dE, tok, dh)

        self.t += 1
        self.Wd = self.adam["Wd"].step(self.Wd, dWd, lr, self.t)
        self.bd = self.adam["bd"].step(self.bd, dbd, lr, self.t)
        self.E = self.adam["E"].step(self.E, dE, lr, self.t)
        return loss

    def eval_loss_at(self, tok, positions):
        h = self.E[tok]
        logits = h[:-1] @ self.Wd + self.bd
        probs = softmax(logits)
        labels = tok[1:]
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
        return nll[positions].mean()


class MiniNCP:
    """Adds the paper's concept pathway on top of the same window=1 token model:
    mean-pool every CHUNK_LEN hidden states into a concept, product-quantize it,
    predict the next concept from the previous one, and causally inject the
    prediction back into the token decoder."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, size=(VOCAB_SIZE, D))
        self.Wd = rng.normal(0, 0.1, size=(D, VOCAB_SIZE))
        self.bd = np.zeros(VOCAB_SIZE)
        self.Wc1 = rng.normal(0, 0.1, size=(D, HIDDEN_C))
        self.bc1 = np.zeros(HIDDEN_C)
        self.Wh = [rng.normal(0, 0.1, size=(HIDDEN_C, N_CODES)) for _ in range(S)]
        self.bh = [np.zeros(N_CODES) for _ in range(S)]
        self.codebooks = [rng.normal(0, 0.3, size=(N_CODES, SEG_DIM)) for _ in range(S)]

        names = ["E", "Wd", "bd", "Wc1", "bc1"]
        self.adam = {k: AdamParam(getattr(self, k).shape) for k in names}
        self.adam_Wh = [AdamParam(w.shape) for w in self.Wh]
        self.adam_bh = [AdamParam(b.shape) for b in self.bh]
        self.adam_codebooks = [AdamParam(c.shape) for c in self.codebooks]
        self.t = 0

    def n_params(self):
        base = self.E.size + self.Wd.size + self.bd.size + self.Wc1.size + self.bc1.size
        base += sum(w.size for w in self.Wh) + sum(b.size for b in self.bh)
        base += sum(c.size for c in self.codebooks)
        return base

    def _quantize(self, c):
        # c: (M, D) -> per-segment nearest codeword, returns assign (M,S) and c_seg (M,S,SEG_DIM)
        M = c.shape[0]
        c_seg = c.reshape(M, S, SEG_DIM)
        assign = np.zeros((M, S), dtype=int)
        for s in range(S):
            d2 = ((c_seg[:, s, None, :] - self.codebooks[s][None, :, :]) ** 2).sum(-1)  # (M,N)
            assign[:, s] = d2.argmin(axis=1)
        return assign, c_seg

    def train_step(self, tok, lr=0.05, alpha=1.0, beta=1.0):
        losses, grads = self._forward_backward(tok, alpha, beta)
        self.t += 1
        self.Wd = self.adam["Wd"].step(self.Wd, grads["Wd"], lr, self.t)
        self.bd = self.adam["bd"].step(self.bd, grads["bd"], lr, self.t)
        self.E = self.adam["E"].step(self.E, grads["E"], lr, self.t)
        self.Wc1 = self.adam["Wc1"].step(self.Wc1, grads["Wc1"], lr, self.t)
        self.bc1 = self.adam["bc1"].step(self.bc1, grads["bc1"], lr, self.t)
        for s in range(S):
            self.Wh[s] = self.adam_Wh[s].step(self.Wh[s], grads["Wh"][s], lr, self.t)
            self.bh[s] = self.adam_bh[s].step(self.bh[s], grads["bh"][s], lr, self.t)
            self.codebooks[s] = self.adam_codebooks[s].step(self.codebooks[s], grads["codebooks"][s], lr, self.t)
        return losses

    def compute_loss(self, tok, alpha=1.0, beta=1.0):
        losses, _ = self._forward_backward(tok, alpha, beta)
        return losses[0]

    def _forward_backward(self, tok, alpha=1.0, beta=1.0):
        T = len(tok)
        M = T // CHUNK_LEN
        h = self.E[tok]  # (T, D)
        c = h[: M * CHUNK_LEN].reshape(M, CHUNK_LEN, D).mean(axis=1)  # (M, D)

        assign, c_seg = self._quantize(c)  # (M,S), (M,S,SEG_DIM)

        # ---- Concept Module: predict c[1:] from c[:-1] (window=1) ----
        prev = c[: M - 1]  # (M-1, D)
        pre_hid = prev @ self.Wc1 + self.bc1
        hid = np.tanh(pre_hid)  # (M-1, HIDDEN_C)

        pi = []
        pred_seg = []
        for s in range(S):
            logits_s = hid @ self.Wh[s] + self.bh[s]
            pi_s = softmax(logits_s)  # (M-1, N)
            pi.append(pi_s)
            pred_seg.append(pi_s @ self.codebooks[s])  # (M-1, SEG_DIM), codebook treated const here
        pred_concept = np.concatenate(pred_seg, axis=1)  # (M-1, D), predicts c[1..M-1]

        # ---- injection: b[t] = pred_concept for concept m=i+1's chunk positions ----
        b = np.zeros((T, D))
        for i in range(M - 1):
            m = i + 1
            start, end = m * CHUNK_LEN, min((m + 1) * CHUNK_LEN, T)
            b[start:end] = pred_concept[i]

        # ---- NTP ----
        fused = h + b
        logits = fused[:-1] @ self.Wd + self.bd
        ntp_loss, dlogits = softmax_cross_entropy(logits, tok[1:])

        dWd = fused[:-1].T @ dlogits
        dbd = dlogits.sum(0)
        dfused = np.zeros((T, D))
        dfused[:-1] = dlogits @ self.Wd.T
        dh = dfused.copy()
        db = dfused.copy()

        # ---- gather db into dpred_concept per chunk ----
        dpred_concept = np.zeros((M - 1, D))
        for i in range(M - 1):
            m = i + 1
            start, end = m * CHUNK_LEN, min((m + 1) * CHUNK_LEN, T)
            dpred_concept[i] = db[start:end].sum(axis=0)

        # ---- NCP loss: MSE(pred_seg_s, stopgrad(c_seg[m])) for m=1..M-1 ----
        target_seg = c_seg[1:]  # (M-1, S, SEG_DIM), stop-gradient by construction (no path back)
        ncp_terms = []
        n_ncp = (M - 1) * S
        for s in range(S):
            diff = pred_seg[s] - target_seg[:, s, :]
            ncp_terms.append((diff ** 2).sum(axis=1))
            dpred_concept[:, s * SEG_DIM:(s + 1) * SEG_DIM] += alpha * (2.0 / n_ncp) * diff
        ncp_loss = alpha * np.mean(np.concatenate(ncp_terms)) if n_ncp > 0 else 0.0

        # ---- backprop dpred_concept -> pi_s -> concept module weights & hid ----
        dhid = np.zeros_like(hid)
        dWh, dbh = [], []
        for s in range(S):
            dpred_seg_s = dpred_concept[:, s * SEG_DIM:(s + 1) * SEG_DIM]
            dpi_s = dpred_seg_s @ self.codebooks[s].T  # (M-1, N)
            pi_s = pi[s]
            dlogits_s = pi_s * (dpi_s - (pi_s * dpi_s).sum(axis=1, keepdims=True))
            dWh.append(hid.T @ dlogits_s)
            dbh.append(dlogits_s.sum(0))
            dhid += dlogits_s @ self.Wh[s].T

        dpre_hid = dhid * (1 - hid ** 2)
        dWc1 = prev.T @ dpre_hid
        dbc1 = dpre_hid.sum(0)
        dprev = dpre_hid @ self.Wc1.T  # (M-1, D) -> flows into c[0:M-1]

        dc = np.zeros((M, D))
        dc[: M - 1] += dprev

        dh_from_c = np.zeros((T, D))
        for m in range(M):
            start, end = m * CHUNK_LEN, (m + 1) * CHUNK_LEN
            dh_from_c[start:end] += dc[m] / CHUNK_LEN

        dh_total = dh + dh_from_c
        dE = np.zeros_like(self.E)
        np.add.at(dE, tok, dh_total)

        # ---- VQ loss: closed-form codebook gradient (stop-grad on c, encoder untouched) ----
        vq_terms = []
        d_codebooks = [np.zeros_like(cb) for cb in self.codebooks]
        MS = M * S
        for s in range(S):
            diff = c_seg[:, s, :] - self.codebooks[s][assign[:, s]]  # (M, SEG_DIM)
            vq_terms.append((diff ** 2).sum(axis=1))
            for n in range(N_CODES):
                mask = assign[:, s] == n
                if mask.any():
                    d_codebooks[s][n] = -(2.0 * beta / MS) * diff[mask].sum(axis=0)
        vq_loss = beta * np.mean(np.concatenate(vq_terms))

        total_loss = ntp_loss + ncp_loss + vq_loss

        grads = {
            "Wd": dWd, "bd": dbd, "E": dE, "Wc1": dWc1, "bc1": dbc1,
            "Wh": dWh, "bh": dbh, "codebooks": d_codebooks,
        }
        return (total_loss, ntp_loss, ncp_loss, vq_loss), grads

    def eval_loss_at(self, tok, positions):
        T = len(tok)
        M = T // CHUNK_LEN
        h = self.E[tok]
        c = h[: M * CHUNK_LEN].reshape(M, CHUNK_LEN, D).mean(axis=1)
        prev = c[: M - 1]
        hid = np.tanh(prev @ self.Wc1 + self.bc1)
        pred_seg = []
        for s in range(S):
            pi_s = softmax(hid @ self.Wh[s] + self.bh[s])
            pred_seg.append(pi_s @ self.codebooks[s])
        pred_concept = np.concatenate(pred_seg, axis=1)

        b = np.zeros((T, D))
        for i in range(M - 1):
            m = i + 1
            start, end = m * CHUNK_LEN, min((m + 1) * CHUNK_LEN, T)
            b[start:end] = pred_concept[i]

        fused = h + b
        logits = fused[:-1] @ self.Wd + self.bd
        probs = softmax(logits)
        labels = tok[1:]
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
        return nll[positions].mean()
