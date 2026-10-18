import numpy as np

from data import make_dataset
from model import TinyVisionCNN, softmax_cross_entropy

# Hyperparameters match the paper exactly (Section 4.5, 5.1): lr=3e-4, batch=6, 20 epochs.
LR = 0.0003
BATCH_SIZE = 6
EPOCHS = 20


def accuracy(model, x, y):
    logits = model.forward(x)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_correct_batch_accumulation(train_x, train_y, seed=0):
    """Gradient accumulators zeroed once per batch (paper Section 4.4, correct)."""
    model = TinyVisionCNN(num_classes=3, seed=seed)
    n = len(train_x)
    rng = np.random.default_rng(seed + 100)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb, yb = train_x[idx], train_y[idx]
            logits = model.forward(xb)
            loss, probs, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits)
            model.adam_step(LR)
        history.append(loss)
    return model, history


def train_buggy_per_image_update(train_x, train_y, seed=0):
    """Bug reproduced from paper Section 4.4: an Adam step fires after every
    single image instead of once per accumulated batch — batch_size becomes
    effectively 1, breaking the intended mini-batch semantics."""
    model = TinyVisionCNN(num_classes=3, seed=seed)
    n = len(train_x)
    rng = np.random.default_rng(seed + 100)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        last_loss = None
        for i in order:
            xb, yb = train_x[i:i + 1], train_y[i:i + 1]
            logits = model.forward(xb)
            loss, probs, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits)
            model.adam_step(LR)
            last_loss = loss
        history.append(last_loss)
    return model, history


def main(n_seeds=8):
    train_x, train_y, val_x, val_y = make_dataset(n_per_class=30, n_val_per_class=3, seed=0)
    print(f"train: {len(train_x)} images, val: {len(val_x)} images (paper Dataset B: ~90 train / 9 val)")
    print(f"averaging over {n_seeds} seeds (val set is only 9 images — single-run numbers are too noisy to trust)\n")

    val_accs_correct, val_accs_buggy = [], []
    loss_var_correct, loss_var_buggy = [], []

    for seed in range(n_seeds):
        model_c, hist_c = train_correct_batch_accumulation(train_x, train_y, seed=seed)
        val_accs_correct.append(accuracy(model_c, val_x, val_y))
        loss_var_correct.append(np.var(hist_c[-5:]))

        model_b, hist_b = train_buggy_per_image_update(train_x, train_y, seed=seed)
        val_accs_buggy.append(accuracy(model_b, val_x, val_y))
        loss_var_buggy.append(np.var(hist_b[-5:]))

    print("=== 정석(배치당 1회 누적) vs 버그(이미지마다 즉시 step), 8-seed 평균 ===")
    print(f"  val accuracy   correct: {np.mean(val_accs_correct)*100:.1f}% ± {np.std(val_accs_correct)*100:.1f}%  (paper reports ~67-78%)")
    print(f"  val accuracy   buggy:   {np.mean(val_accs_buggy)*100:.1f}% ± {np.std(val_accs_buggy)*100:.1f}%")
    print(f"  loss var(last5) correct: {np.mean(loss_var_correct):.5f}")
    print(f"  loss var(last5) buggy:   {np.mean(loss_var_buggy):.5f}")
    print(f"  individual val accs (correct): {[round(a*100,1) for a in val_accs_correct]}")
    print(f"  individual val accs (buggy):   {[round(a*100,1) for a in val_accs_buggy]}")


if __name__ == "__main__":
    main()
