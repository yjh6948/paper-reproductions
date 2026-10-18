import numpy as np

# Architecture and hyperparameters match Ellis (2026), Table 2 / Section 4.5 exactly,
# for INPUT_SIZE=64: Conv1(3x3,4f) -> LeakyReLU -> MaxPool2x2 -> Conv2(3x3,8f)
# -> LeakyReLU -> Flatten(6728) -> Dense(NUM_CLASSES), Adam(lr=3e-4, b1=.9, b2=.999, eps=1e-6).

LEAKY_ALPHA = 0.1


def leaky_relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def leaky_relu_grad(x):
    return np.where(x > 0, 1.0, LEAKY_ALPHA)


def im2col(x, kh, kw):
    n, h, w, c = x.shape
    oh, ow = h - kh + 1, w - kw + 1
    shape = (n, oh, ow, kh, kw, c)
    strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[1], x.strides[2], x.strides[3])
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return patches.reshape(n, oh, ow, kh * kw * c)


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, rng):
        fan_in = in_channels * kernel_size * kernel_size
        self.w = rng.normal(0, np.sqrt(2.0 / fan_in), size=(kernel_size * kernel_size * in_channels, out_channels))
        self.b = np.zeros(out_channels)
        self.k = kernel_size
        self.cin = in_channels
        self.cout = out_channels

    def n_params(self):
        return self.w.size + self.b.size

    def forward(self, x):
        self.x = x
        self.cols = im2col(x, self.k, self.k)  # (N, oh, ow, k*k*cin)
        n, oh, ow, _ = self.cols.shape
        z = self.cols.reshape(n * oh * ow, -1) @ self.w + self.b
        self.pre_act = z.reshape(n, oh, ow, self.cout)
        return leaky_relu(self.pre_act)

    def backward(self, dout):
        n, oh, ow, _ = self.pre_act.shape
        dz = (dout * leaky_relu_grad(self.pre_act)).reshape(n * oh * ow, self.cout)
        cols_flat = self.cols.reshape(n * oh * ow, -1)
        self.dw = cols_flat.T @ dz
        self.db = dz.sum(axis=0)

        dcols = dz @ self.w.T  # (n*oh*ow, k*k*cin)
        dcols = dcols.reshape(n, oh, ow, self.k, self.k, self.cin)
        dx = np.zeros_like(self.x)
        for ky in range(self.k):
            for kx in range(self.k):
                dx[:, ky:ky + oh, kx:kx + ow, :] += dcols[:, :, :, ky, kx, :]
        return dx


class MaxPool2D:
    def __init__(self, size=2):
        self.s = size

    def forward(self, x):
        n, h, w, c = x.shape
        s = self.s
        oh, ow = h // s, w // s
        xr = x[:, :oh * s, :ow * s, :].reshape(n, oh, s, ow, s, c)
        self.x_shape = x.shape
        out = xr.max(axis=(2, 4))
        # recompute argmax mask for backward
        out_b = out.reshape(n, oh, 1, ow, 1, c)
        self.mask = (xr == out_b)
        # break ties by keeping only first max per window
        counts = self.mask.sum(axis=(2, 4), keepdims=True)
        self.mask = self.mask / np.maximum(counts, 1)
        return out

    def backward(self, dout):
        n, oh, ow, c = dout.shape
        s = self.s
        dout_b = dout.reshape(n, oh, 1, ow, 1, c)
        dxr = self.mask * dout_b
        dx = np.zeros(self.x_shape)
        dx[:, :oh * s, :ow * s, :] = dxr.reshape(n, oh * s, ow * s, c)
        return dx


class Dense:
    def __init__(self, in_features, out_features, rng):
        self.w = rng.normal(0, np.sqrt(2.0 / in_features), size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def n_params(self):
        return self.w.size + self.b.size

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dout):
        self.dw = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.w.T


def softmax_cross_entropy(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, probs, dlogits


class AdamParam:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def step(self, w, g, lr, t, b1=0.9, b2=0.999, eps=1e-6):
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * (g * g)
        lr_t = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        w -= lr_t * self.m / (np.sqrt(self.v) + eps)
        return np.clip(w, -10.0, 10.0)


class TinyVisionCNN:
    """Reproduces the paper's architecture at INPUT_SIZE=64: 20,595 params."""

    def __init__(self, num_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2D(3, 4, 3, rng)
        self.pool = MaxPool2D(2)
        self.conv2 = Conv2D(4, 8, 3, rng)
        self.dense = Dense(29 * 29 * 8, num_classes, rng)
        self.adam = {
            "conv1_w": AdamParam(self.conv1.w.shape), "conv1_b": AdamParam(self.conv1.b.shape),
            "conv2_w": AdamParam(self.conv2.w.shape), "conv2_b": AdamParam(self.conv2.b.shape),
            "dense_w": AdamParam(self.dense.w.shape), "dense_b": AdamParam(self.dense.b.shape),
        }
        self.t = 0

    def n_params(self):
        return self.conv1.n_params() + self.conv2.n_params() + self.dense.n_params()

    def forward(self, x):
        a1 = self.conv1.forward(x)
        p1 = self.pool.forward(a1)
        a2 = self.conv2.forward(p1)
        self.a2_shape = a2.shape
        flat = a2.reshape(a2.shape[0], -1)
        logits = self.dense.forward(flat)
        return logits

    def backward(self, dlogits):
        dflat = self.dense.backward(dlogits)
        da2 = dflat.reshape(self.a2_shape)
        dp1 = self.conv2.backward(da2)
        da1 = self.pool.backward(dp1)
        self.conv1.backward(da1)

    def adam_step(self, lr):
        self.t += 1
        self.conv1.w = self.adam["conv1_w"].step(self.conv1.w, self.conv1.dw, lr, self.t)
        self.conv1.b = self.adam["conv1_b"].step(self.conv1.b, self.conv1.db, lr, self.t)
        self.conv2.w = self.adam["conv2_w"].step(self.conv2.w, self.conv2.dw, lr, self.t)
        self.conv2.b = self.adam["conv2_b"].step(self.conv2.b, self.conv2.db, lr, self.t)
        self.dense.w = self.adam["dense_w"].step(self.dense.w, self.dense.dw, lr, self.t)
        self.dense.b = self.adam["dense_b"].step(self.dense.b, self.dense.db, lr, self.t)
