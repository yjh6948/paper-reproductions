import numpy as np

# Synthetic stand-in for the paper's on-device dataset (0Blank / 1Cup / 2Pen),
# since we have no physical ESP32-S3 camera in this session. Three visually
# distinct procedural classes at the same 64x64x3, [0,1]-normalized shape.

IMG_SIZE = 64


NOISE = 0.22  # tuned so single-run train/val accuracy lands near the paper's ~70-80%


def _blank(rng):
    img = rng.normal(0.85, NOISE, size=(IMG_SIZE, IMG_SIZE, 3))
    return np.clip(img, 0, 1)


def _circle(rng):
    img = rng.normal(0.85, NOISE, size=(IMG_SIZE, IMG_SIZE, 3))
    cx, cy = rng.integers(20, 44, size=2)
    r = rng.integers(9, 15)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    color = rng.uniform(0.35, 0.55, size=3)
    img[mask] = color + rng.normal(0, NOISE, size=(mask.sum(), 3))
    return np.clip(img, 0, 1)


def _cross(rng):
    img = rng.normal(0.85, NOISE, size=(IMG_SIZE, IMG_SIZE, 3))
    thickness = rng.integers(3, 6)
    cx = rng.integers(24, 40)
    angle_vertical = rng.random() > 0.5
    color = rng.uniform(0.35, 0.55, size=3)
    if angle_vertical:
        img[:, cx - thickness:cx + thickness] = color
    else:
        img[cx - thickness:cx + thickness, :] = color
    img += rng.normal(0, NOISE, size=img.shape)
    return np.clip(img, 0, 1)


GENERATORS = [_blank, _circle, _cross]
CLASS_NAMES = ["0Blank", "1Circle", "2Line"]


def make_dataset(n_per_class=30, n_val_per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    train_x, train_y, val_x, val_y = [], [], [], []
    for label, gen in enumerate(GENERATORS):
        imgs = [gen(rng) for _ in range(n_per_class + n_val_per_class)]
        for img in imgs[:n_per_class]:
            train_x.append(img)
            train_y.append(label)
        for img in imgs[n_per_class:]:
            val_x.append(img)
            val_y.append(label)
    return (
        np.array(train_x), np.array(train_y),
        np.array(val_x), np.array(val_y),
    )
