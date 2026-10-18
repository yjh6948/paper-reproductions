"""Verify BITCOS is bit-exact (paper's explicit claim, Sec IV-C: "it is
bit-exact, because it decodes the same ternary values as the existing
model") -- across a sweep of zero densities and random seeds."""
import numpy as np
from bitcos import encode, decode

rng = np.random.default_rng(0)
N = 100_000
failures = 0
trials = 0

for z in [0.0, 0.1, 0.297, 0.375, 0.40, 0.515, 0.7, 0.95, 1.0]:
    for trial in range(5):
        trials += 1
        # sample weights: zero with prob z, else uniform +-1
        is_zero = rng.random(N) < z
        signs = rng.choice([-1, 1], size=N)
        weights = np.where(is_zero, 0, signs).astype(np.int8)

        bitmap, sign_bits, n = encode(weights)
        recovered = decode(bitmap, sign_bits, n)

        if not np.array_equal(weights, recovered):
            failures += 1
            print(f"  MISMATCH at z={z}, trial={trial}")

print(f"\n{trials - failures}/{trials} round-trips bit-exact (z sweep 0.0-1.0, N={N} weights each)")
assert failures == 0, "BITCOS round-trip is NOT bit-exact -- contradicts paper's claim"
print("Bit-exactness confirmed.")
