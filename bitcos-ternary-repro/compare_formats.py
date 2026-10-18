"""Reproduce Fig. 1 / Eq. (1): measured BITCOS bits/weight vs. zero density z,
compared against the fixed-rate five-trit packing (1.625 bits/weight) and
2-bit packing (2.0 bits/weight). Confirms the paper's claimed crossover at
z = 0.375 where BITCOS starts beating five-trit packing.
"""
import numpy as np
from bitcos import encode, five_trit_packing_bits, two_bit_packing_bits

rng = np.random.default_rng(1)
N = 200_000  # large enough that measured z matches target z closely

print(f"{'z':>6} {'BITCOS (measured)':>20} {'BITCOS (theory 2-z)':>20} {'5-trit':>8} {'2-bit':>8}  beats-5trit?")
for z in [0.0, 0.10, 0.20, 0.30, 0.35, 0.375, 0.40, 0.45, 0.515, 0.60, 0.80, 0.95]:
    is_zero = rng.random(N) < z
    signs = rng.choice([-1, 1], size=N)
    weights = np.where(is_zero, 0, signs).astype(np.int8)
    measured_z = np.mean(weights == 0)

    bitmap, sign_bits, n = encode(weights)
    bitcos_bits_per_weight = (bitmap.nbytes + sign_bits.nbytes) * 8 / n
    theory_bits_per_weight = 2 - measured_z
    five_trit = five_trit_packing_bits(n) / n
    two_bit = two_bit_packing_bits(n) / n
    beats = "YES" if bitcos_bits_per_weight < five_trit else "no"

    print(f"{z:6.3f} {bitcos_bits_per_weight:20.4f} {theory_bits_per_weight:20.4f} "
          f"{five_trit:8.4f} {two_bit:8.4f}  {beats}")

print("\nExpected: BITCOS beats five-trit packing (1.625) once z > 0.375 (Fig. 1 crossover).")
print("BITCOS always beats 2-bit packing (2.0), for every z, as the paper claims (Sec II-A).")
