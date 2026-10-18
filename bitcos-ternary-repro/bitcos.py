"""BITCOS layout (paper Sec II-A, Eq. 1, Fig. 2).

A ternary tensor (weights in {-1, 0, +1}) is stored as:
  1) a presence bitmap: one bit per weight, set iff the weight is non-zero
  2) a compacted sign vector: one bit per NON-ZERO weight, in tensor order
     (0 = positive, 1 = negative)

Cost per weight at zero density z: B(z) = 1 + (1 - z) = 2 - z bits (Eq. 1).

Simplification vs. the paper: we reproduce the format and its bit-exactness
faithfully (this is the paper's actual "algorithm"), but not the vectorized
unpacking instruction sequences (Figs. 3, 5, 6) -- those are AVX-512/AVX2/Xe2
assembly targeting specific hardware we don't have register-level access to
from Python. We use numpy packbits/unpackbits as the reference (correct but
not vectorized-for-speed) encode/decode.
"""
import numpy as np


def encode(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """weights: 1D int array with values in {-1, 0, 1}.
    Returns (bitmap_bytes, sign_bytes, n) -- both uint8 packed arrays."""
    assert set(np.unique(weights).tolist()) <= {-1, 0, 1}
    n = len(weights)
    presence = (weights != 0).astype(np.uint8)
    nonzero_vals = weights[weights != 0]
    signs = (nonzero_vals < 0).astype(np.uint8)  # 1 = negative
    bitmap_bytes = np.packbits(presence)
    sign_bytes = np.packbits(signs)
    return bitmap_bytes, sign_bytes, n


def decode(bitmap_bytes: np.ndarray, sign_bytes: np.ndarray, n: int) -> np.ndarray:
    """Inverse of encode(). Reconstructs the original ternary array exactly."""
    presence = np.unpackbits(bitmap_bytes)[:n]
    signs = np.unpackbits(sign_bytes)
    out = np.zeros(n, dtype=np.int8)
    nz_idx = np.nonzero(presence)[0]
    vals = np.where(signs[: len(nz_idx)] == 1, -1, 1).astype(np.int8)
    out[nz_idx] = vals
    return out


def storage_bits(bitmap_bytes: np.ndarray, sign_bytes: np.ndarray) -> int:
    return (bitmap_bytes.nbytes + sign_bytes.nbytes) * 8


def five_trit_packing_bits(n: int) -> int:
    """Paper's derivation: 5 trits/byte, blocks of 128 weights.
    ceil(128/5)=26 payload bytes per 128-weight block -> 26*8/128 = 1.625 bits/weight."""
    block = 128
    n_blocks = -(-n // block)  # ceil division
    payload_bytes_per_block = -(-block // 5)  # ceil(128/5) = 26
    return n_blocks * payload_bytes_per_block * 8


def two_bit_packing_bits(n: int) -> int:
    return 2 * n
