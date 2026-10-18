"""Reproduce the paper's CPU roofline model (Sec III-B1, Eq. 5-6, Table II).

Honest scope note: gamma (cycles/iteration) and beta (bytes/cycle bandwidth
per core) are HARDWARE-MEASURED quantities from the paper's own benchmarking
on real Intel CPUs (Emerald Rapids, Arrow Lake, Lunar Lake) -- we have no
access to that hardware or to cycle-accurate instrumentation from Python, so
we take their measured gamma/beta values from Table II as GIVEN INPUTS and
verify that Eq. (5) and Eq. (6), applied to those inputs, reproduce Table
II's B/gamma column and "bound by" classification. This checks the paper's
own arithmetic/model is internally consistent -- it does NOT independently
validate that gamma/beta are correct measurements of real silicon.
"""

# From Table II, z = 0.40
PLATFORMS = [
    # (name, cores, gamma_cycles, beta_bytes_per_cycle, paper_bound_by)
    ("EMR",   64, 4.60, 1.321, "memory"),
    ("ARL-P",  8, 5.00, 0.756, "memory"),
    ("ARL-E", 16, 7.95, 0.903, "instructions (knee)"),
    ("LNL-P",  4, 5.00, 3.224, "instructions"),
    ("LNL-E",  4, 7.95, 3.663, "instructions"),
]


def B(z: float) -> float:
    """Eq. (5): bytes read per 32-weight iteration."""
    return 4 + 4 * (1 - z) + 0.5  # bitmap + signs + fp16 scale


def roofline(gamma: float, beta: float, z: float) -> dict:
    """Eq. (6): T = max(B/beta, gamma); ebw = min(beta, B/gamma)."""
    b = B(z)
    b_over_gamma = b / gamma
    ebw = min(beta, b_over_gamma)
    # "knee" per paper's own Table II wording: within ~4% counts as knee
    ratio = b_over_gamma / beta
    if abs(ratio - 1.0) < 0.04:
        bound_by = "instructions (knee)"
    elif b_over_gamma > beta:
        bound_by = "memory"
    else:
        bound_by = "instructions"
    return {"B": b, "B/gamma": b_over_gamma, "ebw": ebw, "bound_by": bound_by}


if __name__ == "__main__":
    z = 0.40
    print(f"B(z={z}) = {B(z):.2f} bytes / 32 weights  (paper reports 6.90)\n")
    print(f"{'platform':>8} {'gamma':>6} {'beta':>7} {'B/gamma':>8}  {'our bound_by':>20}  {'paper bound_by':>20}  match?")
    all_match = True
    for name, cores, gamma, beta, paper_bound in PLATFORMS:
        r = roofline(gamma, beta, z)
        match = "YES" if r["bound_by"] == paper_bound else "no"
        if match == "no":
            all_match = False
        print(f"{name:>8} {gamma:6.2f} {beta:7.3f} {r['B/gamma']:8.3f}  "
              f"{r['bound_by']:>20}  {paper_bound:>20}  {match}")

    print(f"\nAll platforms match paper's Table II classification: {all_match}")
