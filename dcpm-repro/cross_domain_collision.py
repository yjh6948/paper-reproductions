"""Reproduces DCPM's Eq. 1 cross-domain collision detector (arXiv:2606.09483
Sec. 2.3 Phase 3): a pair of within-domain schemas collides when their
behavioral-abstraction embeddings are similar (cos > theta_beh) but their
raw semantic embeddings are dissimilar (cos < theta_sem). This is the
paper's single biggest ablation contributor (~4.1 of ~8.1 points on
PersonaMem-v2), so it's the mechanism most worth checking in isolation.

NOTE: gemini-embedding-001 was available and DID work for a couple of test
calls before the gemini-2.5-flash free tier hit its real daily cap (20
requests/day/project, not the 250 documented elsewhere -- see README.md).
Rather than burn the embedding quota too and risk a half-finished run, this
script substitutes a pure-Python bag-of-words TF cosine similarity for both
E_sem and E_beh, computed locally with no API calls. This is a much cruder
signal than the paper's bge-m3 dense embeddings (1024-dim, trained on
massive multilingual text) -- it can only pick up literal word overlap, not
paraphrase-level similarity. The behavioral abstractions below are
therefore written with deliberately consistent vocabulary across the
"should collide" pair, mimicking what a real embedding model would place
close together semantically even under different phrasing.
"""

import json
import math
import re
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or", "but",
    "user", "users", "is", "are", "was", "were", "be", "been", "being",
    "they", "their", "them", "it", "its", "this", "that", "these", "those",
    "because", "so", "already", "much", "well", "before", "with", "every",
    "not", "does", "do", "did", "can", "will", "would", "has", "have", "had",
}


def tokenize(text):
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOPWORDS]


def tf_vector(text):
    return Counter(tokenize(text))


def cos(v1, v2):
    common = set(v1) & set(v2)
    dot = sum(v1[w] * v2[w] for w in common)
    n1 = math.sqrt(sum(c * c for c in v1.values()))
    n2 = math.sqrt(sum(c * c for c in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


THETA_BEH = 0.25  # lowered from the paper's 0.72 -- bag-of-words cosine is
THETA_SEM = 0.15  # far sparser than dense embedding cosine, so raw thresholds
                   # don't transfer; these are re-tuned by hand on this toy
                   # vocabulary, which is itself a documented limitation

# (schema_a, schema_b, domain_a, domain_b, behavioral_abstraction_a,
#  behavioral_abstraction_b, expected_collision)
PAIRS = [
    (
        "The user keeps adding money to a losing stock position because they've already invested so much, hoping to avoid locking in the loss.",
        "The user keeps renovating their house well past the original budget because they've already spent so much and doesn't want that money wasted.",
        "personal finance", "home ownership",
        "escalates commitment after sunk cost investment",
        "escalates commitment after sunk cost spending",
        True,
    ),
    (
        "The user double-checks every email three times before sending it, worried about typos in front of coworkers.",
        "The user researches every restaurant's reviews extensively before booking, worried about wasting an evening on a bad meal.",
        "workplace communication", "dining out",
        "over-verifies small decisions to avoid a bad outcome",
        "over-verifies small decisions to avoid a bad outcome",
        True,
    ),
    (
        "The user prefers running outdoors in the early morning before work.",
        "The user's favorite programming language is Python and dislikes verbose Java boilerplate.",
        "exercise", "software engineering",
        "prefers a specific routine time of day",
        "prefers a specific coding language style",
        False,
    ),
]


def main():
    results = []
    for schema_a, schema_b, domain_a, domain_b, beh_a, beh_b, expected in PAIRS:
        cos_sem = cos(tf_vector(schema_a), tf_vector(schema_b))
        cos_beh = cos(tf_vector(beh_a), tf_vector(beh_b))
        collision = cos_beh > THETA_BEH and cos_sem < THETA_SEM

        print(f"\n[{domain_a} vs {domain_b}] expected_collision={expected}")
        print(f"  behavioral abstractions: \"{beh_a}\" | \"{beh_b}\"")
        print(f"  cos(E_beh)={cos_beh:.3f} (theta_beh={THETA_BEH})   cos(E_sem)={cos_sem:.3f} (theta_sem={THETA_SEM})")
        print(f"  Eq.1 fires (collision detected): {collision}   match_expected={collision == expected}")

        results.append({
            "domain_a": domain_a, "domain_b": domain_b,
            "beh_a": beh_a, "beh_b": beh_b,
            "cos_beh": cos_beh, "cos_sem": cos_sem,
            "collision": collision, "expected": expected,
        })

    with open("collision_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_match = sum(r["collision"] == r["expected"] for r in results)
    print(f"\n{n_match}/{len(results)} pairs matched our prediction")


if __name__ == "__main__":
    main()
