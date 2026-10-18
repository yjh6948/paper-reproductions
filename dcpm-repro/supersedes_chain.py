"""Reproduces DCPM's Algorithm 1 (SYSTEM 1 synchronous write pipeline,
arXiv:2606.09483 Sec. 2.2) on a small toy sequence of user turns about one
evolving preference.

NOTE: Gemini calls are NOT used here. The Gemini 2.5 Flash free tier turned
out to be capped at 20 requests/day PER PROJECT (not the 250/day documented
elsewhere) and this session's budget was already spent on the SlopShape
cycle's 20 scoring calls before this script ran (see README.md). The
paper's own extraction/reconcile LLM (M_e, M_r) is a swappable component
(deepseek-v4-flash or kimi-2.5 in the paper); here Claude substitutes for
it directly, reasoning through each turn's ADD/SUPERSEDE/UPDATE decision
inline instead of calling an API. This still exercises the actual
deterministic parts of Algorithm 1: building the doubly linked supersedes
chain and the read-path traversal claim.
"""

import json

TURNS = [
    "I love working from home full time, it's so much more productive for me than the office ever was.",
    "I've started missing the office a bit lately, so I'm going in one day a week now to see the team in person.",
    "Actually the commute turned out to be really draining. I think I prefer full remote after all.",
    "Our new office has a much better setup than the old one. I'm going in three days a week now and genuinely enjoying it.",
    "I've decided I want to be in the office most of the time now -- remote days end up feeling isolating for me.",
]

# Reconcile decisions Claude made in place of M_r (Gemini quota exhausted):
# each new turn's stated preference directly contradicts the most recent
# stored preference, so every step after the first is SUPERSEDE targeting
# the immediately preceding node -- exactly the paper's fitness-preference
# qualitative example (revised across several turns, Sec. 3.5).
FACTS = [
    "User prefers working from home full-time; finds it more productive than the office.",
    "User now goes into the office one day a week to see the team, while still mostly remote.",
    "User has reverted to preferring full remote; the commute was too draining.",
    "User goes into the office three days a week at a new, better office, and is enjoying it.",
    "User wants to be in the office most of the time now; remote days feel isolating.",
]
DECISIONS = [
    ("ADD", None),
    ("SUPERSEDE", 0),
    ("SUPERSEDE", 1),
    ("SUPERSEDE", 2),
    ("SUPERSEDE", 3),
]


def main():
    store = []
    for turn, fact, (op, target) in zip(TURNS, FACTS, DECISIONS):
        node = {"id": len(store), "fact": fact, "supersedes": None, "superseded_by": None}
        if op == "SUPERSEDE" and target is not None:
            node["supersedes"] = target
            store[target]["superseded_by"] = node["id"]
        store.append(node)
        print(f"turn {len(store)}: op={op} target={target} -> \"{fact}\"")

    with open("supersedes_store.json", "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

    print("\n--- doubly linked chain ---")
    for n in store:
        print(f"  id={n['id']} supersedes={n['supersedes']} superseded_by={n['superseded_by']}: {n['fact']}")

    print("\n--- read-path comparison: 'what was the user's stance right after they mentioned the commute?' ---")
    latest = store[-1]
    print(f"update-in-place baseline (keeps only latest fact): \"{latest['fact']}\"")
    print("  -> cannot answer; the commute-driven full-remote reversal (an intermediate state) is gone")

    node = latest
    path = [node["id"]]
    while node["supersedes"] is not None:
        node = store[node["supersedes"]]
        path.append(node["id"])
    print(f"DCPM traversal path from latest back through supersedes chain: {path}")
    target = store[2]
    found = target["id"] in path
    print(f"  -> node id=2 (\"{target['fact']}\") reachable via traversal: {found}")


if __name__ == "__main__":
    main()
