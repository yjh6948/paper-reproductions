import scenario
from workflow import extract_memories, update_policy, score_recall_rejection, INITIAL_POLICY


def answer_from_memory(question: str, memory: list[str], topic_keywords: list[str]) -> str:
    """Deterministic stand-in for Eq. (3)-(4) Retrieve+Respond -- we only need
    (q, y) pairs to build the policy-update history H, so a keyword-match
    lookup against the persistent store is enough; no extra Gemini call."""
    for m in memory:
        if all(kw.lower() in m.lower() for kw in topic_keywords):
            return m
    return "I don't know."


def run_condition(name: str, use_policy_updates: bool):
    print(f"\n=== {name} ===")
    memory: list[str] = []
    policy = INITIAL_POLICY
    session_scores = []

    # --- Week 1 ---
    for text, golden, distractors in scenario.WEEK_1_SESSIONS:
        items = extract_memories(text, policy)
        memory.extend(items)
        s = score_recall_rejection(items, golden, distractors)
        session_scores.append(s)
        print(f"  [W1 session] wrote {len(items)} items -> recall={s['key_recall']:.2f} "
              f"noise_rej={s['noise_rejection']:.2f}")

    # Week-1 QA using current memory, to build policy-update history H
    all_w1_facts = {label: kws for text, golden, _d in scenario.WEEK_1_SESSIONS for label, kws in golden}
    qa_history = []
    for q, label in scenario.WEEK_1_QA:
        a = answer_from_memory(q, memory, all_w1_facts[label])
        qa_history.append((q, a))

    if use_policy_updates:
        policy = update_policy(policy, qa_history, scenario.WEEK_1_FEEDBACK)
        print(f"  [policy update] -> {policy!r}")

    # --- Week 2 (AdaMem uses updated policy; baseline keeps INITIAL_POLICY) ---
    for text, golden, distractors in scenario.WEEK_2_SESSIONS:
        items = extract_memories(text, policy)
        memory.extend(items)
        s = score_recall_rejection(items, golden, distractors)
        session_scores.append(s)
        print(f"  [W2 session] wrote {len(items)} items -> recall={s['key_recall']:.2f} "
              f"noise_rej={s['noise_rejection']:.2f}")

    avg_recall = sum(s["key_recall"] for s in session_scores) / len(session_scores)
    avg_noise_rej = sum(s["noise_rejection"] for s in session_scores) / len(session_scores)
    total_memory = sum(s["memory_count"] for s in session_scores)
    return {"avg_key_recall": avg_recall, "avg_noise_rejection": avg_noise_rej,
            "total_memory_items": total_memory, "final_policy": policy}


if __name__ == "__main__":
    baseline = run_condition("Mem0-style baseline (no policy, no updates)", use_policy_updates=False)
    adamem = run_condition("AdaMem (policy + explicit-feedback update after W1)", use_policy_updates=True)

    print("\n" + "=" * 70)
    print("SUMMARY (cf. paper Table 1 & Table 3)")
    print("=" * 70)
    print(f"{'metric':>22} {'baseline':>10} {'adamem':>10} {'delta':>8}")
    for key, label in [("avg_key_recall", "Key Recall"),
                        ("avg_noise_rejection", "Noise Rejection"),
                        ("total_memory_items", "Memory Volume")]:
        b, a = baseline[key], adamem[key]
        print(f"{label:>22} {b:>10.3f} {a:>10.3f} {a-b:>+8.3f}")

    print(f"\nAdaMem final policy: {adamem['final_policy']!r}")
