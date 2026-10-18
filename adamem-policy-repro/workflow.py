"""Implements the paper's Algorithm 1 / Eq. (1)-(6), using real Gemini 2.5
Flash calls for the extractor E and the policy updater U -- these are
genuinely LLM components in the paper too (not something to fake), matching
the precedent set by the Generative Agents cycle (first cycle) which used
real Gemini calls for importance scoring/reflection rather than simulating
an LLM's judgment with a stub.
"""
import json
import re
from gemini_client import generate

INITIAL_POLICY = "Extract salient facts worth remembering from this conversation."


def extract_memories(session_text: str, policy: str) -> list[str]:
    """Eq. (1): M_i = E(C_i, P)."""
    prompt = f"""You are a memory-writing extractor for a personal assistant.

Memory policy (what to remember): {policy}

Conversation:
{session_text}

Following the policy above, list the facts that should be written to
persistent memory. Output ONLY a JSON array of short strings, one per
memory item. If nothing qualifies under the policy, output [].
JSON array:"""
    raw = generate(prompt)
    return _parse_json_array(raw)


def update_policy(current_policy: str, qa_history: list[tuple], feedback: str) -> str:
    """Eq. (6): P_{j+1} = U(P_j, H_j)."""
    qa_text = "\n".join(f"Q: {q}\nA (from memory): {a}" for q, a in qa_history)
    prompt = f"""You are updating a memory policy for a personal assistant.

Current policy: {current_policy}

This week's Q&A interactions:
{qa_text}

User feedback on how memory should work going forward:
{feedback}

Write a SHORT (1-2 sentence) revised memory policy that incorporates this
feedback. Output ONLY the revised policy text, nothing else."""
    return generate(prompt).strip()


def _parse_json_array(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        return [str(x) for x in items]
    except json.JSONDecodeError:
        return []


def score_recall_rejection(memory_items: list[str], golden: list[tuple], distractors: list[tuple]) -> dict:
    """Deterministic keyword-set matching against hand-authored ground truth
    (see scenario.py docstring for why this replaces the paper's LLM judge)."""
    joined = " || ".join(m.lower() for m in memory_items)

    def matched(tagged_facts):
        hits = 0
        for _label, keywords in tagged_facts:
            if all(kw.lower() in joined for kw in keywords):
                hits += 1
        return hits

    golden_hits = matched(golden)
    distractor_hits = matched(distractors)
    key_recall = golden_hits / len(golden) if golden else 1.0
    noise_rejection = 1 - (distractor_hits / len(distractors)) if distractors else 1.0
    return {
        "key_recall": key_recall,
        "noise_rejection": noise_rejection,
        "golden_hits": golden_hits,
        "golden_total": len(golden),
        "distractor_hits": distractor_hits,
        "distractor_total": len(distractors),
        "memory_count": len(memory_items),
    }
