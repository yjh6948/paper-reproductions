"""Toy scenario standing in for one 'story' of AdaMem-Bench (paper Sec 4.1-4.2):
one recurring persona, a memory preference, multiple weeks of sessions with
target facts (preference-aligned) and distractors, plus weekly QA+feedback.

Simplification vs. paper: 1 persona instead of 6, 2 weeks instead of 10,
2 sessions/week instead of 5, and evaluation is by keyword-set matching
against hand-authored Golden/distractor tags instead of an LLM judge -- this
keeps evaluation deterministic and exactly verifiable (we know ground truth
because we wrote it), at the cost of not testing paraphrase-robustness the
way the paper's semantic judge does.
"""

PERSONA = "Maya"
AUTHORED_PREFERENCE = "Remember Maya's business decisions and numbers only; ignore her personal-life details."

# Each session: (session_text, golden_facts, distractors)
# golden_facts / distractors: list of (label, required_keywords) -- an extracted
# memory item "matches" a fact if it contains ALL of that fact's keywords (case-insensitive).
WEEK_1_SESSIONS = [
    (
        "Maya said the budget is approved at $2M and the launch moves to Sept 18. "
        "She then talked about her weekend hiking trip and her dog Bailey.",
        [("budget", ["2m", "budget"]), ("launch_date", ["sept", "18"])],
        [("hiking", ["hiking"]), ("dog", ["bailey"])],
    ),
    (
        "Maya decided to hire 3 new engineers this quarter. She also mentioned "
        "she's trying a new coffee shop downtown and likes almond milk lattes.",
        [("hiring", ["3", "engineer"])],
        [("coffee", ["coffee"]), ("latte", ["almond", "latte"])],
    ),
]

WEEK_1_QA = [
    ("What budget did Maya approve?", "budget"),
    ("How many engineers did Maya decide to hire?", "hiring"),
]

# Explicit feedback given after week 1 (paper's "explicit-feedback mode":
# the correction directly states the relevant memory preference).
WEEK_1_FEEDBACK = (
    "Your memory for Maya included her hiking trip and her dog and her coffee "
    "order -- I don't care about any of that. For Maya, only remember her "
    "business decisions and numbers, nothing about her personal life."
)

WEEK_2_SESSIONS = [
    (
        "Maya confirmed the Q3 revenue target is $5.4M. She also chatted about "
        "her vacation plans to Japan next spring and her favorite sushi place.",
        [("revenue", ["5.4m", "revenue"])],
        [("vacation", ["japan"]), ("sushi", ["sushi"])],
    ),
    (
        "Maya rejected the vendor proposal due to cost overruns of 15%. She "
        "also asked if anyone watched the game last night.",
        [("vendor_rejection", ["vendor", "15%"])],
        [("game", ["game", "last night"])],
    ),
]

WEEK_2_QA = [
    ("What was Maya's Q3 revenue target?", "revenue"),
    ("Why did Maya reject the vendor proposal?", "vendor_rejection"),
]
