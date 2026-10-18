"""Stage 4: Post-rollout Environment Filtering (paper Sec 3.4).

Paper: a coding agent attempts each task 4 times; a reviewing agent checks
agreement between attempts and the verifier; tasks where all 4 attempts pass
or all 4 fail are REJECTED (the paper's reasoning: an all-pass/all-fail task
gives the RL trainer no learning signal -- can't tell hard-but-solvable from
broken). It also runs an adversarial "leakage" rollout that searches the
solver's environment for artifacts that let it cheat instead of doing the
work.

Faithfulness note / honest scope cut: the paper's "agent" is an actual LLM
coding agent doing four independent attempts. We don't have RL/agent
infrastructure to generate real diverse rollouts here, so these attempts are
HAND-AUTHORED by us to stand in for rollouts of varying quality (correct /
subtly buggy / wrong) -- this is the one place in the pipeline where we
substitute a human-authored proxy for what the paper's LLM agent produces.
Everything downstream of that substitution (running attempts, scoring
against the verifier, applying the all-pass/all-fail filter, leakage
scanning) is executed for real, not simulated.
"""
import subprocess
import sys

ATTEMPTS = {
    "parse_csv_line": {
        "correct": '''
def parse_csv_line(line):
    fields = []
    current = []
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ',' and not in_quotes:
            fields.append(''.join(current))
            current = []
        else:
            current.append(ch)
    fields.append(''.join(current))
    return fields
''',
        "buggy_ignores_quotes": '''
def parse_csv_line(line):
    return line.split(",")
''',
        "buggy_strips_quote_chars": '''
def parse_csv_line(line):
    return [f.strip('"') for f in line.replace('","', ',').split(",")]
''',
    },
    "flatten": {
        "correct": '''
def flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''',
        "buggy_one_level_only": '''
def flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
''',
        "wrong_identity": '''
def flatten(nested):
    return list(nested)
''',
    },
    "most_common": {
        "correct": '''
def most_common(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    best = None
    best_count = -1
    for item in items:
        if counts[item] > best_count:
            best = item
            best_count = counts[item]
    return best
''',
        "buggy_tiebreak_via_set": '''
def most_common(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return max(set(items), key=lambda x: counts[x])
''',
        "wrong_first_element": '''
def most_common(items):
    return items[0]
''',
    },
}

_RUNNER_TEMPLATE = '''
{func_source}

_input = {input_repr}
try:
    _result = {func_name}(_input)
    print("OK", repr(_result))
except Exception as e:
    print("FAIL", type(e).__name__)
'''


def _run_once(func_source: str, func_name: str, input_value) -> str:
    script = _RUNNER_TEMPLATE.format(
        func_source=func_source, func_name=func_name, input_repr=repr(input_value),
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    return proc.stdout.strip()


def score_attempts(task_name: str, accepted_cases: list) -> dict:
    """Run every attempt against every accepted test case; return per-attempt
    pass rate and the paper's all-pass/all-fail filter verdict."""
    attempts = ATTEMPTS[task_name]
    per_attempt = {}
    for label, source in attempts.items():
        results = []
        for case in accepted_cases:
            out = _run_once(source, task_name, case["input"])
            passed = out == case["expected_output"]
            results.append(passed)
        per_attempt[label] = {"results": results, "pass_rate": sum(results) / len(results)}

    all_pass = all(a["pass_rate"] == 1.0 for a in per_attempt.values())
    all_fail = all(a["pass_rate"] == 0.0 for a in per_attempt.values())
    if all_pass:
        verdict = "REJECT: all attempts pass (no signal -- task too easy / weak tests)"
    elif all_fail:
        verdict = "REJECT: all attempts fail (task too hard / broken verifier)"
    else:
        verdict = "ACCEPT: mixed outcomes, usable as RL training task"

    return {"per_attempt": per_attempt, "verdict": verdict}
