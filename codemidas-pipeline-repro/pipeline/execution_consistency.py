"""Stage 3: Environment Preparation / Execution Consistency (paper Sec 3.3).

Paper: each task is checked in SIX FRESH CONTAINERS -- two running the
starting (stub) codebase, four running the reference solution -- and
requires both starting-state runs to fail and all four reference runs to
pass. The paper is explicit that this screens for "unstable execution
outcomes", i.e. flaky/nondeterministic tasks.

Faithfulness note: we reproduce "fresh" literally, using a fresh `python`
subprocess per run (not just a fresh function call in the same process).
This matters: Python's string hashing is randomized per-process
(PYTHONHASHSEED), so `set()`-order bugs like our deliberately-planted
`unique_tags` only show up across *process* boundaries -- calling it twice
in the same interpreter would look perfectly consistent and this check
would falsely pass it. This is the same class of bug noted in
project_research_pipeline's Procedural Graphs cycle (sorted() needed before
random.choice on a set). Running in-process would have silently missed it,
which is exactly the kind of thing this stage exists to catch.
"""
import subprocess
import sys
import textwrap

_RUNNER_TEMPLATE = '''
{func_source}

import sys
_input = {input_repr}
try:
    _result = {func_name}(_input)
    print("OK", repr(_result))
except Exception as e:
    print("FAIL", type(e).__name__)
'''


def _run_once(func_source: str, func_name: str, input_value) -> str:
    script = _RUNNER_TEMPLATE.format(
        func_source=func_source,
        func_name=func_name,
        input_repr=repr(input_value),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=10,
    )
    return proc.stdout.strip()


def check_consistency(task: dict, inputs: list, n_stub_runs: int = 2, n_ref_runs: int = 4) -> dict:
    """Returns per-input: stub outcomes, reference outcomes, and a verdict."""
    results = []
    for inp in inputs:
        stub_outs = [_run_once(task["stub_source"], task["name"], inp) for _ in range(n_stub_runs)]
        ref_outs = [_run_once(task["reference_source"], task["name"], inp) for _ in range(n_ref_runs)]

        stub_all_fail = all(o.startswith("FAIL") for o in stub_outs)
        ref_all_pass = all(o.startswith("OK") for o in ref_outs)
        ref_consistent = len(set(ref_outs)) == 1

        if not ref_all_pass:
            verdict = "REJECT: reference itself failed"
        elif not stub_all_fail:
            verdict = "REJECT: stub unexpectedly passed (no fail-to-pass transition)"
        elif not ref_consistent:
            verdict = f"REJECT: flaky reference ({len(set(ref_outs))} distinct outputs across {n_ref_runs} runs)"
        else:
            verdict = "ACCEPT"

        results.append({
            "input": inp,
            "stub_outputs": stub_outs,
            "reference_outputs": ref_outs,
            "verdict": verdict,
            "expected_output": ref_outs[0] if ref_consistent else None,
        })
    return results
