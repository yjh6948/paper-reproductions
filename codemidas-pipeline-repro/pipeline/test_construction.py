"""Stage 2: Execution-grounded Test Construction (paper Sec 3.2).

Paper: map the task statement's requirements to test inputs; expected
outputs come from executing the reference, not from hand-authored oracles.
Simplification: inputs are hand-picked per function (the paper's agent
invents them by exploring a whole repo; we don't have a repo to explore, so
we author representative + edge-case inputs ourselves). What we DO keep
faithful: this module only picks *inputs* -- the expected outputs are
determined later, in execution_consistency.py, by actually running the
reference (paper explicitly separates "test construction" from "execution
consistency check", and we keep that separation because it's the only way
to detect the flaky function honestly, see that module's docstring).
"""

INPUTS = {
    "parse_csv_line": [
        "a,b,c",
        '"hello, world",b,c',
        "",
        "only_one_field",
        '"a","b,c","d"',
    ],
    "flatten": [
        [1, [2, 3], [4, [5, 6]]],
        [],
        [1, 2, 3],
        [[1, [2, [3, [4]]]]],
    ],
    "dedupe_preserve_order": [
        [1, 2, 2, 3, 1, 4],
        [],
        ["a", "b", "a", "c", "b"],
        [1, 1, 1],
    ],
    "most_common": [
        [1, 2, 2, 3],
        ["x", "y", "x", "y", "x"],
        [7],
        [1, 1, 2, 2],  # tie -> first-encountered (1)
    ],
    "unique_tags": [
        ["a", "b", "a", "c"],
        [1, 2, 3, 1],
        [],
    ],
}


def get_inputs(task_name: str) -> list:
    return INPUTS[task_name]
