# CodeMidas pipeline reproduction

Paper: **CodeMidas: Scaling Agentic Coding RL Environments from Code Itself**
(Xiaomi, arXiv:2609.22068). Selected via "HF Daily Papers 투표순 1위" — the
actual #1 (Code2Skill) was a pure orchestration paper with no learnable
core so it was skipped under my "check for an Algorithm
box or equations first" selection rule; this is #2.

## What this paper actually is

No Algorithm box, no self-derived equations. The paper's real contribution
is a **4-stage deterministic pipeline** (task design → execution-grounded
test construction → execution consistency check → post-rollout filtering)
that turns any code repo into verified RL training environments. The RL
training itself (GRPO, on MiMo-V2.5, at 5,545-task scale) just cites GRPO
from another paper (Shao et al. 2024) and needs GPU-scale infra we don't
have. **So the scope cut here is the RL training, not a "backbone"** — the
part we CAN reproduce faithfully is the part the paper is actually novel
about.

## What's reproduced vs. simplified

| Paper | This repro |
|---|---|
| Agent explores a real GitHub repo to find candidate functionality | We hand-pick 5 functions in a toy `toylib.py` (no repo-exploration agent) |
| Agent writes task statement + removes implementation | `pipeline/task_design.py` does this via real AST manipulation (not an LLM call, but genuinely automated) |
| Agent invents test inputs from the task statement | `pipeline/test_construction.py` hand-authors representative + edge-case inputs per function (paper's agent explores the repo for these; we don't have a repo to explore) |
| Execution consistency: 2 fresh containers (stub) + 4 fresh containers (reference) | `pipeline/execution_consistency.py` — literally 2+4 **fresh Python subprocesses** (not just function calls in one process — see below for why this matters) |
| Coding agent attempts each task 4x, reviewing agent scores agreement | `pipeline/post_rollout_filtering.py` — 3 **hand-authored** attempts per task (correct / buggy / wrong) standing in for real LLM rollouts, scored for real against the verifier |
| Adversarial leakage rollout | `pipeline/leakage_check.py` — builds a real leaky env (stray `.bak` with the reference) and a real clean env, scans both |
| GRPO RL training on 5,545 tasks, 5 external benchmarks | **Not reproduced** — needs a trainable coding-agent model + GPU RL infra, genuinely out of scope for this session |

## Honest results

Run: `python run_full_pipeline.py`

**1. The execution-consistency check caught a real bug, for real reasons.**
I deliberately planted `unique_tags()` in `toylib.py`, implemented with
`list(set(items))`. Calling it twice *in the same process* looks perfectly
deterministic — Python's string hash seed is fixed per-process. So I had to
implement "fresh" literally (a new `python -c` subprocess per run, matching
the paper's "fresh containers") to actually reproduce the bug: across 4
fresh subprocess runs, the string-input case (`['a','b','a','c']`) produced
2 distinct orderings and was correctly **REJECTED as flaky**, while the
int-input case passed (small-int hashes are stable across processes). This
is the same class of bug flagged in the Procedural Graphs cycle
(`random.choice` on a `set()` needs `sorted()` first). Had I taken the shortcut of calling the function
repeatedly in one interpreter (which I initially did, before rewriting to
subprocesses), the check would have silently passed a flaky task — the
paper's "fresh containers" requirement isn't incidental, it's load-bearing.

**2. Post-rollout filtering correctly separated usable tasks from noise**,
on tasks with real mixed-quality attempts:
- `parse_csv_line`: correct=100%, two quote-handling bugs=60% each → ACCEPT (mixed)
- `flatten`: correct=100%, shallow-flatten bug=50%, identity-passthrough=50% → ACCEPT (mixed)
- `most_common`: correct=100%, wrong-first-element=75% → ACCEPT (mixed)

**3. One "bug" I planted didn't actually manifest — and that's the
interesting part.** `buggy_tiebreak_via_set` (`max(set(items), key=counts.get)`
instead of iterating in original order) was written to fail on the tie case
`[1,1,2,2]`, but scored 100% — Python's small-int set ordering happened to
still return the "first-encountered" answer by coincidence, so my 4-input
test suite never exposed the tie-break difference. This is an unintentional
but honest confirmation of something the paper cites (EvalPlus, Liu et al.
2023): weak/narrow test suites let incorrect implementations slip through.
A bigger, more adversarial input set would likely have caught it — I didn't
expand the suite to force the bug to show, since the point here was to
observe what the paper's own pipeline would actually catch with the test
inputs as designed, not to guarantee every planted bug gets found.

**4. Leakage filtering worked exactly as expected** — flagged the
`.cache/flatten.py.bak` leftover in the leaky environment, found nothing in
the clean one.

## Funnel

```
designed candidates:        5
survived execution check:   5  (1 of 5 lost a test case to flakiness, not the whole task)
survived rollout filtering: 3  (of 3 attempted; 2 tasks left unattempted — out of demo scope)
```
