import sys
sys.path.insert(0, ".")

import toylib
from pipeline import task_design, test_construction, execution_consistency, post_rollout_filtering, leakage_check

FUNCS = ["parse_csv_line", "flatten", "dedupe_preserve_order", "most_common", "unique_tags"]
ATTEMPTED = {"parse_csv_line", "flatten", "most_common"}  # tasks with hand-authored rollout attempts

print("=" * 70)
print("STAGE 1-3: Task Design -> Test Construction -> Execution Consistency")
print("=" * 70)

funnel = {"designed": 0, "consistency_accepted": 0, "rollout_accepted": 0}
accepted_tasks = {}

for name in FUNCS:
    funnel["designed"] += 1
    task = task_design.extract_task(toylib, name)
    inputs = test_construction.get_inputs(name)
    results = execution_consistency.check_consistency(task, inputs)

    accepted_cases = [r for r in results if r["verdict"] == "ACCEPT"]
    rejected_cases = [r for r in results if r["verdict"] != "ACCEPT"]

    print(f"\n--- {name} ---")
    print(f"  {len(accepted_cases)}/{len(results)} test cases accepted by execution-consistency check")
    for r in rejected_cases:
        print(f"    REJECTED case input={r['input']!r}: {r['verdict']}")

    if not accepted_cases:
        print(f"  => TASK REJECTED (no consistent test cases survived)")
        continue

    funnel["consistency_accepted"] += 1
    accepted_tasks[name] = {"task": task, "cases": accepted_cases}

print("\n" + "=" * 70)
print("STAGE 4: Post-rollout Environment Filtering")
print("=" * 70)

for name, data in accepted_tasks.items():
    if name not in ATTEMPTED:
        print(f"\n--- {name}: no hand-authored rollout attempts (out of demo scope) ---")
        continue
    print(f"\n--- {name} ---")
    scored = post_rollout_filtering.score_attempts(name, data["cases"])
    for label, info in scored["per_attempt"].items():
        print(f"  attempt '{label}': pass_rate={info['pass_rate']:.2f}  results={info['results']}")
    print(f"  => {scored['verdict']}")
    if scored["verdict"].startswith("ACCEPT"):
        funnel["rollout_accepted"] += 1

print("\n" + "=" * 70)
print("Leakage filtering demo (on 'flatten' task's environment)")
print("=" * 70)
leak_result = leakage_check.demo(accepted_tasks["flatten"]["task"])
print(f"  leaky env  -> flagged files: {leak_result['leaky_env_flagged']} => {leak_result['verdict_leaky']}")
print(f"  clean env  -> flagged files: {leak_result['clean_env_flagged']} => {leak_result['verdict_clean']}")

print("\n" + "=" * 70)
print("FUNNEL SUMMARY (cf. paper Fig 1's retained-task pyramid)")
print("=" * 70)
print(f"  designed candidates:        {funnel['designed']}")
print(f"  survived execution check:   {funnel['consistency_accepted']}")
print(f"  survived rollout filtering: {funnel['rollout_accepted']} (of {len(ATTEMPTED)} attempted)")
