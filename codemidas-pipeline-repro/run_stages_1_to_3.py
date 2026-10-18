import sys
sys.path.insert(0, ".")

import toylib
from pipeline import task_design, test_construction, execution_consistency

FUNCS = ["parse_csv_line", "flatten", "dedupe_preserve_order", "most_common", "unique_tags"]

for name in FUNCS:
    print(f"\n=== {name} ===")
    task = task_design.extract_task(toylib, name)
    inputs = test_construction.get_inputs(name)
    results = execution_consistency.check_consistency(task, inputs)
    for r in results:
        print(f"  input={r['input']!r}")
        print(f"    stub:  {r['stub_outputs']}")
        print(f"    ref:   {r['reference_outputs']}")
        print(f"    -> {r['verdict']}")
