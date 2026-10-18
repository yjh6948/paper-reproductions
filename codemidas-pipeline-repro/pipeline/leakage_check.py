"""Leakage filtering (paper Sec 3.4, "Leakage filtering").

Paper: after removing the core implementation, an adversarial rollout
searches the solver-visible environment for compiled artifacts, caches, or
files left behind by construction agents that could let a solver recover
the answer without doing the work -- e.g. a stale __pycache__/.pyc, a cached
copy in a build output directory, a leftover .orig backup file.

We reproduce this literally rather than simulate it: write BOTH a "leaky"
environment (task dir + an accidentally-retained .bak file containing the
reference implementation) and a "clean" one (properly stripped, matching
the paper's environment-preparation cleanup in Sec 3.3), then run the same
scanner against both and confirm it flags exactly the leaky one.
"""
import os
import shutil
import tempfile


def _scan_for_leakage(env_dir: str, func_name: str, reference_source: str) -> list[str]:
    """Naive but real scanner: does any file in the solver's environment,
    OTHER than the stub itself, contain the real implementation body?"""
    hits = []
    ref_body_marker = reference_source.strip().splitlines()[-1].strip()  # a distinctive line
    for root, _dirs, files in os.walk(env_dir):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                with open(path, "r", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue
            if ref_body_marker in content and "NotImplementedError" not in content:
                hits.append(path)
    return hits


def demo(task: dict) -> dict:
    """Builds a leaky env and a clean env for the same task, scans both."""
    leaky_dir = tempfile.mkdtemp(prefix="codemidas_leaky_")
    clean_dir = tempfile.mkdtemp(prefix="codemidas_clean_")
    try:
        # Both environments: solver-visible stub.
        for d in (leaky_dir, clean_dir):
            with open(os.path.join(d, f"{task['name']}.py"), "w") as f:
                f.write(task["stub_source"])

        # Leaky env ALSO has a stray build-cache-style file with the real
        # implementation (paper's Fig 1 example: dist/serialize.js,
        # .cache/serialize.ts) -- environment-prep cleanup should have
        # removed this but here we simulate a cleanup miss.
        os.makedirs(os.path.join(leaky_dir, ".cache"), exist_ok=True)
        with open(os.path.join(leaky_dir, ".cache", f"{task['name']}.py.bak"), "w") as f:
            f.write(task["reference_source"])

        leaky_hits = _scan_for_leakage(leaky_dir, task["name"], task["reference_source"])
        clean_hits = _scan_for_leakage(clean_dir, task["name"], task["reference_source"])

        return {
            "leaky_env_flagged": leaky_hits,
            "clean_env_flagged": clean_hits,
            "verdict_leaky": "REJECT: leakage detected" if leaky_hits else "ACCEPT (unexpected -- scanner missed it)",
            "verdict_clean": "ACCEPT: no leakage" if not clean_hits else "REJECT (unexpected -- false positive)",
        }
    finally:
        shutil.rmtree(leaky_dir, ignore_errors=True)
        shutil.rmtree(clean_dir, ignore_errors=True)
