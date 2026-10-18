"""Testing-agent step of the Paper2Agent-style pipeline: launches the MCP
server as a real subprocess over stdio, calls the tool through the actual
MCP protocol, and checks results against the reference table (paper's
own "validate against reported results" mechanism, Sec. 2)."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.02  # absolute tolerance on cosine/top1; results are deterministic given seed

REFERENCE = {
    "full": {"cosine_to_target": 0.778, "top1_accuracy": 0.030, "mean_rank": 39.4},
    "no_vsp": {"cosine_to_target": 0.124, "top1_accuracy": 0.006, "mean_rank": 218.2},
    "no_adv": {"cosine_to_target": 0.020, "top1_accuracy": 0.004, "mean_rank": 237.0},
}


async def run():
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "vec2vec_mcp.py")], cwd=str(ROOT)
    )
    failures = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "train_and_evaluate_vec2vec" in names, f"tool missing, got {names}"

            resources = await session.list_resources()
            assert any("reference-results" in str(r.uri) for r in resources.resources), "reference resource missing"

            cases = [
                ("full", {"seed": 0}),
                ("no_vsp", {"seed": 0, "lambda_vsp": 0.0}),
                ("no_adv", {"seed": 0, "use_adv": False}),
            ]
            for label, args in cases:
                result = await session.call_tool("train_and_evaluate_vec2vec", args)
                payload = json.loads(result.content[0].text)
                ref = REFERENCE[label]
                for key in ("cosine_to_target", "top1_accuracy"):
                    got, want = payload[key], ref[key]
                    status = "OK" if abs(got - want) <= TOL else "MISMATCH"
                    print(f"[{label}] {key}: got={got} want={want} tol={TOL} -> {status}")
                    if status == "MISMATCH":
                        failures.append((label, key, got, want))

    if failures:
        print("\nFAILURES:", failures)
        raise SystemExit(1)
    print("\nAll MCP tool outputs match the reference table within tolerance.")


if __name__ == "__main__":
    asyncio.run(run())
