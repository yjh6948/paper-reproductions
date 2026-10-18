"""Paper2Agent-style MCP server for the vec2vec toy reproduction.

Mirrors Paper2Agent's tool-extraction pattern (arXiv:2509.06917): a tutorial
script (train.py) is converted into a single, well-annotated, parameterized
MCP tool with defaults matching the tutorial, plus a resource carrying the
paper's reference numbers and a prompt encoding the ablation workflow.

Source: research/vec2vec-repro/train.py (this session's own reproduction of
Jha et al., "Harnessing the Universal Geometry of Embeddings", arXiv:2505.12540).
"""

import contextlib
import io
from typing import Annotated

from mcp.server.mcpserver import MCPServer

import train as _train

mcp = MCPServer(name="vec2vec-repro")


@mcp.tool()
def train_and_evaluate_vec2vec(
    seed: Annotated[int, "Random seed for data generation and model init"] = 0,
    lambda_vsp: Annotated[float, "Weight of the vector-space-preservation loss; 0.0 disables it"] = _train.LAMBDA_VSP,
    lambda_cc: Annotated[float, "Weight of the cycle-consistency loss"] = _train.LAMBDA_CC,
    use_adv: Annotated[bool, "Whether to include the adversarial (GAN) loss term"] = True,
    n_eval: Annotated[int, "Number of held-out paired samples for evaluation"] = 500,
) -> dict:
    """Train the vec2vec toy model (adapters + shared backbone GAN translating
    between two synthetic embedding spaces) and evaluate translation quality
    on held-out pairs. Reproduces train.py's train()+evaluate() tutorial.

    Returns cosine similarity, top-1 retrieval accuracy, and mean rank of the
    F1 (space1->space2) translation against the target embedding, matching
    the metrics reported in README.md's results table.
    """
    # train.py's train()/evaluate() print progress to stdout, which corrupts
    # the JSON-RPC framing on the stdio MCP transport (found by
    # tests/test_mcp_server.py's real subprocess run). Swallow it here
    # rather than editing the already-validated train.py.
    with contextlib.redirect_stdout(io.StringIO()):
        model, enc = _train.train(seed=seed, lambda_vsp=lambda_vsp, lambda_cc=lambda_cc, use_adv=use_adv)
        results = _train.evaluate(model, enc, n_eval=n_eval, seed=seed + 999)
    return {
        "cosine_to_target": round(results["F1(u)->v: cos"], 4),
        "top1_accuracy": round(results["F1(u)->v: top1"], 4),
        "mean_rank": round(results["F1(u)->v: rank"], 4),
        "n_eval": n_eval,
        "config": {"seed": seed, "lambda_vsp": lambda_vsp, "lambda_cc": lambda_cc, "use_adv": use_adv},
    }


@mcp.resource("paper://vec2vec/reference-results")
def reference_results() -> str:
    """This session's own reproduction reference table (README.md), used by
    the test-verifier-improver step to check the MCP tool's fidelity."""
    return (
        "config,cosine_to_target,top1_accuracy,mean_rank\n"
        "full(adv+rec+cc+vsp),0.778,0.030,39.4\n"
        "no_vsp,0.124,0.006,218.2\n"
        "no_adversarial,0.020,0.004,237.0\n"
    )


@mcp.prompt()
def reproduce_ablation_table() -> str:
    """Guides an agent through reproducing the paper's ablation-style table
    (README.md, mirroring the original paper's Table 6) using the MCP tool."""
    return (
        "Call train_and_evaluate_vec2vec three times with seed=0 and n_eval=500:\n"
        "1. defaults (full model) -> expect cosine~0.78, top1~0.03, rank~39\n"
        "2. lambda_vsp=0.0 -> expect cosine~0.12, top1~0.006, rank~218\n"
        "3. use_adv=False -> expect cosine~0.02, top1~0.004, rank~237\n"
        "Compare each result against paper://vec2vec/reference-results."
    )


if __name__ == "__main__":
    mcp.run()
