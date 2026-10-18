"""Stage 1: Task Design and Codebase Adaptation (paper Sec 3.1).

Paper: an agent inspects the codebase, picks functionality with a public
entry point, removes the core implementation, and produces a task statement.
Simplification: we skip the "agent explores a whole repo" part (no repo to
explore) and go straight to "given one target function, remove & refine" --
this is the part of Fig. 1 module 1 we can reproduce deterministically.
"""
import ast
import inspect
import textwrap


def extract_task(module, func_name: str) -> dict:
    """Given a live module and a function name, produce a CodeMidas-style task."""
    func = getattr(module, func_name)
    source = textwrap.dedent(inspect.getsource(func))
    doc = inspect.getdoc(func) or ""
    signature = str(inspect.signature(func))

    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # "Remove & refine": replace the body with a stub that fails loudly.
    stub_body = [ast.Raise(
        exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                      args=[ast.Constant(value=f"implement {func_name}")],
                      keywords=[]),
        cause=None,
    )]
    stub_def = ast.FunctionDef(
        name=func_def.name,
        args=func_def.args,
        body=stub_body,
        decorator_list=[],
        returns=func_def.returns,
    )
    ast.fix_missing_locations(stub_def)
    stub_source = ast.unparse(stub_def)

    return {
        "name": func_name,
        "statement": f"Implement `{func_name}{signature}`.\n\n{doc}",
        "reference_source": source,
        "stub_source": stub_source,
        "reference_func": func,
    }
