"""Toy 'open-source codebase' that CodeMidas would mine for tasks.

Five small functions, each with a clear public interface and observable
behavior -- standing in for the "implemented functionality with public entry
points" that CodeMidas's Task Design stage looks for (paper Sec 3.1).
"""


def parse_csv_line(line: str) -> list[str]:
    """Split a single CSV line into fields, respecting double-quoted commas."""
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


def flatten(nested: list) -> list:
    """Recursively flatten an arbitrarily nested list."""
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def dedupe_preserve_order(items: list) -> list:
    """Remove duplicates, keeping the first occurrence's position."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def most_common(items: list):
    """Return the most frequent element; ties broken by first appearance."""
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


def unique_tags(items: list) -> list:
    """Return unique items -- BUGGY on purpose: uses set() so order is
    process-dependent for non-int/str-hash-stable content. This mirrors the
    Python set-ordering nondeterminism bug from the Procedural Graphs cycle
    (see project_research_pipeline memory) -- included deliberately so
    CodeMidas's execution-consistency check (Sec 3.3, 6 fresh runs) has
    something real to catch.
    """
    return list(set(items))
