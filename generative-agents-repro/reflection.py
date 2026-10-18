from typing import List

from memory import MemoryObject


def should_reflect(unreflected: List[MemoryObject], threshold: float = 40.0) -> bool:
    total_importance = sum(m.importance or 0.0 for m in unreflected)
    return total_importance >= threshold
