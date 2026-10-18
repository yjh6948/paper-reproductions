from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryObject:
    content: str
    created_at: float
    last_accessed_at: float
    importance: Optional[float] = None
    source: str = "observation"
