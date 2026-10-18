import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import MemoryObject
from retrieval import recency_score, relevance_score, retrieve_top_k


class TestRecencyScore(unittest.TestCase):
    def test_decays_with_time(self):
        mem = MemoryObject(content="x", created_at=0, last_accessed_at=0, importance=5)
        self.assertEqual(recency_score(mem, now=0, decay=0.99), 1.0)
        score_10h = recency_score(mem, now=10, decay=0.99)
        score_20h = recency_score(mem, now=20, decay=0.99)
        self.assertTrue(0 < score_20h < score_10h < 1.0)

    def test_never_negative_time(self):
        mem = MemoryObject(content="x", created_at=5, last_accessed_at=5, importance=5)
        self.assertEqual(recency_score(mem, now=3, decay=0.99), 1.0)


class TestRelevanceScore(unittest.TestCase):
    def test_identical_text_is_1(self):
        mem = MemoryObject(content="the cat sat on the mat", created_at=0, last_accessed_at=0, importance=1)
        self.assertAlmostEqual(relevance_score(mem, "the cat sat on the mat"), 1.0)

    def test_unrelated_text_is_0(self):
        mem = MemoryObject(
            content="Alex refilled the printer with paper", created_at=0, last_accessed_at=0, importance=1
        )
        self.assertEqual(relevance_score(mem, "quantum entanglement in superconductors"), 0.0)


class TestRetrieveTopK(unittest.TestCase):
    def test_prefers_high_importance_and_relevance(self):
        mundane = MemoryObject(content="Alex brushed their teeth", created_at=0, last_accessed_at=0, importance=1.0)
        poignant_relevant = MemoryObject(
            content="Alex's apartment building is being sold, tenants may need to move",
            created_at=0,
            last_accessed_at=0,
            importance=9.0,
        )
        poignant_irrelevant = MemoryObject(
            content="Alex's manager announced a client is cancelling the contract",
            created_at=0,
            last_accessed_at=0,
            importance=9.0,
        )

        memories = [mundane, poignant_relevant, poignant_irrelevant]
        query = "housing and apartment worries"

        top = retrieve_top_k(memories, query, now=0, k=1)
        self.assertIs(top[0][0], poignant_relevant)

    def test_returns_requested_count(self):
        memories = [
            MemoryObject(content=f"event {i}", created_at=i, last_accessed_at=i, importance=float(i % 10 + 1))
            for i in range(5)
        ]
        top = retrieve_top_k(memories, "event", now=5, k=3)
        self.assertEqual(len(top), 3)


if __name__ == "__main__":
    unittest.main()
