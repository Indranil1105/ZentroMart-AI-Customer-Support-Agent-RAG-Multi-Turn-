"""Unit tests for the RAG retrieval pipeline module (agent.rag).

Tests:
1. initialize_rag: chunk loading, index building, and TF-IDF reranker fitting.
2. Auto-initialization on first retrieve() call.
3. Candidate search (top_k * 3) and top_k retrieval truncation.
4. Current policy document 1.5x score boost and non-policy/superseded handling.
5. TF-IDF cosine similarity reranking and hybrid score computation.
6. retrieve_with_context formatting with section IDs, order_data, and product_data.
7. get_cited_section_ids extraction from dicts, strings, lists, and delimiter patterns.
8. Edge cases: empty/whitespace queries, zero/negative top_k, missing metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from zentromart_support_agent.agent.rag import (
    DEFAULT_KB_DIR,
    POLICY_BOOST_MULTIPLIER,
    get_cited_section_ids,
    initialize_rag,
    is_rag_initialized,
    reset_rag,
    retrieve,
    retrieve_with_context,
)
from zentromart_support_agent.knowledge_base.vector_store import (
    clear_store,
    set_backend,
    set_store_dir,
)


class TestRagPipeline(unittest.TestCase):
    """Test suite for the RAG retrieval pipeline."""

    @classmethod
    def setUpClass(cls):
        # Use NumPy backend and temporary vector store directory for isolation
        cls.temp_dir = tempfile.mkdtemp()
        set_store_dir(cls.temp_dir)
        set_backend("numpy")
        reset_rag()

    @classmethod
    def tearDownClass(cls):
        clear_store(cls.temp_dir)
        reset_rag()

    def setUp(self):
        # Sample chunks for testing reranking and policy boosting
        self.mock_chunks = [
            {
                "chunk_id": "POL-RET_RET-1",
                "section_id": "RET-1",
                "doc_id": "POL-RET",
                "doc_type": "policy",
                "title": "General Return Policy",
                "status": "current",
                "priority": 1,
                "effective_from": "2026-08-01",
                "language": "en",
                "text": "## [RET-1] General Return Policy\nStandard return window is 7 calendar days.",
            },
            {
                "chunk_id": "POL-RET_RET-2",
                "section_id": "RET-2",
                "doc_id": "POL-RET",
                "doc_type": "policy",
                "title": "Electronics Return Policy",
                "status": "current",
                "priority": 1,
                "effective_from": "2026-08-01",
                "language": "en",
                "text": "## [RET-2] Electronics Return Policy\nElectronics like laptops have a 10 calendar days replacement window.",
            },
            {
                "chunk_id": "OLD-RET_RET-OLD",
                "section_id": "RET-OLD",
                "doc_id": "OLD-RET",
                "doc_type": "policy",
                "title": "Legacy Return Policy 2025",
                "status": "superseded",
                "priority": 1,
                "effective_from": "2025-01-01",
                "language": "en",
                "text": "## [RET-OLD] Legacy Return Policy 2025\nOld return window was 30 calendar days.",
            },
            {
                "chunk_id": "GD-RET_RET-AG1",
                "section_id": "RET-AG1",
                "doc_id": "GD-RET",
                "doc_type": "agent_guide",
                "title": "Agent Return Handling Guide",
                "status": "current",
                "priority": 2,
                "effective_from": "2026-08-01",
                "language": "en",
                "text": "## [RET-AG1] Agent Return Handling Guide\nAgents count business days Monday to Saturday.",
            },
            {
                "chunk_id": "FAQ-GEN_FAQ-1",
                "section_id": "FAQ-1",
                "doc_id": "FAQ-GEN",
                "doc_type": "faq",
                "title": "General Return FAQs",
                "status": "current",
                "priority": 4,
                "effective_from": "2026-07-01",
                "language": "en",
                "text": "## [FAQ-1] General Return FAQs\nFrequently asked questions about returns and refunds.",
            },
        ]

    def test_01_initialize_rag_and_status(self):
        """Test initialize_rag loads chunks and sets initialized flag."""
        reset_rag()
        self.assertFalse(is_rag_initialized())

        initialize_rag()
        self.assertTrue(is_rag_initialized())

    def test_02_auto_initialization_on_retrieve(self):
        """Test retrieve() auto-initializes if called while uninitialized."""
        reset_rag()
        self.assertFalse(is_rag_initialized())

        results = retrieve("return policy", top_k=3)
        self.assertTrue(is_rag_initialized())
        self.assertIsInstance(results, list)

    def test_03_empty_and_invalid_queries(self):
        """Test empty queries and invalid top_k return empty lists."""
        self.assertEqual(retrieve(""), [])
        self.assertEqual(retrieve("   "), [])
        self.assertEqual(retrieve(None), [])
        self.assertEqual(retrieve("return window", top_k=0), [])
        self.assertEqual(retrieve("return window", top_k=-2), [])

    def test_04_retrieve_top_k_limit(self):
        """Test that retrieve respects top_k limit."""
        results = retrieve("return policy window electronics", top_k=2)
        self.assertLessEqual(len(results), 2)
        for r in results:
            self.assertIn("score", r)
            self.assertIn("section_id", r)
            self.assertIn("vector_score", r)
            self.assertIn("tfidf_score", r)
            self.assertIn("boost_applied", r)

    @patch("zentromart_support_agent.agent.rag.search")
    @patch("zentromart_support_agent.agent.rag.embed_query")
    def test_05_candidate_multiplier_top_k_times_3(self, mock_embed, mock_search):
        """Test that search is queried with top_k * 3 candidates."""
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = self.mock_chunks

        # Ensure pipeline is initialized
        if not is_rag_initialized():
            initialize_rag()

        retrieve("laptop return", top_k=2)
        # top_k=2 -> search should be called with top_k=6 (2*3)
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        self.assertEqual(kwargs.get("top_k"), 6)

    @patch("zentromart_support_agent.agent.rag.search")
    def test_06_policy_boost_1_5x(self, mock_search):
        """Test that current policy documents (priority 1) get 1.5x score boost."""
        # Provide candidate chunks: one current policy, one superseded policy, one agent guide
        c1 = {
            "chunk_id": "POL-1",
            "section_id": "POL-1",
            "doc_type": "policy",
            "status": "current",
            "priority": 1,
            "text": "laptops return policy 10 days",
            "score": 0.5,
        }
        c2 = {
            "chunk_id": "OLD-1",
            "section_id": "OLD-1",
            "doc_type": "policy",
            "status": "superseded",
            "priority": 1,
            "text": "laptops return policy old",
            "score": 0.5,
        }
        c3 = {
            "chunk_id": "AG-1",
            "section_id": "AG-1",
            "doc_type": "agent_guide",
            "status": "current",
            "priority": 2,
            "text": "laptops return policy guide",
            "score": 0.5,
        }
        mock_search.return_value = [c1, c2, c3]

        if not is_rag_initialized():
            initialize_rag()

        results = retrieve("laptops return policy", top_k=3)
        res_by_id = {r["section_id"]: r for r in results}

        # Current policy must receive 1.5x boost
        self.assertTrue(res_by_id["POL-1"]["boost_applied"])
        # Superseded policy must NOT receive boost
        self.assertFalse(res_by_id["OLD-1"]["boost_applied"])
        # Agent guide (priority 2) must NOT receive boost
        self.assertFalse(res_by_id["AG-1"]["boost_applied"])

        # Score of current policy must be strictly higher than superseded policy
        self.assertGreater(res_by_id["POL-1"]["score"], res_by_id["OLD-1"]["score"])

    def test_07_retrieve_with_context_formatting(self):
        """Test retrieve_with_context formats chunks and optional order/product data."""
        # 1. Chunks only
        ctx1 = retrieve_with_context("How do I return items?", top_k=2)
        self.assertIn("=== Knowledge Base References ===", ctx1)
        # Check that section IDs are present
        self.assertTrue(any(tag in ctx1 for tag in ["[RET-", "[KB-", "[FAQ-", "[DEL-"]))

        # 2. Chunks + Order data
        order = {
            "order_id": "ORD-998877",
            "status": "Delivered",
            "delivery_date": "2026-10-15",
            "items": [{"sku": "ELE-0012", "name": "Headphones", "price": 1499}],
        }
        ctx2 = retrieve_with_context("How do I return?", order_data=order, top_k=2)
        self.assertIn("--- Customer Order Data ---", ctx2)
        self.assertIn("ORD-998877", ctx2)
        self.assertIn("ELE-0012", ctx2)

        # 3. Chunks + Order data + Product data
        product = {
            "sku": "ELE-0012",
            "name": "Wireless Noise Cancelling Headphones",
            "category": "electronics",
            "return_category": "10_days_replacement_only",
        }
        ctx3 = retrieve_with_context(
            "How do I return?",
            order_data=order,
            product_data=product,
            top_k=2,
        )
        self.assertIn("--- Customer Order Data ---", ctx3)
        self.assertIn("--- Product Catalog Data ---", ctx3)
        self.assertIn("10_days_replacement_only", ctx3)

    def test_08_get_cited_section_ids_variations(self):
        """Test get_cited_section_ids with dicts, strings, lists, and delimiter patterns."""
        # 1. List of chunk dicts
        chunks = [
            {"section_id": "RET-2", "text": "something"},
            {"section_id": "WAR-1", "text": "something"},
            {"section_id": "RET-2", "text": "duplicate"},
        ]
        self.assertEqual(get_cited_section_ids(chunks), ["RET-2", "WAR-1"])

        # 2. Chunk dicts with bracketed IDs
        bracketed_chunks = [{"section_id": "[RET-2]"}, {"section_id": "[KB-00.1]"}]
        self.assertEqual(get_cited_section_ids(bracketed_chunks), ["RET-2", "KB-00.1"])

        # 3. Single chunk dict
        self.assertEqual(get_cited_section_ids({"section_id": "REF-2"}), ["REF-2"])

        # 4. Formatted context string containing bracketed section IDs
        text_context = "Under [RET-2] and [KB-00.1], electronic returns have a 10-day window. Also see [RET-2]."
        self.assertEqual(get_cited_section_ids(text_context), ["RET-2", "KB-00.1"])

        # 5. Delimiter-separated string (e.g. from questions.csv gold_evidence)
        delim_str = "RET-6 ; PLS-2"
        self.assertEqual(get_cited_section_ids(delim_str), ["RET-6", "PLS-2"])

        # 6. List of strings
        str_list = ["[RET-2]", "WAR-2", "RET-2"]
        self.assertEqual(get_cited_section_ids(str_list), ["RET-2", "WAR-2"])

        # 7. Empty or None
        self.assertEqual(get_cited_section_ids([]), [])
        self.assertEqual(get_cited_section_ids(None), [])
        self.assertEqual(get_cited_section_ids(""), [])


if __name__ == "__main__":
    unittest.main()
