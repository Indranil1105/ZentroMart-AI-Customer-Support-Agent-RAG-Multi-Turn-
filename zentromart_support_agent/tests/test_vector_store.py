"""Unit tests for the knowledge base vector store module.

Tests ChromaDB persistent indexing and search, NumPy fallback with pickle storage,
metadata preservation, document type and status filtering, score calculation,
and interface equivalence across both backends.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from zentromart_support_agent.knowledge_base.embeddings import (
    embed_query,
    embed_texts,
)
from zentromart_support_agent.knowledge_base.vector_store import (
    STORE_DIR,
    build_index,
    clear_store,
    get_backend,
    get_store_dir,
    is_chroma_available,
    is_index_built,
    search,
    set_backend,
    set_store_dir,
)


class BaseVectorStoreTest(unittest.TestCase):
    """Base test class providing temporary directories and mock chunks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        set_store_dir(self.temp_dir)

        self.sample_chunks = [
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
                "text": "Standard return window is 7 calendar days from delivery date for non-electronic items.",
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
                "text": "Electronics such as laptops and mobile phones have a 10 calendar days return window.",
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
                "text": "Agents must calculate business days from Mon-Sat excluding national warehouse holidays.",
            },
            {
                "chunk_id": "FAQ-GEN_FAQ-1",
                "section_id": "FAQ-1",
                "doc_id": "FAQ-GEN",
                "doc_type": "faq",
                "title": "Delivery FAQs",
                "status": "current",
                "priority": 4,
                "effective_from": "2026-07-01",
                "language": "en",
                "text": "Deliveries typically take 2 to 4 business days to major metropolitan cities.",
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
                "text": "Old return policy allowed 30 calendar days for all items without verification.",
            },
        ]

    def tearDown(self):
        clear_store(self.temp_dir)
        set_store_dir(STORE_DIR)
        set_backend("auto")
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestVectorStoreChroma(BaseVectorStoreTest):
    """Tests for ChromaDB vector store implementation."""

    def setUp(self):
        super().setUp()
        set_backend("chroma")

    def test_store_dir_constant(self):
        """Verify the exact required STORE_DIR path constant."""
        expected = os.path.join(
            r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\zentromart_support_agent",
            "chroma_db",
        )
        self.assertEqual(STORE_DIR, expected)

    def test_is_index_built_initial_state(self):
        """Verify is_index_built returns False when store is empty."""
        self.assertFalse(is_index_built())

    def test_build_and_search_basic(self):
        """Test building Chroma index and performing semantic search."""
        build_index(self.sample_chunks)
        self.assertTrue(is_index_built())

        hits = search("how many days to return a laptop?", top_k=3)
        self.assertGreater(len(hits), 0)
        top_hit = hits[0]

        # Electronics return section [RET-2] should be most relevant
        self.assertEqual(top_hit["section_id"], "RET-2")
        self.assertEqual(top_hit["doc_id"], "POL-RET")
        self.assertIn("score", top_hit)
        self.assertIsInstance(top_hit["score"], float)
        self.assertGreater(top_hit["score"], 0.3)

    def test_metadata_fields_preserved(self):
        """Verify all required metadata fields are preserved in search results."""
        build_index(self.sample_chunks)
        hits = search("laptop returns", top_k=1)
        self.assertEqual(len(hits), 1)

        hit = hits[0]
        required_keys = (
            "chunk_id",
            "section_id",
            "doc_id",
            "doc_type",
            "status",
            "priority",
            "effective_from",
            "text",
            "score",
        )
        for key in required_keys:
            self.assertIn(key, hit, f"Missing required key: {key}")

        self.assertEqual(hit["section_id"], "RET-2")
        self.assertEqual(hit["doc_type"], "policy")
        self.assertEqual(hit["status"], "current")
        self.assertEqual(hit["priority"], 1)
        self.assertEqual(hit["effective_from"], "2026-08-01")

    def test_filter_by_doc_type(self):
        """Test filtering results by document type."""
        build_index(self.sample_chunks)

        # Search with agent_guide filter
        hits = search("return", top_k=5, filters={"doc_type": "agent_guide"})
        self.assertGreater(len(hits), 0)
        for h in hits:
            self.assertEqual(h["doc_type"], "agent_guide")
            self.assertEqual(h["section_id"], "RET-AG1")

    def test_filter_by_status(self):
        """Test filtering results by document status."""
        build_index(self.sample_chunks)

        # Search with superseded filter
        hits = search("return policy", top_k=5, filters={"status": "superseded"})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["section_id"], "RET-OLD")
        self.assertEqual(hits[0]["status"], "superseded")

    def test_filter_combined_doc_type_and_status(self):
        """Test filtering by both doc_type and status simultaneously."""
        build_index(self.sample_chunks)

        hits = search(
            "return policy",
            top_k=5,
            filters={"doc_type": "policy", "status": "current"},
        )
        self.assertEqual(len(hits), 2)
        for h in hits:
            self.assertEqual(h["doc_type"], "policy")
            self.assertEqual(h["status"], "current")

    def test_filter_multi_value_doc_type(self):
        """Test filtering by multiple allowed doc_types using a list."""
        build_index(self.sample_chunks)

        hits = search(
            "return",
            top_k=5,
            filters={"doc_type": ["agent_guide", "faq"]},
        )
        self.assertGreater(len(hits), 0)
        for h in hits:
            self.assertIn(h["doc_type"], ["agent_guide", "faq"])

    def test_top_k_limit(self):
        """Verify top_k bounds the returned result count."""
        build_index(self.sample_chunks)

        hits_1 = search("return", top_k=1)
        self.assertEqual(len(hits_1), 1)

        hits_2 = search("return", top_k=2)
        self.assertEqual(len(hits_2), 2)

    def test_search_not_built_raises_runtime_error(self):
        """Verify searching an unbuilt index raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            search("any query", top_k=5)

    def test_empty_query_returns_empty_list(self):
        """Verify empty or blank query returns an empty list without error."""
        build_index(self.sample_chunks)
        self.assertEqual(search("", top_k=5), [])
        self.assertEqual(search("   ", top_k=5), [])

    def test_zero_or_negative_top_k(self):
        """Verify top_k <= 0 returns empty list."""
        build_index(self.sample_chunks)
        self.assertEqual(search("returns", top_k=0), [])
        self.assertEqual(search("returns", top_k=-2), [])

    def test_rebuild_overwrites(self):
        """Verify rebuilding index clears older contents."""
        build_index(self.sample_chunks)
        self.assertEqual(len(search("return", top_k=10)), 5)

        # Rebuild with single chunk
        new_chunk = [self.sample_chunks[0]]
        build_index(new_chunk)
        self.assertEqual(len(search("return", top_k=10)), 1)


class TestVectorStoreNumpyFallback(BaseVectorStoreTest):
    """Tests for NumPy cosine similarity fallback with pickle storage."""

    def setUp(self):
        super().setUp()
        set_backend("numpy")

    def test_numpy_build_and_is_built(self):
        """Verify NumPy index builds and sets is_index_built to True."""
        self.assertFalse(is_index_built())
        build_index(self.sample_chunks)
        self.assertTrue(is_index_built())

    def test_numpy_search_basic(self):
        """Test NumPy cosine search finds the most relevant chunk."""
        build_index(self.sample_chunks)
        hits = search("how many days to return a laptop?", top_k=3)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0]["section_id"], "RET-2")
        self.assertIn("score", hits[0])
        self.assertGreater(hits[0]["score"], 0.3)

    def test_numpy_filters(self):
        """Test doc_type and status filtering in NumPy fallback."""
        build_index(self.sample_chunks)

        hits = search(
            "return",
            top_k=5,
            filters={"doc_type": "policy", "status": "current"},
        )
        self.assertEqual(len(hits), 2)
        for h in hits:
            self.assertEqual(h["doc_type"], "policy")
            self.assertEqual(h["status"], "current")

    def test_numpy_search_not_built_raises(self):
        """Verify search raises RuntimeError if index not built."""
        with self.assertRaises(RuntimeError):
            search("returns")


class TestInterfaceEquivalence(BaseVectorStoreTest):
    """Verifies that ChromaDB and NumPy backends adhere to identical interfaces."""

    def test_both_backends_return_identical_keys(self):
        """Verify ChromaDB and NumPy return identical dictionary keys and types."""
        query = "what is the electronics return window?"

        # Chroma search
        set_backend("chroma")
        build_index(self.sample_chunks)
        chroma_hits = search(query, top_k=2)

        # NumPy search
        set_backend("numpy")
        build_index(self.sample_chunks)
        numpy_hits = search(query, top_k=2)

        self.assertEqual(len(chroma_hits), len(numpy_hits))

        for c_hit, n_hit in zip(chroma_hits, numpy_hits):
            self.assertEqual(c_hit["section_id"], n_hit["section_id"])
            self.assertEqual(c_hit["doc_id"], n_hit["doc_id"])
            self.assertEqual(c_hit["doc_type"], n_hit["doc_type"])
            self.assertEqual(c_hit["status"], n_hit["status"])
            self.assertEqual(c_hit["priority"], n_hit["priority"])
            self.assertEqual(c_hit["effective_from"], n_hit["effective_from"])
            self.assertEqual(set(c_hit.keys()), set(n_hit.keys()))
            # Scores should match within float tolerance
            self.assertAlmostEqual(c_hit["score"], n_hit["score"], places=4)


class TestEmbeddingsModule(unittest.TestCase):
    """Tests for the knowledge_base.embeddings module."""

    def test_embed_texts_and_query(self):
        """Test batch embed_texts and single embed_query."""
        texts = ["ZentroMart returns", "Fast shipping"]
        embeddings = embed_texts(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)
        self.assertEqual(len(embeddings[1]), 384)

        query_emb = embed_query("What is the return policy?")
        self.assertEqual(len(query_emb), 384)

    def test_empty_embed_texts(self):
        """Test embedding empty list returns empty result."""
        res = embed_texts([])
        self.assertEqual(len(res), 0)


if __name__ == "__main__":
    unittest.main()
