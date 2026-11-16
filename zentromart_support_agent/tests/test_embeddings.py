"""Unit tests for the knowledge base embedding module.

Tests:
    - Lazy loading of SentenceTransformer("all-MiniLM-L6-v2")
    - Module-level caching of the model
    - Fallback to TF-IDF when sentence-transformers is unavailable or fails
    - embed_texts() output shapes, types, and edge cases
    - embed_query() output shapes, types, and semantic retrieval
    - Alignment between embed_texts() and embed_query() in both ST and TF-IDF modes
    - Cache management and manual dependency injection
    - End-to-end integration with Knowledge Base chunks
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from zentromart_support_agent.knowledge_base.embeddings import (
    DEFAULT_MODEL_NAME,
    _get_tfidf_vectorizer,
    clear_model_cache,
    embed_query,
    embed_texts,
    get_embedding_model,
    set_embedding_model,
    set_tfidf_vectorizer,
)


class TestEmbeddingModelLazyLoading(unittest.TestCase):
    """Test lazy loading and module caching of the embedding model."""

    def tearDown(self):
        clear_model_cache()

    def test_get_embedding_model_loads_successfully(self):
        """SentenceTransformer model should load and not be None."""
        model = get_embedding_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "encode"))

    def test_get_embedding_model_caching(self):
        """Subsequent calls must return the identical cached instance."""
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        self.assertIs(model1, model2)

    def test_clear_model_cache(self):
        """Clearing model cache should remove the cached model."""
        model1 = get_embedding_model()
        self.assertIsNotNone(model1)
        clear_model_cache()
        # After clear, cache is reset
        from zentromart_support_agent.knowledge_base import embeddings

        self.assertIsNone(embeddings._model)

    def test_sentence_transformers_import_error(self):
        """When sentence_transformers cannot be imported, return None."""
        clear_model_cache()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'sentence_transformers'")):
                model = get_embedding_model()
                self.assertIsNone(model)

    def test_sentence_transformers_initialization_exception(self):
        """When SentenceTransformer raises an exception during initialization, return None."""
        clear_model_cache()
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Corrupt weights")):
            model = get_embedding_model()
            self.assertIsNone(model)


class TestEmbedTexts(unittest.TestCase):
    """Test embed_texts with SentenceTransformer model."""

    @classmethod
    def setUpClass(cls):
        # Pre-warm model once for all test methods in this class
        get_embedding_model()

    def test_embed_texts_standard(self):
        texts = [
            "Return policy for electronics items within 7 calendar days [RET-1].",
            "Customer wallet refunds are processed immediately upon inspection [REF-2].",
        ]
        embeddings = embed_texts(texts)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_embed_texts_empty_list(self):
        """An empty list should return an empty array with shape (0, 384)."""
        embeddings = embed_texts([])
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (0, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_embed_texts_single_item(self):
        """A list with one text should return shape (1, 384)."""
        embeddings = embed_texts(["Single test document."])
        self.assertEqual(embeddings.shape, (1, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_embed_texts_with_none_and_empty_strings(self):
        """Non-string or None items in texts list should be handled gracefully."""
        texts = ["Valid document", None, "", "   "]
        embeddings = embed_texts(texts)
        self.assertEqual(embeddings.shape, (4, 384))
        self.assertEqual(embeddings.dtype, np.float32)


class TestEmbedQuery(unittest.TestCase):
    """Test embed_query with SentenceTransformer model."""

    @classmethod
    def setUpClass(cls):
        get_embedding_model()

    def test_embed_query_standard(self):
        query = "How do I return a damaged product?"
        vec = embed_query(query)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (384,))
        self.assertEqual(vec.dtype, np.float32)

    def test_embed_query_empty_string(self):
        """Empty query string should produce a valid 1D vector."""
        vec = embed_query("")
        self.assertEqual(vec.shape, (384,))
        self.assertEqual(vec.dtype, np.float32)

    def test_embed_query_none(self):
        """None query should be handled as empty string without raising an exception."""
        vec = embed_query(None)  # type: ignore
        self.assertEqual(vec.shape, (384,))
        self.assertEqual(vec.dtype, np.float32)

    def test_semantic_retrieval_ranking(self):
        """Semantically relevant document must have a higher cosine score than an irrelevant one."""
        docs = [
            "Return window for standard orders is 7 calendar days from delivery.",
            "Warehouse is closed on Diwali and Independence Day holidays.",
        ]
        doc_embeddings = embed_texts(docs)
        query_embedding = embed_query("What is the deadline for returning my order?")

        # Compute cosine similarity
        norm_docs = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norm_q = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_docs, norm_q)

        self.assertGreater(similarities[0], similarities[1])


class TestTfidfFallback(unittest.TestCase):
    """Test TF-IDF fallback behavior when sentence-transformers is unavailable or fails."""

    def setUp(self):
        clear_model_cache()
        # Inject None model to simulate sentence-transformers being unavailable
        from zentromart_support_agent.knowledge_base import embeddings

        embeddings._model = None
        embeddings._model_load_attempted = True

    def tearDown(self):
        clear_model_cache()

    @patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None)
    def test_fallback_embed_texts(self, mock_get_model):
        """embed_texts should fall back to TF-IDF when get_embedding_model returns None."""
        texts = [
            "Return policy for electronics items within 7 calendar days.",
            "Customer wallet refund is processed immediately after pickup.",
        ]
        embeddings = embed_texts(texts)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertGreater(embeddings.shape[1], 0)
        self.assertEqual(embeddings.dtype, np.float32)

    @patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None)
    def test_fallback_embed_query_aligned_with_texts(self, mock_get_model):
        """Query embedding must have identical feature dimension as prior embed_texts call."""
        corpus = [
            "Damaged item replacement within 48 hours of delivery.",
            "ZentroMart delivery executive reverse pickup procedure.",
        ]
        corpus_vectors = embed_texts(corpus)
        query_vector = embed_query("damaged item replacement")

        self.assertIsInstance(query_vector, np.ndarray)
        self.assertEqual(query_vector.shape, (corpus_vectors.shape[1],))
        self.assertEqual(query_vector.dtype, np.float32)

        # Semantic retrieval ranking test on TF-IDF
        scores = np.dot(corpus_vectors, query_vector)
        self.assertGreater(scores[0], scores[1])

    @patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None)
    def test_fallback_embed_query_before_texts(self, mock_get_model):
        """Calling embed_query before embed_texts should still return a valid 1D vector."""
        vec = embed_query("hello world")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.ndim, 1)
        self.assertEqual(vec.dtype, np.float32)

    @patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None)
    def test_fallback_empty_texts_and_empty_query(self, mock_get_model):
        """Empty texts and empty query should be handled gracefully without crashing."""
        empty_texts_res = embed_texts([])
        self.assertEqual(empty_texts_res.shape[0], 0)

        empty_query_res = embed_query("")
        self.assertIsInstance(empty_query_res, np.ndarray)
        self.assertEqual(empty_query_res.ndim, 1)

    @patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None)
    def test_fallback_out_of_vocabulary_query(self, mock_get_model):
        """Out of vocabulary query returns zero vector of matching dimension."""
        corpus = ["alpha beta gamma"]
        corpus_vecs = embed_texts(corpus)
        expected_dim = corpus_vecs.shape[1]
        oov_vec = embed_query("xyz unrelated nonexisting")
        self.assertEqual(oov_vec.shape, (expected_dim,))
        self.assertTrue(np.all(oov_vec == 0.0))


class TestDependencyInjectionAndOverrides(unittest.TestCase):
    """Test manual model and vectorizer injection."""

    def tearDown(self):
        clear_model_cache()

    def test_set_embedding_model(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((2, 10), dtype=np.float32)

        set_embedding_model(mock_model, "mock-model")
        res = embed_texts(["doc1", "doc2"])
        mock_model.encode.assert_called_once()
        self.assertEqual(res.shape, (2, 10))

    def test_set_tfidf_vectorizer(self):
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value.toarray.return_value = np.zeros((2, 5), dtype=np.float32)

        set_tfidf_vectorizer(mock_vectorizer)
        clear_model_cache()
        set_tfidf_vectorizer(mock_vectorizer)

        with patch("zentromart_support_agent.knowledge_base.embeddings.get_embedding_model", return_value=None):
            res = embed_texts(["doc1", "doc2"])
            self.assertEqual(res.shape, (2, 5))


class TestEndToEndKnowledgeBaseIntegration(unittest.TestCase):
    """End-to-end integration test with real KB markdown chunks."""

    def test_embed_kb_chunks_and_query(self):
        from zentromart_support_agent.knowledge_base.chunker import load_and_chunk_kb

        chunks = load_and_chunk_kb()
        self.assertGreater(len(chunks), 0)

        # Take a subset of 5 chunks for quick embedding
        sample_chunks = chunks[:5]
        chunk_texts = [f"{c['title']} - {c['text']}" for c in sample_chunks]

        embeddings = embed_texts(chunk_texts)
        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.shape[1], 384)

        # Query embedding
        q_vec = embed_query("return policy refund window")
        self.assertEqual(q_vec.shape, (384,))

        # Dot product
        scores = np.dot(embeddings, q_vec)
        self.assertEqual(scores.shape, (5,))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
