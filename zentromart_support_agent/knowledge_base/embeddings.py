"""Embedding module for ZentroMart AI Customer Support Agent knowledge base.

Provides lazy-loading of the SentenceTransformer embedding model ('all-MiniLM-L6-v2')
with module-level caching and a graceful TF-IDF fallback via scikit-learn when
sentence-transformers is unavailable or encounters runtime errors.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Default model name per ZentroMart configuration
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Module-level model and fallback caches
_model: Optional[Any] = None
_model_name: Optional[str] = None
_model_load_attempted: bool = False
_tfidf_vectorizer: Optional[Any] = None


def get_embedding_model(model_name: Optional[str] = None) -> Optional[Any]:
    """Lazy-load and return the SentenceTransformer embedding model.

    Maintains a module-level cache (`_model`) so that subsequent invocations
    reuse the previously loaded model without incurring initialization overhead.
    If `sentence-transformers` is not installed, or if the model weights cannot
    be loaded, logs an informative warning and returns None.

    Args:
        model_name: Optional model identifier. Defaults to 'all-MiniLM-L6-v2'
            or the value configured in zentromart_support_agent.config.

    Returns:
        SentenceTransformer model instance if successfully loaded; otherwise None.
    """
    global _model, _model_name, _model_load_attempted

    # If a model is already cached:
    # 1. If caller didn't specify a model_name, return the cached model.
    # 2. If caller specified a model_name matching the cached model, return it.
    if _model is not None:
        if model_name is None or model_name == _model_name:
            return _model

    # Determine target model name
    if model_name is None:
        try:
            from zentromart_support_agent.config import EMBEDDING_MODEL_NAME

            target_model_name = EMBEDDING_MODEL_NAME
        except (ImportError, AttributeError):
            target_model_name = DEFAULT_MODEL_NAME
    else:
        target_model_name = model_name

    # Check for sentence_transformers library
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "sentence-transformers is unavailable (%s). Returning None (TF-IDF fallback will be used).",
            exc,
        )
        _model = None
        _model_name = target_model_name
        _model_load_attempted = True
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected error importing sentence-transformers (%s). Returning None.",
            exc,
        )
        _model = None
        _model_name = target_model_name
        _model_load_attempted = True
        return None

    # Attempt to load the SentenceTransformer model
    try:
        # First attempt with local_files_only=True to prevent slow network timeouts in sandboxed/offline environments
        logger.info("Loading SentenceTransformer model '%s' (local_files_only=True)...", target_model_name)
        model = SentenceTransformer(target_model_name, local_files_only=True)
    except Exception as local_exc:
        logger.debug(
            "Local-only load for '%s' failed or not cached (%s). Attempting standard load...",
            target_model_name,
            local_exc,
        )
        try:
            model = SentenceTransformer(target_model_name)
        except Exception as load_exc:
            logger.warning(
                "Failed to initialize SentenceTransformer('%s'): %s. Returning None.",
                target_model_name,
                load_exc,
            )
            _model = None
            _model_name = target_model_name
            _model_load_attempted = True
            return None

    _model = model
    _model_name = target_model_name
    _model_load_attempted = True
    logger.info("Successfully loaded and cached SentenceTransformer model '%s'.", target_model_name)
    return _model


def _get_tfidf_vectorizer() -> Any:
    """Retrieve or create the module-level TF-IDF vectorizer fallback.

    Returns:
        A fitted or un-fitted TfidfVectorizer instance.
    """
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            _tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                max_features=384,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("scikit-learn is unavailable (%s); cannot construct TF-IDF fallback.", exc)
            raise RuntimeError("Neither sentence-transformers nor scikit-learn is available for embeddings.") from exc
    return _tfidf_vectorizer


def embed_texts(texts: list[str], model: Optional[Any] = None) -> np.ndarray:
    """Encode a collection of text documents into vector embeddings.

    Uses the lazy-loaded SentenceTransformer model if available. If
    sentence-transformers is not installed, fails to load, or encounters an
    encoding error, automatically falls back to TF-IDF vectorization via scikit-learn.

    Args:
        texts: A list of text strings to embed.
        model: Optional SentenceTransformer model instance. If None, retrieves
            the cached model via get_embedding_model().

    Returns:
        np.ndarray: 2D array of shape (len(texts), embedding_dimension) containing
            the float32 embedding vectors. For an empty list, returns an empty array.
    """
    if not texts:
        active_model = model if model is not None else get_embedding_model()
        dim = 384
        if active_model is not None:
            if hasattr(active_model, "get_embedding_dimension"):
                try:
                    dim = active_model.get_embedding_dimension()
                except Exception:
                    dim = 384
            elif hasattr(active_model, "get_sentence_embedding_dimension"):
                try:
                    dim = active_model.get_sentence_embedding_dimension()
                except Exception:
                    dim = 384
        return np.empty((0, dim), dtype=np.float32)

    # Coerce all elements to strings to avoid type issues with None or non-str
    normalized_texts = [str(t) if t is not None else "" for t in texts]

    # Try encoding with SentenceTransformer
    active_model = model if model is not None else get_embedding_model()
    if active_model is not None:
        try:
            embeddings = active_model.encode(
                normalized_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.warning(
                "SentenceTransformer.encode failed (%s). Falling back to TF-IDF vectorization.",
                exc,
            )

    # Fallback to TF-IDF vectorization
    logger.info("Encoding %d texts using TF-IDF fallback.", len(normalized_texts))
    vectorizer = _get_tfidf_vectorizer()

    try:
        sparse_matrix = vectorizer.fit_transform(normalized_texts)
        dense_array = sparse_matrix.toarray()
    except ValueError:
        # Handles cases where all texts are empty strings or purely stop words
        dense_array = np.zeros((len(normalized_texts), 1), dtype=np.float32)
    except Exception as exc:
        logger.error("TF-IDF vectorization failed: %s", exc)
        dense_array = np.zeros((len(normalized_texts), 1), dtype=np.float32)

    return np.asarray(dense_array, dtype=np.float32)


def embed_query(query: str, model: Optional[Any] = None) -> np.ndarray:
    """Encode a single search query into a vector embedding.

    Uses the lazy-loaded SentenceTransformer model if available. If
    sentence-transformers is unavailable or fails, falls back to TF-IDF
    vectorization aligned with previously encoded documents.

    Args:
        query: Query string to embed.
        model: Optional SentenceTransformer model instance. If None, retrieves
            the cached model via get_embedding_model().

    Returns:
        np.ndarray: 1D array of shape (embedding_dimension,) containing the float32
            embedding vector for the query.
    """
    normalized_query = str(query) if query is not None else ""

    # Try encoding with SentenceTransformer
    active_model = model if model is not None else get_embedding_model()
    if active_model is not None:
        try:
            embedding = active_model.encode(
                normalized_query,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # Ensure 1D array
            arr = np.asarray(embedding, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze()
            return arr
        except Exception as exc:
            logger.warning(
                "SentenceTransformer.encode failed for query (%s). Falling back to TF-IDF vectorization.",
                exc,
            )

    # Fallback to TF-IDF vectorization
    logger.info("Encoding query '%s' using TF-IDF fallback.", normalized_query[:50])
    vectorizer = _get_tfidf_vectorizer()

    # If the vectorizer has already been fitted on a corpus of texts, transform against that vocabulary
    if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_:
        try:
            dense_array = vectorizer.transform([normalized_query]).toarray()[0]
        except Exception as exc:
            logger.warning("TF-IDF transform failed (%s); returning zero vector.", exc)
            dense_array = np.zeros((len(vectorizer.vocabulary_),), dtype=np.float32)
    else:
        # Vectorizer not yet fitted, fit directly on the query
        try:
            dense_array = vectorizer.fit_transform([normalized_query]).toarray()[0]
        except ValueError:
            dense_array = np.zeros((1,), dtype=np.float32)

    return np.asarray(dense_array, dtype=np.float32)


def clear_model_cache() -> None:
    """Clear the module-level model and TF-IDF vectorizer caches.

    Useful for testing, freeing memory, or reloading models with different configurations.
    """
    global _model, _model_name, _model_load_attempted, _tfidf_vectorizer
    _model = None
    _model_name = None
    _model_load_attempted = False
    _tfidf_vectorizer = None
    logger.debug("Cleared module-level embedding caches.")


def set_embedding_model(model: Any, model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Manually inject an embedding model into the module cache.

    Primarily intended for dependency injection, test mocking, and custom models.

    Args:
        model: Embedding model instance (e.g. SentenceTransformer or mock).
        model_name: Optional name tag for the model.
    """
    global _model, _model_name, _model_load_attempted
    _model = model
    _model_name = model_name
    _model_load_attempted = True


def set_tfidf_vectorizer(vectorizer: Any) -> None:
    """Manually inject a TF-IDF vectorizer into the fallback cache.

    Args:
        vectorizer: TfidfVectorizer instance or mock.
    """
    global _tfidf_vectorizer
    _tfidf_vectorizer = vectorizer
