"""Knowledge Base Vector Store Module for ZentroMart AI Customer Support Agent.

This module provides persistent semantic indexing and similarity search for
knowledge base chunks using ChromaDB as the primary vector database, with an
automatic fallback to NumPy cosine similarity and pickle-based storage when
ChromaDB is unavailable or disabled.

Both backends provide an identical search and retrieval interface, supporting
metadata preservation, section ID tracking, document precedence prioritization,
and filtering by document type and status.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Ensure package root is in sys.path for robust direct imports
_CURRENT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CURRENT_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

# Required import from knowledge_base.embeddings
try:
    from knowledge_base.embeddings import embed_query, embed_texts
except ImportError:
    from zentromart_support_agent.knowledge_base.embeddings import (
        embed_query,
        embed_texts,
    )

# Standard Persistent Storage Directory
STORE_DIR: str = os.path.join(
    r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\zentromart_support_agent",
    "chroma_db",
)

# Collection and File Constants
COLLECTION_NAME: str = "zentromart_kb"
PICKLE_FILENAME: str = "numpy_vector_store.pkl"

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.config import Settings
    _CHROMADB_INSTALLED = True
except ImportError:
    _CHROMADB_INSTALLED = False

# Module-level configuration state
_STORE_DIR: str = STORE_DIR
_ACTIVE_BACKEND: str = "auto"  # 'auto', 'chroma', or 'numpy'
_CACHED_NUMPY_STORE: Optional[Dict[str, Any]] = None
_CACHED_NUMPY_PATH: Optional[str] = None


def get_store_dir() -> str:
    """Return the current vector store directory path.

    Returns:
        str: Absolute path to the storage directory.
    """
    return _STORE_DIR


def set_store_dir(path: Union[str, Path]) -> None:
    """Override the vector store directory path.

    Args:
        path: Directory path where ChromaDB or pickle data will be stored.
    """
    global _STORE_DIR, _CACHED_NUMPY_STORE, _CACHED_NUMPY_PATH
    _STORE_DIR = str(Path(path).resolve())
    _CACHED_NUMPY_STORE = None
    _CACHED_NUMPY_PATH = None
    logger.debug("Vector store directory set to: %s", _STORE_DIR)


def get_backend() -> str:
    """Return the currently configured backend mode ('auto', 'chroma', or 'numpy').

    Returns:
        str: Active backend identifier.
    """
    return _ACTIVE_BACKEND


def set_backend(backend: str) -> None:
    """Set the vector store backend mode.

    Args:
        backend: One of 'auto', 'chroma', or 'numpy'.

    Raises:
        ValueError: If backend is not one of the supported options.
    """
    global _ACTIVE_BACKEND
    cleaned = backend.strip().lower()
    if cleaned not in ("auto", "chroma", "numpy"):
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: 'auto', 'chroma', 'numpy'.")
    _ACTIVE_BACKEND = cleaned
    logger.info("Vector store active backend set to: %s", _ACTIVE_BACKEND)


def is_chroma_available() -> bool:
    """Check if ChromaDB is installed and operational.

    Returns:
        bool: True if ChromaDB is available, False otherwise.
    """
    if not _CHROMADB_INSTALLED:
        return False
    if _ACTIVE_BACKEND == "numpy":
        return False
    return True


def _get_pickle_path(target_dir: Optional[str] = None) -> str:
    """Return the absolute path to the NumPy fallback pickle file."""
    base = target_dir or _STORE_DIR
    return os.path.join(base, PICKLE_FILENAME)


def _load_numpy_store(target_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the NumPy pickle vector store from disk into memory cache."""
    global _CACHED_NUMPY_STORE, _CACHED_NUMPY_PATH

    pickle_path = _get_pickle_path(target_dir)
    if _CACHED_NUMPY_STORE is not None and _CACHED_NUMPY_PATH == pickle_path:
        return _CACHED_NUMPY_STORE

    if not os.path.isfile(pickle_path):
        return None

    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "chunks" in data and "embeddings" in data:
            _CACHED_NUMPY_STORE = data
            _CACHED_NUMPY_PATH = pickle_path
            return _CACHED_NUMPY_STORE
    except Exception as exc:
        logger.error("Failed to load NumPy vector store pickle from '%s': %s", pickle_path, exc)

    return None


def _save_numpy_pickle(
    store_dir: str,
    chunks: List[Dict[str, Any]],
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Serialize chunks and embeddings to disk using pickle."""
    global _CACHED_NUMPY_STORE, _CACHED_NUMPY_PATH

    os.makedirs(store_dir, exist_ok=True)
    pickle_path = _get_pickle_path(store_dir)

    norm_embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(norm_embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    norm_embeddings = norm_embeddings / norms

    data = {
        "chunks": chunks,
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
        "embeddings": norm_embeddings,
        "backend": "numpy",
        "count": len(chunks),
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    _CACHED_NUMPY_STORE = data
    _CACHED_NUMPY_PATH = pickle_path
    logger.info("Saved %d chunks to NumPy vector store pickle at %s", len(chunks), pickle_path)


def _build_chroma_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB-compatible query filter dictionary from key-value filters.

    Args:
        filters: Dictionary of metadata filters (e.g. {'doc_type': 'policy', 'status': 'current'}).

    Returns:
        Optional[Dict[str, Any]]: Formatted ChromaDB where clause or None.
    """
    if not filters:
        return None

    clauses: List[Dict[str, Any]] = []

    for key, val in filters.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            val_list = [str(v) for v in val]
            if len(val_list) == 1:
                clauses.append({key: val_list[0]})
            elif len(val_list) > 1:
                clauses.append({key: {"$in": val_list}})
        else:
            clauses.append({key: val})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _matches_filters(chunk_meta: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Check if a chunk's metadata satisfies the given filter constraints.

    Args:
        chunk_meta: Metadata dictionary of the chunk.
        filters: Filter criteria to evaluate.

    Returns:
        bool: True if the chunk matches all filter criteria, False otherwise.
    """
    if not filters:
        return True

    for key, expected_val in filters.items():
        if expected_val is None:
            continue

        actual_val = chunk_meta.get(key)

        if isinstance(expected_val, (list, tuple, set)):
            str_expected = [str(x).lower().strip() for x in expected_val]
            if str(actual_val).lower().strip() not in str_expected:
                return False
        else:
            if str(actual_val).lower().strip() != str(expected_val).lower().strip():
                return False

    return True


def build_index(chunks: List[Dict[str, Any]]) -> None:
    """Build a persistent vector index from knowledge base chunks.

    Stores chunk text and metadata (section_id, doc_id, doc_type, status, priority,
    effective_from) in a persistent ChromaDB index. If ChromaDB is unavailable,
    uses NumPy cosine similarity with pickle storage.

    Args:
        chunks: List of chunk dictionaries containing 'text', 'section_id', 'doc_id',
            'doc_type', 'status', 'priority', 'effective_from', and optional fields.

    Raises:
        ValueError: If chunks is None.
    """
    if chunks is None:
        raise ValueError("Chunks argument cannot be None.")

    target_dir = get_store_dir()
    os.makedirs(target_dir, exist_ok=True)

    if not chunks:
        logger.warning("Empty chunks list provided. Clearing existing vector store index.")
        clear_store()
        return

    logger.info("Building vector index for %d chunks in directory: %s", len(chunks), target_dir)

    # Prepare chunk IDs, documents, and normalized metadata
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    seen_ids: Dict[str, int] = {}

    for i, chunk in enumerate(chunks):
        doc_id = str(chunk.get("doc_id") or "").strip()
        section_id = str(chunk.get("section_id") or "").strip()

        base_id = str(chunk.get("chunk_id") or "").strip()
        if not base_id:
            base_id = f"{doc_id}_{section_id}" if (doc_id and section_id) else f"chunk_{i}"

        # Guarantee unique IDs
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            unique_id = f"{base_id}_dup{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 1
            unique_id = base_id

        ids.append(unique_id)

        raw_text = str(chunk.get("text") or "").strip()
        texts.append(raw_text)

        # Standard metadata extraction according to project specifications
        doc_type = str(chunk.get("doc_type") or "").strip()
        status = str(chunk.get("status") or "").strip()
        title = str(chunk.get("title") or "").strip()
        language = str(chunk.get("language") or "en").strip()

        raw_priority = chunk.get("priority")
        try:
            priority = int(raw_priority) if raw_priority is not None else 99
        except (ValueError, TypeError):
            priority = 99

        raw_eff = chunk.get("effective_from")
        effective_from = str(raw_eff).strip() if raw_eff is not None else ""

        # ChromaDB requires primitive metadata values: str, int, float, bool
        meta_dict: Dict[str, Any] = {
            "chunk_id": unique_id,
            "section_id": section_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "status": status,
            "priority": priority,
            "effective_from": effective_from,
            "title": title,
            "language": language,
        }
        metadatas.append(meta_dict)

    # Compute dense embeddings for all chunk texts
    raw_embeddings = embed_texts(texts)
    if len(raw_embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding length mismatch: expected {len(texts)} embeddings, got {len(raw_embeddings)}."
        )

    if isinstance(raw_embeddings, np.ndarray):
        embeddings: List[List[float]] = raw_embeddings.tolist()
    else:
        embeddings = [list(e) for e in raw_embeddings]

    # Determine whether to build ChromaDB index or NumPy fallback
    use_chroma = is_chroma_available()

    if use_chroma:
        try:
            client = chromadb.PersistentClient(path=target_dir)

            # Recreate collection to ensure clean persistent state
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.debug("Existing ChromaDB collection '%s' deleted for rebuild.", COLLECTION_NAME)
            except Exception:
                pass

            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            # ChromaDB supports batch additions
            batch_size = 500
            for start_idx in range(0, len(ids), batch_size):
                end_idx = min(start_idx + batch_size, len(ids))
                collection.add(
                    ids=ids[start_idx:end_idx],
                    documents=texts[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                    embeddings=embeddings[start_idx:end_idx],
                )

            # Also maintain NumPy pickle store as a hot fallback
            _save_numpy_pickle(target_dir, chunks, ids, texts, metadatas, embeddings)

            logger.info(
                "Successfully built persistent ChromaDB index with %d chunks in %s",
                len(ids),
                target_dir,
            )
            return

        except Exception as exc:
            logger.warning(
                "Failed to build ChromaDB persistent index (%s). Falling back to NumPy store.",
                exc,
            )

    # Fallback to NumPy cosine similarity with pickle storage
    _save_numpy_pickle(target_dir, chunks, ids, texts, metadatas, embeddings)
    logger.info("Successfully built NumPy pickle index with %d chunks in %s", len(ids), target_dir)


def is_index_built() -> bool:
    """Check whether a valid vector store index is built and ready for search.

    Returns:
        bool: True if an index exists with at least one chunk, False otherwise.
    """
    target_dir = get_store_dir()
    if not os.path.exists(target_dir):
        return False

    # Check ChromaDB if enabled and available
    if is_chroma_available():
        try:
            client = chromadb.PersistentClient(path=target_dir)
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() > 0:
                return True
        except Exception:
            pass

    # Check NumPy pickle storage
    pickle_path = _get_pickle_path(target_dir)
    if os.path.isfile(pickle_path):
        store = _load_numpy_store(target_dir)
        if store is not None and store.get("count", 0) > 0:
            return True

    return False


def _search_chroma(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute semantic search against the persistent ChromaDB collection."""
    target_dir = get_store_dir()
    client = chromadb.PersistentClient(path=target_dir)
    collection = client.get_collection(COLLECTION_NAME)

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    # Encode query string
    raw_query_vector = embed_query(query)
    if isinstance(raw_query_vector, np.ndarray):
        query_vector = raw_query_vector.tolist()
    else:
        query_vector = list(raw_query_vector)

    # Build Chroma where clause
    where_clause = _build_chroma_where(filters)

    n_results = min(top_k, total_chunks)

    query_params: Dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
    }
    if where_clause is not None:
        query_params["where"] = where_clause

    results = collection.query(**query_params)

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

    hits: List[Dict[str, Any]] = []
    for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
        # In cosine distance space, similarity = 1.0 - distance
        score = float(1.0 - float(dist))

        eff = meta.get("effective_from")
        effective_from = eff if eff != "" else None

        item: Dict[str, Any] = {
            "chunk_id": str(meta.get("chunk_id") or chunk_id),
            "section_id": str(meta.get("section_id") or ""),
            "doc_id": str(meta.get("doc_id") or ""),
            "doc_type": str(meta.get("doc_type") or ""),
            "title": str(meta.get("title") or ""),
            "status": str(meta.get("status") or ""),
            "priority": int(meta.get("priority", 99)),
            "effective_from": effective_from,
            "language": str(meta.get("language") or "en"),
            "text": doc_text,
            "score": score,
        }
        hits.append(item)

    # Sort results by score descending, with precedence priority ascending for ties
    hits.sort(key=lambda x: (-x["score"], x["priority"]))
    return hits[:top_k]


def _search_numpy(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute semantic search against the NumPy pickle store."""
    target_dir = get_store_dir()
    store = _load_numpy_store(target_dir)

    if store is None or not store.get("chunks"):
        return []

    chunks: List[Dict[str, Any]] = store["chunks"]
    metadatas: List[Dict[str, Any]] = store["metadatas"]
    norm_embeddings: np.ndarray = store["embeddings"]

    # Filter candidate indices based on metadata criteria
    candidate_indices: List[int] = []
    for idx, meta in enumerate(metadatas):
        if _matches_filters(meta, filters):
            candidate_indices.append(idx)

    if not candidate_indices:
        return []

    # Encode and normalize query vector
    q_vec = np.array(embed_query(query), dtype=np.float32)
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm > 0.0:
        q_vec = q_vec / q_norm
    else:
        q_vec[0] = 1.0

    # Calculate cosine similarity with candidate embeddings
    cand_vectors = norm_embeddings[candidate_indices]
    similarities = np.dot(cand_vectors, q_vec)

    hits: List[Dict[str, Any]] = []
    for sub_idx, original_idx in enumerate(candidate_indices):
        orig_chunk = chunks[original_idx]
        meta = metadatas[original_idx]
        sim_score = float(similarities[sub_idx])

        eff = meta.get("effective_from") or orig_chunk.get("effective_from")
        effective_from = eff if eff != "" else None

        item: Dict[str, Any] = {
            "chunk_id": str(orig_chunk.get("chunk_id") or meta.get("chunk_id") or ""),
            "section_id": str(orig_chunk.get("section_id") or meta.get("section_id") or ""),
            "doc_id": str(orig_chunk.get("doc_id") or meta.get("doc_id") or ""),
            "doc_type": str(orig_chunk.get("doc_type") or meta.get("doc_type") or ""),
            "title": str(orig_chunk.get("title") or meta.get("title") or ""),
            "status": str(orig_chunk.get("status") or meta.get("status") or ""),
            "priority": int(orig_chunk.get("priority") or meta.get("priority", 99)),
            "effective_from": effective_from,
            "language": str(orig_chunk.get("language") or meta.get("language") or "en"),
            "text": str(orig_chunk.get("text") or meta.get("text") or ""),
            "score": sim_score,
        }
        hits.append(item)

    # Sort results by score descending, with precedence priority ascending for ties
    hits.sort(key=lambda x: (-x["score"], x["priority"]))
    return hits[:top_k]


def search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Perform semantic search across knowledge base chunks with score ranking and filtering.

    Args:
        query: User or support question string.
        top_k: Maximum number of search results to return (default: 10).
        filters: Optional metadata filter dict (supports 'doc_type', 'status', etc.).
            Values can be exact strings or lists of strings for multi-category matching.

    Returns:
        List[Dict[str, Any]]: Top-k matching chunks with similarity 'score', sorted
        descending by relevance score with precedence rank tie-breaking.

    Raises:
        RuntimeError: If vector store index has not been built yet.
    """
    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if not is_index_built():
        raise RuntimeError(
            "Vector store index is not built. Please call build_index(chunks) first."
        )

    # Select backend according to configuration and availability
    if is_chroma_available():
        try:
            return _search_chroma(query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning(
                "ChromaDB search encountered an error (%s). Falling back to NumPy search.",
                exc,
            )

    return _search_numpy(query, top_k=top_k, filters=filters)


def clear_store(target_dir: Optional[str] = None) -> None:
    """Clear all index files and in-memory caches from the target directory.

    Args:
        target_dir: Optional storage directory path. If None, uses current store dir.
    """
    global _CACHED_NUMPY_STORE, _CACHED_NUMPY_PATH

    dir_to_clear = target_dir or get_store_dir()

    # Clear ChromaDB collection
    if is_chroma_available() and os.path.exists(dir_to_clear):
        try:
            client = chromadb.PersistentClient(path=dir_to_clear)
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # Clear pickle file
    pickle_path = _get_pickle_path(dir_to_clear)
    if os.path.isfile(pickle_path):
        try:
            os.remove(pickle_path)
        except Exception as exc:
            logger.debug("Could not remove pickle file %s: %s", pickle_path, exc)

    _CACHED_NUMPY_STORE = None
    _CACHED_NUMPY_PATH = None
    logger.debug("Cleared vector store index at: %s", dir_to_clear)
