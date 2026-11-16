"""RAG (Retrieval-Augmented Generation) Pipeline for ZentroMart Support Agent.

This module implements the knowledge retrieval pipeline for customer support queries:
1. initialize_rag(): Loads knowledge base chunks, builds the vector store index,
   and initializes the TF-IDF reranker (scikit-learn TfidfVectorizer + cosine_similarity).
2. retrieve(query, top_k=5): Encodes query into dense embedding, searches top_k * 3
   candidate chunks via vector similarity, applies TF-IDF reranking against candidate texts,
   boosts current policy documents (priority 1 receives a 1.5x score boost), and returns top_k chunks.
3. retrieve_with_context(query, order_data=None, product_data=None): Formats retrieved chunks
   into a structured context prompt with section IDs for accurate LLM citations, incorporating
   verified customer order details and product catalog information when provided.
4. get_cited_section_ids(retrieved_chunks): Extracts unique section identifiers
   (e.g. 'RET-2', 'KB-00.1') from chunks, dictionaries, or formatted context strings.

Document Precedence Hierarchy:
    policy (priority 1) > agent_guide (2) > help (3) > faq (4) > marketing (5)
Current policy documents (priority 1, status='current') receive a 1.5x score multiplier
during reranking to resolve conflicting guidelines in favor of authoritative policies.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Required: sys.path insert for zentromart_support_agent
sys.path.insert(
    0,
    r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\zentromart_support_agent",
)

# Imports from knowledge_base package
try:
    from knowledge_base.chunker import load_and_chunk_kb
    from knowledge_base.embeddings import embed_query, embed_texts
    from knowledge_base.vector_store import build_index, is_index_built, search
except ImportError:
    from zentromart_support_agent.knowledge_base.chunker import load_and_chunk_kb
    from zentromart_support_agent.knowledge_base.embeddings import (
        embed_query,
        embed_texts,
    )
    from zentromart_support_agent.knowledge_base.vector_store import (
        build_index,
        is_index_built,
        search,
    )

# Optional scikit-learn components for TF-IDF reranking
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# Default knowledge base path
DEFAULT_KB_DIR: str = r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026\kb"

# Policy boost multiplier per project specifications
POLICY_BOOST_MULTIPLIER: float = 1.5

# Section ID extraction regex pattern
SECTION_ID_PATTERN: re.Pattern = re.compile(r"\[([A-Za-z0-9_.-]+)\]")

# Module-level state variables
_all_chunks: Optional[List[Dict[str, Any]]] = None
_tfidf_vectorizer: Optional[Any] = None
_is_initialized: bool = False
_kb_dir_used: Optional[str] = None


def is_rag_initialized() -> bool:
    """Check whether the RAG pipeline is initialized and ready for retrieval.

    Returns:
        bool: True if chunks and reranker are initialized, False otherwise.
    """
    return _is_initialized and _tfidf_vectorizer is not None


def reset_rag() -> None:
    """Reset the module-level RAG cache and initialization state.

    Primarily used for testing, cache eviction, or reloading with custom datasets.
    """
    global _all_chunks, _tfidf_vectorizer, _is_initialized, _kb_dir_used
    _all_chunks = None
    _tfidf_vectorizer = None
    _is_initialized = False
    _kb_dir_used = None
    logger.debug("RAG module state has been reset.")


def initialize_rag(
    kb_dir: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> None:
    """Initialize the RAG retrieval pipeline and TF-IDF reranker.

    1. Loads knowledge base document chunks via chunker.load_and_chunk_kb.
    2. Builds or verifies the vector index via vector_store.build_index.
    3. Sets up the TF-IDF reranker using scikit-learn TfidfVectorizer fitted on chunk texts.

    Args:
        kb_dir: Path to the knowledge base directory. Defaults to
            'c:\\Users\\Dell\\Downloads\\Capstone_Project_Dataset_FTTT_2026\\Capstone_Project_Dataset_FTTT_2026\\kb'.
        force_rebuild: If True, forces re-indexing in the vector store even if an index exists.

    Raises:
        FileNotFoundError: If the specified knowledge base directory does not exist.
        RuntimeError: If scikit-learn is not available for TF-IDF reranking.
    """
    global _all_chunks, _tfidf_vectorizer, _is_initialized, _kb_dir_used

    target_kb_dir = str(Path(kb_dir).resolve()) if kb_dir is not None else DEFAULT_KB_DIR

    logger.info("Initializing RAG pipeline with knowledge base from: %s", target_kb_dir)

    # 1. Load chunks from KB
    chunks = load_and_chunk_kb(target_kb_dir)
    if not chunks:
        logger.warning("No knowledge base chunks loaded from %s.", target_kb_dir)

    # 2. Build or verify vector store index
    if force_rebuild or not is_index_built():
        logger.info("Building vector store index for %d chunks...", len(chunks))
        build_index(chunks)
    else:
        logger.info("Vector store index already built. Reusing existing vector index.")

    # 3. Set up TF-IDF reranker
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for TF-IDF reranker but is not installed.")

    chunk_texts: List[str] = [str(c.get("text") or "") for c in chunks]

    # Initialize TF-IDF vectorizer with unicode token pattern to support multilingual queries
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=True,
    )

    if chunk_texts:
        vectorizer.fit(chunk_texts)
        logger.debug("TF-IDF vectorizer fitted on %d knowledge base chunks.", len(chunk_texts))
    else:
        # Fallback dummy vocabulary if chunks are empty
        vectorizer.fit(["empty knowledge base"])

    _all_chunks = chunks
    _tfidf_vectorizer = vectorizer
    _kb_dir_used = target_kb_dir
    _is_initialized = True
    logger.info("RAG pipeline initialization completed successfully.")


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    vector_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant knowledge base chunks using dense search and TF-IDF reranking.

    Retrieval Workflow:
    1. Auto-initializes the RAG pipeline if not already initialized.
    2. Encodes query into dense vector embedding via embed_query.
    3. Retrieves top_k * 3 candidate chunks via vector similarity search.
    4. Computes TF-IDF cosine similarity between candidate texts and query.
    5. Combines semantic vector similarity and TF-IDF keyword relevance.
    6. Applies a 1.5x score boost to current policy documents (priority 1, status='current').
    7. Sorts by boosted relevance score descending (with priority rank tie-breaking) and returns top_k.

    Args:
        query: User question or customer support message.
        top_k: Number of final ranked chunks to return (default: 5).
        filters: Optional metadata filters for vector search (e.g. {'status': 'current'}).
        vector_weight: Weight assigned to dense semantic similarity (default: 0.5).
        tfidf_weight: Weight assigned to lexical TF-IDF similarity (default: 0.5).

    Returns:
        List[Dict[str, Any]]: Top-k retrieved chunks, each enriched with 'score',
        'vector_score', 'tfidf_score', and 'boost_applied'.
    """
    if not query or not query.strip():
        logger.debug("Empty query provided to retrieve(); returning empty list.")
        return []

    if top_k <= 0:
        return []

    # Auto-initialize on first retrieve() call if not initialized
    if not is_rag_initialized():
        logger.info("RAG pipeline not initialized. Auto-initializing now...")
        initialize_rag()

    cleaned_query = query.strip()

    # 1. Embed query
    try:
        _ = embed_query(cleaned_query)
    except Exception as exc:
        logger.warning("Query embedding generation encountered warning/error: %s", exc)

    # 2. Search top_k * 3 candidates via vector store
    candidate_k = max(top_k * 3, 1)
    candidates = search(cleaned_query, top_k=candidate_k, filters=filters)

    if not candidates:
        logger.debug("Vector search returned zero candidates for query: %s", cleaned_query[:50])
        return []

    # 3. TF-IDF rerank
    cand_texts = [str(c.get("text") or "") for c in candidates]

    if _tfidf_vectorizer is not None and cand_texts:
        try:
            cand_tfidf = _tfidf_vectorizer.transform(cand_texts)
            query_tfidf = _tfidf_vectorizer.transform([cleaned_query])
            tfidf_sims = cosine_similarity(cand_tfidf, query_tfidf).flatten()
        except Exception as exc:
            logger.warning("TF-IDF similarity calculation failed (%s); using zero scores.", exc)
            tfidf_sims = np.zeros(len(cand_texts), dtype=np.float32)
    else:
        tfidf_sims = np.zeros(len(cand_texts), dtype=np.float32)

    # 4. Score combination and 1.5x current policy doc boost
    reranked_chunks: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        v_score = float(cand.get("score", 0.0))
        t_score = float(tfidf_sims[idx]) if idx < len(tfidf_sims) else 0.0

        # Base hybrid score combining dense and sparse similarities
        base_score = (vector_weight * v_score) + (tfidf_weight * t_score)

        # Check if candidate is a current policy document (priority 1)
        # Key rule: Priority 1 gets 1.5x score boost.
        priority = cand.get("priority")
        doc_type = str(cand.get("doc_type") or "").strip().lower()
        status = str(cand.get("status") or "").strip().lower()

        is_priority_1 = (priority == 1) or (doc_type == "policy")
        is_current = status in ("current", "") or cand.get("status") is None

        if is_priority_1 and is_current:
            final_score = base_score * POLICY_BOOST_MULTIPLIER
            boost_applied = True
        else:
            final_score = base_score
            boost_applied = False

        chunk_copy = dict(cand)
        chunk_copy["vector_score"] = v_score
        chunk_copy["tfidf_score"] = t_score
        chunk_copy["score"] = float(final_score)
        chunk_copy["boost_applied"] = boost_applied
        reranked_chunks.append(chunk_copy)

    # Sort descending by final score, using priority rank (ascending) for ties
    reranked_chunks.sort(
        key=lambda item: (-item["score"], item.get("priority", 99))
    )

    # Return top_k
    return reranked_chunks[:top_k]


def _format_data_block(title: str, data: Any) -> str:
    """Format structured order or product data into a readable context block.

    Args:
        title: Header title for the section (e.g. 'Customer Order Data').
        data: Dictionary, list, or string representing order or product details.

    Returns:
        str: Formatted text block.
    """
    if data is None:
        return ""

    if isinstance(data, str):
        cleaned = data.strip()
        return f"--- {title} ---\n{cleaned}" if cleaned else ""

    lines: List[str] = [f"--- {title} ---"]

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for sub_item in v:
                    if isinstance(sub_item, dict):
                        sub_str = ", ".join(f"{sk}: {sv}" for sk, sv in sub_item.items())
                        lines.append(f"  - {sub_str}")
                    else:
                        lines.append(f"  - {sub_item}")
            elif isinstance(v, dict):
                lines.append(f"{k}:")
                for sk, sv in v.items():
                    lines.append(f"  {sk}: {sv}")
            else:
                lines.append(f"{k}: {v}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                item_str = ", ".join(f"{k}: {v}" for k, v in item.items())
                lines.append(f"- {item_str}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append(str(data))

    return "\n".join(lines)


def retrieve_with_context(
    query: str,
    order_data: Optional[Any] = None,
    product_data: Optional[Any] = None,
    top_k: int = 5,
) -> str:
    """Retrieve relevant chunks and assemble complete prompt context for customer support.

    Formats retrieved chunks with distinct section IDs (e.g. '[RET-2]') to enable accurate
    citations and policy grounding. Appends structured order details and product specifications
    when provided.

    Args:
        query: User query or support conversation turn.
        order_data: Optional order record (dict, list, or formatted string) from tool lookup.
        product_data: Optional product details (dict, list, or formatted string) from catalog.
        top_k: Number of knowledge base chunks to retrieve (default: 5).

    Returns:
        str: Fully formatted context string ready to be injected into an LLM prompt.
    """
    retrieved_chunks = retrieve(query, top_k=top_k)

    context_sections: List[str] = []

    # 1. Format Retrieved Knowledge Base Chunks
    if retrieved_chunks:
        kb_lines: List[str] = ["=== Knowledge Base References ==="]
        for i, chunk in enumerate(retrieved_chunks, start=1):
            sec_id = str(chunk.get("section_id") or "UNKNOWN").strip().strip("[]")
            doc_id = str(chunk.get("doc_id") or "").strip()
            doc_type = str(chunk.get("doc_type") or "").strip()
            title = str(chunk.get("title") or "").strip()
            text = str(chunk.get("text") or "").strip()

            meta_parts = []
            if doc_type:
                meta_parts.append(doc_type)
            if doc_id:
                meta_parts.append(doc_id)
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

            header = f"[{sec_id}] {title}{meta_str}:"
            kb_lines.append(header)
            kb_lines.append(text)
            kb_lines.append("")  # Blank separator line

        context_sections.append("\n".join(kb_lines).strip())
    else:
        context_sections.append("=== Knowledge Base References ===\nNo matching knowledge base documents found.")

    # 2. Format Order Data (if provided)
    if order_data is not None:
        order_block = _format_data_block("Customer Order Data", order_data)
        if order_block:
            context_sections.append(order_block)

    # 3. Format Product Data (if provided)
    if product_data is not None:
        prod_block = _format_data_block("Product Catalog Data", product_data)
        if prod_block:
            context_sections.append(prod_block)

    return "\n\n".join(context_sections)


def get_cited_section_ids(
    retrieved_chunks: Union[List[Dict[str, Any]], Dict[str, Any], List[str], str, None],
) -> List[str]:
    """Extract unique section IDs in order of appearance from chunks or text.

    Supports:
        - List of chunk dictionaries containing 'section_id' or 'chunk_id' keys.
        - Single chunk dictionary.
        - Text string containing citation tags (e.g. '[RET-2]', '[KB-00.1]').
        - List of section ID strings.

    Args:
        retrieved_chunks: Chunks, dictionary, string, or list of strings.

    Returns:
        List[str]: Cleaned, unique section identifiers (e.g. ['RET-2', 'KB-00.1']).
    """
    if retrieved_chunks is None:
        return []

    extracted_ids: List[str] = []

    # If a raw string is passed (e.g. context or response text)
    if isinstance(retrieved_chunks, str):
        matches = SECTION_ID_PATTERN.findall(retrieved_chunks)
        if matches:
            return list(dict.fromkeys(matches))

        # Check for delimiter-separated tokens like 'RET-2 ; KB-00.1'
        tokens = [t.strip() for t in re.split(r"[,;|\n]+", retrieved_chunks) if t.strip()]
        valid_tokens: List[str] = []
        for t in tokens:
            cleaned = re.sub(r"^\[|\]$", "", t).strip()
            if cleaned and re.match(r"^[A-Za-z0-9_.-]+$", cleaned):
                valid_tokens.append(cleaned)
        return list(dict.fromkeys(valid_tokens))

    # Normalize to list of items
    chunks_list: Sequence[Any]
    if isinstance(retrieved_chunks, dict):
        chunks_list = [retrieved_chunks]
    elif isinstance(retrieved_chunks, (list, tuple)):
        chunks_list = retrieved_chunks
    else:
        return []

    for item in chunks_list:
        if isinstance(item, dict):
            sec_id = item.get("section_id")
            if sec_id:
                cleaned = str(sec_id).strip().strip("[]")
                if cleaned:
                    extracted_ids.append(cleaned)
            elif item.get("chunk_id"):
                cid = str(item["chunk_id"]).strip()
                if "_" in cid:
                    cid = cid.split("_", 1)[1]
                extracted_ids.append(cid)
            elif item.get("text"):
                matches = SECTION_ID_PATTERN.findall(str(item["text"]))
                extracted_ids.extend(matches)
        elif isinstance(item, str):
            cleaned = item.strip()
            matches = SECTION_ID_PATTERN.findall(cleaned)
            if matches:
                extracted_ids.extend(matches)
            else:
                bare = cleaned.strip("[]").strip()
                if bare and re.match(r"^[A-Za-z0-9_.-]+$", bare):
                    extracted_ids.append(bare)

    # Deduplicate while preserving original order
    return list(dict.fromkeys(extracted_ids))
