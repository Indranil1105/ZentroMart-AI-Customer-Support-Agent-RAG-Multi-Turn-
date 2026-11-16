"""Knowledge Base package for ZentroMart AI Customer Support Agent.

Exports chunking, embedding generation, vector indexing, and semantic retrieval
utilities for markdown knowledge base documents.
"""

from zentromart_support_agent.knowledge_base.chunker import (
    CHUNK_KEYS,
    DOC_TYPE_PRIORITY,
    FRONTMATTER_FIELDS,
    H2_HEADING_REGEX,
    SECTION_ID_REGEX,
    chunk_document,
    clear_cache,
    filter_chunks,
    get_all_section_ids,
    get_chunk_by_section_id,
    get_doc_priority,
    get_kb_dir,
    load_and_chunk_kb,
    parse_frontmatter,
    set_kb_dir,
)
from zentromart_support_agent.knowledge_base.embeddings import (
    clear_model_cache,
    embed_query,
    embed_texts,
    get_embedding_model,
)
from zentromart_support_agent.knowledge_base.vector_store import (
    COLLECTION_NAME,
    PICKLE_FILENAME,
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

__all__ = [
    # Chunker
    "CHUNK_KEYS",
    "DOC_TYPE_PRIORITY",
    "FRONTMATTER_FIELDS",
    "H2_HEADING_REGEX",
    "SECTION_ID_REGEX",
    "chunk_document",
    "clear_cache",
    "filter_chunks",
    "get_all_section_ids",
    "get_chunk_by_section_id",
    "get_doc_priority",
    "get_kb_dir",
    "load_and_chunk_kb",
    "parse_frontmatter",
    "set_kb_dir",
    # Embeddings
    "clear_model_cache",
    "embed_query",
    "embed_texts",
    "get_embedding_model",
    # Vector Store
    "COLLECTION_NAME",
    "PICKLE_FILENAME",
    "STORE_DIR",
    "build_index",
    "clear_store",
    "get_backend",
    "get_store_dir",
    "is_chroma_available",
    "is_index_built",
    "search",
    "set_backend",
    "set_store_dir",
]
