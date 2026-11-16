"""Knowledge Base Chunking Module for ZentroMart AI Customer Support Agent.

This module parses markdown knowledge base documents, extracts YAML frontmatter
metadata, and splits document content into semantic chunks based on '##' section
headings containing unique '[SECTION-ID]' markers (e.g. [RET-2], [KB-00.1]).

Key Design Principles:
1. Robust YAML frontmatter parser handling unquoted colons and missing fields.
2. Preserves '###' and deeper sub-sections within their parent '##' section chunk.
3. Preserves '[SECTION-ID]' markers in headings and assigns accurate precedence priorities:
   policy (1) > agent_guide (2) > help (3) > faq (4) > marketing (5).
4. Module-level caching for fast subsequent lookups and in-memory indexing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

# Configure logger
logger = logging.getLogger(__name__)

# Fallback path definitions if config cannot be imported
_DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "ZENTROMART_DATASET_DIR",
        r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026",
    )
).resolve()

_KB_DIR: Path = _DEFAULT_DATASET_DIR / "kb"

try:
    from zentromart_support_agent.config import KB_DIR as _CONFIG_KB_DIR
    _KB_DIR = Path(_CONFIG_KB_DIR).resolve()
except ImportError:
    pass

# Document Type Precedence Mapping (1 = highest precedence, 5 = lowest)
# Key rule: policy (1) > agent_guide (2) > help (3) > faq (4) > marketing (5)
DOC_TYPE_PRIORITY: Dict[str, int] = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
}

# Section ID extraction regex pattern
# Matches patterns like [RET-2], [KB-00.1], [FAQH-1], [RETv3-2]
SECTION_ID_REGEX: re.Pattern = re.compile(r"\[([A-Z0-9a-z\-\.]+)\]")

# Heading pattern matching '##' section headings containing a [SECTION-ID]
H2_HEADING_REGEX: re.Pattern = re.compile(
    r"^##\s+.*?\[([A-Z0-9a-z\-\.]+)\][^\r\n]*",
    re.MULTILINE,
)

# Standard YAML frontmatter fields expected in ZentroMart KB
FRONTMATTER_FIELDS: Tuple[str, ...] = (
    "doc_id",
    "title",
    "doc_type",
    "audience",
    "language",
    "version",
    "effective_from",
    "status",
    "supersedes",
    "superseded_by",
)

# Standard chunk dictionary keys returned by chunker
CHUNK_KEYS: Tuple[str, ...] = (
    "chunk_id",
    "section_id",
    "doc_id",
    "doc_type",
    "title",
    "status",
    "effective_from",
    "language",
    "text",
    "priority",
)

# Module-level cache
_CACHED_CHUNKS: Optional[List[Dict[str, Any]]] = None
_CACHED_INDEX_BY_SECTION_ID: Dict[str, Dict[str, Any]] = {}
_CACHED_KB_DIR: Optional[str] = None


def get_kb_dir() -> Path:
    """Return the currently configured knowledge base directory path.

    Returns:
        Path: Path to the directory containing knowledge base markdown files.
    """
    return _KB_DIR


def set_kb_dir(path: Union[str, Path]) -> None:
    """Override the knowledge base directory path and clear the chunk cache.

    Args:
        path: New path to the directory containing knowledge base markdown files.
    """
    global _KB_DIR
    _KB_DIR = Path(path).resolve()
    clear_cache()
    logger.info("Knowledge base directory updated to: %s", _KB_DIR)


def clear_cache() -> None:
    """Clear all cached chunks and in-memory section indices."""
    global _CACHED_CHUNKS, _CACHED_INDEX_BY_SECTION_ID, _CACHED_KB_DIR
    _CACHED_CHUNKS = None
    _CACHED_INDEX_BY_SECTION_ID.clear()
    _CACHED_KB_DIR = None
    logger.debug("Knowledge base chunk cache cleared.")


def get_doc_priority(doc_type: Optional[str]) -> int:
    """Calculate numerical priority rank for a given document type.

    Precedence order:
    1 = policy
    2 = agent_guide
    3 = help
    4 = faq
    5 = marketing
    99 = meta / unknown fallback

    Args:
        doc_type: Document category string (e.g. 'policy', 'agent_guide').

    Returns:
        int: Numerical priority rank (1 = highest precedence).
    """
    if not doc_type:
        return 99
    return DOC_TYPE_PRIORITY.get(doc_type.strip().lower(), 99)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter and body from a markdown document string.

    Robustly handles unquoted colons in scalar values (e.g. 'title: Reverse pickup: how it works')
    by falling back to line-by-line parsing if the standard YAML parser encounters a syntax error.

    Args:
        content: The full raw text of the markdown document.

    Returns:
        Tuple[Dict[str, Any], str]: A dictionary with frontmatter fields and the remaining
        markdown body text.
    """
    default_fm: Dict[str, Any] = {field: None for field in FRONTMATTER_FIELDS}

    # Normalize newlines
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")

    if not normalized_content.startswith("---"):
        return default_fm, normalized_content

    parts = normalized_content.split("---", 2)
    if len(parts) < 3:
        return default_fm, normalized_content

    fm_raw = parts[1]
    body = parts[2]

    raw_data: Dict[str, Any] = {}

    # Attempt standard PyYAML safe_load first
    try:
        loaded = yaml.safe_load(fm_raw)
        if isinstance(loaded, dict):
            raw_data = loaded
    except Exception as exc:
        logger.debug("Standard YAML load failed, using resilient line parser: %s", exc)

    # Line-by-line fallback and supplement to handle edge cases like unquoted colons in titles
    for line in fm_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Strip outer quotes if present
            if len(val) >= 2 and (
                (val.startswith('"') and val.endswith('"'))
                or (val.startswith("'") and val.endswith("'"))
            ):
                val = val[1:-1]
            if val in ("", "~", "null", "None"):
                val = None
            if key not in raw_data or raw_data[key] is None:
                raw_data[key] = val

    # Format and normalize each expected field
    for field in FRONTMATTER_FIELDS:
        val = raw_data.get(field)
        if val is not None:
            if hasattr(val, "isoformat"):
                val = val.isoformat()
            else:
                val = str(val).strip()
            default_fm[field] = val
        else:
            default_fm[field] = None

    return default_fm, body


def chunk_document(
    content: str,
    filepath: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Split a single markdown document into section chunks based on '##' headings.

    Sub-sections (e.g. '###' headings) and their contents are kept together with their
    parent '##' section chunk.

    Args:
        content: The raw text of the markdown document.
        filepath: Optional path to the source file for logging/debugging context.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries containing standard keys.
    """
    fm, body = parse_frontmatter(content)

    doc_id = str(fm.get("doc_id") or "").strip()
    doc_type = str(fm.get("doc_type") or "").strip()
    title = str(fm.get("title") or "").strip()
    status = str(fm.get("status") or "").strip()
    effective_from = fm.get("effective_from")
    if effective_from is not None:
        effective_from = str(effective_from).strip()
    language = str(fm.get("language") or "en").strip()

    priority = get_doc_priority(doc_type)

    matches = list(H2_HEADING_REGEX.finditer(body))
    if not matches:
        logger.debug(
            "No '## [SECTION-ID]' headings found in document: %s",
            filepath or doc_id or "unknown",
        )
        return []

    chunks: List[Dict[str, Any]] = []

    for i, match in enumerate(matches):
        section_id = match.group(1).strip()
        start_idx = match.start()
        end_idx = matches[i + 1].start() if (i + 1 < len(matches)) else len(body)

        chunk_text = body[start_idx:end_idx].strip()
        chunk_id = f"{doc_id}_{section_id}" if doc_id else section_id

        chunk_dict: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "section_id": section_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "title": title,
            "status": status,
            "effective_from": effective_from,
            "language": language,
            "text": chunk_text,
            "priority": priority,
        }
        chunks.append(chunk_dict)

    return chunks


def load_and_chunk_kb(kb_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load and chunk all markdown files in the knowledge base directory.

    Caches results in memory after the first load for instantaneous access.

    Args:
        kb_dir: Path to the knowledge base directory. If None, uses the configured default KB_DIR.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries across all knowledge base files.

    Raises:
        FileNotFoundError: If the specified or default directory does not exist.
    """
    global _CACHED_CHUNKS, _CACHED_INDEX_BY_SECTION_ID, _CACHED_KB_DIR

    target_dir = Path(kb_dir).resolve() if kb_dir is not None else get_kb_dir()
    target_dir_str = str(target_dir)

    # Return cached chunks if directory matches and cache is already populated
    if _CACHED_CHUNKS is not None and _CACHED_KB_DIR == target_dir_str:
        return _CACHED_CHUNKS

    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist or is not a directory: {target_dir}"
        )

    all_chunks: List[Dict[str, Any]] = []
    index_by_section: Dict[str, Dict[str, Any]] = {}

    md_files = sorted(target_dir.glob("*.md"))
    if not md_files:
        logger.warning("No markdown files found in knowledge base directory: %s", target_dir)

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            doc_chunks = chunk_document(content, filepath=md_file)
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                sec_id = chunk["section_id"]
                index_by_section[sec_id] = chunk
                index_by_section[sec_id.lower()] = chunk
                # Also index by chunk_id
                index_by_section[chunk["chunk_id"]] = chunk
                index_by_section[chunk["chunk_id"].lower()] = chunk
        except Exception as exc:
            logger.error("Error processing knowledge base file '%s': %s", md_file, exc)
            raise

    # Populate module cache
    _CACHED_CHUNKS = all_chunks
    _CACHED_INDEX_BY_SECTION_ID = index_by_section
    _CACHED_KB_DIR = target_dir_str

    logger.info(
        "Successfully loaded and chunked %d sections from %d files in %s",
        len(all_chunks),
        len(md_files),
        target_dir,
    )
    return _CACHED_CHUNKS


def get_chunk_by_section_id(
    section_id: str,
    kb_dir: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single knowledge base chunk by its section identifier.

    Supports exact section ID matching (e.g. 'RET-2'), case-insensitive matching
    (e.g. 'ret-2'), and chunk_id matching (e.g. 'POL-RET_RET-2').

    Args:
        section_id: The section identifier to search for (e.g. 'RET-2', 'KB-00.1').
        kb_dir: Optional directory to load from if not already cached.

    Returns:
        Optional[Dict[str, Any]]: The matching chunk dictionary, or None if not found.
    """
    if not section_id:
        return None

    # Ensure chunks are loaded into memory
    global _CACHED_CHUNKS, _CACHED_INDEX_BY_SECTION_ID
    if _CACHED_CHUNKS is None or (kb_dir is not None and str(Path(kb_dir).resolve()) != _CACHED_KB_DIR):
        load_and_chunk_kb(kb_dir=kb_dir)

    clean_sec_id = section_id.strip()

    # Fast O(1) index lookup
    if clean_sec_id in _CACHED_INDEX_BY_SECTION_ID:
        return _CACHED_INDEX_BY_SECTION_ID[clean_sec_id]

    lower_sec_id = clean_sec_id.lower()
    if lower_sec_id in _CACHED_INDEX_BY_SECTION_ID:
        return _CACHED_INDEX_BY_SECTION_ID[lower_sec_id]

    # Fallback linear scan if index miss
    for chunk in (_CACHED_CHUNKS or []):
        if chunk.get("section_id") == clean_sec_id:
            return chunk
        if chunk.get("chunk_id") == clean_sec_id:
            return chunk
        if chunk.get("section_id", "").lower() == lower_sec_id:
            return chunk
        if chunk.get("chunk_id", "").lower() == lower_sec_id:
            return chunk

    return None


def get_all_section_ids(kb_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return a list of all section IDs present in the knowledge base.

    Args:
        kb_dir: Optional directory to load from if not already cached.

    Returns:
        List[str]: List of unique section IDs in order of document appearance.
    """
    chunks = load_and_chunk_kb(kb_dir=kb_dir)
    return [chunk["section_id"] for chunk in chunks]


def filter_chunks(
    doc_type: Optional[str] = None,
    status: Optional[str] = None,
    language: Optional[str] = None,
    kb_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Filter knowledge base chunks by document type, status, or language.

    Args:
        doc_type: Optional document type filter (e.g. 'policy', 'help').
        status: Optional status filter (e.g. 'current', 'superseded').
        language: Optional language filter (e.g. 'en', 'hi').
        kb_dir: Optional knowledge base directory path.

    Returns:
        List[Dict[str, Any]]: Filtered list of chunk dictionaries.
    """
    chunks = load_and_chunk_kb(kb_dir=kb_dir)
    filtered = chunks

    if doc_type is not None:
        target_doc_type = doc_type.strip().lower()
        filtered = [c for c in filtered if c.get("doc_type", "").lower() == target_doc_type]

    if status is not None:
        target_status = status.strip().lower()
        filtered = [c for c in filtered if c.get("status", "").lower() == target_status]

    if language is not None:
        target_lang = language.strip().lower()
        filtered = [c for c in filtered if c.get("language", "").lower() == target_lang]

    return filtered
