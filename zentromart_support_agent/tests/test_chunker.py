"""Unit tests for the knowledge base chunker module.

Tests YAML frontmatter extraction, section splitting by '##' headings with [SECTION-ID],
sub-section preservation ('###' headings), precedence priorities, in-memory caching,
and retrieval by section ID.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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


class TestFrontmatterParser(unittest.TestCase):
    """Tests for YAML frontmatter parsing and normalization."""

    def test_standard_frontmatter(self):
        content = (
            "---\n"
            "doc_id: POL-RET\n"
            "title: Returns policy\n"
            "doc_type: policy\n"
            "audience: customer\n"
            "language: en\n"
            "version: 4.0\n"
            "effective_from: 2026-08-01\n"
            "status: current\n"
            "supersedes: POL-RET-v3\n"
            "---\n"
            "# Returns policy\n"
        )
        fm, body = parse_frontmatter(content)
        self.assertEqual(fm["doc_id"], "POL-RET")
        self.assertEqual(fm["title"], "Returns policy")
        self.assertEqual(fm["doc_type"], "policy")
        self.assertEqual(fm["audience"], "customer")
        self.assertEqual(fm["language"], "en")
        self.assertEqual(fm["version"], "4.0")
        self.assertEqual(fm["effective_from"], "2026-08-01")
        self.assertEqual(fm["status"], "current")
        self.assertEqual(fm["supersedes"], "POL-RET-v3")
        self.assertIsNone(fm["superseded_by"])
        self.assertIn("# Returns policy", body)

    def test_unquoted_colon_in_title(self):
        """Ensure resilient parsing when titles contain unquoted colons."""
        content = (
            "---\n"
            "doc_id: HLP-RVP\n"
            "title: Reverse pickup: how it works\n"
            "doc_type: help\n"
            "audience: customer\n"
            "language: en\n"
            "version: 1.2\n"
            "effective_from: 2026-08-01\n"
            "status: current\n"
            "---\n"
            "## [RVP-1] Booking\n"
            "Booking details here.\n"
        )
        fm, body = parse_frontmatter(content)
        self.assertEqual(fm["doc_id"], "HLP-RVP")
        self.assertEqual(fm["title"], "Reverse pickup: how it works")
        self.assertEqual(fm["doc_type"], "help")
        self.assertEqual(fm["version"], "1.2")

    def test_missing_frontmatter(self):
        """Content without frontmatter delimiters returns empty metadata and entire body."""
        content = "## [RET-1] Scope\nJust content without header."
        fm, body = parse_frontmatter(content)
        self.assertIsInstance(fm, dict)
        for key in FRONTMATTER_FIELDS:
            self.assertIn(key, fm)
            self.assertIsNone(fm[key])
        self.assertEqual(body, content)


class TestDocumentChunking(unittest.TestCase):
    """Tests for splitting document markdown content into section chunks."""

    def test_subsections_preserved_with_parent(self):
        """Verify '###' sub-sections stay inside the parent '##' section chunk."""
        content = (
            "---\n"
            "doc_id: POL-TEST\n"
            "title: Test Policy\n"
            "doc_type: policy\n"
            "language: en\n"
            "status: current\n"
            "---\n"
            "\n"
            "## [TST-1] Parent Section One\n"
            "Parent body text.\n"
            "\n"
            "### Subsection 1.1\n"
            "Detailed subsection text.\n"
            "\n"
            "### Subsection 1.2\n"
            "Another subsection text.\n"
            "\n"
            "## [TST-2] Parent Section Two\n"
            "Second parent body text.\n"
        )
        chunks = chunk_document(content)
        self.assertEqual(len(chunks), 2)

        chunk1 = chunks[0]
        self.assertEqual(chunk1["section_id"], "TST-1")
        self.assertEqual(chunk1["chunk_id"], "POL-TEST_TST-1")
        self.assertIn("### Subsection 1.1", chunk1["text"])
        self.assertIn("Detailed subsection text.", chunk1["text"])
        self.assertIn("### Subsection 1.2", chunk1["text"])
        self.assertNotIn("Parent Section Two", chunk1["text"])

        chunk2 = chunks[1]
        self.assertEqual(chunk2["section_id"], "TST-2")
        self.assertEqual(chunk2["chunk_id"], "POL-TEST_TST-2")
        self.assertIn("Second parent body text.", chunk2["text"])

    def test_regex_section_id_formats(self):
        """Verify SECTION_ID_REGEX and H2_HEADING_REGEX match all project ID variants."""
        sample_headers = [
            "## [RET-1] Scope",
            "## [RET-2] Return windows by category",
            "## [KB-00.1] Which document wins",
            "## [RETv3-2] Return windows by category",
            "## [FAQH-1] क्या मैं ऑर्डर कैंसिल कर सकता हूँ?",
            "## [MKT-1] Shop more, worry less",
        ]
        expected_ids = ["RET-1", "RET-2", "KB-00.1", "RETv3-2", "FAQH-1", "MKT-1"]
        for header, expected_id in zip(sample_headers, expected_ids):
            m = SECTION_ID_REGEX.search(header)
            self.assertIsNotNone(m, f"Failed to match section ID in '{header}'")
            self.assertEqual(m.group(1), expected_id)

            m_h2 = H2_HEADING_REGEX.search(header)
            self.assertIsNotNone(m_h2, f"Failed to match H2 heading in '{header}'")
            self.assertEqual(m_h2.group(1), expected_id)


class TestLoadAndChunkKB(unittest.TestCase):
    """End-to-end tests loading the actual 21 markdown files from the dataset."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_load_all_chunks_count_and_keys(self):
        """Verify all 84 sections across 21 files are loaded with the exact 10 keys."""
        chunks = load_and_chunk_kb()
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 84)

        for chunk in chunks:
            # Check exact keys present
            self.assertEqual(set(chunk.keys()), set(CHUNK_KEYS))
            self.assertTrue(len(chunk["chunk_id"]) > 0)
            self.assertTrue(len(chunk["section_id"]) > 0)
            self.assertTrue(len(chunk["doc_id"]) > 0)
            self.assertTrue(len(chunk["doc_type"]) > 0)
            self.assertTrue(len(chunk["title"]) > 0)
            self.assertIn(chunk["status"], ("current", "superseded"))
            self.assertIn(chunk["language"], ("en", "hi"))
            self.assertTrue(len(chunk["text"]) > 0)
            self.assertIsInstance(chunk["priority"], int)

    def test_priority_precedence(self):
        """Verify priority numbers: policy=1, agent_guide=2, help=3, faq=4, marketing=5."""
        chunks = load_and_chunk_kb()

        policy_chunks = [c for c in chunks if c["doc_type"] == "policy"]
        agent_guide_chunks = [c for c in chunks if c["doc_type"] == "agent_guide"]
        help_chunks = [c for c in chunks if c["doc_type"] == "help"]
        faq_chunks = [c for c in chunks if c["doc_type"] == "faq"]
        marketing_chunks = [c for c in chunks if c["doc_type"] == "marketing"]
        meta_chunks = [c for c in chunks if c["doc_type"] == "meta"]

        self.assertGreater(len(policy_chunks), 0)
        self.assertGreater(len(agent_guide_chunks), 0)
        self.assertGreater(len(help_chunks), 0)
        self.assertGreater(len(faq_chunks), 0)
        self.assertGreater(len(marketing_chunks), 0)
        self.assertGreater(len(meta_chunks), 0)

        for c in policy_chunks:
            self.assertEqual(c["priority"], 1)
        for c in agent_guide_chunks:
            self.assertEqual(c["priority"], 2)
        for c in help_chunks:
            self.assertEqual(c["priority"], 3)
        for c in faq_chunks:
            self.assertEqual(c["priority"], 4)
        for c in marketing_chunks:
            self.assertEqual(c["priority"], 5)
        for c in meta_chunks:
            self.assertEqual(c["priority"], 99)

    def test_superseded_status(self):
        """Check superseded policy chunks have status='superseded'."""
        ret_v3 = get_chunk_by_section_id("RETv3-2")
        self.assertIsNotNone(ret_v3)
        self.assertEqual(ret_v3["status"], "superseded")
        self.assertEqual(ret_v3["doc_id"], "POL-RET-v3")

        ret_v4 = get_chunk_by_section_id("RET-2")
        self.assertIsNotNone(ret_v4)
        self.assertEqual(ret_v4["status"], "current")
        self.assertEqual(ret_v4["doc_id"], "POL-RET")

    def test_hindi_faq(self):
        """Verify language='hi' and Devanagari text for Hindi FAQ chunks."""
        faqh = get_chunk_by_section_id("FAQH-1")
        self.assertIsNotNone(faqh)
        self.assertEqual(faqh["language"], "hi")
        self.assertEqual(faqh["doc_type"], "faq")
        self.assertEqual(faqh["priority"], 4)
        self.assertIn("ऑर्डर कैंसिल", faqh["text"])


class TestGetChunkBySectionId(unittest.TestCase):
    """Tests for looking up chunks by section identifier."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_lookup_existing_sections(self):
        """Lookup core section IDs used in evaluation gold evidence."""
        ret2 = get_chunk_by_section_id("RET-2")
        self.assertIsNotNone(ret2)
        self.assertEqual(ret2["section_id"], "RET-2")
        self.assertEqual(ret2["doc_id"], "POL-RET")
        self.assertIn("Electronics and accessories | 7 days", ret2["text"])

        kb001 = get_chunk_by_section_id("KB-00.1")
        self.assertIsNotNone(kb001)
        self.assertEqual(kb001["section_id"], "KB-00.1")
        self.assertEqual(kb001["doc_id"], "KB-00")
        self.assertIn("Which document wins", kb001["text"])

        war2 = get_chunk_by_section_id("WAR-2")
        self.assertIsNotNone(war2)
        self.assertEqual(war2["section_id"], "WAR-2")
        self.assertIn("ZentroMart Assured", war2["text"])

    def test_case_insensitive_lookup(self):
        """Lookup by lowercase section ID should succeed."""
        chunk = get_chunk_by_section_id("ret-2")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["section_id"], "RET-2")

        chunk_kb = get_chunk_by_section_id("kb-00.1")
        self.assertIsNotNone(chunk_kb)
        self.assertEqual(chunk_kb["section_id"], "KB-00.1")

    def test_lookup_by_chunk_id(self):
        """Lookup using composite chunk_id should succeed."""
        chunk = get_chunk_by_section_id("POL-RET_RET-2")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["section_id"], "RET-2")

    def test_lookup_nonexistent(self):
        """Lookup of invalid or empty IDs returns None."""
        self.assertIsNone(get_chunk_by_section_id("NONEXISTENT-99"))
        self.assertIsNone(get_chunk_by_section_id(""))
        self.assertIsNone(get_chunk_by_section_id("   "))


class TestCachingBehavior(unittest.TestCase):
    """Tests for lazy loading and module caching."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_caching_returns_same_instance(self):
        """Repeated calls to load_and_chunk_kb should return cached list."""
        chunks1 = load_and_chunk_kb()
        chunks2 = load_and_chunk_kb()
        self.assertIs(chunks1, chunks2)

    def test_clear_cache(self):
        """clear_cache removes cached data forcing reload."""
        chunks1 = load_and_chunk_kb()
        clear_cache()
        chunks2 = load_and_chunk_kb()
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertIsNot(chunks1, chunks2)


class TestErrorHandling(unittest.TestCase):
    """Tests for directory and error validations."""

    def test_nonexistent_directory_raises(self):
        """Providing a non-existent directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_and_chunk_kb(Path("non_existent_kb_dir_404_test"))


class TestFilteringAndHelpers(unittest.TestCase):
    """Tests for get_all_section_ids and filter_chunks utilities."""

    def test_get_all_section_ids(self):
        sec_ids = get_all_section_ids()
        self.assertEqual(len(sec_ids), 84)
        self.assertEqual(len(set(sec_ids)), 84)
        self.assertIn("RET-2", sec_ids)
        self.assertIn("KB-00.1", sec_ids)

    def test_filter_chunks(self):
        hindi_chunks = filter_chunks(language="hi")
        self.assertEqual(len(hindi_chunks), 4)
        for c in hindi_chunks:
            self.assertEqual(c["language"], "hi")

        superseded_chunks = filter_chunks(status="superseded")
        self.assertEqual(len(superseded_chunks), 2)
        for c in superseded_chunks:
            self.assertEqual(c["status"], "superseded")


if __name__ == "__main__":
    unittest.main()
