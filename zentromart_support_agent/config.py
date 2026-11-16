"""Central configuration module for ZentroMart AI Customer Support Agent.

This module defines directory paths, reference constants, document type precedence,
embedding configurations, and business calculation rules for the ZentroMart project.
"""

from datetime import date
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

# Base Project and Dataset Paths
PROJECT_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = Path(
    os.environ.get(
        "ZENTROMART_DATASET_DIR",
        r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026",
    )
).resolve()

# Dataset Subdirectories
KB_DIR: Path = DATASET_DIR / "kb"
DATA_DIR: Path = DATASET_DIR / "data"
PRACTICE_DIR: Path = DATASET_DIR / "practice"
TOOLS_DIR: Path = DATASET_DIR / "tools"

# Chroma Vector Database Directory
CHROMA_PERSIST_DIR: Path = PROJECT_DIR / "chroma_db"

# Data CSV Files
CUSTOMERS_CSV: Path = DATA_DIR / "customers.csv"
ORDERS_CSV: Path = DATA_DIR / "orders.csv"
ORDER_ITEMS_CSV: Path = DATA_DIR / "order_items.csv"
PRODUCTS_CSV: Path = DATA_DIR / "products.csv"
PINCODES_CSV: Path = DATA_DIR / "pincodes.csv"
HOLIDAYS_CSV: Path = DATA_DIR / "holidays.csv"
SUPPORT_LOGS_CSV: Path = DATA_DIR / "support_logs.csv"

# Practice Files
QUESTIONS_CSV: Path = PRACTICE_DIR / "questions.csv"
CONVERSATIONS_CSV: Path = PRACTICE_DIR / "conversations.csv"

# Global Freeze Date for Calculations
# All timelines (deadlines, delays, refund calculations) must use this reference date.
REFERENCE_DATE: date = date(2026, 10, 20)

# Embedding Model Configuration
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Document Precedence Hierarchy
# When two documents disagree, precedence is: policy > agent_guide > help > faq > marketing
DOC_TYPE_PRECEDENCE: List[str] = [
    "policy",
    "agent_guide",
    "help",
    "faq",
    "marketing",
]

DOC_PRECEDENCE_RANKS: Dict[str, int] = {
    doc_type: rank for rank, doc_type in enumerate(DOC_TYPE_PRECEDENCE, start=1)
}

# Calendar and Business Day Rules
# Business days are Monday (0) through Saturday (5), excluding warehouse holidays.
BUSINESS_DAYS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Return windows are counted in calendar days.
RETURN_WINDOW_UNIT: str = "calendar_days"

# Section ID Preservation Pattern (e.g., [RET-2], [KB-00.1])
SECTION_ID_PATTERN: re.Pattern = re.compile(r"\[([A-Za-z0-9_.-]+)\]")


def get_doc_precedence_rank(doc_type: Optional[str]) -> int:
    """Return the precedence rank of a document type.

    Lower number indicates higher precedence.
    Unknown or unspecified types default to a low priority (99).

    Args:
        doc_type: The document type string (e.g. 'policy', 'agent_guide').

    Returns:
        int: Precedence rank (1 = highest).
    """
    if not doc_type:
        return 99
    return DOC_PRECEDENCE_RANKS.get(doc_type.lower().strip(), 99)


def validate_paths() -> Dict[str, bool]:
    """Validate that expected directories and data files exist on the filesystem.

    Returns:
        Dict[str, bool]: Mapping of path descriptions to boolean existence status.
    """
    paths_to_check = {
        "dataset_dir": DATASET_DIR.is_dir(),
        "kb_dir": KB_DIR.is_dir(),
        "data_dir": DATA_DIR.is_dir(),
        "practice_dir": PRACTICE_DIR.is_dir(),
        "tools_dir": TOOLS_DIR.is_dir(),
        "customers_csv": CUSTOMERS_CSV.is_file(),
        "orders_csv": ORDERS_CSV.is_file(),
        "order_items_csv": ORDER_ITEMS_CSV.is_file(),
        "products_csv": PRODUCTS_CSV.is_file(),
        "pincodes_csv": PINCODES_CSV.is_file(),
        "holidays_csv": HOLIDAYS_CSV.is_file(),
        "questions_csv": QUESTIONS_CSV.is_file(),
        "conversations_csv": CONVERSATIONS_CSV.is_file(),
    }
    return paths_to_check
