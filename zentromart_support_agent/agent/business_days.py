"""Business days and date calculation utilities for ZentroMart AI Customer Support Agent.

This module provides business-day and calendar-day arithmetic adhering to ZentroMart's
operational policies:
    - Reference Date: 20 October 2026 (freeze date for all calculations).
    - Business Days: Monday through Saturday, excluding company holidays.
    - Non-Business Days: Sundays and warehouse/company holidays.
    - Return Windows: Counted in calendar days from the delivery date.
    - Refund / Shipping Timelines: Counted in business days from initiation.

Holidays are loaded on module import from the dataset's holidays.csv.
"""

from datetime import date, datetime, timedelta
import csv
import logging
import os
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Module constant for the project's frozen reference date
REFERENCE_DATE: date = date(2026, 10, 20)

# Default known holidays for ZentroMart in 2026 (fallback if CSV is missing)
DEFAULT_KNOWN_HOLIDAYS: Dict[date, str] = {
    date(2026, 8, 15): "Independence Day",
    date(2026, 10, 2): "Gandhi Jayanti",
    date(2026, 10, 19): "Warehouse closure (festival)",
    date(2026, 11, 9): "Warehouse closure (festival)",
}

# Primary file paths to search for holidays.csv
PRIMARY_HOLIDAYS_CSV: Path = Path(
    r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026\data\holidays.csv"
)
RELATIVE_HOLIDAYS_CSV: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "Capstone_Project_Dataset_FTTT_2026"
    / "data"
    / "holidays.csv"
)


def _to_date(val: Union[date, datetime, str]) -> date:
    """Normalize a date, datetime, or ISO string to a datetime.date object.

    Args:
        val: An instance of date, datetime, or an ISO-8601 formatted string (YYYY-MM-DD).

    Returns:
        date: Standardized date object.

    Raises:
        TypeError: If val is not a date, datetime, or string.
        ValueError: If val is a string that cannot be parsed into a valid date.
    """
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        cleaned = val.strip()
        try:
            return date.fromisoformat(cleaned)
        except ValueError:
            # Handle possible ISO datetime strings like '2026-10-20T12:00:00'
            try:
                return datetime.fromisoformat(cleaned).date()
            except ValueError:
                raise ValueError(
                    f"Invalid date string format: {val!r}. Expected YYYY-MM-DD."
                )
    raise TypeError(
        f"Expected date, datetime, or date string, got {type(val).__name__}: {val!r}"
    )


def load_holidays(
    filepath: Optional[Union[str, Path]] = None
) -> Tuple[FrozenSet[date], Dict[date, str]]:
    """Load holiday dates and names from a CSV file.

    Searches in the following order:
      1. Explicit filepath argument (if provided).
      2. Environment variable 'HOLIDAYS_CSV_PATH'.
      3. Primary path in dataset directory.
      4. Relative path from this file.
      5. Fallback to DEFAULT_KNOWN_HOLIDAYS if file cannot be found or read.

    Args:
        filepath: Optional custom path to holidays.csv.

    Returns:
        Tuple[FrozenSet[date], Dict[date, str]]: A tuple containing:
            - A frozenset of holiday dates for O(1) membership checks.
            - A dictionary mapping each holiday date to its description.
    """
    candidate_paths = []
    if filepath:
        candidate_paths.append(Path(filepath))
    env_path = os.environ.get("HOLIDAYS_CSV_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.append(PRIMARY_HOLIDAYS_CSV)
    candidate_paths.append(RELATIVE_HOLIDAYS_CSV)

    resolved_path: Optional[Path] = None
    for p in candidate_paths:
        try:
            if p.is_file():
                resolved_path = p
                break
        except (OSError, PermissionError):
            continue

    if resolved_path is None:
        logger.warning(
            "holidays.csv not found at candidate locations. Using default known holidays."
        )
        return frozenset(DEFAULT_KNOWN_HOLIDAYS.keys()), dict(DEFAULT_KNOWN_HOLIDAYS)

    holiday_set = set()
    holiday_dict = {}
    try:
        with open(resolved_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_date = row.get("date", "").strip()
                name = row.get("name", "").strip()
                if raw_date:
                    try:
                        d = date.fromisoformat(raw_date)
                        holiday_set.add(d)
                        holiday_dict[d] = name
                    except ValueError:
                        logger.warning("Skipping invalid holiday date row: %s", row)
        return frozenset(holiday_set), holiday_dict
    except Exception as exc:
        logger.warning(
            "Failed to read holidays from %s (%s). Using default known holidays.",
            resolved_path,
            exc,
        )
        return frozenset(DEFAULT_KNOWN_HOLIDAYS.keys()), dict(DEFAULT_KNOWN_HOLIDAYS)


# Load holidays once on module import
HOLIDAYS, HOLIDAY_NAMES = load_holidays()


def is_business_day(d: Union[date, datetime, str]) -> bool:
    """Determine whether a given date is a business day for ZentroMart.

    Business days are Monday through Saturday, excluding declared holidays
    (holidays: 2026-08-15, 2026-10-02, 2026-10-19, 2026-11-09).
    Sundays are NOT business days.

    Args:
        d: Date to inspect (date, datetime, or ISO string YYYY-MM-DD).

    Returns:
        bool: True if the date is Monday-Saturday and not a holiday, False otherwise.
    """
    dt = _to_date(d)
    # weekday(): Monday is 0, ..., Saturday is 5, Sunday is 6
    if dt.weekday() == 6:
        return False
    return dt not in HOLIDAYS


def count_business_days(
    start: Union[date, datetime, str],
    end: Union[date, datetime, str],
) -> int:
    """Count business days between start and end (exclusive start, inclusive end).

    Range: (start, end]
      - If start == end, returns 0.
      - If start < end, counts every day d in (start, end] where is_business_day(d) is True.
      - If start > end, returns -count_business_days(end, start).

    Args:
        start: Start date (exclusive).
        end: End date (inclusive).

    Returns:
        int: Number of business days between start and end.
    """
    s = _to_date(start)
    e = _to_date(end)

    if s == e:
        return 0

    if s > e:
        return -count_business_days(e, s)

    count = 0
    current = s + timedelta(days=1)
    while current <= e:
        if is_business_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def add_business_days(start: Union[date, datetime, str], n: int) -> date:
    """Add n business days to a start date.

    The start date itself is not counted (exclusive start).
    If n > 0, advances forward until n business days have elapsed.
    If n < 0, steps backward until abs(n) business days have elapsed.
    If n == 0, returns the start date.

    Args:
        start: Base date (date, datetime, or ISO string).
        n: Number of business days to add (positive, negative, or zero).

    Returns:
        date: The target date resulting from adding n business days.

    Raises:
        TypeError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}: {n!r}")

    current = _to_date(start)
    if n == 0:
        return current

    step = timedelta(days=1 if n > 0 else -1)
    remaining = abs(n)

    while remaining > 0:
        current += step
        if is_business_day(current):
            remaining -= 1

    return current


def days_since(
    event_date: Union[date, datetime, str],
    ref_date: Union[date, datetime, str] = REFERENCE_DATE,
) -> int:
    """Calculate elapsed calendar days from event_date up to ref_date.

    Formula: (ref_date - event_date).days

    Args:
        event_date: Date when the event occurred (e.g. order placed, delivered).
        ref_date: Baseline date for comparison. Defaults to REFERENCE_DATE (2026-10-20).

    Returns:
        int: Calendar days elapsed (positive if event is in the past,
             0 if same day, negative if event is in the future).
    """
    ev = _to_date(event_date)
    ref = _to_date(ref_date)
    return (ref - ev).days


def business_days_since(
    event_date: Union[date, datetime, str],
    ref_date: Union[date, datetime, str] = REFERENCE_DATE,
) -> int:
    """Calculate elapsed business days from event_date up to ref_date.

    Business days are Monday-Saturday excluding company holidays.
    Follows exclusive start, inclusive end convention: (event_date, ref_date].

    Args:
        event_date: Date when the event occurred.
        ref_date: Baseline date for comparison. Defaults to REFERENCE_DATE (2026-10-20).

    Returns:
        int: Business days elapsed (positive if event is in the past,
             0 if same day, negative if event is in the future).
    """
    ev = _to_date(event_date)
    ref = _to_date(ref_date)
    return count_business_days(ev, ref)


__all__ = [
    "REFERENCE_DATE",
    "HOLIDAYS",
    "HOLIDAY_NAMES",
    "load_holidays",
    "is_business_day",
    "count_business_days",
    "add_business_days",
    "days_since",
    "business_days_since",
]
