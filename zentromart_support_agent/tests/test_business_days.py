"""Unit tests for zentromart_support_agent.agent.business_days module."""

from datetime import date, datetime
import unittest

from zentromart_support_agent.agent.business_days import (
    REFERENCE_DATE,
    HOLIDAYS,
    HOLIDAY_NAMES,
    is_business_day,
    count_business_days,
    add_business_days,
    days_since,
    business_days_since,
    load_holidays,
)


class TestBusinessDays(unittest.TestCase):
    """Test suite covering all business day and calendar day calculations."""

    def test_reference_date(self):
        """Verify the global reference date is frozen at 20 October 2026."""
        self.assertEqual(REFERENCE_DATE, date(2026, 10, 20))

    def test_holidays_loaded(self):
        """Verify the 4 required company holidays are loaded on import."""
        expected_holidays = {
            date(2026, 8, 15),  # Independence Day (Saturday)
            date(2026, 10, 2),  # Gandhi Jayanti (Friday)
            date(2026, 10, 19),  # Warehouse closure (Monday)
            date(2026, 11, 9),  # Warehouse closure (Monday)
        }
        for h in expected_holidays:
            self.assertIn(h, HOLIDAYS)
        self.assertGreaterEqual(len(HOLIDAYS), 4)

    def test_is_business_day_regular_weekdays_and_saturdays(self):
        """Monday through Saturday are business days unless declared a holiday."""
        # 2026-10-12 is Monday (not a holiday)
        self.assertTrue(is_business_day(date(2026, 10, 12)))
        # 2026-10-14 is Wednesday
        self.assertTrue(is_business_day(date(2026, 10, 14)))
        # 2026-10-17 is Saturday
        self.assertTrue(is_business_day(date(2026, 10, 17)))
        # 2026-10-20 is Tuesday
        self.assertTrue(is_business_day(date(2026, 10, 20)))
        # 2026-10-24 is Saturday
        self.assertTrue(is_business_day(date(2026, 10, 24)))

    def test_is_business_day_sundays(self):
        """Sundays are NEVER business days."""
        # 2026-10-11 is Sunday
        self.assertFalse(is_business_day(date(2026, 10, 11)))
        # 2026-10-18 is Sunday
        self.assertFalse(is_business_day(date(2026, 10, 18)))
        # 2026-10-25 is Sunday
        self.assertFalse(is_business_day(date(2026, 10, 25)))

    def test_is_business_day_holidays(self):
        """Declared holidays are NOT business days, even if falling Mon-Sat."""
        # Saturday holiday
        self.assertFalse(is_business_day(date(2026, 8, 15)))
        # Friday holiday
        self.assertFalse(is_business_day(date(2026, 10, 2)))
        # Monday holiday (warehouse closure festival)
        self.assertFalse(is_business_day(date(2026, 10, 19)))
        # Monday holiday (warehouse closure festival)
        self.assertFalse(is_business_day(date(2026, 11, 9)))

    def test_is_business_day_string_and_datetime(self):
        """String and datetime inputs should be parsed and processed correctly."""
        self.assertTrue(is_business_day("2026-10-20"))
        self.assertFalse(is_business_day("2026-10-19"))  # holiday
        self.assertFalse(is_business_day("2026-10-18"))  # Sunday
        self.assertTrue(is_business_day(datetime(2026, 10, 20, 15, 30)))
        self.assertFalse(is_business_day(datetime(2026, 10, 19, 9, 0)))

    def test_count_business_days_same_date(self):
        """Count between identical dates (exclusive start, inclusive end) is 0."""
        self.assertEqual(count_business_days(date(2026, 10, 20), date(2026, 10, 20)), 0)

    def test_count_business_days_across_weekend_and_holiday(self):
        """Count business days between 17 Oct (Sat) and 20 Oct (Tue).
        Exclusive start (17th), inclusive end (20th):
        - 18 Oct: Sunday (not business day)
        - 19 Oct: Holiday (not business day)
        - 20 Oct: Tuesday (business day #1)
        Total: 1 business day.
        """
        self.assertEqual(count_business_days(date(2026, 10, 17), date(2026, 10, 20)), 1)

    def test_count_business_days_refund_scenarios(self):
        """Test business day spans matching practice QA scenarios:
        14 Oct (Wed) to 17 Oct (Sat): 15(Thu), 16(Fri), 17(Sat) -> 3 business days.
        14 Oct (Wed) to 21 Oct (Wed): 15, 16, 17, 20, 21 -> 5 business days.
        17 Oct (Sat) to 22 Oct (Thu): 20, 21, 22 -> 3 business days.
        17 Oct (Sat) to 24 Oct (Sat): 20, 21, 22, 23, 24 -> 5 business days.
        17 Oct (Sat) to 27 Oct (Tue): 20, 21, 22, 23, 24, 26, 27 -> 7 business days.
        """
        self.assertEqual(count_business_days(date(2026, 10, 14), date(2026, 10, 17)), 3)
        self.assertEqual(count_business_days(date(2026, 10, 14), date(2026, 10, 21)), 5)
        self.assertEqual(count_business_days(date(2026, 10, 17), date(2026, 10, 22)), 3)
        self.assertEqual(count_business_days(date(2026, 10, 17), date(2026, 10, 24)), 5)
        self.assertEqual(count_business_days(date(2026, 10, 17), date(2026, 10, 27)), 7)

    def test_count_business_days_start_greater_than_end(self):
        """When start > end, return signed negative count."""
        self.assertEqual(count_business_days(date(2026, 10, 20), date(2026, 10, 17)), -1)

    def test_add_business_days(self):
        """Test adding business days matching refund SLAs."""
        start = date(2026, 10, 17)  # Saturday
        # 1 business day -> skips Sun 18, Mon 19 (holiday) -> Tue 20 Oct
        self.assertEqual(add_business_days(start, 1), date(2026, 10, 20))
        # 3 business days -> Thu 22 Oct
        self.assertEqual(add_business_days(start, 3), date(2026, 10, 22))
        # 5 business days -> Sat 24 Oct
        self.assertEqual(add_business_days(start, 5), date(2026, 10, 24))
        # 7 business days -> Tue 27 Oct
        self.assertEqual(add_business_days(start, 7), date(2026, 10, 27))

        # Adding 0 returns same date
        self.assertEqual(add_business_days(start, 0), start)

        # Subtracting business days
        self.assertEqual(add_business_days(date(2026, 10, 20), -1), date(2026, 10, 17))

    def test_days_since_calendar_days(self):
        """Test calendar days calculation against default REFERENCE_DATE (2026-10-20)."""
        # Event 10 days ago
        self.assertEqual(days_since(date(2026, 10, 10)), 10)
        # Event on reference date
        self.assertEqual(days_since(date(2026, 10, 20)), 0)
        # Event in future
        self.assertEqual(days_since(date(2026, 10, 25)), -5)
        # Custom reference date
        self.assertEqual(days_since(date(2026, 10, 5), ref_date=date(2026, 10, 15)), 10)
        # String input
        self.assertEqual(days_since("2026-10-10"), 10)

    def test_business_days_since(self):
        """Test business days calculation against default REFERENCE_DATE (2026-10-20)."""
        # 17 Oct (Sat) to 20 Oct (Tue) -> 1 business day
        self.assertEqual(business_days_since(date(2026, 10, 17)), 1)
        # 14 Oct (Wed) to 20 Oct (Tue) -> 15, 16, 17, 20 -> 4 business days
        self.assertEqual(business_days_since(date(2026, 10, 14)), 4)
        # 20 Oct to 20 Oct -> 0 business days
        self.assertEqual(business_days_since(date(2026, 10, 20)), 0)
        # String input
        self.assertEqual(business_days_since("2026-10-17"), 1)

    def test_invalid_inputs(self):
        """Test proper error handling for invalid types and formats."""
        with self.assertRaises(TypeError):
            is_business_day(12345)  # type: ignore

        with self.assertRaises(ValueError):
            is_business_day("not-a-date")

        with self.assertRaises(TypeError):
            add_business_days(date(2026, 10, 20), "3")  # type: ignore


if __name__ == "__main__":
    unittest.main()
