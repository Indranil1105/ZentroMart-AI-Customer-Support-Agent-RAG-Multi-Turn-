"""Unit tests for the ZentroMart data_loader module using unittest."""

import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from zentromart_support_agent.data_loader import (
    LIST_SEPARATOR,
    clear_cache,
    format_list_cell,
    get_customer_orders,
    get_customers,
    get_data_dir,
    get_holidays,
    get_order_items,
    get_order_with_items,
    get_orders,
    get_pincodes,
    get_product_by_sku,
    get_products,
    get_support_logs,
    load_all,
    parse_list_cell,
    set_data_dir,
    split_multivalue,
)
import zentromart_support_agent.data_loader.loader as loader_module


class TestDataLoaderBasics(unittest.TestCase):
    """Test basic loading and row counts for all 7 CSV datasets."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_get_products(self):
        df = get_products()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 70)
        self.assertIn("sku", df.columns)
        self.assertIn("price_inr", df.columns)
        self.assertIn("warranty_months", df.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(df["price_inr"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["warranty_months"]))
        # Check IDs remain strings
        self.assertIsInstance(df["sku"].iloc[0], str)

    def test_get_customers(self):
        df = get_customers()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 300)
        self.assertIn("customer_id", df.columns)
        self.assertIn("pincode", df.columns)
        self.assertIn("phone", df.columns)
        # Text/ID columns must remain strings
        self.assertIsInstance(df["customer_id"].iloc[0], str)
        self.assertIsInstance(df["pincode"].iloc[0], str)
        self.assertIsInstance(df["phone"].iloc[0], str)

    def test_get_orders(self):
        df = get_orders()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertIn("order_id", df.columns)
        self.assertIn("delivery_pincode", df.columns)
        self.assertIn("total_inr", df.columns)
        self.assertIn("subtotal_inr", df.columns)
        # Check string IDs
        self.assertIsInstance(df["order_id"].iloc[0], str)
        self.assertIsInstance(df["delivery_pincode"].iloc[0], str)
        # Check numeric conversions
        self.assertTrue(pd.api.types.is_integer_dtype(df["total_inr"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["subtotal_inr"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["delivery_attempts"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["coupon_discount_inr"]))

    def test_get_order_items(self):
        df = get_order_items()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1783)
        self.assertIn("order_id", df.columns)
        self.assertIn("sku", df.columns)
        self.assertIn("line_no", df.columns)
        self.assertIn("qty", df.columns)
        self.assertIn("unit_price_inr", df.columns)
        # String IDs
        self.assertIsInstance(df["order_id"].iloc[0], str)
        self.assertIsInstance(df["sku"].iloc[0], str)
        # Numeric integers
        self.assertTrue(pd.api.types.is_integer_dtype(df["line_no"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["qty"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["unit_price_inr"]))

    def test_get_pincodes(self):
        df = get_pincodes()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 36)
        self.assertIn("pincode", df.columns)
        self.assertIsInstance(df["pincode"].iloc[0], str)

    def test_get_holidays(self):
        df = get_holidays()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertIn("date", df.columns)
        self.assertIn("name", df.columns)

    def test_get_support_logs(self):
        df = get_support_logs()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 244)
        self.assertIn("log_id", df.columns)
        self.assertIn("csat", df.columns)
        self.assertIsInstance(df["log_id"].iloc[0], str)
        self.assertTrue(pd.api.types.is_integer_dtype(df["csat"]))

    def test_load_all(self):
        all_data = load_all()
        self.assertIsInstance(all_data, dict)
        expected_keys = {"products", "customers", "orders", "order_items", "pincodes", "holidays", "support_logs"}
        self.assertEqual(set(all_data.keys()), expected_keys)
        for name, df in all_data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)


class TestLazyLoadingAndCache(unittest.TestCase):
    """Test lazy loading and caching behavior."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_lazy_loading(self):
        # Cache starts empty
        self.assertEqual(len(loader_module._CACHE), 0)
        
        # Accessing only products should only load products
        df_prod = get_products()
        self.assertIn("products", loader_module._CACHE)
        self.assertNotIn("orders", loader_module._CACHE)
        self.assertEqual(len(loader_module._CACHE), 1)

        # Accessing orders adds orders to cache
        df_orders = get_orders()
        self.assertIn("orders", loader_module._CACHE)
        self.assertEqual(len(loader_module._CACHE), 2)

    def test_cache_reuse(self):
        df1 = get_products()
        cache_id1 = id(loader_module._CACHE["products"])
        
        # Second call should reuse cached DataFrame
        df2 = get_products()
        cache_id2 = id(loader_module._CACHE["products"])
        self.assertEqual(cache_id1, cache_id2)

    def test_reload_flag(self):
        get_products()
        orig_obj = loader_module._CACHE["products"]
        
        # Reload=True replaces the cached object
        get_products(reload=True)
        new_obj = loader_module._CACHE["products"]
        self.assertIsNot(orig_obj, new_obj)

    def test_clear_cache(self):
        get_products()
        get_customers()
        self.assertEqual(len(loader_module._CACHE), 2)
        clear_cache()
        self.assertEqual(len(loader_module._CACHE), 0)


class TestMultiValueParsing(unittest.TestCase):
    """Test parsing and formatting of ' ; ' delimited cells."""

    def test_parse_list_cell_basic(self):
        raw = "2026-10-12T08:37 placed ; 2026-10-12T10:08 packed ; 2026-10-13T20:46 shipped"
        parsed = parse_list_cell(raw)
        self.assertEqual(parsed, [
            "2026-10-12T08:37 placed",
            "2026-10-12T10:08 packed",
            "2026-10-13T20:46 shipped",
        ])

    def test_parse_list_cell_empty_and_none(self):
        self.assertEqual(parse_list_cell(""), [])
        self.assertEqual(parse_list_cell(None), [])
        self.assertEqual(parse_list_cell("   "), [])
        self.assertEqual(parse_list_cell(pd.NA), [])

    def test_parse_list_cell_single_value(self):
        self.assertEqual(parse_list_cell("single_event"), ["single_event"])

    def test_parse_list_cell_already_list(self):
        self.assertEqual(parse_list_cell(["a", "b"]), ["a", "b"])

    def test_split_multivalue_alias(self):
        self.assertEqual(split_multivalue("A ; B"), ["A", "B"])

    def test_format_list_cell(self):
        items = ["A", "B", "C"]
        formatted = format_list_cell(items)
        self.assertEqual(formatted, "A ; B ; C")
        self.assertEqual(format_list_cell([]), "")


class TestJoinHelpers(unittest.TestCase):
    """Test get_order_with_items, get_customer_orders, and get_product_by_sku."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_get_order_with_items_success(self):
        order_dict = get_order_with_items("69520867")
        self.assertIsInstance(order_dict, dict)
        self.assertEqual(order_dict["order_id"], "69520867")
        self.assertEqual(order_dict["customer_id"], "C100124")
        self.assertEqual(order_dict["total_inr"], 3778)
        self.assertIsInstance(order_dict["total_inr"], int)
        
        # Check timeline list parsing
        self.assertIsInstance(order_dict["timeline"], list)
        self.assertEqual(len(order_dict["timeline"]), 3)
        self.assertEqual(
            order_dict["timeline_raw"],
            "2026-10-12T08:37 placed ; 2026-10-12T10:08 packed ; 2026-10-13T20:46 shipped"
        )
        self.assertEqual(order_dict["timeline_list"], order_dict["timeline"])

        # Check items
        self.assertIn("items", order_dict)
        items = order_dict["items"]
        self.assertEqual(len(items), 2)
        line1 = next(it for it in items if it["line_no"] == 1)
        self.assertEqual(line1["sku"], "HOM-0044")
        self.assertEqual(line1["qty"], 1)
        self.assertEqual(line1["unit_price_inr"], 3039)
        self.assertIsInstance(line1["unit_price_inr"], int)

    def test_get_order_with_items_as_int(self):
        order_dict = get_order_with_items(69520867)
        self.assertEqual(order_dict["order_id"], "69520867")

    def test_get_order_with_items_not_found(self):
        res = get_order_with_items("99999999")
        self.assertEqual(res, {})
        self.assertEqual(get_order_with_items(""), {})

    def test_get_order_with_items_include_customer(self):
        order_dict = get_order_with_items("69520867", include_customer=True)
        self.assertIn("customer", order_dict)
        self.assertEqual(order_dict["customer"]["customer_id"], "C100124")
        self.assertIn("customer_first_name", order_dict)
        self.assertIn("is_plus", order_dict)

    def test_get_customer_orders_success(self):
        df_orders = get_customer_orders("C100124")
        self.assertIsInstance(df_orders, pd.DataFrame)
        self.assertEqual(len(df_orders), 4)
        self.assertEqual(set(df_orders["order_id"]), {"69520867", "48812628", "13396965", "57269519"})

    def test_get_customer_orders_not_found(self):
        df_orders = get_customer_orders("NONEXISTENT_CUST")
        self.assertIsInstance(df_orders, pd.DataFrame)
        self.assertEqual(len(df_orders), 0)
        self.assertIn("order_id", df_orders.columns)

    def test_get_product_by_sku_success(self):
        prod = get_product_by_sku("ELE-0001")
        self.assertIsInstance(prod, dict)
        self.assertEqual(prod["sku"], "ELE-0001")
        self.assertEqual(prod["title"], "Nimbus X2 Wireless Earbuds, Beige")
        self.assertEqual(prod["price_inr"], 2199)
        self.assertIsInstance(prod["price_inr"], int)
        self.assertEqual(prod["warranty_months"], 12)
        self.assertIsInstance(prod["warranty_months"], int)

    def test_get_product_by_sku_not_found(self):
        self.assertIsNone(get_product_by_sku("NONEXISTENT_SKU"))
        self.assertIsNone(get_product_by_sku(""))


class TestErrorHandlingAndDirectoryOverrides(unittest.TestCase):
    """Test setting custom directory and error handling."""

    def test_invalid_directory_raises_error(self):
        orig_dir = get_data_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                set_data_dir(Path(tmpdir) / "nonexistent")
                with self.assertRaises(FileNotFoundError):
                    get_products()
            finally:
                set_data_dir(orig_dir)


if __name__ == "__main__":
    unittest.main()
