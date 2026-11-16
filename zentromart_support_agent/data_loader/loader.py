"""Data loader module for ZentroMart AI Customer Support Agent.

This module provides lazy-loading, caching, accessor functions, join helpers,
and multi-value cell parsing for all ZentroMart CSV datasets.

Key Design Principles:
1. Loads all CSVs with pandas `dtype=str, keep_default_na=False` to preserve
   raw formats without unwanted string coercion (e.g. 'NA' codes or leading zeros).
2. Converts numeric columns (e.g. price_inr, total_inr, qty) to integers
   (using standard `int64` or nullable `Int64` where missing values exist).
3. Preserves all identifiers, pincodes, phone numbers, and payment references as text strings.
4. Uses module-level caching for lazy on-demand loading of datasets.
5. Provides utilities to parse and format multi-value cells separated by ' ; '.
6. Provides join helpers for orders with items, customer orders, and product lookups.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)

# Fallback path definitions if config cannot be imported
_DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "ZENTROMART_DATASET_DIR",
        r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026",
    )
).resolve()

_DATA_DIR: Path = _DEFAULT_DATASET_DIR / "data"

try:
    from zentromart_support_agent.config import DATA_DIR as _CONFIG_DATA_DIR
    _DATA_DIR = Path(_CONFIG_DATA_DIR).resolve()
except ImportError:
    # Use fallback path if zentromart_support_agent is not installed in sys.path
    pass

# Module-level cache for lazy-loaded DataFrames
_CACHE: Dict[str, pd.DataFrame] = {}

# Delimiter used in multi-value CSV cells
LIST_SEPARATOR: str = " ; "

# Identifier and text columns that must never be coerced to numeric
TEXT_COLUMNS: set[str] = {
    "order_id",
    "customer_id",
    "delivery_pincode",
    "pincode",
    "phone",
    "sku",
    "return_sku",
    "damage_sku",
    "cancellation_rrn",
    "rto_rrn",
    "return_rrn",
    "log_id",
}

# Mapping of table names to numeric columns
# non_nullable: columns with no missing values (converted to int64)
# nullable: columns that may have missing values (converted to Int64)
NUMERIC_COLUMNS_SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "products": {
        "non_nullable": ["price_inr", "warranty_months"],
        "nullable": [],
    },
    "order_items": {
        "non_nullable": ["line_no", "qty", "unit_price_inr"],
        "nullable": [],
    },
    "orders": {
        "non_nullable": [
            "delivery_attempts",
            "subtotal_inr",
            "delivery_fee_inr",
            "cod_fee_inr",
            "total_inr",
            "contacts_last_7_days",
        ],
        "nullable": [
            "coupon_discount_inr",
            "coupon_min_order_inr",
            "cancellation_refund_amount_inr",
            "rto_refund_amount_inr",
            "express_late_credit_inr",
            "return_line_no",
            "return_qty",
            "return_refund_amount_inr",
            "return_reverse_pickup_fee_inr",
            "damage_hours_after_delivery",
        ],
    },
    "support_logs": {
        "non_nullable": ["csat"],
        "nullable": [],
    },
    "customers": {
        "non_nullable": [],
        "nullable": [],
    },
    "pincodes": {
        "non_nullable": [],
        "nullable": [],
    },
    "holidays": {
        "non_nullable": [],
        "nullable": [],
    },
}


def get_data_dir() -> Path:
    """Return the currently configured data directory path.

    Returns:
        Path: Path to the directory containing data CSV files.
    """
    return _DATA_DIR


def set_data_dir(path: Union[str, Path]) -> None:
    """Override the data directory path and clear the cache.

    Args:
        path: New path to the directory containing data CSV files.
    """
    global _DATA_DIR
    _DATA_DIR = Path(path).resolve()
    clear_cache()
    logger.info("Data directory updated to: %s", _DATA_DIR)


def clear_cache() -> None:
    """Clear all cached DataFrames from memory."""
    _CACHE.clear()
    logger.debug("Data loader cache cleared.")


def parse_list_cell(
    value: Union[str, Sequence[str], None],
    separator: str = LIST_SEPARATOR,
) -> List[str]:
    """Parse a cell value containing list items separated by a delimiter.

    Args:
        value: Cell value to parse (string, sequence, or None/NA).
        separator: Delimiter separating individual items (default: ' ; ').

    Returns:
        List[str]: Cleaned list of item strings, or empty list if input is empty/None.

    Examples:
        >>> parse_list_cell("2026-10-12 placed ; 2026-10-13 shipped")
        ['2026-10-12 placed', '2026-10-13 shipped']
        >>> parse_list_cell("")
        []
        >>> parse_list_cell(None)
        []
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if pd.isna(value) or value == "":
        return []
    val_str = str(value)
    if separator in val_str:
        return [item.strip() for item in val_str.split(separator) if item.strip()]
    stripped = val_str.strip()
    return [stripped] if stripped else []


def format_list_cell(
    items: Sequence[Any],
    separator: str = LIST_SEPARATOR,
) -> str:
    """Format a sequence of items into a single string separated by delimiter.

    Args:
        items: Sequence of items to format.
        separator: Delimiter to join items with (default: ' ; ').

    Returns:
        str: Formatted string, or empty string if sequence is empty.

    Examples:
        >>> format_list_cell(['WAR-2', 'products.csv:ELE-0012'])
        'WAR-2 ; products.csv:ELE-0012'
        >>> format_list_cell([])
        ''
    """
    if not items:
        return ""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return separator.join(cleaned)


def split_multivalue(
    value: Union[str, Sequence[str], None],
    separator: str = LIST_SEPARATOR,
) -> List[str]:
    """Alias for parse_list_cell to split multi-value cells.

    Args:
        value: Cell value to split.
        separator: Delimiter separating values (default: ' ; ').

    Returns:
        List[str]: List of individual values.
    """
    return parse_list_cell(value, separator=separator)


def _load_csv_file(table_name: str) -> pd.DataFrame:
    """Load a CSV file, apply initial string parsing, and convert numeric types.

    Args:
        table_name: Name of the table (e.g. 'products', 'orders').

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame with converted numeric types.

    Raises:
        FileNotFoundError: If the CSV file does not exist in DATA_DIR.
        ValueError: If there is an issue parsing the CSV file.
    """
    file_path = _DATA_DIR / f"{table_name}.csv"
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file '{table_name}.csv' not found at expected path: {file_path}. "
            f"Please verify DATA_DIR or ZENTROMART_DATASET_DIR environment variable."
        )

    try:
        # Load all columns as strings with keep_default_na=False to prevent
        # automatic coercion of empty strings or text like 'NA' into NaN.
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{file_path}': {exc}") from exc

    # Convert numeric columns where appropriate while strictly keeping IDs as strings
    schema = NUMERIC_COLUMNS_SCHEMA.get(table_name, {})
    non_nullable_cols = schema.get("non_nullable", [])
    nullable_cols = schema.get("nullable", [])

    for col in non_nullable_cols:
        if col in df.columns:
            # Non-nullable numeric columns: convert to standard int64
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    for col in nullable_cols:
        if col in df.columns:
            # Nullable numeric columns: convert to pandas nullable Int64 (preserves <NA>)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def _get_or_load(table_name: str, reload: bool = False, copy: bool = True) -> pd.DataFrame:
    """Retrieve a DataFrame from cache or load it lazily from disk.

    Args:
        table_name: Table name to retrieve.
        reload: If True, forces reloading from disk and updates cache.
        copy: If True, returns an independent copy to protect cache integrity.

    Returns:
        pd.DataFrame: Cached or freshly loaded DataFrame.
    """
    if reload or table_name not in _CACHE:
        logger.debug("Loading table '%s' from disk...", table_name)
        _CACHE[table_name] = _load_csv_file(table_name)

    df = _CACHE[table_name]
    return df.copy() if copy else df


# ==============================================================================
# Accessor Functions (Each returns a DataFrame)
# ==============================================================================

def get_products(reload: bool = False) -> pd.DataFrame:
    """Return products DataFrame.

    Columns:
        sku (str): Product SKU ID (e.g. 'ELE-0001').
        title (str): Product name.
        brand (str): Fictional brand name.
        return_category (str): Policy category code (RET-2).
        price_inr (int): Price in Indian Rupees.
        fulfilled_by (str): Seller name or 'ZentroMart'.
        assured (str): 'True' if Assured badge applies.
        warranty_months (int): Brand warranty duration in months.
        description (str): Short product description.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Products dataset.
    """
    return _get_or_load("products", reload=reload)


def get_customers(reload: bool = False) -> pd.DataFrame:
    """Return customers DataFrame.

    Columns:
        customer_id (str): Customer ID (e.g. 'C100000').
        name (str): Customer name.
        phone (str): Registered phone number string.
        email (str): Registered email address.
        city (str): City.
        state (str): State.
        pincode (str): Pincode string.
        is_plus (str): 'True' if ZentroMart Plus member.
        plus_since (str): Date membership started, or empty.
        preferred_language (str): Language code (e.g. 'en', 'hi', 'kn').
        created_at (str): Account creation date.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Customers dataset.
    """
    return _get_or_load("customers", reload=reload)


def get_orders(reload: bool = False) -> pd.DataFrame:
    """Return orders DataFrame.

    Columns:
        order_id (str): 8-digit order identifier.
        customer_id (str): Customer ID foreign key.
        placed_at (str): ISO timestamp of order placement.
        status (str): Status as of reference date (20 Oct 2026).
        payment_method (str): upi, card, netbanking, wallet, cod.
        delivery_type (str): standard, express, slot.
        delivery_pincode (str): Delivery destination pincode.
        promised_date (str): Promised delivery date (YYYY-MM-DD).
        shipped_on (str): Date shipped, or empty string.
        delivered_on (str): Date delivered, or empty string.
        delivery_attempts (int): Number of delivery attempts.
        subtotal_inr (int): Item subtotal in rupees.
        coupon_code (str): Applied coupon, or empty string.
        coupon_discount_inr (Int64): Discount amount, or NA.
        coupon_min_order_inr (Int64): Minimum order requirement, or NA.
        delivery_fee_inr (int): Shipping fee in rupees.
        cod_fee_inr (int): Cash on delivery fee in rupees.
        total_inr (int): Total charged amount in rupees.
        contacts_last_7_days (int): Support contacts in last 7 days.
        ...
        timeline (str): Event timeline separated by ' ; '.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Orders dataset.
    """
    return _get_or_load("orders", reload=reload)


def get_order_items(reload: bool = False) -> pd.DataFrame:
    """Return order_items DataFrame.

    Columns:
        order_id (str): Order identifier foreign key.
        line_no (int): Line item sequence number (1-based).
        sku (str): Product SKU identifier.
        title (str): Product title.
        return_category (str): Return policy category code.
        qty (int): Quantity ordered.
        unit_price_inr (int): Unit price in rupees.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Order items dataset.
    """
    return _get_or_load("order_items", reload=reload)


def get_pincodes(reload: bool = False) -> pd.DataFrame:
    """Return pincodes DataFrame.

    Columns:
        city (str): City name.
        state (str): State name.
        pincode (str): 6-digit pincode string.
        metro (str): 'True' if metro tier location.
        serviceable (str): 'True' if serviceable.
        cod_available (str): 'True' if COD is available.
        express_available (str): 'True' if express delivery available.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Pincodes dataset.
    """
    return _get_or_load("pincodes", reload=reload)


def get_holidays(reload: bool = False) -> pd.DataFrame:
    """Return holidays DataFrame.

    Columns:
        date (str): Holiday date (YYYY-MM-DD).
        name (str): Name of public/warehouse holiday.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Warehouse holidays dataset.
    """
    return _get_or_load("holidays", reload=reload)


def get_support_logs(reload: bool = False) -> pd.DataFrame:
    """Return support_logs DataFrame.

    Columns:
        log_id (str): Support interaction ID (e.g. 'L200040').
        channel (str): voice, chat, email.
        started_at (str): Interaction start timestamp.
        language (str): Language code.
        customer_id (str): Customer ID.
        order_id (str): Order ID, or empty.
        intent (str): Primary user intent.
        handled_by (str): human_agent or bot.
        transcript (str): Conversation dialogue.
        resolution (str): Outcome code.
        escalated (str): 'True' if escalated.
        csat (int): Customer satisfaction score (1-5).
        qa_reviewed (str): 'True' if reviewed.
        qa_findings (str): Quality findings, or empty.

    Args:
        reload: If True, force reloads from disk.

    Returns:
        pd.DataFrame: Support interaction logs dataset.
    """
    return _get_or_load("support_logs", reload=reload)


def load_all(reload: bool = False) -> Dict[str, pd.DataFrame]:
    """Load and return all 7 datasets as a dictionary of DataFrames.

    Args:
        reload: If True, force reloads all datasets from disk.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of dataset names to DataFrames.
    """
    return {
        "products": get_products(reload=reload),
        "customers": get_customers(reload=reload),
        "orders": get_orders(reload=reload),
        "order_items": get_order_items(reload=reload),
        "pincodes": get_pincodes(reload=reload),
        "holidays": get_holidays(reload=reload),
        "support_logs": get_support_logs(reload=reload),
    }


# ==============================================================================
# Helper Function for Converting Dict Values Cleanly
# ==============================================================================

def _row_to_clean_dict(row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a row Series or raw dict into a clean Python dictionary.

    - Empty strings ('') and pandas missing values (<NA>, NaN) become None.
    - Boolean strings ('True', 'False') become boolean True / False.
    - Preserves string IDs and integers.

    Args:
        row: Series or dictionary to clean.

    Returns:
        Dict[str, Any]: Clean dictionary.
    """
    raw_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    out: Dict[str, Any] = {}

    for k, v in raw_dict.items():
        if pd.isna(v) or v == "":
            out[k] = None
        elif k not in TEXT_COLUMNS and str(v) in ("True", "False"):
            out[k] = (str(v) == "True")
        elif hasattr(v, "item"):
            # Convert NumPy/pandas scalar types (e.g. np.int64) to native Python types
            out[k] = v.item()
        else:
            out[k] = v

    return out


# ==============================================================================
# Join Helpers
# ==============================================================================

def get_order_with_items(
    order_id: Union[str, int],
    parse_lists: bool = True,
    include_customer: bool = False,
) -> Dict[str, Any]:
    """Look up an order and join its associated line items.

    Args:
        order_id: Order identifier (e.g. '69520867' or integer 69520867).
        parse_lists: If True, multi-value cells like 'timeline' are parsed into
            a list of strings, while raw string is retained in 'timeline_raw'.
        include_customer: If True, attaches customer information to the dictionary.

    Returns:
        Dict[str, Any]: Dictionary representing the order with an 'items' list of
        line item dictionaries. Returns an empty dictionary `{}` if order not found.

    Examples:
        >>> order = get_order_with_items('69520867')
        >>> order['order_id']
        '69520867'
        >>> len(order['items'])
        2
        >>> isinstance(order['timeline'], list)
        True
    """
    oid = str(order_id).strip()
    if not oid:
        return {}

    orders_df = get_orders()
    matching_orders = orders_df[orders_df["order_id"] == oid]

    if matching_orders.empty:
        logger.debug("Order ID '%s' not found.", oid)
        return {}

    order_row = matching_orders.iloc[0]
    order_dict = _row_to_clean_dict(order_row)

    # Handle multi-value cells like 'timeline'
    raw_timeline = order_row.get("timeline", "")
    timeline_items = parse_list_cell(raw_timeline)
    order_dict["timeline_list"] = timeline_items
    order_dict["timeline_raw"] = raw_timeline

    if parse_lists:
        order_dict["timeline"] = timeline_items
    else:
        order_dict["timeline"] = raw_timeline

    # Fetch and join order items
    items_df = get_order_items()
    matching_items = items_df[items_df["order_id"] == oid]

    items_list: List[Dict[str, Any]] = []
    if not matching_items.empty:
        for _, item_row in matching_items.iterrows():
            item_clean = _row_to_clean_dict(item_row)
            items_list.append(item_clean)

    order_dict["items"] = items_list

    # Optionally attach customer details
    if include_customer and order_dict.get("customer_id"):
        cid = str(order_dict["customer_id"]).strip()
        customers_df = get_customers()
        matching_cust = customers_df[customers_df["customer_id"] == cid]
        if not matching_cust.empty:
            cust_dict = _row_to_clean_dict(matching_cust.iloc[0])
            order_dict["customer"] = cust_dict
            order_dict["customer_first_name"] = (cust_dict.get("name") or "").split()[0]
            order_dict["is_plus"] = cust_dict.get("is_plus")

    return order_dict


def get_customer_orders(customer_id: Union[str, int]) -> pd.DataFrame:
    """Return all orders placed by a specific customer.

    Args:
        customer_id: Customer ID (e.g. 'C100124').

    Returns:
        pd.DataFrame: Filtered orders DataFrame for that customer.
        Returns an empty DataFrame with orders columns if no orders are found.

    Examples:
        >>> orders = get_customer_orders('C100124')
        >>> len(orders)
        4
    """
    cid = str(customer_id).strip()
    orders_df = get_orders()

    if not cid:
        return orders_df.iloc[0:0].copy()

    filtered = orders_df[orders_df["customer_id"] == cid]
    return filtered.copy().reset_index(drop=True)


def get_product_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    """Look up a single product by SKU ID.

    Args:
        sku: Product SKU code (e.g. 'ELE-0001').

    Returns:
        Optional[Dict[str, Any]]: Clean dictionary of product details,
        or None if SKU is not found.

    Examples:
        >>> prod = get_product_by_sku('ELE-0001')
        >>> prod['title']
        'Nimbus X2 Wireless Earbuds, Beige'
        >>> prod['price_inr']
        2199
        >>> get_product_by_sku('NONEXISTENT') is None
        True
    """
    s = str(sku).strip()
    if not s:
        return None

    products_df = get_products()
    matching = products_df[products_df["sku"] == s]

    if matching.empty:
        return None

    product_row = matching.iloc[0]
    return _row_to_clean_dict(product_row)
