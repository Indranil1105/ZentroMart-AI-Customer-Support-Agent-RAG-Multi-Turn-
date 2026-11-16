"""Data loader package for ZentroMart AI Customer Support Agent."""

try:
    from zentromart_support_agent.data_loader.loader import (
        LIST_SEPARATOR, clear_cache, format_list_cell, get_customer_orders,
        get_customers, get_data_dir, get_holidays, get_order_items,
        get_order_with_items, get_orders, get_pincodes, get_product_by_sku,
        get_products, get_support_logs, load_all, parse_list_cell,
        set_data_dir, split_multivalue,
    )
except ImportError:
    from data_loader.loader import (
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

__all__ = [
    "LIST_SEPARATOR",
    "clear_cache",
    "format_list_cell",
    "get_customer_orders",
    "get_customers",
    "get_data_dir",
    "get_holidays",
    "get_order_items",
    "get_order_with_items",
    "get_orders",
    "get_pincodes",
    "get_product_by_sku",
    "get_products",
    "get_support_logs",
    "load_all",
    "parse_list_cell",
    "set_data_dir",
    "split_multivalue",
]
