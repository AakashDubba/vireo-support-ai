"""
utils.py — Data loading, validation, PII masking, and shared constants.

All constants that must remain consistent across modules live here.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Taxonomy (working draft — validated against actual data 2026-10-01) ─────────
# Two renames from the original proposed list:
#   "Connectivity"  -> "Connectivity & Pairing"  (adds most common symptom)
#   "Audio Quality" -> "Audio Quality & Hardware Defect" (broadens to physical defects)
# Taxonomy will be FROZEN only after 20-ticket tuning is complete.
CATEGORIES = [
    "Delivery & Shipping",
    "Billing & Payments",
    "Returns & Refunds",
    "Charging & Battery",
    "Connectivity & Pairing",
    "Audio Quality & Hardware Defect",
    "App, Firmware & Software",
    "Warranty & Repair",
    "Account & Login",
    "Product Enquiry & General",
]

# Prompt version — v2.1 (Master hierarchical classification & policy precedence).
# Cache key includes this; changing it invalidates cached results for this version.
PROMPT_VERSION = "v2.1"

# ── Policy constants (support-policy.pdf §4) ─────────────────────────────────
BLENDED_COST_PER_CONTACT_INR = 290   # per contact, all channels blended
AGENT_COST_PER_HOUR_INR      = 165   # fully loaded, per agent-hour
TRANSFER_COST_INR            = 305   # per inter-team hand-off

# ── Groq model pricing defaults (override via .env) ──────────────────────────
GROQ_PRICE_PER_1M_INPUT  = 0.075   # USD per 1M input  tokens
GROQ_PRICE_PER_1M_OUTPUT = 0.30    # USD per 1M output tokens

# ── Primary business-reporting window ────────────────────────────────────────
PRIMARY_WINDOW_START = pd.Timestamp("2025-01-01")
PRIMARY_WINDOW_END   = pd.Timestamp("2026-06-30 23:59:59")

# ── PII patterns ─────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w.\-]+\.\w{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+91[\s\-]?|0)?[6-9]\d{9}")


# ─────────────────────────────────────────────────────────────────────────────
# PII masking
# ─────────────────────────────────────────────────────────────────────────────

def mask_pii(text: str) -> str:
    """
    Mask email addresses and Indian mobile phone numbers in free text.

    Limitations:
    - Only covers standard email format and 10-digit Indian mobiles with
      optional +91 / 0 prefix.
    - Does not mask embedded names, addresses, order IDs, or account numbers.
    - Best-effort only; may miss non-standard formats.
    """
    if not isinstance(text, str):
        return ""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_tickets(data_dir: Path) -> pd.DataFrame:
    """
    Load tickets.csv, parse timestamps, derive handle_hours, flag primary window.

    Handle time definition (policy §10): first_response_at → resolved_at.
    Negative durations and unresolved tickets are flagged but NOT dropped here;
    callers must filter using handle_time_valid.
    """
    path = data_dir / "tickets.csv"
    df = pd.read_csv(path, dtype=str)
    logger.info(f"Loaded {len(df):,} tickets from {path}")

    # Parse timestamps
    for col in ["created_at", "first_response_at", "resolved_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse numeric fields
    for col in ["transfers", "csat_score", "refund_amount_inr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle time: policy §10 — first_response_at → resolved_at
    df["handle_hours"] = (
        (df["resolved_at"] - df["first_response_at"]).dt.total_seconds() / 3600
    )

    # Valid handle-time flag: both timestamps present, non-negative duration
    df["handle_time_valid"] = (
        df["first_response_at"].notna()
        & df["resolved_at"].notna()
        & (df["handle_hours"] >= 0)
    )

    # Primary window flag
    df["in_primary_window"] = (
        (df["created_at"] >= PRIMARY_WINDOW_START)
        & (df["created_at"] <= PRIMARY_WINDOW_END)
    )

    # Ensure string IDs are clean
    for col in ["ticket_id", "customer_id", "agent_id", "product_sku"]:
        df[col] = df[col].astype(str).str.strip()

    _log_handle_time_stats(df)
    return df


def _log_handle_time_stats(df: pd.DataFrame) -> None:
    total = len(df)
    no_first_resp = df["first_response_at"].isna().sum()
    no_resolved   = df["resolved_at"].isna().sum()
    negative      = (df["handle_hours"] < 0).sum()
    valid         = df["handle_time_valid"].sum()
    logger.info(
        f"Handle-time stats: valid={valid:,}, "
        f"missing first_response_at={no_first_resp:,}, "
        f"unresolved={no_resolved:,}, "
        f"negative (excluded)={negative:,}"
    )


def load_agents(data_dir: Path) -> pd.DataFrame:
    """Load agents.csv. Numeric tier field is parsed; open-ended to_date is NaT."""
    path = data_dir / "agents.csv"
    df = pd.read_csv(path, dtype=str)
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce").astype("Int64")
    df["agent_id"] = df["agent_id"].str.strip()
    logger.info(f"Loaded {len(df):,} agent roster rows ({df['agent_id'].nunique()} unique agents)")
    return df


def load_customers(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "customers.csv"
    df = pd.read_csv(path, dtype=str)
    df["customer_id"] = df["customer_id"].str.strip()
    return df


def load_orders(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "orders.csv"
    df = pd.read_csv(path, dtype=str)
    df["order_id"] = df["order_id"].str.strip()
    df["order_value_inr"] = pd.to_numeric(df["order_value_inr"], errors="coerce")
    return df


def load_products(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "products.csv"
    return pd.read_csv(path, dtype=str)


# ─────────────────────────────────────────────────────────────────────────────
# Join validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_joins(
    tickets: pd.DataFrame,
    agents: pd.DataFrame,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> Dict[str, object]:
    """
    Validate all foreign-key joins; check for orphans and fan-out.

    Returns a report dict. Callers should warn on any orphan > 0 or fanout != 1.0.
    Transfer-data is NOT checked here — its missingness is expected for legacy_fd.
    """
    report: Dict[str, object] = {}

    # tickets → agents
    ticket_agents  = set(tickets["agent_id"].dropna().unique())
    roster_agents  = set(agents["agent_id"].unique())
    report["agent_orphans"] = len(ticket_agents - roster_agents)

    # tickets → customers
    ticket_cids   = set(tickets["customer_id"].dropna().unique())
    customer_ids  = set(customers["customer_id"].unique())
    report["customer_orphans"] = len(ticket_cids - customer_ids)

    # tickets → orders (only for tickets that supply an order_id)
    tickets_with_order = tickets[tickets["order_id"].notna() & (tickets["order_id"] != "nan")]
    order_ids = set(orders["order_id"].unique())
    report["order_orphans"] = len(set(tickets_with_order["order_id"].unique()) - order_ids)

    # fan-out: merge and check row count
    if len(tickets_with_order) > 0:
        joined = tickets_with_order.merge(orders, on="order_id", how="left")
        report["order_fanout"] = round(len(joined) / len(tickets_with_order), 4)
    else:
        report["order_fanout"] = None

    # tickets → products
    ticket_skus  = set(tickets["product_sku"].dropna().unique()) - {"nan"}
    product_skus = set(products["sku"].unique())
    report["sku_orphans"] = len(ticket_skus - product_skus)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_classification_text(row: pd.Series) -> str:
    """
    Build the masked text for LLM / embedding classification.

    Primary signal: customer_message. Secondary: agent_notes.
    Both fields are masked for PII before use.
    Agent names and customer names are NOT included (never added to this text).
    """
    msg   = mask_pii(str(row.get("customer_message") or "").strip())
    notes = mask_pii(str(row.get("agent_notes")      or "").strip())
    parts = []
    if msg:
        parts.append(f"Customer message: {msg}")
    if notes:
        parts.append(f"Agent notes: {notes}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Date-coverage report
# ─────────────────────────────────────────────────────────────────────────────

def report_date_coverage(tickets: pd.DataFrame) -> dict:
    """Return a summary of date range, window membership, and source system split."""
    return {
        "total_tickets":          int(len(tickets)),
        "date_min":               str(tickets["created_at"].min()),
        "date_max":               str(tickets["created_at"].max()),
        "primary_window_tickets": int(tickets["in_primary_window"].sum()),
        "out_of_window_tickets":  int((~tickets["in_primary_window"]).sum()),
        "helpdesk_tickets":       int((tickets["source_system"] == "helpdesk").sum()),
        "legacy_fd_tickets":      int((tickets["source_system"] == "legacy_fd").sum()),
        "handle_time_stats": {
            "valid":              int(tickets["handle_time_valid"].sum()),
            "unresolved":         int(tickets["resolved_at"].isna().sum()),
            "negative_excluded":  int((tickets["handle_hours"] < 0).sum()),
        },
    }
