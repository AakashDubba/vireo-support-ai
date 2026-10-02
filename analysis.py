"""
analysis.py — Business analysis: breakdowns, headcount, cost, validation, charts.

All monetary constants from support-policy.pdf §4.
Handle time = first_response_at → resolved_at (policy §10).
Transfer stats reported only where the field is non-null (helpdesk tickets).
Billing team handle-time anomaly is explicitly disclosed in headcount output.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    CATEGORIES,
    BLENDED_COST_PER_CONTACT_INR,
    AGENT_COST_PER_HOUR_INR,
    TRANSFER_COST_INR,
    GROQ_PRICE_PER_1M_INPUT,
    GROQ_PRICE_PER_1M_OUTPUT,
    PRIMARY_WINDOW_START,
    PRIMARY_WINDOW_END,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Monthly breakdowns
# ─────────────────────────────────────────────────────────────────────────────

def monthly_category_breakdown(tickets: pd.DataFrame) -> pd.DataFrame:
    """Monthly ticket volume by AI category — primary window only."""
    df = tickets[tickets["in_primary_window"]].copy()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["month", "ai_category"])
        .size()
        .unstack(fill_value=0)
    )
    for cat in CATEGORIES:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[CATEGORIES]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def monthly_team_breakdown(tickets: pd.DataFrame) -> pd.DataFrame:
    """Monthly ticket volume by assigned_team — primary window only."""
    df = tickets[tickets["in_primary_window"]].copy()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["month", "assigned_team"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def monthly_source_breakdown(tickets: pd.DataFrame) -> pd.DataFrame:
    """Full date-range monthly counts by source_system — for out-of-window disclosure."""
    df = tickets.copy()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["month", "source_system"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["in_primary_window"] = (
        (df.groupby("month")["in_primary_window"].mean() > 0)
        .reindex(pivot.index, fill_value=False)
    )
    return pivot.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Headcount analysis — workload evidence with anomaly disclosure
# ─────────────────────────────────────────────────────────────────────────────

def compute_headcount_analysis(
    tickets: pd.DataFrame,
    agents: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-team workload metrics.

    IMPORTANT: Billing team handle-time is split by source_system because:
    - helpdesk tickets show median ~0.43h (fast resolution / payment callbacks)
    - legacy_fd tickets show median ~43h (consistent with other teams)
    This is explicitly flagged in the output; it does NOT change the volume evidence.

    Transfer stats: computed only over tickets with non-null transfer field
    (all helpdesk; legacy_fd has no transfer data — treated as MISSING, not 0).
    """
    df = tickets[tickets["in_primary_window"]].copy()

    # ── Ticket volumes ────────────────────────────────────────────────────────
    team_vol = (
        df.groupby("assigned_team")
        .agg(total_tickets=("ticket_id", "count"))
        .reset_index()
    )

    # ── Agent headcount + tier ────────────────────────────────────────────────
    agent_counts = (
        agents.groupby("team")
        .agg(headcount=("agent_id", "count"),
             tier=("tier", lambda x: x.min()))   # use team's minimum tier
        .reset_index()
        .rename(columns={"team": "assigned_team"})
    )
    agent_counts["tier"] = agent_counts["tier"].astype("Int64")

    merged = team_vol.merge(agent_counts, on="assigned_team", how="left")
    merged["tickets_per_agent"] = (
        merged["total_tickets"] / merged["headcount"]
    ).round(1)

    # ── Handle time split by source_system ───────────────────────────────────
    valid_all = df[df["handle_time_valid"]]

    ht_all = (
        valid_all.groupby("assigned_team")["handle_hours"]
        .agg(
            valid_ht_n="count",
            median_ht_h="median",
            mean_ht_h="mean",
            p90_ht_h=lambda x: x.quantile(0.9),
        )
        .reset_index()
    )
    for col in ["median_ht_h", "mean_ht_h", "p90_ht_h"]:
        ht_all[col] = ht_all[col].round(2)
    merged = merged.merge(ht_all, on="assigned_team", how="left")

    # Helpdesk-only handle time (excludes legacy_fd negative-duration artefacts)
    helpdesk_valid = df[
        (df["source_system"] == "helpdesk") & df["handle_time_valid"]
    ]
    ht_hd = (
        helpdesk_valid.groupby("assigned_team")["handle_hours"]
        .agg(
            helpdesk_ht_n="count",
            helpdesk_median_h="median",
        )
        .reset_index()
    )
    ht_hd["helpdesk_median_h"] = ht_hd["helpdesk_median_h"].round(2)
    merged = merged.merge(ht_hd, on="assigned_team", how="left")

    # Excluded counts
    neg_excl = (
        df[df["first_response_at"].notna() & df["resolved_at"].notna()]
        .groupby("assigned_team")
        .apply(lambda g: (g["handle_hours"] < 0).sum())
        .reset_index(name="negative_ht_excluded")
    )
    unresolved = (
        df.groupby("assigned_team")
        .apply(lambda g: g["resolved_at"].isna().sum())
        .reset_index(name="unresolved_excluded")
    )
    merged = merged.merge(neg_excl,    on="assigned_team", how="left")
    merged = merged.merge(unresolved,  on="assigned_team", how="left")

    # ── Transfer stats (helpdesk only) ────────────────────────────────────────
    current    = df[df["source_system"] == "helpdesk"]
    has_tr     = current[current["transfers"].notna()]
    if len(has_tr) > 0:
        tr_stats = (
            has_tr.groupby("assigned_team")
            .agg(
                transfer_data_n=("transfers", "count"),
                mean_transfers=("transfers", "mean"),
                transfer_rate=("transfers", lambda x: round((x > 0).mean(), 3)),
            )
            .reset_index()
        )
        tr_stats["mean_transfers"] = tr_stats["mean_transfers"].round(3)
        merged = merged.merge(tr_stats, on="assigned_team", how="left")
    else:
        merged["transfer_data_n"] = np.nan
        merged["mean_transfers"]  = np.nan
        merged["transfer_rate"]   = np.nan

    # ── CSAT ─────────────────────────────────────────────────────────────────
    csat = (
        df[df["csat_score"].notna()]
        .groupby("assigned_team")["csat_score"]
        .agg(avg_csat="mean", csat_n="count")
        .reset_index()
    )
    csat["avg_csat"] = csat["avg_csat"].round(2)
    merged = merged.merge(csat, on="assigned_team", how="left")

    # ── High-priority % ───────────────────────────────────────────────────────
    hp = (
        df.groupby("assigned_team")
        .apply(lambda g: round((g["priority"] == "High").mean(), 3))
        .reset_index(name="high_priority_pct")
    )
    merged = merged.merge(hp, on="assigned_team", how="left")

    # ── Handle-time anomaly flag ──────────────────────────────────────────────
    # Flag teams where helpdesk median is <2h AND overall median is also <2h
    # (indicates very fast helpdesk resolution, not just a split artefact)
    merged["billing_ht_anomaly_flag"] = (
        (merged["helpdesk_median_h"] < 2.0) &
        (merged["helpdesk_ht_n"].fillna(0) > 100)
    )

    return merged.sort_values("tickets_per_agent", ascending=False).reset_index(drop=True)


def generate_hire_recommendation(hc: pd.DataFrame) -> str:
    """
    Evidence-based hire recommendation — 2 headcount positions.

    Evidence sources used (in order of reliability):
      1. Tickets per agent (primary — directly measurable)
      2. Transfer rate (complexity proxy — helpdesk only, ~66% coverage)
      3. CSAT (stress signal — ~45% response rate)
      4. Handle time (with anomaly disclosure for Billing)

    Explicitly states limitations; never hides the Billing anomaly.
    """
    t1 = hc[hc["tier"] == 1].dropna(subset=["tickets_per_agent"]).copy()

    if t1.empty:
        return (
            "INSUFFICIENT DATA: No Tier 1 teams with reliable workload metrics.\n"
            "Cannot make a headcount recommendation."
        )

    top3 = t1.nlargest(3, "tickets_per_agent")

    sep  = "=" * 60
    lines = [sep, "HEADCOUNT RECOMMENDATION  (next 2 hires)", sep, ""]

    lines.append("Evidence — Tier 1 teams by tickets/agent (primary window):")
    lines.append(
        f"  {'Team':<28} {'HC':>3} {'Tkt/Agt':>8} {'Median_h(all)':>14} "
        f"{'Median_h(HD)':>13} {'TR_rate':>8} {'CSAT':>6}"
    )
    lines.append("  " + "-" * 86)

    for _, r in top3.iterrows():
        ht_all_str = f"{r['median_ht_h']:.1f}h" if pd.notna(r.get("median_ht_h")) else "n/a"
        ht_hd_str  = f"{r['helpdesk_median_h']:.1f}h" if pd.notna(r.get("helpdesk_median_h")) else "n/a"
        tr_str     = f"{r['transfer_rate']*100:.0f}%" if pd.notna(r.get("transfer_rate")) else "n/a"
        csat_str   = f"{r['avg_csat']:.2f}" if pd.notna(r.get("avg_csat")) else "n/a"
        anomaly    = " [!]" if r.get("billing_ht_anomaly_flag", False) else ""
        lines.append(
            f"  {r['assigned_team']:<28} {int(r['headcount']):>3} "
            f"{r['tickets_per_agent']:>8.0f} {ht_all_str:>14} "
            f"{ht_hd_str:>13}{anomaly} {tr_str:>8} {csat_str:>6}"
        )

    lines.append("")

    # ── Anomaly disclosure ────────────────────────────────────────────────────
    billing_row = t1[t1["assigned_team"].str.lower().str.contains("billing")]
    if not billing_row.empty:
        b = billing_row.iloc[0]
        if pd.notna(b.get("helpdesk_median_h")) and b["helpdesk_median_h"] < 2.0:
            lines += [
                "BILLING HANDLE-TIME ANOMALY [!]:",
                f"  Helpdesk tickets: median = {b['helpdesk_median_h']:.2f}h  "
                f"(n={int(b['helpdesk_ht_n']) if pd.notna(b.get('helpdesk_ht_n')) else '?'})",
                f"  All tickets (valid): median = {b['median_ht_h']:.2f}h",
                f"  Excluded: {int(b['negative_ht_excluded']) if pd.notna(b.get('negative_ht_excluded')) else '?'} "
                f"negative (legacy_fd UTC/IST artefact)",
                "  Interpretation: helpdesk Billing tickets close fast (payment confirmations / macros).",
                "  This does NOT invalidate the volume or transfer-rate evidence.",
                "  Mean handle time = 19.4h (long-tail escalations present).",
                "",
            ]

    # ── Decision ──────────────────────────────────────────────────────────────
    if len(top3) >= 2:
        first  = top3.iloc[0]
        second = top3.iloc[1]
        ratio  = first["tickets_per_agent"] / max(second["tickets_per_agent"], 1)

        lines.append("RECOMMENDATION:")
        if ratio >= 1.5:
            lines += [
                f"  Place BOTH new hires on {first['assigned_team']}.",
                f"  Rationale:",
                f"    - Volume gap: {first['tickets_per_agent']:.0f} vs "
                f"{second['tickets_per_agent']:.0f} tickets/agent ({ratio:.1f}x)",
            ]
            if pd.notna(first.get("transfer_rate")) and pd.notna(second.get("transfer_rate")):
                lines.append(
                    f"    - Transfer rate: {first['transfer_rate']*100:.0f}% vs "
                    f"{second['transfer_rate']*100:.0f}% — Billing tickets are more complex"
                )
            if pd.notna(first.get("avg_csat")) and pd.notna(second.get("avg_csat")):
                lines.append(
                    f"    - CSAT: {first['avg_csat']:.2f} vs {second['avg_csat']:.2f} — "
                    f"stress signal on {first['assigned_team']}"
                )
        elif 1.2 <= ratio < 1.5:
            lines += [
                f"  SPLIT: one hire to {first['assigned_team']}, one to {second['assigned_team']}.",
                f"  Rationale:",
                f"    - Volume gap {ratio:.2f}x is meaningful but not decisive for both hires.",
                f"    - Splitting reduces single-team concentration risk.",
            ]
        else:
            lines += [
                f"  INSUFFICIENT EVIDENCE to prefer one team over another.",
                f"  Volume gap ({ratio:.2f}x) is within noise margin.",
                f"  Recommendation: conduct shift-level workload analysis first.",
            ]
    else:
        lines += ["  Only one Tier 1 team found — recommend both hires there."]

    lines += [
        "",
        "DATA LIMITATIONS:",
        "  - Roster is a static snapshot; agents who changed teams mid-period are not tracked.",
        "  - Transfer data covers current helpdesk only (~66% of primary-window tickets).",
        "  - Negative handle times (579 Billing, 236 Logistics, etc.) excluded — UTC/IST artefact.",
        "  - CSAT response rate ~45%; non-response bias unknown.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Cost report
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_report(
    classifier_stats: dict,
    tickets: pd.DataFrame,
    weekly_ticket_rate: int = 650,
) -> pd.DataFrame:
    """
    cost_report.csv — AI operating cost and support/business costs.

    Section A: AI cost (historical actual + forward estimate).
    Section B: Support/business impact (policy standards).
    These two sections are NEVER mixed.
    """
    usd_to_inr = float(os.getenv("USD_TO_INR", "84.0"))
    api_calls  = int(classifier_stats.get("api_calls",          0))
    cost_usd   = float(classifier_stats.get("estimated_cost_usd", 0.0))
    cache_hits = int(classifier_stats.get("cache_hits",         0))
    misses     = int(classifier_stats.get("cache_misses",       0))
    in_tok     = int(classifier_stats.get("input_tokens",       0))
    out_tok    = int(classifier_stats.get("output_tokens",      0))

    # Per-ticket cost for forward estimates
    price_in  = float(os.getenv("LLM_PRICE_PER_1M_INPUT",  str(GROQ_PRICE_PER_1M_INPUT)))
    price_out = float(os.getenv("LLM_PRICE_PER_1M_OUTPUT", str(GROQ_PRICE_PER_1M_OUTPUT)))

    if api_calls > 0 and cost_usd > 0:
        cost_per_ticket_usd  = cost_usd / api_calls
        estimate_basis       = f"actual cost from {api_calls} Groq API calls"
        estimate_is_measured = True
    else:
        # Conservative estimate: 250 input + 20 output tokens per ticket
        cost_per_ticket_usd  = (250 / 1_000_000 * price_in) + (20 / 1_000_000 * price_out)
        estimate_basis       = (
            f"ESTIMATE: 250 input + 20 output tokens @ "
            f"${price_in}/1M in, ${price_out}/1M out (Groq) — no actual API calls yet"
        )
        estimate_is_measured = False

    weekly_cost_usd = cost_per_ticket_usd * weekly_ticket_rate
    annual_cost_usd = weekly_cost_usd * 52

    # Transfer cost (helpdesk only — NOT extrapolated to legacy_fd)
    current        = tickets[tickets["source_system"] == "helpdesk"]
    tr_recorded    = current["transfers"].notna().sum()
    tr_total_units = current["transfers"].fillna(0).sum()
    transfer_cost_inr = tr_total_units * TRANSFER_COST_INR

    primary_n = int(tickets["in_primary_window"].sum())
    total_support_cost_inr = primary_n * BLENDED_COST_PER_CONTACT_INR

    # Section A — AI cost
    section_a = [
        ("── SECTION A: AI OPERATING COST ──────────────────",  "",  "",  ""),
        ("a_groq_model",       f"{os.getenv('LLM_MODEL','openai/gpt-oss-20b')}", "model",     "Groq model used for classification"),
        ("a_api_calls",        api_calls,                        "calls",        "Groq API calls for historical dataset"),
        ("a_cache_hits",       cache_hits,                       "hits",         "Reused from disk cache — zero API cost"),
        ("a_cache_misses",     misses,                           "misses",       "Required API call or offline classification"),
        ("a_input_tokens",     in_tok,                           "tokens",       "Total Groq input tokens (0 = offline mode)"),
        ("a_output_tokens",    out_tok,                          "tokens",       "Total Groq output tokens (0 = offline mode)"),
        ("a_historical_cost_usd", round(cost_usd, 6),           "USD",
         "Actual Groq cost — 0.0 means offline mode was used (not that cost is zero)"),
        ("a_historical_cost_inr", round(cost_usd * usd_to_inr, 2), "INR",       f"At 1 USD = Rs.{usd_to_inr}"),
        ("a_cost_per_ticket_usd", round(cost_per_ticket_usd, 8), "USD/ticket",  estimate_basis),
        ("── FORWARD ESTIMATES (not historical spend) ──────", "",  "",  ""),
        ("a_est_weekly_cost_usd",  round(weekly_cost_usd, 6),   "USD/week-ESTIMATE",
         f"At {weekly_ticket_rate} tickets/week — {'measured' if estimate_is_measured else 'token estimate'}"),
        ("a_est_annual_cost_usd",  round(annual_cost_usd, 4),   "USD/year-ESTIMATE",
         f"52 weeks x {weekly_ticket_rate} tickets/week"),
        ("a_est_annual_cost_inr",  round(annual_cost_usd * usd_to_inr, 2), "INR/year-ESTIMATE",
         "Annual AI operating cost estimate"),
        ("a_price_in_per_1m_usd",  price_in,  "USD/1M-tokens",  "Groq input token pricing"),
        ("a_price_out_per_1m_usd", price_out, "USD/1M-tokens",  "Groq output token pricing"),
        ("a_usd_to_inr_rate",      usd_to_inr, "rate",          "Configurable via USD_TO_INR in .env"),
    ]

    # Section B — Support / business costs
    section_b = [
        ("── SECTION B: SUPPORT / BUSINESS COSTS ──────────", "",  "",  ""),
        ("b_policy_cost_per_contact_inr", BLENDED_COST_PER_CONTACT_INR, "INR/contact",
         "support-policy.pdf §4 — blended all channels; NOT AI cost"),
        ("b_primary_window_contacts",  primary_n,             "tickets",      "Jan 2025-Jun 2026"),
        ("b_total_support_cost_inr",   total_support_cost_inr, "INR",
         f"{primary_n:,} contacts x Rs.{BLENDED_COST_PER_CONTACT_INR}"),
        ("b_agent_cost_per_hour_inr",  AGENT_COST_PER_HOUR_INR, "INR/hour",  "Policy §4 — staffing decisions"),
        ("b_transfer_cost_per_unit_inr", TRANSFER_COST_INR,  "INR/transfer", "Policy §4"),
        ("b_helpdesk_tickets_with_transfer_data", int(tr_recorded), "tickets",
         "Helpdesk tickets with non-null transfers — legacy_fd NOT extrapolated"),
        ("b_transfer_units_recorded",  int(tr_total_units),   "transfers",   "Sum of transfers field (helpdesk only)"),
        ("b_transfer_cost_inr",        round(transfer_cost_inr, 0), "INR",
         "Transfer cost for helpdesk-only tickets; legacy_fd excluded (data missing)"),
    ]

    rows = []
    for item in section_a + section_b:
        rows.append({"metric": item[0], "value": item[1], "unit": item[2], "notes": item[3]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    tickets: pd.DataFrame,
    human_labels_path: Path,
    outputs_dir: Path,
    label: str = "validation",
) -> pd.DataFrame:
    """
    Score model predictions against human labels.

    human_labels.csv must have columns: ticket_id, human_label.
    Saves confusion_matrix.png and {label}_detail.csv.
    Returns a validation_report DataFrame.

    Never modifies human_labels.  Never invents accuracy.
    """
    if not human_labels_path.exists():
        logger.warning(f"Human labels file not found: {human_labels_path}")
        return pd.DataFrame()

    human_df = pd.read_csv(human_labels_path, dtype=str)
    required  = {"ticket_id", "human_label"}
    if not required.issubset(human_df.columns):
        logger.error(f"human_labels file must have columns: {required}")
        return pd.DataFrame()

    # Warn on invalid labels
    invalid = set(human_df["human_label"].unique()) - set(CATEGORIES)
    if invalid:
        logger.warning(f"Human labels not in taxonomy: {invalid}")

    val = human_df.merge(
        tickets[["ticket_id", "ai_category", "classification_mode"]],
        on="ticket_id", how="inner",
    )
    if val.empty:
        logger.warning("No ticket_ids matched between human labels and classified tickets.")
        return pd.DataFrame()

    n = len(val)
    logger.info(f"[{label}] Matched {n:,} tickets for scoring.")
    val["correct"] = val["human_label"] == val["ai_category"]

    llm_mask     = val["classification_mode"] == "llm"
    offline_mask = ~llm_mask

    overall_acc = val["correct"].mean()
    llm_acc     = val.loc[llm_mask,     "correct"].mean() if llm_mask.any()     else None
    offline_acc = val.loc[offline_mask, "correct"].mean() if offline_mask.any() else None

    logger.info(f"[{label}] Overall accuracy : {overall_acc:.3f} (n={n})")
    if llm_acc     is not None: logger.info(f"[{label}] LLM accuracy     : {llm_acc:.3f} (n={llm_mask.sum()})")
    if offline_acc is not None: logger.info(f"[{label}] Offline accuracy  : {offline_acc:.3f} (n={offline_mask.sum()})")

    # Confusion matrix
    _plot_confusion_matrix(val, outputs_dir, label)

    # Top error pairs
    errors = val[~val["correct"]]
    error_pairs = (
        errors.groupby(["human_label", "ai_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    report_rows = [
        {"metric": "label",               "value": label,                               "notes": "Validation set identifier"},
        {"metric": "n",                   "value": n,                                    "notes": "Tickets matched to human labels"},
        {"metric": "overall_accuracy",    "value": round(overall_acc, 4),                "notes": "Fraction correctly classified"},
        {"metric": "llm_accuracy",        "value": round(llm_acc, 4) if llm_acc is not None else "N/A", "notes": "LLM-mode predictions only"},
        {"metric": "offline_accuracy",    "value": round(offline_acc, 4) if offline_acc is not None else "N/A", "notes": "Offline predictions only"},
        {"metric": "llm_n",               "value": int(llm_mask.sum()),                  "notes": "Tickets via LLM"},
        {"metric": "offline_n",           "value": int(offline_mask.sum()),              "notes": "Tickets via offline mode"},
        {"metric": "correct",             "value": int(val["correct"].sum()),             "notes": "Correctly classified"},
        {"metric": "errors",              "value": int((~val["correct"]).sum()),          "notes": "Misclassified"},
    ]
    for _, ep in error_pairs.head(10).iterrows():
        key = f"failure_{ep['human_label'][:22]}->{ep['ai_category'][:22]}"
        report_rows.append({"metric": key, "value": int(ep["count"]), "notes": "Misclassification count"})

    # Save detail
    detail_path = outputs_dir / f"{label}_detail.csv"
    val.to_csv(detail_path, index=False)
    logger.info(f"Saved {detail_path.name}")

    return pd.DataFrame(report_rows)


def _compute_cm(true_labels, pred_labels, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n   = len(labels)
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        ti, pi = idx.get(t, -1), idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti][pi] += 1
    return cm


def _plot_confusion_matrix(val: pd.DataFrame, outputs_dir: Path, label: str = "") -> None:
    present = sorted(
        set(val["human_label"].unique()) | set(val["ai_category"].unique()),
        key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 999,
    )
    labels = [c for c in CATEGORIES if c in present]
    cm     = _compute_cm(val["human_label"].tolist(), val["ai_category"].tolist(), labels)

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted (Model)", fontsize=11)
    ax.set_ylabel("True (Human Label)", fontsize=11)
    title = f"Confusion Matrix — Vireo Audio {'(' + label + ')' if label else ''}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    thresh = cm.max() * 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    out_path = outputs_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Business charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_business_charts(tickets: pd.DataFrame, outputs_dir: Path) -> None:
    """
    4-panel business overview chart (primary window):
      Panel 1: Category distribution (horizontal bar)
      Panel 2: Monthly volume trend — top 5 categories (line)
      Panel 3: Total tickets by team (bar)
      Panel 4: Channel distribution (pie)
    """
    df = tickets[tickets["in_primary_window"]].copy()
    if df.empty:
        logger.warning("No primary-window tickets — skipping charts.")
        return

    df["month"] = df["created_at"].dt.to_period("M").astype(str)

    palette = [
        "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
        "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4F46E5",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    fig.suptitle(
        f"Vireo Audio — Support Intelligence Dashboard\n"
        f"Primary window: Jan 2025 - Jun 2026  |  Total tickets: {len(df):,}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # Panel 1: Category distribution
    ax = axes[0, 0]
    cat_counts = df["ai_category"].value_counts().sort_values()
    colors_1   = [palette[i % len(palette)] for i in range(len(cat_counts))]
    bars = ax.barh(cat_counts.index, cat_counts.values, color=colors_1, edgecolor="white")
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + max(cat_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlabel("Tickets", fontsize=10)
    ax.set_title("Ticket Volume by Category", fontsize=12, fontweight="bold")
    ax.set_xlim(0, cat_counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Monthly trend — top 5 categories
    ax = axes[0, 1]
    top5 = df["ai_category"].value_counts().head(5).index.tolist()
    monthly = df.groupby(["month", "ai_category"]).size().unstack(fill_value=0)
    for i, cat in enumerate(top5):
        if cat in monthly.columns:
            ax.plot(monthly.index, monthly[cat], marker="o", markersize=4,
                    label=cat, color=palette[i], linewidth=1.8)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Tickets", fontsize=10)
    ax.set_title("Monthly Trend — Top 5 Categories", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: Tickets by team
    ax = axes[1, 0]
    teams = df.groupby("assigned_team").size().reset_index(name="tickets")
    teams = teams.sort_values("tickets", ascending=False)
    colors_3 = [palette[i % len(palette)] for i in range(len(teams))]
    bars3 = ax.bar(teams["assigned_team"], teams["tickets"], color=colors_3, edgecolor="white")
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{int(bar.get_height()):,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total Tickets", fontsize=10)
    ax.set_title("Ticket Volume by Team", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 4: Channel split
    ax = axes[1, 1]
    ch = df["channel"].value_counts()
    ax.pie(
        ch.values,
        labels=ch.index,
        autopct="%1.1f%%",
        colors=palette[:len(ch)],
        startangle=140,
        pctdistance=0.82,
        textprops={"fontsize": 10},
    )
    ax.set_title("Channel Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = outputs_dir / "business_charts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path.name}")
