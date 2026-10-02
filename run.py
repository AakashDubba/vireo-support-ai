"""
run.py — CLI entry point for the Vireo Audio Support Ticket Intelligence System.

Usage:
    python run.py --data ./data
    python run.py --data ./data --offline
    python run.py --data ./data --validate --human-labels ./data/human_labels.csv

Outputs written to ./outputs/ by default.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Load .env before any other project imports (python-dotenv optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils import (
    CATEGORIES, PROMPT_VERSION,
    load_tickets, load_agents, load_customers, load_orders, load_products,
    validate_joins, report_date_coverage,
)
from classifier import TicketClassifier
from analysis import (
    monthly_category_breakdown,
    monthly_team_breakdown,
    compute_headcount_analysis,
    generate_hire_recommendation,
    compute_cost_report,
    run_validation,
    plot_business_charts,
)

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vireo Audio Support Ticket Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline first run (downloads sentence-transformer model ~90 MB on first run):
  python run.py --data ./data

  # Force offline (skip API even if key present):
  python run.py --data ./data --offline

  # With Groq API (set LLM_API_KEY in .env, LLM_PROVIDER=groq):
  python run.py --data ./data

  # Run final validation against human labels:
  python run.py --data ./data --validate

  # Skip classification — serve from cache only (fastest re-run):
  python run.py --data ./data --skip-classify --validate
""",
    )
    p.add_argument("--data",          required=True, type=Path,
                   help="Directory containing tickets.csv, agents.csv, etc.")
    p.add_argument("--output",        type=Path, default=Path("./outputs"),
                   help="Output directory (default: ./outputs)")
    p.add_argument("--cache",         type=Path, default=Path("./cache"),
                   help="Cache directory (default: ./cache)")
    p.add_argument("--offline",       action="store_true",
                   help="Force offline mode (sentence transformers / TF-IDF)")
    p.add_argument("--skip-classify", action="store_true",
                   help="Skip classification; serve from cache only")
    p.add_argument("--validate",      action="store_true",
                   help="Run validation against human labels")
    p.add_argument("--validate-only", action="store_true",
                   help="Run validation ONLY on the human-labeled subset (does not classify full dataset)")
    p.add_argument("--human-labels",  type=Path, default=Path("./data/human_labels.csv"),
                   help="CSV with ticket_id + human_label columns")
    p.add_argument("--weekly-rate",   type=int, default=650,
                   help="Projected weekly ticket volume for cost estimate (default 650)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    api_key  = os.getenv("LLM_API_KEY", "").strip()
    offline  = args.offline or not bool(api_key)
    provider = os.getenv("LLM_PROVIDER", "groq")
    model    = os.getenv("LLM_MODEL",    "openai/gpt-oss-20b")

    _banner("Vireo Audio -- Support Ticket Intelligence System")
    logger.info(f"Data directory  : {args.data.resolve()}")
    logger.info(f"Output directory: {args.output.resolve()}")
    logger.info(f"Cache directory : {args.cache.resolve()}")
    logger.info(f"Prompt version  : {PROMPT_VERSION} (working draft)")
    logger.info(f"Provider        : {provider}  model={model}")
    if not api_key:
        logger.warning(
            "LLM_API_KEY is empty -- running in OFFLINE mode. "
            "Set LLM_API_KEY in .env to enable Groq classification."
        )
    logger.info(f"Offline mode    : {offline}")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    _step("1 / 7", "Loading data")
    tickets   = load_tickets(args.data)
    agents    = load_agents(args.data)
    customers = load_customers(args.data)
    orders    = load_orders(args.data)
    products  = load_products(args.data)

    cov = report_date_coverage(tickets)
    logger.info(f"  Total tickets        : {cov['total_tickets']:,}")
    logger.info(f"  Actual date range    : {cov['date_min']}  to  {cov['date_max']}")
    logger.info(f"  Primary window       : Jan 2025 - Jun 2026  -> {cov['primary_window_tickets']:,} tickets")
    logger.info(f"  Out-of-window        : {cov['out_of_window_tickets']:,} tickets (reported; not deleted)")
    logger.info(f"  helpdesk / legacy_fd : {cov['helpdesk_tickets']:,} / {cov['legacy_fd_tickets']:,}")

    ht_stats = cov.get("handle_time_stats", {})
    if ht_stats:
        logger.info(
            f"  Handle-time valid    : {ht_stats.get('valid', 0):,}  "
            f"/ neg_excl={ht_stats.get('negative_excluded', 0):,}  "
            f"/ unresolved={ht_stats.get('unresolved', 0):,}"
        )

    # ── Targeted Validation Only Mode ─────────────────────────────────────────
    if args.validate_only:
        _step("VALIDATION ONLY", f"Scoring against {args.human_labels.name}")
        if not args.human_labels.exists():
            logger.error(f"Human labels file not found: {args.human_labels}")
            sys.exit(1)
        human_df = pd.read_csv(args.human_labels, dtype=str)
        val_ids = set(human_df["ticket_id"].str.strip())
        val_subset = tickets[tickets["ticket_id"].isin(val_ids)].copy()
        logger.info(f"  Matched {len(val_subset)} tickets to human validation set.")
        val_classifier = TicketClassifier(
            cache_dir=args.cache,
            offline=offline,
            bypass_cache=not offline,
            strict_llm=not offline,
        )
        val_subset = val_classifier.classify_dataframe(val_subset)
        val_report = run_validation(val_subset, args.human_labels, args.output)
        if not val_report.empty:
            val_report.to_csv(args.output / "validation_report.csv", index=False)
            logger.info("  Saved validation_report.csv")
        logger.info(f"Targeted validation finished. Output artifacts saved to {args.output.resolve()}.\n")
        return

    # ── Step 2: Validate joins ────────────────────────────────────────────────
    _step("2 / 7", "Validating joins")
    joins = validate_joins(tickets, agents, customers, orders, products)
    for key, val in joins.items():
        if "orphan" in key:
            status = "OK" if val == 0 else "WARN: ORPHANS"
        elif "fanout" in key:
            status = "OK" if val == 1.0 else f"WARN: FAN-OUT {val}"
        else:
            status = str(val)
        logger.info(f"  {key:<25}: {val}  {status}")

    # ── Step 3: Classify ──────────────────────────────────────────────────────
    _step("3 / 7", "Classifying tickets")
    classifier = TicketClassifier(
        cache_dir=args.cache,
        offline=offline or args.skip_classify,
    )
    tickets = classifier.classify_dataframe(tickets)

    mode_counts = tickets["classification_mode"].value_counts()
    for mode, cnt in mode_counts.items():
        logger.info(f"  Mode '{mode}': {cnt:,} tickets")
    logger.info(f"  Cache hits     : {classifier.stats['cache_hits']:,}")
    logger.info(f"  API calls      : {classifier.stats['api_calls']:,}")
    logger.info(f"  LLM successes  : {classifier.stats['llm_successes']:,}")
    logger.info(f"  Failures       : {classifier.stats['failures']:,}")
    if classifier.stats["api_calls"] > 0:
        logger.info(f"  Input tokens   : {classifier.stats['input_tokens']:,}")
        logger.info(f"  Output tokens  : {classifier.stats['output_tokens']:,}")
        logger.info(f"  Est. cost      : ${classifier.stats['estimated_cost_usd']:.6f} USD")

    out_cat = args.output / "categorized_tickets.csv"
    tickets.to_csv(out_cat, index=False)
    logger.info(f"  Saved -> {out_cat.name}")

    # ── Step 4: Monthly breakdowns ────────────────────────────────────────────
    _step("4 / 7", "Monthly breakdowns")
    monthly_cat  = monthly_category_breakdown(tickets)
    monthly_team = monthly_team_breakdown(tickets)
    monthly_cat.to_csv(args.output / "monthly_category_breakdown.csv",  index=False)
    monthly_team.to_csv(args.output / "monthly_team_breakdown.csv",     index=False)
    logger.info(f"  Saved monthly_category_breakdown.csv ({len(monthly_cat)} months)")
    logger.info(f"  Saved monthly_team_breakdown.csv ({len(monthly_team)} months)")

    # ── Step 5: Headcount analysis ────────────────────────────────────────────
    _step("5 / 7", "Headcount analysis")
    hc = compute_headcount_analysis(tickets, agents)
    hc.to_csv(args.output / "headcount_analysis.csv", index=False)
    logger.info("  Saved headcount_analysis.csv")

    rec = generate_hire_recommendation(hc)
    for line in rec.split("\n"):
        logger.info(line)
    (args.output / "hire_recommendation.txt").write_text(rec, encoding="utf-8")

    # ── Step 6: Cost report ───────────────────────────────────────────────────
    _step("6 / 7", "Cost report")
    cost_df = compute_cost_report(classifier.stats, tickets, args.weekly_rate)
    cost_df.to_csv(args.output / "cost_report.csv", index=False)
    logger.info("  Saved cost_report.csv")
    _log_cost_summary(cost_df)

    # ── Step 7: Validation ────────────────────────────────────────────────────
    _step("7 / 7", "Validation")
    if args.validate:
        val_report = run_validation(tickets, args.human_labels, args.output)
        if not val_report.empty:
            val_report.to_csv(args.output / "validation_report.csv", index=False)
            logger.info("  Saved validation_report.csv")
        else:
            logger.warning("  Validation produced no results -- check human_labels.csv path")
    else:
        logger.info("  Skipped (use --validate to score against human_labels.csv)")

    # ── Charts ────────────────────────────────────────────────────────────────
    logger.info("\nGenerating charts...")
    plot_business_charts(tickets, args.output)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    required_outputs = [
        "categorized_tickets.csv",
        "monthly_category_breakdown.csv",
        "monthly_team_breakdown.csv",
        "headcount_analysis.csv",
        "hire_recommendation.txt",
        "cost_report.csv",
        "business_charts.png",
    ]
    if args.validate:
        required_outputs += ["validation_report.csv", "confusion_matrix.png", "validation_detail.csv"]

    all_present = True
    for f in required_outputs:
        path = args.output / f
        if path.exists():
            size = f"{max(1, path.stat().st_size // 1024):,} KB"
        else:
            size = "MISSING"
            all_present = False
        logger.info(f"  {f:<45} {size}")

    if not all_present:
        logger.error("  Some required output files are MISSING. Check logs above.")

    if not api_key:
        logger.warning(
            "\n  NOTE: Classification used OFFLINE mode (no API key)."
            "\n  To use Groq: add LLM_API_KEY to .env and re-run."
            "\n  Cached offline results will be replaced by LLM results on next run."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(msg)
    logger.info("=" * 60)


def _step(num: str, desc: str) -> None:
    logger.info(f"\n[{num}] {desc}...")


def _log_cost_summary(cost_df: pd.DataFrame) -> None:
    """Log the most important cost rows without exposing API keys."""
    want = {
        "a_historical_cost_usd",
        "a_historical_cost_inr",
        "a_est_annual_cost_usd",
        "a_est_annual_cost_inr",
        "b_total_support_cost_inr",
    }
    for _, row in cost_df[cost_df["metric"].isin(want)].iterrows():
        logger.info(f"  {row['metric']:<40}: {row['value']}  ({row['unit']})")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
