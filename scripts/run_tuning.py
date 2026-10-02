"""
scripts/run_tuning.py — Prompt tuning workflow for the 20-ticket tuning set.

Evaluates the system prompt using the real Groq API on data/tuning_set.csv.
Bypasses old offline cache to ensure valid LLM predictions.
Does not silently fall back to offline embeddings if Groq fails.

Usage:
    python -X utf8 scripts/run_tuning.py                     # runs Groq LLM (bypasses cache)
    python -X utf8 scripts/run_tuning.py --force-llm         # explicit forced Groq LLM run
    python -X utf8 scripts/run_tuning.py --no-cache          # bypass cache
    python -X utf8 scripts/run_tuning.py --use-cache         # allow reading valid LLM cache
    python -X utf8 scripts/run_tuning.py --offline           # force offline embeddings
"""

import argparse
import json
import os
import sys
from pathlib import Path

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_tuning")


try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from utils import CATEGORIES, PROMPT_VERSION, load_tickets, build_classification_text
from classifier import TicketClassifier, SYSTEM_PROMPT


def parse_args():
    p = argparse.ArgumentParser(description="Tuning-set classification and analysis")
    p.add_argument("--cache",      type=Path, default=ROOT / "cache", help="Cache directory")
    p.add_argument("--offline",    action="store_true", help="Force offline mode (sentence transformers / TF-IDF)")
    p.add_argument("--force-llm",  action="store_true", help="Force fresh Groq LLM API calls, bypassing cache")
    p.add_argument("--no-cache",   action="store_true", help="Bypass cache for tuning tickets")
    p.add_argument("--use-cache",  action="store_true", help="Allow reading from cache if valid LLM cache is present")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if args.offline:
        offline = True
        strict_llm = False
        bypass_cache = args.no_cache
    else:
        if not api_key:
            print("\n[ERROR] LLM_API_KEY is not set or empty in .env!")
            print("        Groq LLM tuning cannot run without a valid API key.")
            print("        To run in offline mode instead, explicitly pass --offline.\n")
            sys.exit(1)
        offline = False
        strict_llm = True
        # For tuning in LLM mode, default is to bypass old cache so we evaluate real Groq API calls,
        # unless user explicitly asked to use cache via --use-cache.
        bypass_cache = True if (not args.use_cache) else (args.no_cache or args.force_llm)

    tuning_path = ROOT / "data" / "tuning_set.csv"
    if not tuning_path.exists():
        print("[ERROR] data/tuning_set.csv not found. Run scripts/sample_and_label.py first.")
        sys.exit(1)

    tuning = pd.read_csv(tuning_path, dtype=str)
    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    provider = os.getenv("LLM_PROVIDER", "groq")

    print(f"\n{'='*60}")
    print(f"TUNING WORKFLOW  —  {len(tuning)} tickets")
    print(f"Prompt version   : {PROMPT_VERSION} (working draft)")
    print(f"Provider         : {provider if not offline else 'offline'}")
    print(f"Model            : {model_name if not offline else 'all-MiniLM-L6-v2'}")
    print(f"Mode             : {'offline-embedding' if offline else 'Groq LLM'}")
    print(f"Bypass cache     : {bypass_cache}")
    print(f"Strict LLM       : {strict_llm}")
    print(f"{'='*60}\n")

    # Load the full ticket dataset to pass to classifier (it needs all metadata cols)
    tickets_all = load_tickets(ROOT / "data")
    tuning_ids = set(tuning["ticket_id"].str.strip())
    subset = tickets_all[tickets_all["ticket_id"].isin(tuning_ids)].copy()
    if subset.empty:
        print("[ERROR] Tuning ticket IDs not found in tickets.csv")
        sys.exit(1)

    classifier = TicketClassifier(
        cache_dir=args.cache,
        offline=offline,
        bypass_cache=bypass_cache,
        strict_llm=strict_llm,
    )

    out_path = ROOT / "outputs" / "tuning_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_rows = []

    for i, (_, t_row) in enumerate(tuning.iterrows(), 1):
        tid = t_row["ticket_id"]
        sub_row = subset[subset["ticket_id"] == tid]
        if sub_row.empty:
            continue
        try:
            classified = classifier.classify_dataframe(sub_row)
        except Exception as exc:
            print(f"\n[FATAL ERROR] Groq LLM classification failed on {tid}: {exc}")
            print("Tuning stopped. No offline fallback was used.\n")
            sys.exit(1)

        c_row = classified.iloc[0]
        pred_cat = c_row["ai_category"]
        conf = c_row["ai_confidence"]
        mode = c_row["classification_mode"]
        cached = c_row["cache_hit"]
        is_ok = (t_row["human_label"] == pred_cat)

        row_res = {
            "ticket_id": tid,
            "human_label": t_row["human_label"],
            "existing_tag": t_row.get("existing_tag", ""),
            "customer_message": t_row.get("customer_message", ""),
            "ai_category": pred_cat,
            "ai_confidence": conf,
            "classification_mode": mode,
            "cache_hit": cached,
            "correct": is_ok,
        }
        results_rows.append(row_res)
        pd.DataFrame(results_rows).to_csv(out_path, index=False)

        ok_str = "YES" if is_ok else "NO "
        logger.info(f"[{i}/{len(tuning)}] {tid}: AI='{pred_cat}' | Human='{t_row['human_label']}' | OK={ok_str} (conf={float(conf):.2f})")
        sys.stdout.flush()

    result = pd.DataFrame(results_rows)
    n = len(result)
    n_correct = result["correct"].sum()
    accuracy = n_correct / n

    # Verification: check if all predictions came from Groq LLM
    n_llm = (result["classification_mode"] == "llm").sum()
    n_offline = (result["classification_mode"] != "llm").sum()
    all_genuine_llm = (n_llm == n) if not offline else False

    print(f"\nResults: {n_correct}/{n} correct  ({accuracy:.1%} accuracy)")
    if not offline:
        if all_genuine_llm:
            print("Verification: ALL 20 predictions were GENUINELY generated by Groq LLM.\n")
        else:
            print(f"Verification: WARNING! {n_offline} predictions did NOT come from Groq LLM!\n")


    # Print per-ticket details
    print(f"{'Ticket':<14} {'Human Label':<34} {'AI Prediction':<34} {'Conf':>5} {'OK':>3} {'Mode':<18} {'Source':<10} {'Cached':>7}")
    print("-" * 135)
    for _, row in result.iterrows():
        ok_str   = "YES" if row["correct"] else "NO "
        conf_str = f"{float(row['ai_confidence']):.2f}" if pd.notna(row.get("ai_confidence")) else "n/a"
        cached   = "yes" if str(row.get("cache_hit")).lower() == "true" else "no"
        source   = "Groq API" if row["classification_mode"] == "llm" else "Offline"
        print(f"{row['ticket_id']:<14} {str(row['human_label']):<34} {str(row['ai_category']):<34} "
              f"{conf_str:>5} {ok_str:>3} {str(row['classification_mode']):<18} {source:<10} {cached:>7}")

    # Error analysis
    errors = result[~result["correct"]]
    if not errors.empty:
        print(f"\n{'='*60}")
        print(f"MISCLASSIFICATIONS ({len(errors)}/{n}):")
        print(f"{'='*60}")
        for _, row in errors.iterrows():
            print(f"\n  Ticket  : {row['ticket_id']}")
            print(f"  Human   : {row['human_label']}")
            print(f"  AI said : {row['ai_category']}  (conf={float(row['ai_confidence']):.2f})")
            msg = str(row.get("customer_message", "")).strip().replace("\n", " ")
            print(f"  Message : {msg[:140]}")

        # Pattern analysis
        print(f"\n{'='*60}")
        print("CONFUSION PAIRS:")
        pairs = errors.groupby(["human_label", "ai_category"]).size().reset_index(name="count")
        for _, r in pairs.sort_values("count", ascending=False).iterrows():
            print(f"  {r['human_label']:<34} -> {r['ai_category']:<34}  ({int(r['count'])} times)")
    else:
        print("\nAll tuning tickets classified correctly!")

    # Category distribution
    print(f"\n{'='*60}")
    print("CATEGORY DISTRIBUTION (tuning set):")
    print(f"  {'Category':<38} {'Human':>6} {'AI':>6} {'Match':>6}")
    for cat in CATEGORIES:
        h = (result["human_label"]  == cat).sum()
        a = (result["ai_category"]  == cat).sum()
        m = result[(result["human_label"] == cat) & result["correct"]].shape[0]
        if h > 0 or a > 0:
            print(f"  {cat:<38} {h:>6} {a:>6} {m:>6}")

    print(f"\n{'='*60}")
    print(f"SUMMARY  |  {PROMPT_VERSION}  |  mode={'offline' if offline else 'Groq LLM'}  |  accuracy={accuracy:.1%}")
    print(f"{'='*60}\n")

    if not offline:
        print(f"API calls      : {classifier.stats['api_calls']}")
        print(f"Cache hits     : {classifier.stats['cache_hits']}")
        print(f"LLM successes  : {classifier.stats['llm_successes']}")
        print(f"Failures       : {classifier.stats['failures']}")
        print(f"Input tokens   : {classifier.stats['input_tokens']:,}")
        print(f"Output tokens  : {classifier.stats['output_tokens']:,}")
        print(f"Est. cost (USD): ${classifier.stats['estimated_cost_usd']:.6f}\n")

    # Save tuning results CSV for reference
    out_path = ROOT / "outputs" / "tuning_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}\n")


if __name__ == "__main__":
    main()
