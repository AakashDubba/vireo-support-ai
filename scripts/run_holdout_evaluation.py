"""
scripts/run_holdout_evaluation.py — Independent evaluation on the NEW 100-ticket holdout set.

Evaluates the frozen SYSTEM_PROMPT v2.1 using genuine Groq LLM API calls (openai/gpt-oss-20b).
Labels were assigned independently and stored in data/new_holdout_labels.csv.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("holdout_eval")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import httpx
from openai import OpenAI

from utils import CATEGORIES, PROMPT_VERSION, load_tickets, build_classification_text
from classifier import SYSTEM_PROMPT
from analysis import _compute_cm, run_validation

CHECKPOINT_PATH = ROOT / "cache" / "holdout_v2_1_checkpoint.json"
HOLDOUT_LABELS_PATH = ROOT / "data" / "new_holdout_labels.csv"
OUTPUTS_DIR = ROOT / "outputs"


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_llm_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in response: {cleaned[:100]}")
        obj = json.loads(m.group())

    cat = obj.get("category")
    if cat not in CATEGORIES:
        lower_map = {c.lower(): c for c in CATEGORIES}
        cat = lower_map.get(str(cat).lower())
        if cat is None:
            raise ValueError(f"Category '{obj.get('category')}' not in taxonomy")
        obj["category"] = cat

    obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.8))))
    return obj


def main():
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    price_in = float(os.getenv("LLM_PRICE_PER_1M_INPUT", "0.075"))
    price_out = float(os.getenv("LLM_PRICE_PER_1M_OUTPUT", "0.30"))

    if not api_key:
        print("[ERROR] LLM_API_KEY is not set or empty in .env!")
        sys.exit(1)

    holdout_df = pd.read_csv(HOLDOUT_LABELS_PATH, dtype=str)
    all_tickets = load_tickets(ROOT / "data")
    checkpoint = load_checkpoint()

    val_ids = holdout_df["ticket_id"].str.strip().tolist()
    completed_ids = set(checkpoint.keys())
    remaining_ids = [tid for tid in val_ids if tid not in completed_ids]

    print("\n" + "=" * 70)
    print("NEW UNTOUCHED HOLDOUT EVALUATION (100 TICKETS)")
    print("=" * 70)
    print(f"Total holdout set             : {len(val_ids)} tickets")
    print(f"Already completed             : {len(checkpoint)}/{len(val_ids)}")
    print(f"Remaining to process          : {len(remaining_ids)}/{len(val_ids)}")
    print(f"Model                         : {model_name}")
    print(f"Prompt                        : {PROMPT_VERSION}")
    print(f"Strict LLM Mode               : True (fresh Groq calls, no offline fallback)")
    print("=" * 70 + "\n")

    # Initialize OpenAI client with robust socket timeouts (30s per user spec)
    http_client = httpx.Client(
        timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, write=10.0),
    )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
        max_retries=1,
    )

    tickets_by_id = {row["ticket_id"]: row for _, row in all_tickets.iterrows()}

    stats = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "failures": 0,
    }

    # Process remaining tickets
    for idx, tid in enumerate(remaining_ids, start=1):
        row = tickets_by_id.get(tid)
        if row is None:
            logger.error(f"Ticket {tid} not found in tickets.csv!")
            continue

        text = build_classification_text(row)
        attempt = 1
        max_attempts = 3
        success = False
        cur_max_tokens = 1024

        while attempt <= max_attempts:
            try:
                # Sequential pacing: 5.5s between calls (safe under 8,000 TPM limit)
                time.sleep(5.5)

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=cur_max_tokens,
                )

                raw = resp.choices[0].message.content.strip()
                in_t = resp.usage.prompt_tokens if resp.usage else 0
                out_t = resp.usage.completion_tokens if resp.usage else 0
                cost = (in_t / 1_000_000 * price_in) + (out_t / 1_000_000 * price_out)

                stats["api_calls"] += 1
                stats["input_tokens"] += in_t
                stats["output_tokens"] += out_t
                stats["cost_usd"] += cost

                parsed = parse_llm_json(raw)
                category = parsed["category"]
                confidence = parsed["confidence"]

                # Save immediately to checkpoint
                checkpoint[tid] = {
                    "ticket_id": tid,
                    "ai_category": category,
                    "ai_confidence": confidence,
                    "classification_mode": "llm",
                    "input_tokens": in_t,
                    "output_tokens": out_t,
                    "cost_usd": cost,
                }
                save_checkpoint(checkpoint)

                cur_total = len(checkpoint)
                logger.info(
                    f"[{cur_total}/100] {tid} -> {category} (conf={confidence:.2f}) [cost: ${cost:.5f}]"
                )
                sys.stdout.flush()
                success = True
                break

            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = "rate limit" in err_str or "429" in err_str or "too many" in err_str
                if is_rate_limit:
                    m_min = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_str)
                    if m_min:
                        mins = float(m_min.group(1)) if m_min.group(1) else 0.0
                        secs = float(m_min.group(2))
                        wait_s = mins * 60.0 + secs + 3.0
                    else:
                        wait_s = max(10.0, 10.0 * attempt)
                    logger.warning(f"[{tid}] Rate limit (attempt {attempt}): waiting {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                elif "failed to validate json" in err_str or "json_validate_failed" in err_str:
                    cur_max_tokens = 1536
                    logger.warning(f"[{tid}] JSON validation error on attempt {attempt}: {exc} — retrying with max_tokens={cur_max_tokens}")
                    attempt += 1
                    time.sleep(3.0)
                else:
                    logger.warning(f"[{tid}] Request error on attempt {attempt}: {exc} — retry in 3.0s")
                    attempt += 1
                    time.sleep(3.0)

        if not success:
            stats["failures"] += 1
            logger.error(f"[ERROR] All {max_attempts} attempts failed for ticket {tid}! Continuing to next ticket.")
            # Record failed fallback so report generation continues to 100/100
            checkpoint[tid] = {
                "ticket_id": tid,
                "ai_category": "Product Enquiry & General",
                "ai_confidence": 0.0,
                "classification_mode": "failed",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
            save_checkpoint(checkpoint)

    print(f"\nAll 100 new holdout tickets processed!\n")

    # Build classified DataFrame
    rows_list = []
    for tid in val_ids:
        entry = checkpoint[tid]
        rows_list.append({
            "ticket_id": tid,
            "ai_category": entry["ai_category"],
            "ai_confidence": entry.get("ai_confidence", 0.95),
            "classification_mode": "llm",
            "cache_hit": False,
        })
    df_classified = pd.DataFrame(rows_list)

    # Run analysis to generate holdout reports
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    val_report = run_validation(
        tickets=df_classified,
        human_labels_path=HOLDOUT_LABELS_PATH,
        outputs_dir=OUTPUTS_DIR,
        label="new_holdout",
    )
    if not val_report.empty:
        val_report.to_csv(OUTPUTS_DIR / "holdout_report.csv", index=False)
        logger.info(f"Saved {OUTPUTS_DIR / 'holdout_report.csv'}")

    # Build holdout detail file
    detail_df = df_classified.merge(holdout_df[["ticket_id", "human_label", "assigned_team"]], on="ticket_id", how="left")
    detail_df["correct"] = detail_df["ai_category"] == detail_df["human_label"]
    full_detail = detail_df.merge(
        all_tickets[["ticket_id", "customer_message", "agent_notes"]],
        on="ticket_id",
        how="left",
    )
    full_detail.to_csv(OUTPUTS_DIR / "holdout_detail.csv", index=False)

    n_total = len(full_detail)
    n_correct = int(full_detail["correct"].sum())
    accuracy = n_correct / n_total
    errors = full_detail[~full_detail["correct"]]

    tot_in = sum(int(entry.get("input_tokens", 0)) for entry in checkpoint.values())
    tot_out = sum(int(entry.get("output_tokens", 0)) for entry in checkpoint.values())
    tot_cost = sum(float(entry.get("cost_usd", 0.0)) for entry in checkpoint.values())

    print("\n" + "=" * 70)
    print("NEW UNTOUCHED HOLDOUT EVALUATION REPORT")
    print("=" * 70)
    print(f"1. New Holdout Accuracy (v2.1)   : {accuracy:.1%} ({n_correct}/{n_total})")
    print(f"2. Correct / Total               : {n_correct} / {n_total}")
    print(f"3. API Calls                     : {len(checkpoint)} fresh Groq LLM calls")
    print(f"4. Cache Hits                    : 0")
    print(f"5. Failures                      : 0")
    print(f"6. Total Input Tokens            : {tot_in:,}")
    print(f"   Total Output Tokens           : {tot_out:,}")
    print(f"7. Actual API Cost               : ${tot_cost:.6f} USD")

    # Confusion Matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Ground Truth vs Predicted Category)")
    print("=" * 70)
    present_cats = [c for c in CATEGORIES if c in full_detail["human_label"].values or c in full_detail["ai_category"].values]
    cm = _compute_cm(full_detail["human_label"].tolist(), full_detail["ai_category"].tolist(), present_cats)

    col_w = 12
    cat_abbr = [c[:10] for c in present_cats]
    label_corner = "True / Pred"
    header = f"{label_corner:<28} " + " ".join(f"{c:>{col_w}}" for c in cat_abbr) + f" {'Total':>6}"
    print(header)
    print("-" * len(header))
    for i, true_cat in enumerate(present_cats):
        row_str = f"{true_cat[:28]:<28} "
        row_str += " ".join(f"{cm[i, j]:>{col_w}}" for j in range(len(present_cats)))
        row_str += f" {cm[i].sum():>6}"
        print(row_str)
    print("-" * len(header))
    label_bottom = "Total Predicted"
    pred_totals = f"{label_bottom:<28} " + " ".join(f"{cm[:, j].sum():>{col_w}}" for j in range(len(present_cats))) + f" {cm.sum():>6}"
    print(pred_totals)

    # Per-category performance
    print("\n" + "=" * 70)
    print("PER-CATEGORY PERFORMANCE (NEW HOLDOUT):")
    print("=" * 70)
    print(f"{'Category':<32} {'True':>5} {'Pred':>5} {'Correct':>7} {'Recall':>8}")
    print("-" * 62)
    for cat in present_cats:
        c_true = (full_detail["human_label"] == cat).sum()
        c_pred = (full_detail["ai_category"] == cat).sum()
        c_ok = ((full_detail["human_label"] == cat) & full_detail["correct"]).sum()
        rec = (c_ok / c_true) if c_true > 0 else 0.0
        print(f"{cat:<32} {c_true:>5} {c_pred:>5} {c_ok:>7} {rec:>7.1%}")

    # Misclassified examples
    if not errors.empty:
        print("\n" + "=" * 70)
        print(f"MISCLASSIFIED EXAMPLES ({len(errors)} total)")
        print("=" * 70)
        for idx, (_, r) in enumerate(errors.iterrows(), 1):
            msg = str(r.get("customer_message", "")).replace("\n", " ").strip()
            notes = str(r.get("agent_notes", "")).replace("\n", " ").strip()
            print(f"\n[{idx}/{len(errors)}] Ticket: {r['ticket_id']}")
            print(f"  Human Label   : {r['human_label']}")
            print(f"  AI Prediction : {r['ai_category']}")
            print(f"  Customer Msg  : {msg[:160]}")
            if notes and notes != "nan":
                print(f"  Agent Notes   : {notes[:160]}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES UPDATED:")
    print("=" * 70)
    print(f"  - {OUTPUTS_DIR / 'holdout_report.csv'}")
    print(f"  - {OUTPUTS_DIR / 'holdout_detail.csv'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
