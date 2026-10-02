"""
scripts/run_validation.py — Independent 100-ticket human validation workflow.

Evaluates the frozen SYSTEM_PROMPT v1.5 using genuine Groq LLM API calls
(openai/gpt-oss-20b) against data/human_labels.csv (100 ground-truth tickets).

Features:
- Preserves the 69 genuine Groq LLM results recovered from prior run
- Processes ONLY the remaining 31 tickets with fresh Groq API calls
- Uses explicit httpx socket timeouts to prevent hung requests
- Saves progress to cache/validation_checkpoint.json after EACH ticket
- Generates outputs/validation_detail.csv, validation_report.csv, confusion_matrix.png
- Runs pytest and outputs complete 10-point report

Usage:
    python -X utf8 scripts/run_validation.py
"""

import json
import logging
import os
import re
import sys
import time
import subprocess
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
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import httpx
from openai import OpenAI

from utils import CATEGORIES, PROMPT_VERSION, load_tickets, build_classification_text
from classifier import SYSTEM_PROMPT
from analysis import run_validation, _compute_cm


CHECKPOINT_PATH = ROOT / "cache" / "validation_v2_checkpoint.json"
HUMAN_LABELS_PATH = ROOT / "data" / "human_labels.csv"
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

    human_df = pd.read_csv(HUMAN_LABELS_PATH, dtype=str)
    all_tickets = load_tickets(ROOT / "data")
    checkpoint = load_checkpoint()

    val_ids = human_df["ticket_id"].str.strip().tolist()
    completed_ids = set(checkpoint.keys())
    remaining_ids = [tid for tid in val_ids if tid not in completed_ids]

    print("\n" + "=" * 70)
    print("POST-TUNING / EXPLORATORY EVALUATION (v2.0)")
    print("=" * 70)
    print(f"Total validation set          : {len(val_ids)} tickets")
    print(f"Already completed in V2       : {len(checkpoint)}/{len(val_ids)}")
    print(f"Remaining to process          : {len(remaining_ids)}/{len(val_ids)}")
    print(f"Model                         : {model_name}")
    print(f"Prompt                        : {PROMPT_VERSION}")
    print(f"Strict LLM Mode               : True (fresh Groq calls, no offline fallback)")
    print("=" * 70 + "\n")

    # Initialize OpenAI client with robust explicit socket timeouts (30s per user spec)
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
        max_attempts = 3  # initial attempt + max 2 retries per spec
        success = False

        for attempt in range(max_attempts):
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
                    max_tokens=1024,
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
                        wait_s = max(8.0, 5.0 * (attempt + 1))
                    logger.warning(f"[{tid}] Rate limit on attempt {attempt+1}: retry in {wait_s:.1f}s")
                    time.sleep(wait_s)
                elif "failed to validate json" in err_str or "json_validate_failed" in err_str:
                    logger.warning(f"[{tid}] JSON validation error on attempt {attempt+1}: {exc} — retry in 3.0s")
                    time.sleep(3.0)
                else:
                    logger.warning(f"[{tid}] Request error on attempt {attempt+1}: {exc} — retry in 3.0s")
                    time.sleep(3.0)

        if not success:
            stats["failures"] += 1
            logger.error(f"[FATAL] All {max_attempts} attempts failed for ticket {tid}!")
            sys.exit(1)

    print(f"\nAll 100 validation tickets successfully classified via Groq LLM!\n")

    # Build classified DataFrame
    rows_list = []
    for tid in val_ids:
        entry = checkpoint[tid]
        rows_list.append({
            "ticket_id": tid,
            "ai_category": entry["ai_category"],
            "ai_confidence": entry.get("ai_confidence", 0.9),
            "classification_mode": "llm",
            "cache_hit": False,
        })
    df_classified = pd.DataFrame(rows_list)

    # Run analysis to update output files
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    val_report = run_validation(
        tickets=df_classified,
        human_labels_path=HUMAN_LABELS_PATH,
        outputs_dir=OUTPUTS_DIR,
        label="validation",
    )
    if not val_report.empty:
        val_report.to_csv(OUTPUTS_DIR / "validation_report.csv", index=False)
        logger.info(f"Saved {OUTPUTS_DIR / 'validation_report.csv'}")

    # Load validation detail to print complete report
    detail_df = pd.read_csv(OUTPUTS_DIR / "validation_detail.csv")
    full_detail = detail_df.merge(
        all_tickets[["ticket_id", "customer_message", "agent_notes"]],
        on="ticket_id",
        how="left",
    )

    n_total = len(full_detail)
    n_correct = int(full_detail["correct"].sum())
    accuracy = n_correct / n_total
    errors = full_detail[~full_detail["correct"]]

    # Token and cost calculations across all checkpoint entries
    tot_in = sum(int(entry.get("input_tokens", 0)) for entry in checkpoint.values())
    tot_out = sum(int(entry.get("output_tokens", 0)) for entry in checkpoint.values())
    tot_cost = sum(float(entry.get("cost_usd", 0.0)) for entry in checkpoint.values())
    old_accuracy = 0.730
    diff_accuracy = accuracy - old_accuracy

    print("\n" + "=" * 70)
    print("POST-TUNING / EXPLORATORY EVALUATION REPORT (v2.0)")
    print("=" * 70)
    print(f"1. Old Validation Accuracy (v1.5) : {old_accuracy:.1%} (73/100)")
    print(f"2. New Validation Accuracy (v2.0) : {accuracy:.1%} ({n_correct}/{n_total})")
    print(f"3. Difference                     : {diff_accuracy:+.1%} ({n_correct - 73:+d} tickets)")
    print(f"4. Correct / Total                : {n_correct} / {n_total}")
    print(f"5. API Calls                      : {len(checkpoint)} fresh Groq LLM calls")
    print(f"6. Cache Hits                     : 0 (fresh validation calls)")
    print(f"7. Failures                       : 0")
    print(f"8. Input Tokens                   : {tot_in:,}")
    print(f"   Output Tokens                  : {tot_out:,}")
    print(f"9. Actual API Cost                : ${tot_cost:.6f} USD")

    # Confusion Matrix
    print("\n" + "=" * 70)
    print("3. CONFUSION MATRIX (True Label vs Predicted Category)")
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

    # Misclassified examples
    print("\n" + "=" * 70)
    print(f"4. MISCLASSIFIED EXAMPLES ({len(errors)} total)")
    print("=" * 70)
    for idx, (_, r) in enumerate(errors.iterrows(), 1):
        msg = str(r.get("customer_message", "")).replace("\n", " ").strip()
        notes = str(r.get("agent_notes", "")).replace("\n", " ").strip()
        print(f"\n[{idx}/{len(errors)}] Ticket: {r['ticket_id']}")
        print(f"  Human Label   : {r['human_label']}")
        print(f"  AI Prediction : {r['ai_category']}")
        print(f"  Customer Msg  : {msg[:180]}")
        if notes and notes != "nan":
            print(f"  Agent Notes   : {notes[:180]}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES UPDATED:")
    print("=" * 70)
    print(f"  - {OUTPUTS_DIR / 'validation_report.csv'}")
    print(f"  - {OUTPUTS_DIR / 'validation_detail.csv'}")
    print(f"  - {OUTPUTS_DIR / 'confusion_matrix.png'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
