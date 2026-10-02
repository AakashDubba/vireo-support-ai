"""
audit.py — Pre-LLM checks for the 5 items in the user request.
Run with:  python -X utf8 scripts/audit.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from utils import CATEGORIES, PRIMARY_WINDOW_START, PRIMARY_WINDOW_END

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Human labels: derived from text, not copied from category column?
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 66)
print("CHECK 2: Human label source verification")
print("=" * 66)

tickets  = pd.read_csv(ROOT / "data" / "tickets.csv", dtype=str)
detail   = pd.read_csv(ROOT / "data" / "human_labels_detail.csv", dtype=str)
labels   = pd.read_csv(ROOT / "data" / "human_labels.csv", dtype=str)

# Merge detail with source tickets to get original category
merged = detail.merge(
    tickets[["ticket_id", "category"]],
    on="ticket_id",
    how="left",
    suffixes=("_detail", "_original"),
)

# The 'existing_tag' in detail was captured at sampling time; verify it matches tickets.csv
tag_match_vs_original = (merged["existing_tag"] == merged["category"]).mean()
print(f"  existing_tag field matches tickets.csv category    : {tag_match_vs_original:.1%}")

# Agreement between human_label and original category tag
agree_rate = (detail["human_label"] == detail["existing_tag"]).mean()
print(f"  human_label agrees with existing tag               : {agree_rate:.1%}")
print(f"  (This should be LOW if labels are independently derived.)")

# Show the disagreement cases — if labels were just copied we'd see 100% agreement
disagreements = detail[detail["human_label"] != detail["existing_tag"]]
print(f"\n  Disagreements (human label != existing tag)        : {len(disagreements)} / {len(detail)}")
print()

# Category distribution comparison
print("  Distribution comparison  (human_label vs existing_tag):")
hd = detail["human_label"].value_counts()
ed = detail["existing_tag"].value_counts()
all_cats = sorted(set(hd.index) | set(ed.index))
for c in all_cats:
    print(f"    {c:<40} human={hd.get(c,0):>3}  existing={ed.get(c,0):>3}")

# Spot-check a few disagreements to show labels are text-based
print("\n  5 disagreement examples (showing message text):")
for _, r in disagreements.head(5).iterrows():
    print(f"    [{r['ticket_id']}] existing='{r['existing_tag']}' → human='{r['human_label']}'")
    print(f"      msg: {str(r['customer_message'])[:100]}")

# Taxonomy membership check
invalid_labels = set(detail["human_label"].unique()) - set(CATEGORIES)
print(f"\n  Labels outside taxonomy                            : {invalid_labels or 'none'}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Zero overlap between tuning set (20) and validation set (100)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 66)
print("CHECK 3: Tuning/validation overlap")
print("=" * 66)

tuning = pd.read_csv(ROOT / "data" / "tuning_set.csv", dtype=str)
val    = pd.read_csv(ROOT / "data" / "human_labels.csv", dtype=str)

tuning_ids = set(tuning["ticket_id"])
val_ids    = set(val["ticket_id"])
overlap    = tuning_ids & val_ids

print(f"  Tuning set size     : {len(tuning_ids)}")
print(f"  Validation set size : {len(val_ids)}")
print(f"  Overlap             : {len(overlap)}")
print(f"  Result              : {'PASS — zero overlap' if not overlap else 'FAIL — overlapping tickets: ' + str(overlap)}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Headcount analysis audit
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 66)
print("CHECK 4: Headcount analysis audit")
print("=" * 66)

agents  = pd.read_csv(ROOT / "agents.csv", dtype=str)
agents["tier"] = pd.to_numeric(agents["tier"], errors="coerce")

# Load tickets with timestamps parsed
tk = pd.read_csv(ROOT / "data" / "tickets.csv", dtype=str)
for col in ["created_at", "first_response_at", "resolved_at"]:
    tk[col] = pd.to_datetime(tk[col], errors="coerce")

tk["handle_hours"] = (
    (tk["resolved_at"] - tk["first_response_at"]).dt.total_seconds() / 3600
)
tk["handle_time_valid"] = (
    tk["first_response_at"].notna()
    & tk["resolved_at"].notna()
    & (tk["handle_hours"] >= 0)
)
tk["in_primary_window"] = (
    (tk["created_at"] >= PRIMARY_WINDOW_START)
    & (tk["created_at"] <= PRIMARY_WINDOW_END)
)
tk["transfers"] = pd.to_numeric(tk["transfers"], errors="coerce")
tk["csat_score"] = pd.to_numeric(tk["csat_score"], errors="coerce")

# Primary-window subset
pw = tk[tk["in_primary_window"]].copy()
print(f"  Primary window (Jan 2025 – Jun 2026)   : {len(pw):,} tickets")
print(f"  Date range check: min={pw['created_at'].min().date()}  max={pw['created_at'].max().date()}")

# Ticket volumes per team
vol = pw.groupby("assigned_team").size().reset_index(name="tickets")

# Agent headcount per team (from agents.csv)
hc_agents = agents.groupby("team").agg(
    headcount=("agent_id", "count"),
    min_tier=("tier", "min"),
    max_tier=("tier", "max"),
).reset_index().rename(columns={"team": "assigned_team"})

# Handle-time stats (valid only)
valid_ht = pw[pw["handle_time_valid"]]
ht_stats = valid_ht.groupby("assigned_team")["handle_hours"].agg(
    valid_n="count",
    median_h="median",
    mean_h="mean",
    p90_h=lambda x: x.quantile(0.9),
    negative_excluded="count",  # just a placeholder label
).reset_index()
# Re-derive negative excluded separately
neg_excluded = (
    pw[pw["first_response_at"].notna() & pw["resolved_at"].notna()]
    .groupby("assigned_team")
    .apply(lambda g: (g["handle_hours"] < 0).sum())
    .reset_index(name="negative_ht_excluded")
)

# Transfer stats (helpdesk tickets only — where data is available)
current = pw[pw["source_system"] == "helpdesk"]
has_tr  = current[current["transfers"].notna()]
if len(has_tr) > 0:
    tr_stats = has_tr.groupby("assigned_team").agg(
        tr_n=("transfers", "count"),
        mean_transfers=("transfers", "mean"),
        transfer_rate=("transfers", lambda x: (x > 0).mean()),
    ).reset_index()
else:
    tr_stats = pd.DataFrame(columns=["assigned_team", "tr_n", "mean_transfers", "transfer_rate"])

# CSAT
csat = pw[pw["csat_score"].notna()].groupby("assigned_team")["csat_score"].agg(
    avg_csat="mean", csat_n="count"
).reset_index()

# Join everything
report = vol.merge(hc_agents, on="assigned_team", how="left")
report = report.merge(ht_stats[["assigned_team", "valid_n", "median_h", "mean_h", "p90_h"]], on="assigned_team", how="left")
report = report.merge(neg_excluded, on="assigned_team", how="left")
report = report.merge(tr_stats, on="assigned_team", how="left")
report = report.merge(csat, on="assigned_team", how="left")

report["tickets_per_agent"] = (report["tickets"] / report["headcount"]).round(1)
report["handle_h_notes"]    = report.apply(
    lambda r: f"valid={int(r['valid_n']) if pd.notna(r['valid_n']) else 'n/a'} / neg_excl={int(r['negative_ht_excluded']) if pd.notna(r['negative_ht_excluded']) else 'n/a'}", axis=1
)

print(f"\n  Transfer data: {len(has_tr):,} helpdesk tickets with non-null transfers / {len(current):,} total helpdesk in window")
print(f"  Transfer data covers: {len(has_tr)/len(pw)*100:.1f}% of primary-window tickets")
print()

# Print the full table
print(f"  {'Team':<30} {'Tier':>4} {'HC':>3} {'Tickets':>8} {'Tkt/Agt':>8} {'Median_h':>9} {'P90_h':>6} {'Valid_n':>7} {'Neg_excl':>9} {'TR_rate':>8} {'CSAT':>6}")
print("  " + "-"*100)
for _, r in report.sort_values("tickets_per_agent", ascending=False).iterrows():
    tier_str = str(r["min_tier"]) if r["min_tier"] == r["max_tier"] else f"{r['min_tier']}-{r['max_tier']}"
    tr_str   = f"{r['transfer_rate']*100:.0f}%" if pd.notna(r.get("transfer_rate")) else "n/a"
    csat_str = f"{r['avg_csat']:.2f}" if pd.notna(r.get("avg_csat")) else "n/a"
    med_str  = f"{r['median_h']:.1f}" if pd.notna(r.get("median_h")) else "n/a"
    p90_str  = f"{r['p90_h']:.1f}" if pd.notna(r.get("p90_h")) else "n/a"
    neg_str  = str(int(r['negative_ht_excluded'])) if pd.notna(r.get('negative_ht_excluded')) else "n/a"
    vn_str   = str(int(r['valid_n'])) if pd.notna(r.get('valid_n')) else "n/a"
    print(f"  {r['assigned_team']:<30} {tier_str:>4} {int(r['headcount']):>3} {int(r['tickets']):>8} {r['tickets_per_agent']:>8.1f} {med_str:>9} {p90_str:>6} {vn_str:>7} {neg_str:>9} {tr_str:>8} {csat_str:>6}")

# Check: does the headcount recommendation "Billing gets both" hold up?
billing_row  = report[report["assigned_team"].str.lower().str.contains("billing")].iloc[0]
logistics_row = report[report["assigned_team"].str.lower().str.contains("logistics")].iloc[0]
returns_row  = report[report["assigned_team"].str.lower().str.contains("returns")].iloc[0]

print("\n  Top 3 by tickets/agent:")
print(f"    Billing     : {billing_row['tickets_per_agent']:.0f} tkt/agt | tier={billing_row['min_tier']} | median_h={billing_row['median_h']:.1f}h | transfer_rate={billing_row['transfer_rate']*100:.0f}% | CSAT={billing_row['avg_csat']:.2f}" if pd.notna(billing_row.get("transfer_rate")) else f"    Billing     : {billing_row['tickets_per_agent']:.0f} tkt/agt | tier={billing_row['min_tier']}")
print(f"    Logistics   : {logistics_row['tickets_per_agent']:.0f} tkt/agt | tier={logistics_row['min_tier']} | median_h={logistics_row['median_h']:.1f}h | transfer_rate={logistics_row['transfer_rate']*100:.0f}% | CSAT={logistics_row['avg_csat']:.2f}" if pd.notna(logistics_row.get("transfer_rate")) else f"    Logistics   : {logistics_row['tickets_per_agent']:.0f} tkt/agt | tier={logistics_row['min_tier']}")

# Check tier assignments
print("\n  Tier breakdown across teams:")
tier_check = agents.groupby(["team", "tier"]).size().unstack(fill_value=0)
print(tier_check.to_string())

# Handle-time concern: Billing has very low median (0.5h) — flag this
print(f"\n  *** Billing team median handle time: {billing_row['median_h']:.1f}h")
print(f"      This is suspiciously low vs Logistics ({logistics_row['median_h']:.1f}h).")
print(f"      Check: are Billing tickets possibly resolved via auto-close/macro?")
print(f"      Valid handle-time observations for Billing: {int(billing_row['valid_n']) if pd.notna(billing_row['valid_n']) else 'n/a'}")

# Check negative excluded for Billing
print(f"      Negative handle-time excluded (Billing): {int(billing_row['negative_ht_excluded']) if pd.notna(billing_row.get('negative_ht_excluded')) else 'n/a'}")

# Transfer rate concern for Billing
print(f"\n  *** Billing transfer_rate={billing_row['transfer_rate']*100:.0f}% vs Logistics={logistics_row['transfer_rate']*100:.0f}%")
print(f"      Billing transfers recorded over {int(billing_row['tr_n'])} tickets (helpdesk only)")

# Issue: headcount_analysis.csv had "2425" tickets for Billing but raw data shows...
billing_all = tk[tk["assigned_team"].str.lower().str.contains("billing", na=False) & tk["in_primary_window"]]["ticket_id"].nunique()
print(f"\n  Billing tickets in primary window (raw): {billing_all:,}")
print(f"  Billing tickets in headcount_analysis.csv output: {int(billing_row['tickets']):,}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Cost report units
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 66)
print("CHECK 5: Cost report unit audit")
print("=" * 66)

cost_df = pd.read_csv(ROOT / "outputs" / "cost_report.csv")
print(cost_df[["metric", "value", "unit", "notes"]].to_string(index=False))

# Check: are USD and INR properly separated?
usd_rows = cost_df[cost_df["unit"].str.contains("USD", na=False)]
inr_rows = cost_df[cost_df["unit"].str.contains("INR", na=False)]
print(f"\n  USD-unit rows: {len(usd_rows)}")
print(f"  INR-unit rows: {len(inr_rows)}")

# Check: is the GPT-4o-mini estimate being used as final cost?
ai_cost_row = cost_df[cost_df["metric"] == "ai_historical_cost_usd"]
if not ai_cost_row.empty:
    actual_cost = float(ai_cost_row["value"].iloc[0])
    print(f"\n  ai_historical_cost_usd = {actual_cost}")
    if actual_cost == 0.0:
        print("  [OK] No LLM was called; historical cost is correctly 0.0 (offline run)")
    else:
        print("  [WARN] Non-zero LLM cost recorded — verify this is real API cost")

# Check projected cost uses correct basis
proj_row = cost_df[cost_df["metric"] == "ai_projected_annual_cost_usd"]
if not proj_row.empty:
    print(f"\n  ai_projected_annual_cost_usd = {proj_row['value'].iloc[0]}")
    print(f"  notes: {proj_row['notes'].iloc[0]}")
    print("  [ISSUE] Projected cost uses GPT-4o-mini token estimate (preliminary).")
    print("  Fix: label this as 'estimate' not 'cost' and scope it clearly.")

# Confirm ₹290/contact basis
contact_row = cost_df[cost_df["metric"] == "support_blended_cost_per_contact_inr"]
if not contact_row.empty:
    v = float(contact_row["value"].iloc[0])
    print(f"\n  support_blended_cost_per_contact_inr = {v}")
    print(f"  [{'OK' if v == 290 else 'FAIL'}] Should be 290 per policy §4")

print("\nAudit complete.")
