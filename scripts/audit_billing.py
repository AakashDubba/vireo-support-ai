"""
Dig into Billing team handle-time distribution and source_system split.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from utils import PRIMARY_WINDOW_START, PRIMARY_WINDOW_END

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

pw = tk[tk["in_primary_window"]].copy()
billing = pw[pw["assigned_team"] == "Billing"]

print("=== Billing team handle-time deep dive ===")
print(f"Total Billing tickets (primary window): {len(billing)}")

# Source system split
print("\nSource system split:")
print(billing["source_system"].value_counts().to_string())

# Handle-time breakdown by source system
for sys_name, grp in billing.groupby("source_system"):
    valid_ht = grp[grp["handle_time_valid"]]
    neg_ht   = grp[grp["first_response_at"].notna() & grp["resolved_at"].notna() & (grp["handle_hours"] < 0)]
    print(f"\n  source_system={sys_name!r}  total={len(grp)} valid={len(valid_ht)} neg_excl={len(neg_ht)}")
    if len(valid_ht) > 0:
        print(f"    median={valid_ht['handle_hours'].median():.2f}h  mean={valid_ht['handle_hours'].mean():.2f}h  p5={valid_ht['handle_hours'].quantile(0.05):.2f}h  p90={valid_ht['handle_hours'].quantile(0.90):.2f}h  max={valid_ht['handle_hours'].max():.1f}h")
        # Breakdown of very short handle times
        short = (valid_ht["handle_hours"] < 0.1).sum()
        print(f"    handle_h < 6 min (auto-close likely): {short} ({short/len(valid_ht):.0%})")

# Compare with Logistics
print("\n\n=== Logistics team handle-time deep dive ===")
logistics = pw[pw["assigned_team"] == "Logistics"]
for sys_name, grp in logistics.groupby("source_system"):
    valid_ht = grp[grp["handle_time_valid"]]
    neg_ht   = grp[grp["first_response_at"].notna() & grp["resolved_at"].notna() & (grp["handle_hours"] < 0)]
    print(f"\n  source_system={sys_name!r}  total={len(grp)} valid={len(valid_ht)} neg_excl={len(neg_ht)}")
    if len(valid_ht) > 0:
        print(f"    median={valid_ht['handle_hours'].median():.2f}h  mean={valid_ht['handle_hours'].mean():.2f}h  p5={valid_ht['handle_hours'].quantile(0.05):.2f}h  p90={valid_ht['handle_hours'].quantile(0.90):.2f}h")

# Transfer-rate concern: Billing has 32% vs Logistics 8%
print("\n\n=== Transfer rate reality check ===")
current = pw[pw["source_system"] == "helpdesk"]
billing_cur   = current[current["assigned_team"] == "Billing"]
logistics_cur = current[current["assigned_team"] == "Logistics"]
print(f"Billing   helpdesk: {len(billing_cur)} tickets, transfers non-null: {billing_cur['transfers'].notna().sum()}")
tk["transfers"] = pd.to_numeric(tk["transfers"], errors="coerce")
pw_transfers = pw.copy()
pw_transfers["transfers"] = pd.to_numeric(pw_transfers["transfers"], errors="coerce")
billing_transfers = pw_transfers[(pw_transfers["assigned_team"]=="Billing") & pw_transfers["transfers"].notna()]
print(f"  transfer > 0: {(billing_transfers['transfers'] > 0).sum()}/{len(billing_transfers)} = {(billing_transfers['transfers']>0).mean():.0%}")
print(f"  mean transfers per ticket: {billing_transfers['transfers'].mean():.2f}")
logistics_transfers = pw_transfers[(pw_transfers["assigned_team"]=="Logistics") & pw_transfers["transfers"].notna()]
print(f"Logistics helpdesk: transfers >0: {(logistics_transfers['transfers']>0).sum()}/{len(logistics_transfers)} = {(logistics_transfers['transfers']>0).mean():.0%}")
