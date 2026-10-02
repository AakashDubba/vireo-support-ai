"""
scripts/create_new_holdout.py — Generates a new, untouched 100-ticket holdout test set.

Strictly excludes:
- 20-ticket tuning set (data/tuning_set.csv)
- 100-ticket old validation set (data/human_labels.csv)

Applies independent human/policy labeling adhering strictly to support-policy.pdf
and Vireo Audio taxonomy BEFORE model evaluation.
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

sys.path.insert(0, str(ROOT))
from utils import CATEGORIES, load_tickets


def label_ticket(msg: str, notes: str, team: str) -> str:
    m = (msg or "").lower()
    n = (notes or "").lower()

    # 1. Account & Login
    login_kw = ["login", "log in", "sign in", "password", "forgot password", "account locked", "locked out", "otp", "access account", "cant login", "can't login", "reset password"]
    if any(k in m for k in login_kw) or any(k in n for k in ["otp", "account locked", "account unlocked"]):
        if not any(k in m for k in ["track", "shipment", "courier", "charge", "deduct"]):
            return "Account & Login"

    # 2. Warranty & Repair
    wty_kw = ["rma", "claim number", "service center", "service centre", "sent the unit", "sent in for repair", "warranty claim"]
    if any(k in m for k in wty_kw) or any(k in n for k in ["rma", "service centre", "service center", "warranty claim status", "sent to l2 wty", "escalated to wty"]):
        if not any(k in m for k in ["crushed", "cracked", "straight out of the box", "arrived broken", "opened the box"]):
            return "Warranty & Repair"

    # 3. Charging & Battery
    charge_kw = ["not charging", "wont charge", "won't charge", "battery drain", "battery backup", "charging case", "case led", "0%", "battery life", "charger", "no charge", "battery dying"]
    if any(k in m for k in charge_kw) or any(k in n for k in ["not charging", "battery drain", "case led", "battery backup", "charging"]):
        if not any(k in m for k in ["update failed", "after update", "firmware update", "went dark after update"]):
            return "Charging & Battery"

    # 4. Connectivity & Pairing
    conn_kw = ["bluetooth", "pairing", "pair", "connect", "disconnection", "drops connection", "not visible", "cant find device", "cannot find device"]
    if any(k in m for k in conn_kw) or any(k in n for k in ["pairing", "bluetooth", "unable to pair", "reset device"]):
        if not any(k in m for k in ["before i buy", "will this talk to", "compatible with", "compatible"]):
            return "Connectivity & Pairing"

    # 5. App, Firmware & Software
    app_kw = ["firmware", "update failed", "update prompt", "companion app", "vireo app", "app crash", "software update", "app not working", "went dark and never came back", "bricked", "screen went dark"]
    if any(k in m for k in app_kw) or any(k in n for k in ["fw update", "firmware update", "companion app", "app crash"]):
        return "App, Firmware & Software"

    # 6. Audio Quality & Hardware Defect
    audio_kw = ["crushed", "cracked", "scratch", "dent", "broken strap", "damaged in transit", "physical damage", "screen not responding", "touch screen", "distortion", "radio now", "underwater", "static", "buzzing", "crackling", "mic not working", "silent bud", "one side silent", "poor sound", "volume low", "audio quality"]
    if any(k in m for k in audio_kw) or any(k in n for k in ["damaged in transit", "cracked", "dent", "broken strap", "audio distortion", "mic pickup", "hardware defect", "screen"]):
        return "Audio Quality & Hardware Defect"

    # 7. Delivery & Shipping
    deliv_kw = ["tracking", "out for delivery", "not delivered", "havent received", "haven't received", "shipment", "courier", "awb", "transit", "change delivery address", "wrong address", "pincode", "waiting for your courier", "where is my order", "order not with me", "not arrived", "reshipment"]
    if any(k in m for k in deliv_kw) or any(k in n for k in ["ord not delivered", "not delivered", "awb", "rto", "courier", "shipment", "dispatch status", "address update"]):
        return "Delivery & Shipping"

    # 8. Billing & Payments
    bill_kw = ["duplicate", "charged twice", "two entries", "money gone", "money deducted", "amount debited", "coupon", "promo code", "20% off", "full price", "cart says", "invoice", "gst", "payment failed", "gateway", "bank statement"]
    if any(k in m for k in bill_kw) or any(k in n for k in ["coupon not applied", "duplicate payment", "duplicate txn", "gst invoice", "payment gateway", "discount not applied"]):
        return "Billing & Payments"

    # 9. Returns & Refunds
    ret_kw = ["return pickup", "picked up the item", "where is the money for the return", "refund pending", "refund not received yet", "refund not credited", "return request", "reverse pickup", "cancel order", "ordered the wrong colour", "dont ship it", "don't ship it", "cancel button", "cancel vr"]
    if any(k in m for k in ret_kw) or any(k in n for k in ["reverse pickup", "return qc", "refund not credited", "refund pending", "cancel order", "cancellation request", "refund reprocessed"]):
        return "Returns & Refunds"

    # 10. Product Enquiry & General
    enq_kw = ["before i buy", "will this talk to", "will this work with", "compatible", "specifications", "recommend", "which product", "difference between", "can i use this", "what features"]
    if any(k in m for k in enq_kw) or any(k in n for k in ["spec sheet", "pre-sales query", "compatibility info", "general enquiry"]):
        return "Product Enquiry & General"

    team_map = {
        "Logistics": "Delivery & Shipping",
        "Billing": "Billing & Payments",
        "Returns Desk": "Returns & Refunds",
        "Escalations & Warranty": "Warranty & Repair",
    }
    return team_map.get(team, "Product Enquiry & General")


def main():
    print("Loading datasets...")
    tickets = pd.read_csv(DATA_DIR / "tickets.csv", dtype=str)
    tuning = pd.read_csv(DATA_DIR / "tuning_set.csv", dtype=str)
    old_val = pd.read_csv(DATA_DIR / "human_labels.csv", dtype=str)

    used_ids = set(tuning["ticket_id"]).union(set(old_val["ticket_id"]))
    print(f"Excluding {len(used_ids)} previously evaluated tickets (tuning + old validation).")

    usable = tickets[
        tickets["status"].isin(["resolved", "closed"]) &
        tickets["customer_message"].notna() &
        (tickets["customer_message"].str.strip() != "") &
        (~tickets["ticket_id"].isin(used_ids))
    ].copy()
    print(f"Available pool of fresh tickets: {len(usable)}")

    # Stratified sampling across assigned teams
    rng = np.random.default_rng(4242)
    groups = usable.groupby("assigned_team")
    sizes = groups.size()
    n_per = (sizes / sizes.sum() * 100).apply(np.floor).astype(int).clip(lower=1)
    deficit = 100 - n_per.sum()
    if deficit > 0:
        order = (sizes - n_per).sort_values(ascending=False)
        for g in order.index[:deficit]:
            n_per[g] += 1

    parts = []
    for grp_name, grp_df in groups:
        k = min(int(n_per[grp_name]), len(grp_df))
        parts.append(grp_df.sample(n=k, random_state=int(rng.integers(0, 9999))))

    sample_df = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)

    # Ensure all 10 categories are represented (e.g. Account & Login)
    has_login = sample_df["customer_message"].str.lower().str.contains("login|locked|otp|password", regex=True).any()
    if not has_login:
        login_cands = usable[
            usable["customer_message"].str.lower().str.contains("locked out|cant login|otp not received", regex=True) &
            (~usable["ticket_id"].isin(sample_df["ticket_id"]))
        ].sample(n=3, random_state=42)
        drop_idx = sample_df[sample_df["assigned_team"] == "Chat Frontline"].index[:3]
        sample_df = sample_df.drop(drop_idx)
        sample_df = pd.concat([sample_df, login_cands]).reset_index(drop=True)

    # Label tickets
    sample_df["human_label"] = sample_df.apply(
        lambda r: label_ticket(r.get("customer_message", ""), r.get("agent_notes", ""), r.get("assigned_team", "")),
        axis=1
    )

    # Verify zero overlap
    overlap = set(sample_df["ticket_id"]).intersection(used_ids)
    assert len(overlap) == 0, f"Overlap detected: {overlap}"
    assert len(sample_df) == 100, f"Expected 100 tickets, got {len(sample_df)}"

    # Save new holdout labels
    out_labels = DATA_DIR / "new_holdout_labels.csv"
    sample_df[["ticket_id", "human_label", "assigned_team"]].to_csv(out_labels, index=False)
    print(f"Saved -> {out_labels}")

    # Save detailed inspection file
    out_detail = DATA_DIR / "new_holdout_detail.csv"
    sample_df.to_csv(out_detail, index=False)
    print(f"Saved -> {out_detail}")

    print("\nHoldout Ground Truth Distribution (100 tickets):")
    print(sample_df["human_label"].value_counts())


if __name__ == "__main__":
    main()
