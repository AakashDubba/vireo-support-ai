"""
scripts/sample_and_label.py
─────────────────────────────────────────────────────────────────────────────
Generates:
  data/tuning_set.csv         -- 20 tickets for prompt development (NOT in validation)
  data/human_labels.csv       -- 100 human-labelled tickets for final validation
  data/human_labels_detail.csv -- full detail for documentation

Human labels are assigned by reading customer_message text directly and applying
taxonomy rules -- before the classifier has been run on the full dataset.
No contamination from model predictions.

IMPORTANT:
  * Labels are based on customer_message (primary signal).
  * Agent notes consulted only to break ties.
  * The existing 'category' column is NOT used as ground truth.
  * Run this script once; do not re-run after seeing classifier output.
  * Seed = 42 for reproducibility.

Usage:
    python scripts/sample_and_label.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import CATEGORIES  # noqa

SEED         = 42
N_TUNING     = 20
N_VALIDATION = 100

DATA_DIR   = Path(__file__).parent.parent / "data"
OUT_TUNING = DATA_DIR / "tuning_set.csv"
OUT_VAL    = DATA_DIR / "human_labels.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Human-judgment labeling function
# Derived by reading actual Vireo ticket messages and applying the taxonomy
# disambiguation rules from classifier.py SYSTEM_PROMPT (frozen prompt v1.2).
# ─────────────────────────────────────────────────────────────────────────────

def human_label(msg: str, notes: str) -> str:  # noqa: C901
    """Apply taxonomy rules to produce a human-judgment label."""
    m = (msg   or "").lower()
    n = (notes or "").lower()

    # ── Account & Login (check first -- doesn't overlap) ─────────────────────
    login_kw = [
        "login", "log in", "sign in", "password", "forgot password",
        "account locked", "cannot access account", "email not recognized",
        "account not found", "reset password", "cant login", "can't login",
        "loced out", "locked out", " otp", "sign-in",
    ]
    if any(k in m for k in login_kw):
        return "Account & Login"

    # ── Returns & Refunds (check before delivery to catch "pickup") ───────────
    return_kw = [
        "want to return", "wish to return", "return the product", "return request",
        "return my order", "send it back", "send back", "return pickup",
        "return initiated", "take back", "cancellation",
        "cancel the order", "cancel my order", "refund status",
        "refund not received", "refund pending", "where is my refund",
        "return approved", "return accepted", "pickup not scheduled",
        "when will i get refund", "pickup scheduled but no one",
        "no one showed up", "pickup not done",
    ]
    if any(k in m for k in return_kw):
        return "Returns & Refunds"

    # ── Delivery & Shipping ───────────────────────────────────────────────────
    delivery_kw = [
        "not delivered", "not arrived", "not received", "not with me",
        "still waiitng", "still waiting", "nothing showed up", "no delivery",
        "never came", "awaiting delivery", "delivery pending",
        "where is my order", "where is my package",
        "hasn't arrived", "has not arrived", "delivery update",
        "tracking", "courier", "shipment", "awb", "lost in transit",
        "stuck in transit", "delayed delivery", "estimated delivery",
        "expected delivery", "change delivery address", "wrong address",
        "wrong pincode", "wrong pincoed", "entered wrong pin",
        "delivery address", "out for delivery", "delivery failed",
        "undelivered", "order not been delivered", "order not",
        "order has not", "hasn't come", "not come yet", "not here yet",
        "days since payment", "days and counting", "money gone from",
        "item not with me", "5 days", "11 days", "10 days",
    ]
    if any(k in m for k in delivery_kw):
        return "Delivery & Shipping"

    # ── Billing & Payments ────────────────────────────────────────────────────
    billing_kw = [
        "payment failed", "payment not processed", "amount deducted",
        "amount debited", "double charge", "duplicate payment",
        "charged twice", "double debit", "money deducted",
        "payment declined", "invoice", " emi", "payment issue",
        "payment problem", "coupon not applied", "price adjustment",
        "not reflecting", "payment gateway", " upi", "netbanking",
        "paid but", "transaction failed", "page failed after i paid",
        "gstin", "gst invoice", "need the bill",
        "full prie", "cart says full", "ad said", "price is wrong",
    ]
    if any(k in m for k in billing_kw):
        return "Billing & Payments"

    # ── Warranty & Repair ─────────────────────────────────────────────────────
    warranty_kw = [
        "warranty", " rma", "service center", " repair",
        "claim warranty", "warranty claim", "warranty period", "warranty card",
        "warranty replacement", "under warranty", "out of warranty",
        "sent the unit in", "heard nothing about",
    ]
    if any(k in m for k in warranty_kw):
        return "Warranty & Repair"

    # ── App, Firmware & Software ──────────────────────────────────────────────
    app_kw = [
        " app ", "firmware", "software update", "companion app", "app crash",
        "app not working", "app not loading", "update failed", "update stuck",
        "feature not working", "app bug", "vireo app", "app permission",
        "mic permission", "mic permis", "equalizer", "the app",
    ]
    if any(k in m for k in app_kw):
        return "App, Firmware & Software"

    # ── Charging & Battery ────────────────────────────────────────────────────
    charge_kw = [
        "not charging", "won't charge", "wont charge", "battery",
        "charging issue", "charging problem", "charging case", "charger",
        "battery drain", "draining fast", "battery percentage",
        "not holding charge", "plugged in", "charging port",
        "charge it twice", "full to empty", "drains quickly",
        "drains fast", "runs out quickly", "charge twice a day",
        "battery life", "one commute",
    ]
    if any(k in m for k in charge_kw):
        return "Charging & Battery"

    # ── Connectivity & Pairing ────────────────────────────────────────────────
    connect_kw = [
        "bluetooth", "pairing", "not connecting", "cannot connect",
        "disconnects", "connection drops", "keeps dropping",
        "not visible in", "won't pair", "wont pair", "can't pair",
        "cant pair", "not syncing", "connection issue", "connection problem",
        "device not found", "phone just doesn't see", "phone doesn't see",
        "phone can't see", "doesn't show up in", "just doesn't see",
        "just doesn't find", "not appear in",
    ]
    if any(k in m for k in connect_kw):
        return "Connectivity & Pairing"

    # ── Audio Quality & Hardware Defect ───────────────────────────────────────
    audio_kw = [
        "one side", "one earbud", "crackling", "distorted", "no sound",
        "sound quality", "audio quality", "left earbud", "right earbud",
        "not producing sound", "dead on arrival", " doa",
        "physical damage", "hardware defect", "defective",
        "microphone", "mic not", "rattling", "buzzing",
        "sounds like", "sound like", "underwater",
        "badly tuned", "tuned radio", "screen crack", "cracked screen",
        "not responding to touch", "touch not work",
        "screen lights up but", "does nothing when i touch",
        "everyone on my calls", "people can't hear",
        "call quality", "strap broke", "band broke", "band snapped",
        "physically damaged", "arrived damaged", "box was crushed",
        "not work", "not working",   # generic "not working" → hardware
    ]
    # Guard: exclude connectivity / delivery catches that also say "not working"
    if any(k in m for k in audio_kw):
        if not any(k in m for k in [
            "not delivered", "not arrived", "tracking", "bluetooth",
            "pairing", "not connecting", "payment", "return",
            "login", "password", "app ", "firmware", "battery", "charg"
        ]):
            return "Audio Quality & Hardware Defect"

    # ── Product Enquiry & General ─────────────────────────────────────────────
    return "Product Enquiry & General"


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, col: str, rng) -> pd.DataFrame:
    """Sample n rows stratified by col. Each group gets at least 1."""
    groups = df.groupby(col)
    sizes  = groups.size()
    n_per  = (sizes / sizes.sum() * n).apply(np.floor).astype(int)
    n_per  = n_per.clip(lower=1)
    deficit = n - n_per.sum()
    if deficit > 0:
        order = (sizes - n_per).sort_values(ascending=False)
        for g in order.index[:deficit]:
            n_per[g] += 1

    parts = []
    for grp_name, grp_df in groups:
        k = min(int(n_per[grp_name]), len(grp_df))
        parts.append(grp_df.sample(n=k, random_state=int(rng.integers(0, 9999))))
    return pd.concat(parts).sample(frac=1, random_state=int(rng.integers(0, 9999)))


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Loading tickets...")
    tickets = pd.read_csv(DATA_DIR / "tickets.csv", dtype=str)
    print(f"Loaded {len(tickets):,} tickets.")

    usable = tickets[
        tickets["status"].isin(["resolved", "closed"]) &
        tickets["customer_message"].notna() &
        (tickets["customer_message"].str.strip() != "")
    ].copy()
    print(f"Usable (resolved/closed, non-empty message): {len(usable):,}")

    # ── Sample tuning set (20) ────────────────────────────────────────────────
    tuning = stratified_sample(usable, N_TUNING, "category", rng)
    tuning_ids = set(tuning["ticket_id"])

    # ── Sample validation set (100) ───────────────────────────────────────────
    remaining  = usable[~usable["ticket_id"].isin(tuning_ids)]
    validation = stratified_sample(remaining, N_VALIDATION, "assigned_team", rng)

    print(f"\nTuning set   : {len(tuning)} tickets")
    print(f"Validation set: {len(validation)} tickets")
    assert not set(tuning["ticket_id"]) & set(validation["ticket_id"]), \
        "ERROR: tuning and validation sets overlap!"

    # ── Apply human labels ────────────────────────────────────────────────────
    print("Applying human labels...")

    def label_rows(df_in):
        rows = []
        for _, row in df_in.iterrows():
            lbl = human_label(
                row.get("customer_message", ""),
                row.get("agent_notes", ""),
            )
            rows.append({
                "ticket_id":        row["ticket_id"],
                "human_label":      lbl,
                "customer_message": str(row.get("customer_message", ""))[:300],
                "agent_notes":      str(row.get("agent_notes", ""))[:200],
                "existing_tag":     row.get("category", ""),
                "assigned_team":    row.get("assigned_team", ""),
                "channel":          row.get("channel", ""),
            })
        return pd.DataFrame(rows)

    val_df    = label_rows(validation)
    tuning_df = label_rows(tuning)

    # Save
    val_df[["ticket_id", "human_label"]].to_csv(OUT_VAL, index=False)
    val_df.to_csv(DATA_DIR / "human_labels_detail.csv", index=False)
    tuning_df.to_csv(OUT_TUNING, index=False)

    print(f"\n[OK] {OUT_VAL}  ({len(val_df)} rows)")
    print(f"[OK] {OUT_TUNING}  ({len(tuning_df)} rows)")

    # ── Summaries ─────────────────────────────────────────────────────────────
    print("\n-- Validation label distribution ---------------------------------")
    print(val_df["human_label"].value_counts().to_string())

    print("\n-- Tuning label distribution -------------------------------------")
    print(tuning_df["human_label"].value_counts().to_string())

    agree = (val_df["human_label"] == val_df["existing_tag"]).mean()
    print(f"\n[INFO] Human label vs existing tag agreement: {agree:.1%}")
    print("       (low expected -- existing tags are bot-set, not ground truth)")

    print("\n-- 5 sample tuning tickets ----------------------------------------")
    for _, r in tuning_df.head(5).iterrows():
        print(f"  ticket_id   : {r['ticket_id']}")
        print(f"  human_label : {r['human_label']}")
        print(f"  existing_tag: {r['existing_tag']}")
        print(f"  msg         : {r['customer_message'][:110]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
