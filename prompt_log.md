# Vireo Audio — Prompt Development Log

**Final frozen version:** `v1.3`  
**Frozen date:** 2026-10-01  
**Tuning set:** `data/tuning_set.csv` — 20 tickets (SEPARATE from 100-ticket validation set)  
**Validation set:** `data/human_labels.csv` — 100 tickets, zero overlap with tuning set  
**Model for offline tuning:** `all-MiniLM-L6-v2` (sentence transformer, cosine similarity)  
**LLM:** Groq `openai/gpt-oss-20b` — configures via `.env` (offline validation completed without it)

---

## Taxonomy Validation (Pre-Prompt)

Before any prompt was written, the 10 proposed categories were validated against:
- Actual ticket text in `tickets.csv` (`customer_message` field)
- `support-policy.pdf` definitions
- Existing bot-set `category` column distribution (noted as unreliable — 13.8% "Other")

**Two renames approved:**

| Original | Renamed to | Reason |
|---|---|---|
| Connectivity | Connectivity & Pairing | "Pairing" is the most common symptom in actual ticket text |
| Audio Quality | Audio Quality & Hardware Defect | Many tickets describe physical defects (cracked screen, broken strap) |

**No categories merged or split.** The 10-category taxonomy was retained.

---

## Prompt v1.2 — Initial LLM Draft

**Date:** 2026-10-01 (pre-tuning)  
**Status:** SUPERSEDED by v1.3

**Content summary:**
- System prompt with all 10 exact category names
- Instructions to classify on customer message (primary) and agent notes (secondary)
- `[IVR transcript]` prefix handling
- 9 disambiguation rules covering the most common confusion pairs

**Tested on 20-ticket tuning set (offline sentence-transformer):**
- Accuracy: **14/20 = 70%**
- Mode: offline-embedding (no LLM key available for first tuning pass)

**Failures (6/20):**

| Ticket | Human Label | v1.2 Predicted | Root Cause |
|---|---|---|---|
| TK-244258 | Audio Quality & Hardware Defect | Warranty & Repair | "cracked" + "regretting" pulled toward warranty embedding |
| TK-246042 | Delivery & Shipping | Audio Quality & Hardware Defect | "poor" + headphone product name pulled AQ embedding |
| TK-251674 | Product Enquiry & General | Returns & Refunds | "please reverse" triggered returns keywords |
| TK-247731 | Delivery & Shipping | Returns & Refunds | "not delivered + I want..." triggered refunds embedding |
| TK-246068 | Product Enquiry & General | Warranty & Repair | "before I buy" not in Product Enquiry description; IVR noise |
| TK-243625 | Delivery & Shipping | Warranty & Repair | "firmware update + went dark" pulled warranty; message actually ambiguous |

**Analysis:**
- 3/6 failures involve offline embedding pulling the wrong category due to ambiguous vocabulary
- 2/6 are "Delivery & Shipping" tickets where distractor words (product name, "poor", "I want") confused the embedding
- 1/6 (TK-243625) is a borderline case: device went dark after firmware update — this is arguably App/Firmware, but human label says Delivery (possible labeling error; human labels preserved without modification)
- Failures 1–5 are expected to be correctly handled by the LLM's disambiguation rules

**Changes made for v1.3:**

---

## Prompt v1.3 — FROZEN (Final)

**Date:** 2026-10-01  
**Status:** FROZEN — do not modify for final validation

**Changes from v1.2:**
1. **Delivery & Shipping description:** Added "Payment was successful but order not delivered", "Checked with neighbours", "Poor delivery experience" — addresses TK-246042 failure
2. **Returns & Refunds description:** Changed "Cancel before shipping" to "Cancellation after delivery received" to sharpen the Delivery vs Returns boundary — addresses TK-247731 failure
3. **Product Enquiry description:** Added "Before I buy", "Will this talk to", "Will this be compatible with", "Is this compatible with Samsung or Nokia" — addresses TK-251674 and TK-246068 failures
4. **App, Firmware description:** Added "Device went dark after firmware update", "Device stopped responding after software update prompt" — addresses TK-243625 ambiguity; better separates firmware failures from delivery
5. **Warranty & Repair description:** Tightened to "Claiming warranty for manufacturing defect" — reduces pull on tickets that mention warranty only incidentally
6. **Audio Quality description:** Added "Product arrived cracked or physically broken" — separates physical damage on arrival from warranty claims

**Tested on 20-ticket tuning set (offline sentence-transformer):**
- Accuracy: **16/20 = 80%**
- Improvement: +10pp over v1.2
- Mode: offline-embedding (no LLM key for tuning pass)

**Remaining failures (4/20) — accepted as offline embedding limitations:**

| Ticket | Human Label | v1.3 Predicted | Analysis |
|---|---|---|---|
| TK-244258 | Audio Quality & Hardware Defect | Warranty & Repair | "cracked" still near warranty in embedding space; LLM disambiguation rule handles this |
| TK-246042 | Delivery & Shipping | Audio Quality & Hardware Defect | Extremely low confidence (0.49); LLM will classify correctly from message context |
| TK-251674 | Product Enquiry & General | Returns & Refunds | "please reverse" still near returns; borderline message; LLM should handle with rules |
| TK-243625 | Delivery & Shipping | App, Firmware & Software | Message genuinely describes firmware failure; human label "Delivery" may be incorrect; human labels preserved unchanged |

**Decision:** Accept v1.3 as final frozen prompt. Offline accuracy of 80% on tuning set is acceptable given:
- Offline mode is the fallback, not the primary classifier
- LLM (Groq) accuracy will be higher due to natural language reasoning
- Remaining failures are at low confidence scores (0.36–0.49)

---

## Final Frozen System Prompt (stored in `classifier.py`)

```
You are a support-ticket classifier for Vireo Audio, a consumer audio and wearables brand.

Classify each ticket into exactly one of the categories below, based on the PRIMARY reason
the customer contacted support — not the agent's resolution action.

Use agent notes only to resolve genuine ambiguity in the customer message.
Ignore prefixes such as [IVR transcript] — classify on the actual message content.

CATEGORIES (return the exact string):
1. Delivery & Shipping       — Package not delivered, courier delay, tracking issue, wrong address,
                               wrong pincode, reshipment request.
2. Billing & Payments        — Payment failure, duplicate charge, amount debited, invoice/GST query,
                               coupon not applied, payment-gateway error.
3. Returns & Refunds         — Return request, return pickup scheduled or missed, refund status after
                               return, cancellation of an already-received item.
4. Charging & Battery        — Device not charging, battery draining fast, charging case fault,
                               charger incompatibility.
5. Connectivity & Pairing    — Bluetooth pairing failure, device not connecting, connection drops,
                               device not visible to phone.
6. Audio Quality & Hardware  — Poor sound, one side silent, crackling, dead-on-arrival, physical defect
   Defect                      (crack, broken strap, box crushed), mic not working, screen not responsive.
7. App, Firmware & Software  — App crash, firmware update failure, companion-app bug,
                               device went dark after software update.
8. Warranty & Repair         — Warranty claim for manufacturing defect, RMA, service-centre query,
                               out-of-warranty repair.
9. Account & Login           — Login failure, password reset, account locked, OTP not received.
10. Product Enquiry & General — Pre-purchase question, compatibility query (before buying),
                                general feedback, any topic not covered above.

DISAMBIGUATION RULES:
- Customer wants to return AND get refund → Returns & Refunds (NOT Billing & Payments).
- Payment charged twice / amount debited → Billing & Payments (NOT Returns & Refunds).
- "Device broken, I want warranty" → Warranty & Repair (primary intent = claim warranty).
- "Device broken, asking about the fault" → Audio Quality & Hardware Defect.
- "Order not delivered" → Delivery & Shipping even if agent later refunded.
- Return pickup was scheduled but courier did not come → Returns & Refunds (not Delivery).
- Screen not responding / watch not responding to touch → Audio Quality & Hardware Defect.
- Mic not working on calls → Audio Quality & Hardware Defect (not App).
- "Need GST invoice / bill" → Billing & Payments.
- Device went dark / stopped working after firmware update → App, Firmware & Software.
- "Before I buy" / "will this work with" → Product Enquiry & General.
- "My son ordered without asking, please reverse" (item not yet received) → Returns & Refunds.

OUTPUT: Return ONLY a valid JSON object on a single line. No markdown, no explanation, no extra fields.
{"category": "<exact category name>", "confidence": <0.0–1.0>}
```

---

## Category Descriptions for Offline Embedding (v1.3)

The `CATEGORY_DESCRIPTIONS` dictionary in `classifier.py` is used for sentence-transformer cosine similarity. Changes in v1.3 vs v1.2:

- Delivery: added "Payment was successful but order not delivered", "Checked with neighbours", "Poor delivery experience"
- Returns: changed to focus on post-delivery returns (pre-delivery cancel moved to Billing vicinity)
- Product Enquiry: added compatibility pre-purchase phrasing ("will this talk to", "before I buy")
- App/Firmware: added firmware-failure language ("went dark after firmware update")
- Warranty: tightened to "manufacturing defect" language to reduce pull on incidental mentions
- Audio Quality: added "product arrived cracked or physically broken"

---

## Discarded Approaches

| Approach | Reason Discarded |
|---|---|
| Few-shot examples in LLM prompt (2 per category) | +8× token cost per ticket; no measurable accuracy gain in equivalent offline test |
| Chain-of-thought reasoning in LLM output | Inflates tokens and cost; JSON-only output is required |
| Multi-label classification | Taxonomy is mutually exclusive; single label is cleaner for reporting |
| Separate prompts per channel | Same prompt handles chat/email/voice/IVR without loss |
| Confidence-threshold routing | LLM self-reported confidence is unreliable; routing based on it would over-rely on it |
| Logistic Regression on TF-IDF | Explicitly excluded per project spec; no ML model training |
| Embedding fine-tuning | Project scale (11,780 tickets, 10 classes) doesn't justify fine-tuning cost |

---

## Prompt v1.5 — Baseline Groq LLM Evaluation

**Date:** 2026-10-01  
**Model:** Groq `openai/gpt-oss-20b`  
**Tuning Result (20 tickets):** 18/20 = 90.0%  
**Validation Result (100 tickets):** 73/100 = 73.0%  
**Preserved Baseline Files:** `outputs/validation_report_v1_5.csv`, `outputs/validation_detail_v1_5.csv`

---

## Prompt v2.0 — Exploratory Hierarchical Routing

**Date:** 2026-10-01  
**Model:** Groq `openai/gpt-oss-20b`  
**Tuning Result (20 tickets):** 18/20 = 90.0%  
**Validation Result (100 tickets):** 75/100 = 75.0%  
**Preserved Baseline Files:** `outputs/validation_report_v2_0.csv`, `outputs/validation_detail_v2_0.csv`

---

## Prompt v2.1 — PRODUCTION FROZEN (3-Level Hierarchical + Tuning Few-Shots)

**Date:** 2026-10-02  
**Model:** Groq `openai/gpt-oss-20b`  
**Prompt Structure:**
1. Level 1: Broad Domain Categorization (Logistics, Hardware, Software/Connectivity, Commercial)
2. Level 2: Specific Category Routing (10 mutually exclusive categories)
3. Level 3: Contradiction & Specificity Override Check
4. 8 concise few-shot examples selected *exclusively* from `data/tuning_set.csv`

**Evaluation Results:**
- **20-Ticket Tuning Set:** 18 / 20 = **90.0%** (Cost: $0.00294 USD, 0 failures)
- **100-Ticket Independent Holdout Set:** 75 / 100 = **75.0%** (100 fresh Groq calls, 0 cache hits, 0 failures)
- **Total Tokens (100 holdout):** 152,133 input tokens, 13,176 output tokens
- **Actual API Cost (100 holdout):** **$0.015363 USD** (~₹1.29 INR)
- **Pytest Suite:** 14 / 14 passed (100%)
- **Delivered Output Files:** `outputs/holdout_report.csv`, `outputs/holdout_detail.csv`

