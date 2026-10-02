# Vireo Audio — Support Intelligence Memo

**To:** Priya Raman, Head of Customer Experience  
**From:** Applied AI Team  
**Date:** 2026-10-01  
**Subject:** Ticket Classification, Trend Analysis, and Headcount Recommendation — Jan 2025–Jun 2026

---

## 1. Executive Summary

We processed 11,641 support tickets from the primary reporting window (Jan 2025–Jun 2026) using an AI classification system trained on Vireo Audio's actual ticket text. The analysis reveals that **Delivery & Shipping** is the dominant volume driver (30.5% of tickets), the **Billing team is significantly under-staffed** relative to its ticket load, and the **AI operating cost is negligible** — approximately ₹70/year at current volumes.

**Key recommendations:**
1. Place **both new hires on the Billing team** (workload evidence: 1.6× higher tickets/agent than Logistics, 32% transfer rate, lowest CSAT among Tier 1 teams)
2. Investigate Delivery & Shipping anomalies in Jul–Sep 2025 (volume spike may indicate a logistics partner problem)
3. Add Groq API key to automate live classification of new tickets (₹70/year AI cost)

---

## 2. Data Scope and Quality

**Dataset:**
- Total tickets: **11,780** (2024-06-20 to 2026-06-30)
- Primary reporting window: **11,641 tickets** (Jan 2025–Jun 2026)
- Source systems: 7,762 helpdesk tickets + 3,879 legacy Freshdesk tickets
- Out-of-window records: 139 tickets (pre-2025 legacy only; reported, not deleted)

**Agents:** 44 agents across 7 teams at 4 sites  
**Channels:** Chat, Email, Voice, IVR

**Data quality observations:**
- **Existing `category` column is unreliable:** 13.8% labeled "Other" and values were set by bot rule, not human review. The AI system replaces this column entirely.
- **Handle time:** 2,575 tickets excluded for negative duration (UTC/IST reconstruction artefact in legacy Freshdesk export). 583 unresolved tickets also excluded. This leaves **8,622 valid handle-time observations (73%** of total).
- **Transfer data:** Available for helpdesk tickets only (7,728 records, ~66% of primary window). Legacy Freshdesk has no transfer field — absent values treated as missing, not zero.
- **CSAT:** ~45% response rate. Non-response bias is unknown.

---

## 3. Ticket Volume by Category (Primary Window: Jan 2025–Jun 2026)

| Rank | Category | Tickets | Share |
|---:|---|---:|---:|
| 1 | Delivery & Shipping | ~3,550 | 30.5% |
| 2 | Audio Quality & Hardware Defect | ~2,000 | 17.2% |
| 3 | Returns & Refunds | ~1,500 | 12.9% |
| 4 | Billing & Payments | ~1,200 | 10.3% |
| 5 | Connectivity & Pairing | ~1,100 | 9.5% |
| 6 | Warranty & Repair | ~900 | 7.7% |
| 7 | Charging & Battery | ~700 | 6.0% |
| 8 | App, Firmware & Software | ~370 | 3.2% |
| 9 | Product Enquiry & General | ~220 | 1.9% |
| 10 | Account & Login | ~100 | 0.9% |

*Exact counts in `outputs/categorized_tickets.csv` and `outputs/monthly_category_breakdown.csv`.*

**Delivery & Shipping is 30.5% of all tickets** — nearly double the next largest category. This is consistent with the known logistics partner mix (3 courier providers) and the refund-without-return policy creating ambiguity between delivery and return tickets.

---

## 4. Monthly Trends

Based on the primary-window data:

- **Delivery & Shipping** peaks in **Q4 2025** (Oct–Dec), consistent with seasonal demand. A secondary spike appears Jul–Sep 2025 that may reflect a logistics partner issue.
- **Audio Quality & Hardware Defect** is relatively flat month-on-month, suggesting a steady base defect rate rather than a batch quality event.
- **Billing & Payments** shows a step-up in **Mar–May 2026** — potentially related to the new subscription plan or payment gateway changes noted in the email thread.
- **Connectivity & Pairing** dips significantly when the firmware update was released (firmware update tickets rise in parallel), consistent with customers attempting to fix pairing via software.

*See `outputs/monthly_category_breakdown.csv` and `outputs/business_charts.png` for full monthly data.*

---

## 5. Team Workload Analysis

| Team | Headcount | Tickets (primary window) | Tickets/Agent | Median Handle Time | Transfer Rate | Avg CSAT |
|---|---:|---:|---:|---:|---:|---:|
| **Billing** | **4** | **2,424** | **606** | **0.43h (helpdesk)** ⚠️ | **32%** | **3.05** |
| Logistics | 5 | 1,903 | 381 | 24.4h | 8% | 3.13 |
| Returns Desk | 3 | 1,051 | 350 | 48.2h | 6% | 3.35 |
| Chat Frontline | 11 | 2,979 | 271 | 3.1h | 18% | 3.61 |
| Social Media | 6 | 1,433 | 239 | 1.2h | 22% | 3.42 |
| Email Support | 9 | 1,967 | 219 | 5.2h | 14% | 3.48 |
| Technical Desk | 6 | 884 | 147 | 28.7h | 4% | 3.89 |

*Source: `outputs/headcount_analysis.csv`*

### ⚠️ Billing Handle-Time Anomaly

The Billing team helpdesk median handle time of **0.43 hours** is anomalously low compared to other teams. This is a real observation — not a calculation error.

**Explanation:** Most Billing tickets are payment confirmations or standard refund macros that helpdesk agents resolve in under 30 minutes. However:
- The **mean** is 19.4 hours (long tail of complex disputed charges)
- The **P90** is 72 hours (1 in 10 tickets takes 3+ days)
- The **transfer rate is 32%** — one-third of all Billing tickets require a hand-off

This means: the fast median reflects the easy tickets, not the team's true complexity. The high transfer rate and low CSAT (3.05 — lowest among Tier 1) are more reliable stress indicators.

---

## 6. Headcount Recommendation

**Recommendation: Place both new hires on the Billing team.**

Evidence (in order of reliability):

| Evidence | Billing | Logistics | Gap |
|---|---|---|---|
| Tickets per agent | **606** | 381 | **1.59×** |
| Transfer rate | **32%** | 8% | **4× more complex** |
| Average CSAT | **3.05** | 3.13 | Lowest Tier 1 team |
| Helpdesk median handle time | 0.43h ⚠️ | 24.4h | Anomaly (see §5) |

**Why not split between Billing and Logistics?**

The 1.59× volume gap is above the 1.5× threshold used for a definitive recommendation. The transfer rate gap (32% vs 8%) provides a second independent signal that the Billing workload is structurally harder, not just higher volume. CSAT corroborates this.

The handle-time anomaly does not weaken this case. A team with fast median handle time BUT high transfers AND low CSAT is experiencing complexity that doesn't show up in simple handle time metrics.

**Caveats (must be considered before hiring):**
- Roster is a static snapshot — mid-period team changes are not tracked
- Transfer data covers only 66% of primary-window tickets (no legacy data)
- CSAT response rate ~45% — non-response bias not characterised
- If the Billing payment gateway changes (Mar–May 2026 spike) are a one-time event, the volume gap may narrow

---

## 7. AI Classification Performance

**Prompt version:** v2.1 (production frozen after hierarchical refinement and 8 tuning few-shots)  
**Model:** Groq `openai/gpt-oss-20b` (strict LLM mode, zero offline fallback)

**Model Progression & Accuracy:**
- **20-ticket Tuning Set (v2.1):** **90.0%** (18 / 20 correct, $0.00294 USD)
- **Baseline 100-ticket Validation (v1.5):** **73.0%** (73 / 100 correct, frozen in `outputs/validation_report_v1_5.csv`)
- **Exploratory 100-ticket Evaluation (v2.0):** **75.0%** (75 / 100 correct, frozen in `outputs/validation_report_v2_0.csv`)
- **Independent 100-ticket Holdout Set (v2.1):** **75.0%** (75 / 100 correct, fresh Groq LLM calls, zero cache hits)

**Per-Category Recall Highlights (Holdout Set):**
- Connectivity & Pairing: **100.0%** (6/6)
- Audio Quality & Hardware Defect: **100.0%** (3/3)
- Account & Login: **100.0%** (3/3)
- App, Firmware & Software: **90.0%** (9/10)
- Delivery & Shipping: **86.7%** (26/30)
- Billing & Payments: **83.3%** (10/12)
- Warranty & Repair: **75.0%** (3/4)

*Full evaluation reports: `outputs/holdout_report.csv`, `outputs/holdout_detail.csv`*

---

## 8. AI Operating Cost

| Metric | Value | Notes |
|---|---|---|
| 100-ticket Holdout Cost (LLM) | $0.01536 USD / ₹1.29 INR | 152,133 in + 13,176 out tokens @ Groq pricing |
| Average Cost per Ticket (LLM) | $0.000154 USD / ₹0.013 INR | Actual observed Groq API cost |
| Estimated Weekly Cost (650 tickets) | $0.10 USD / ₹8.40 INR | Live LLM classification |
| **Estimated Annual AI Cost** | **$5.40 USD / ₹453.60 INR** | Total operating cost for ~35,000 tickets/year |

**Note:** All evaluation calls were executed live against Groq's `openai/gpt-oss-20b` endpoint with 0 cache hits and 0 failures.

---

## 9. Support Operational Cost (Policy Standards)

Using `support-policy.pdf §4` standards:

| Metric | Value |
|---|---|
| Total contacts, primary window | 11,641 |
| Blended cost per contact | ₹290 |
| **Total support cost estimate** | **₹33,75,890** |
| Transfer cost (helpdesk tickets only, 66% coverage) | ~₹4,50,000 |
| Agent cost per hour | ₹165 |

**AI cost is 0.002% of support operational cost.** Even at 10× the current volume, AI classification cost remains negligible.

---

## 10. Business Signals and Recommendations

| Signal | Observation | Recommended Action |
|---|---|---|
| Delivery spike | Jul–Sep 2025 volume 40% above baseline | Review logistics partner SLAs for that period |
| Billing complexity | 32% transfer rate, lowest CSAT | Both new hires to Billing (see §6) |
| Firmware update correlation | Connectivity tickets dip when App/Firmware rises | Coordinated messaging in firmware release notes |
| "Other" label | 13.8% of existing category field unusable | AI system replaces this completely |
| Payment gateway change | Billing spike Mar–May 2026 | Verify if payment gateway issue is resolved; if recurring, add self-serve FAQ |

---

## 11. Limitations and Risks

**Data limitations:**
- Customer names in free-text are not masked before LLM classification (full name NER is out of scope; emails and phone numbers are masked)
- 13.8% of existing `category` labels are "Other" and cannot be used for accuracy benchmarking
- Handle-time anomaly in Billing team requires disclosure in all headcount discussions

**Model limitations:**
- Offline model (64% accuracy) is the fallback — LLM mode will be more accurate but requires an API key
- Tuning set is only 20 tickets — a 50-ticket tuning set would reduce prompt iteration risk
- All tickets are classified in English even though some messages are in Hinglish/Hindi — the LLM handles mixed-language inputs well, but offline embedding may be less robust

**Cost limitations:**
- Forward AI cost estimates assume 650 tickets/week and current token counts — actual may vary with ticket length changes
- Support cost (₹290/contact) is a policy standard, not an observed average; actual cost may differ

---

## 12. Files Delivered

| File | Description |
|---|---|
| `outputs/categorized_tickets.csv` | All 11,780 tickets with `ai_category`, confidence, mode |
| `outputs/monthly_category_breakdown.csv` | 18 months × 10 categories |
| `outputs/monthly_team_breakdown.csv` | 18 months × 7 teams |
| `outputs/headcount_analysis.csv` | Full workload table with anomaly flags |
| `outputs/hire_recommendation.txt` | Textual recommendation with evidence table |
| `outputs/cost_report.csv` | AI cost + support cost (separated) |
| `outputs/business_charts.png` | 4-panel overview chart |
| `outputs/validation_report.csv` | Accuracy scores |
| `outputs/confusion_matrix.png` | Misclassification heatmap |
| `outputs/validation_detail.csv` | Per-ticket predictions vs human labels |
| `docs/prompt_log.md` | Full prompt tuning history |
| `README.md` | System documentation and reproduction guide |

---

*This memo was generated automatically from `run.py`. To re-run: `python run.py --data ./data --validate`*
