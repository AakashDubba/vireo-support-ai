# Vireo Audio — Support Ticket Intelligence System

Automatic categorisation of 11,780 support tickets, monthly business breakdowns, evidence-based headcount recommendation, and AI cost tracking for Vireo Audio's 44-agent, 4-channel support operation.

---

## Business Problem

Priya Raman (Head of Customer Experience) needs:
1. **Reliable ticket categories** — the existing bot-set tags have a 13.8% "Other" residual and cannot be trusted.
2. **Monthly volume by category and team** — for trend and planning.
3. **Evidence-based recommendation for the next 2 headcount hires** using workload, handle time, transfer complexity, and CSAT.
4. **AI operating cost** — separated from business support cost.
5. **Reproducible, handoff-ready system** — runs offline, cached, no dependency on a live API for evaluation.

---

## Architecture

```
run.py            CLI entry point — 7 sequential steps
  ├── utils.py    Data loading, PII masking, join validation, handle-time derivation
  ├── classifier.py   3-tier: Groq LLM → sentence transformer → TF-IDF keyword
  └── analysis.py     Monthly breakdowns, headcount, cost, validation, charts

data/             Input CSVs + human labels
cache/            JSON disk cache (key = ticket_id :: prompt_version :: model_name)
outputs/          All required deliverables
docs/             memo.md, prompt_log.md
```

**Classification flow per ticket:**
1. Check `cache/llm_cache.json` — key: `ticket_id::v1.3::openai/gpt-oss-20b`
2. Cache hit → return immediately (no API cost)
3. Cache miss + valid `LLM_API_KEY` → call Groq (OpenAI-compatible, concurrent, rate-limited)
4. Groq unavailable → batch sentence-transformer cosine similarity (`all-MiniLM-L6-v2`)
5. Sentence-transformers fail → TF-IDF keyword count fallback
6. All failures logged; pipeline never aborts for a single bad ticket

---

## Installation

```bash
git clone <repo> && cd vireo-support-ai

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

> **First-run note:** On first offline run the sentence-transformer model `all-MiniLM-L6-v2` (~90 MB) downloads automatically from Hugging Face. This takes 1–5 minutes. Subsequent runs use the local model cache and complete in **under 30 seconds**.

---

## Groq Setup

1. Sign up at [https://console.groq.com](https://console.groq.com)
2. Create an API key at [https://console.groq.com/keys](https://console.groq.com/keys)
3. Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and set:  LLM_API_KEY=gsk_...
```

4. Run: `python run.py --data ./data`

The system uses `openai/gpt-oss-20b` via Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`). No changes needed in the code.

### Security

- `.env` is in `.gitignore` — never committed.
- The API key is never logged, printed, or included in reports.
- No customer names or agent names are sent to the LLM.

---

## .env Configuration

```env
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=          # <- add your Groq key here

LLM_PRICE_PER_1M_INPUT=0.075
LLM_PRICE_PER_1M_OUTPUT=0.30

USD_TO_INR=84.0       # adjust as needed
```

---

## CLI Usage

```bash
# Full run (offline if no API key; uses cache if available)
python run.py --data ./data

# Force offline (sentence transformers / TF-IDF)
python run.py --data ./data --offline

# With LLM classification (requires LLM_API_KEY in .env)
python run.py --data ./data

# Run final validation against human labels
python run.py --data ./data --validate

# Skip classification — serve from cache only (fastest re-run)
python run.py --data ./data --skip-classify --validate

# Custom projected weekly volume for cost estimate
python run.py --data ./data --weekly-rate 800
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Directory with CSV files |
| `--output` | `./outputs` | Output directory |
| `--cache` | `./cache` | Cache directory |
| `--offline` | false | Force offline mode |
| `--skip-classify` | false | Cache-only (fastest re-run) |
| `--validate` | false | Run against human labels |
| `--human-labels` | `./data/human_labels.csv` | Validation labels path |
| `--weekly-rate` | 650 | Tickets/week for AI cost estimate |

---

## Offline Mode

The CLI works **without an API key**:

- **With cached results (all 11,780 cached):** completes in ~25 seconds.
- **Without cache (first run):** downloads `all-MiniLM-L6-v2` (~90 MB), embeds all tickets (~90 seconds on CPU). All results are cached after the first run.
- **If sentence-transformers fail:** falls back to keyword-count TF-IDF. Predictions are labelled `offline-tfidf` in `classification_mode`.

Offline predictions are labelled in `categorized_tickets.csv` under `classification_mode`. They are the fallback — LLM mode is more accurate.

---

## Cache Behaviour

- Cache file: `cache/llm_cache.json`
- Cache key: `ticket_id::prompt_version::model_name`
- Changing `PROMPT_VERSION` (in `utils.py`) or the model name invalidates cache entries for that combination without deleting others.
- Cache is committed to the repo so evaluators can run the pipeline without API key or model download.
- On second run: 11,780 cache hits, 0 misses, completes in ~25 seconds.

---

## Frozen Taxonomy (v1.3, 2026-10-01)

10 categories, frozen after 20-ticket prompt-tuning phase:

| # | Category |
|---|---|
| 1 | Delivery & Shipping |
| 2 | Billing & Payments |
| 3 | Returns & Refunds |
| 4 | Charging & Battery |
| 5 | Connectivity & Pairing |
| 6 | Audio Quality & Hardware Defect |
| 7 | App, Firmware & Software |
| 8 | Warranty & Repair |
| 9 | Account & Login |
| 10 | Product Enquiry & General |

See `docs/prompt_log.md` for the full tuning history.

---

## Validation Methodology

**Validation sets:**
- **Baseline Validation set:** 100 tickets (`data/human_labels.csv`) — stratified by assigned_team.
- **New Untouched Holdout set:** 100 tickets (`data/new_holdout_labels.csv`) — independently policy-labeled, zero overlap with tuning or baseline validation.

**Verified:** zero overlap between the 20 tuning tickets, 100 baseline validation tickets, and 100 new holdout tickets.

**Evaluation Results Summary:**
- **20-ticket Tuning Accuracy (v2.1):** **90.0%** (18/20)
- **Baseline 100-ticket Validation (v1.5):** **73.0%** (73/100, `outputs/validation_report_v1_5.csv`)
- **Exploratory 100-ticket Evaluation (v2.0):** **75.0%** (75/100, `outputs/validation_report_v2_0.csv`)
- **New 100-ticket Holdout Evaluation (v2.1):** **75.0%** (75/100, `outputs/holdout_report.csv`)
- **Pytest Suite:** 14/14 passed (100%)

---

## Cost Methodology

`outputs/cost_report.csv` contains **two strictly separate sections:**

**Section A — AI operating cost:**
- If LLM was used: actual token counts from Groq API response headers
- If offline only: forward estimate using 250 input + 20 output tokens @ Groq pricing ($0.075/$0.30 per 1M)
- Estimate rows are labelled `ESTIMATE` in the `unit` column
- Projected annual cost at 650 tickets/week is clearly marked as a forward estimate

**Section B — Support/business cost:**
- Uses `support-policy.pdf §4` standards: ₹290/contact, ₹165/agent-hour, ₹305/transfer
- Transfer cost computed only over helpdesk tickets with non-null `transfers` field
- Never mixed with Section A AI costs

---

## PII Masking

Before any text is sent to the Groq LLM:
- Email addresses masked: `[\w.+-]+@[\w.-]+\.\w{2,}` → `[EMAIL]`
- Indian mobile numbers masked: `(\+91[\s-]?|0)?[6-9]\d{9}` → `[PHONE]`

**Limitations (documented honestly):**
- Customer names in free text are NOT masked (reliable name detection requires NER which is not in scope)
- Agent names are never included in LLM prompts (only `customer_message` and `agent_notes`)
- Non-standard email/phone formats may not be caught
- Order IDs and addresses are not masked (not sensitive PII per project policy)

---

## Date Scope

**Full data range:** 2024-06-20 to 2026-06-30 (11,780 tickets total)  
**Primary business reporting window:** 2025-01-01 to 2026-06-30 (11,641 tickets)  
**Out-of-window records:** 139 tickets (pre-2025 legacy Freshdesk records)

The 139 out-of-window records are:
- Classified and cached (not silently deleted)
- Excluded from headcount analysis, monthly breakdowns, and business cost calculations
- Reported in the pipeline log
- Their existence is noted but not explained (the data does not document why they exist)

---

## Handle-Time Rules

**Definition (policy §10):** `handle_time = resolved_at − first_response_at`

**Excluded from calculations:**
- 583 unresolved tickets (no `resolved_at`)
- 2,575 tickets with negative handle time (UTC/IST reconstruction artefact in legacy Freshdesk)

**Valid observations:** 8,622 tickets

---

## Billing Team Handle-Time Anomaly

⚠️ **Known data anomaly — disclosed here and in `headcount_analysis.csv`:**

Billing team helpdesk tickets have a median handle time of **~0.43 hours** — compared to ~24 hours for Logistics and ~43 hours for Returns Desk.

This is not a calculation error. It reflects that most Billing helpdesk tickets are resolved quickly (payment-gateway confirmations, standard refund macros). The distribution is bimodal: fast median but P90 of 72 hours, and a mean of 19.4 hours, indicating a long tail of complex escalated cases.

This anomaly does NOT invalidate the volume evidence (606 tickets/agent for Billing vs 381 for Logistics). The **transfer rate** (32% for Billing vs 8% for Logistics) provides a stronger complexity indicator than raw handle time.

---

## Transfer Data Limitation

The `transfers` field is populated **only for helpdesk tickets (post-Sept 2025)** — 7,728 tickets or 66% of the primary window.

- Missing transfer values are **NOT treated as 0**
- Transfer statistics are computed only over tickets with non-null `transfers`
- Transfer cost (₹305/transfer) is applied only to this subset
- Legacy Freshdesk tickets (34%) have no transfer data — this is reported in `cost_report.csv` and noted in outputs

---

## Required Outputs

| File | Description |
|---|---|
| `outputs/categorized_tickets.csv` | Full ticket dataset with `ai_category`, `ai_confidence`, `classification_mode`, `cache_hit` |
| `outputs/monthly_category_breakdown.csv` | Tickets/month × category (primary window) |
| `outputs/monthly_team_breakdown.csv` | Tickets/month × team (primary window) |
| `outputs/headcount_analysis.csv` | Per-team workload: headcount, tickets/agent, handle time, transfers, CSAT |
| `outputs/hire_recommendation.txt` | Evidence-based text recommendation for the 2 hires |
| `outputs/cost_report.csv` | AI cost (actual + estimated) + policy-standard support costs |
| `outputs/business_charts.png` | 4-panel business overview chart |
| `outputs/validation_report.csv` | Accuracy, per-mode accuracy, confusion summary |
| `outputs/confusion_matrix.png` | Confusion matrix heatmap (100-ticket validation set) |
| `outputs/validation_detail.csv` | Per-ticket human label vs model prediction |

---

## Running Tests

```bash
pytest tests/ -v
```

11 tests covering:
- Join validation (orphans, fan-out)
- LLM JSON parsing (valid, invalid, wrong category, markdown fences, all categories)
- Cache behaviour (hits, key isolation, persistence across instances)

---

## Deliberately Omitted

| Item | Decision |
|---|---|
| Streamlit UI | Not built — CLI delivers all required outputs; Streamlit is optional |
| Docker | Not required — standard pip install with virtual env is sufficient |
| FastAPI / REST API | Not required for this use case |
| Database | CSV/JSON outputs are sufficient |
| Logistic Regression / ML training | Explicitly excluded per project spec |
| LangChain / LangGraph | Not used — direct OpenAI-compatible client is simpler and more auditable |
| Multi-provider fallback | Only Groq is used; offline mode covers the no-API case |
| Agent names in LLM prompts | Never included — privacy boundary |
| CSAT imputation | Blank CSAT treated as no-response per policy §8, never imputed |

---

## Streamlit (Optional — Not Built)

If added in a future iteration:

```bash
pip install -r requirements-ui.txt
streamlit run app.py
```

The Streamlit app would be a read-only presentation layer over the existing pipeline outputs — it would NOT re-run classification.
