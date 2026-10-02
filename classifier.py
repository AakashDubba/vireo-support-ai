"""
classifier.py — Groq-backed ticket classifier with offline fallback.

Classification flow for every ticket:
  1. Check JSON disk cache  (key = ticket_id :: prompt_version :: model_name)
  2. Cache hit → return immediately, no API call.
  3. Cache miss + valid API key → call Groq (OpenAI-compatible endpoint).
     Concurrent requests via ThreadPoolExecutor with rate-limit back-off.
  4. Groq unavailable / no key → batch sentence-transformer cosine similarity.
  5. Sentence-transformers unavailable → lightweight TF-IDF keyword fallback.
  6. Every failure is recorded; a single bad ticket NEVER aborts the run.

Offline predictions are labelled in classification_mode column.
"""

import json
import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd

from utils import CATEGORIES, PROMPT_VERSION, build_classification_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — v2.1 (Master hierarchical classification & policy precedence)
# See docs/prompt_log.md for full version history.
# Taxonomy frozen after tuning evaluation.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the master support-ticket classifier for Vireo Audio, a consumer audio and wearables brand.

OBJECTIVE:
Classify each customer ticket into exactly one approved category based on the customer's PRIMARY UNDERLYING SUPPORT ISSUE, adhering strictly to Vireo Customer Support Operating Policy.
Do NOT classify based on the customer's demanded resolution.

THE 10 APPROVED CATEGORIES:
1. Delivery & Shipping — Shipment transit, courier delay, tracking updates, wrong delivery address, missing package, courier not arriving.
2. Billing & Payments — Payment gateway failure, duplicate debit/charge, invoice or GST query, coupon/discount not applied at checkout, price adjustment.
3. Returns & Refunds — Return requests for received items, reverse pickup scheduling/missed pickup, refund status tracking for returned goods, return/refund process itself.
4. Charging & Battery — Device not charging, rapid battery drain, charging case defect, charger/cable incompatibility, battery health.
5. Connectivity & Pairing — Bluetooth pairing failure, device not connecting or dropping connection, device invisible in Bluetooth scan.
6. Audio Quality & Hardware Defect — Sound distortion, static, crackling, mic failure, one earbud silent, dead on arrival (DOA), physical transit damage (crushed box, cracked screen, broken strap), screen not responding.
7. App, Firmware & Software — Companion app crash/bug, firmware update failed or stuck, device bricked / screen went dark after update, software feature issues.
8. Warranty & Repair — Warranty claims, RMA status/claim number inquiries, service center delays, unit sent in for repair.
9. Account & Login — Account locked, OTP not received, password reset, login authentication failure.
10. Product Enquiry & General — Pre-purchase questions, compatibility checks before buying ("will this work with..."), product specifications, features, general non-technical feedback.

HIERARCHICAL CLASSIFICATION STRATEGY:
LEVEL 1 — DETERMINE BROAD INTENT:
A. Delivery & Logistics: shipment transit, courier delays, tracking issues, missing packages -> Delivery & Shipping
B. Transaction & Commercial: payment charges, gateway disputes, invoices -> Billing & Payments; return/refund/cancellation workflow -> Returns & Refunds
C. Technical & Device Support: battery/power -> Charging & Battery; Bluetooth -> Connectivity & Pairing; physical/sound defect -> Audio Quality & Hardware Defect; app/firmware -> App, Firmware & Software; warranty/RMA claim -> Warranty & Repair
D. Account & Access: login, OTP, password, account locked -> Account & Login
E. General Information: genuine pre-purchase specs, compatibility before purchase -> Product Enquiry & General

LEVEL 2 — ROOT CAUSE OVER REQUESTED REMEDY (STRIP THE REMEDY TEST):
Customers routinely demand "refund", "replacement", "cancel", "reverse", or "money back". These are REQUESTED RESOLUTIONS, not the underlying defect.
Always ask: "If I strip away the demand for a refund/replacement, what concrete problem caused the customer to reach out?"
- Physical damage/defect upon arrival + refund demand -> Audio Quality & Hardware Defect
- Delivery/courier delay + refund demand -> Delivery & Shipping
- Charging/battery failure + replacement demand -> Charging & Battery
- Firmware update failure + replacement demand -> App, Firmware & Software
- Quoted RMA / warranty repair delay + refund demand -> Warranty & Repair
- Duplicate payment / cart coupon dispute -> Billing & Payments
- Only classify as Returns & Refunds when the return/refund/cancellation process itself is the primary issue.

LEVEL 3 — CONTRADICTION & SPECIFICITY CHECK:
"Product Enquiry & General" is STRICTLY RESERVED for genuine general/pre-purchase enquiries (e.g. "before I buy, will this work with my phone?"). It is NEVER a catch-all. Any concrete operational, technical, delivery, billing, or warranty problem MUST ALWAYS override Product Enquiry & General.
An order number (e.g. VR890562) is merely transaction context and does not imply Delivery & Shipping unless delivery/transit issues are explicitly described.

REPRESENTATIVE FEW-SHOT EXAMPLES (from tuning set):
Example 1 — Hardware defect with replacement/refund demand:
Customer: "box was crushed and the watch is cracked. I already refused nothing as I didn't notice till later. I want a replacement or refund, nothing else."
Output: {"category": "Audio Quality & Hardware Defect", "confidence": 0.95}

Example 2 — Audio defect with acoustic distortion:
Customer: "SOUNDS LIKE A BADLY TUNED RADIO NOW - KINDLY LOOK INTO IT"
Output: {"category": "Audio Quality & Hardware Defect", "confidence": 0.95}

Example 3 — Delivery delay despite payment/refund mention:
Customer: "Order VR881259. payment successful but order not delivered. I already checked with neighbours. I want a replacement or refund, nothing else."
Output: {"category": "Delivery & Shipping", "confidence": 0.95}

Example 4 — Checkout discount dispute:
Customer: "your ad said 20% off and the cart says full price"
Output: {"category": "Billing & Payments", "confidence": 0.95}

Example 5 — Bricked device from update:
Customer: "Update failed and now it won't turn on. I kept it in the case. Not what I expected from Vireo. Please resolve asap."
Output: {"category": "App, Firmware & Software", "confidence": 0.95}

Example 6 — Active Warranty / RMA claim status:
Customer: "Claim number RMA82615, anyone alive there?. I emailed twice. I want my money back. Please revert"
Output: {"category": "Warranty & Repair", "confidence": 0.95}

Example 7 — Account lockout with irrelevant remedy demand:
Customer: "Locked out of my own account since 3 weeks back. I have tried 5 times. I request you to kindly arrange a replacement."
Output: {"category": "Account & Login", "confidence": 0.95}

Example 8 — Pre-purchase compatibility question:
Customer: "sir ji before i buy - will this talk to a 2019 samsung"
Output: {"category": "Product Enquiry & General", "confidence": 0.95}

OUTPUT FORMAT:
Return ONLY a valid single-line JSON object. No explanation, no markdown fence, no extra keys:
{"category": "<exact category name>", "confidence": <0.0–1.0>}"""

# ─────────────────────────────────────────────────────────────────────────────
# Category descriptions for offline embedding
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "Delivery & Shipping": (
        "Package not delivered. Order has not arrived. Not received yet. Still waiting for order. "
        "Payment was successful but order not delivered. Shipment delayed. Courier lost the item. "
        "Tracking page shows no update. Wrong delivery address. Wrong pincode. Item still in transit. "
        "Awaiting delivery. Reshipment requested. AWB number not updating. Order not received. "
        "Out for delivery for many days. Change delivery address. Courier did not come. "
        "Checked with neighbours but not found. Poor delivery experience."
    ),
    "Billing & Payments": (
        "Payment failed but money deducted. Amount debited from account. Double charge. "
        "Duplicate payment. Invoice not received. GST invoice needed. GSTIN on bill. "
        "Coupon code not applied. Price is wrong. EMI not processed. Payment gateway error. "
        "Paid but order not confirmed. Transaction failed. Cart shows wrong price."
    ),
    "Returns & Refunds": (
        "Want to return the product. Return request. Return pickup not scheduled. "
        "Return pickup was missed. No one showed up for pickup. Refund not received after return. "
        "Cancellation after delivery received. Money back after receiving item. Return label not provided. "
        "Return approved but refund pending. Sent the item back. "
        "Reverse logistics request. Return the item I already received."
    ),
    "Charging & Battery": (
        "Device is not charging. Battery draining too fast. Charging case not working. "
        "Charger not compatible. Goes from full to empty quickly. Plugged in but not charging. "
        "Battery percentage not increasing. Charging port problem. Not holding charge. "
        "Have to charge it twice a day. Battery drains during one commute."
    ),
    "Connectivity & Pairing": (
        "Cannot pair with Bluetooth. Device not connecting to phone. Earbuds not found by phone. "
        "Keeps disconnecting. Connection drops frequently. Cannot find in Bluetooth list. "
        "Pairing failed. Wireless drops during music. Phone does not see it. "
        "Device not visible. Will not pair. Connection drops."
    ),
    "Audio Quality & Hardware Defect": (
        "Sound quality is bad. One earbud not working. One side is silent. Crackling noise. "
        "Distorted audio. Physical damage on arrival. Arrived damaged. Box was crushed. Dead on arrival. "
        "No sound from one side. Rattling in earbuds. Microphone not working on calls. "
        "People cannot hear me. Sounds like a badly tuned radio. Screen cracked. "
        "Screen not responding to touch. Watch not responding. Strap broke. Band snapped. "
        "Mic not working. Everyone says I sound underwater. Hardware defect. "
        "Product arrived cracked or physically broken."
    ),
    "App, Firmware & Software": (
        "App is not working. Companion app crash. Firmware update failed. Software bug. "
        "App not loading. Feature missing from app. Cannot update firmware. Version issue. "
        "Settings not saving in app. App stuck on update screen. App permission issue. "
        "Equalizer not working. Vireo app crash. Device went dark after firmware update. "
        "Device stopped responding after software update prompt."
    ),
    "Warranty & Repair": (
        "Claiming warranty for manufacturing defect. Product under warranty. Need warranty replacement. "
        "RMA request. Repair quote. Out-of-warranty repair. Service center. Warranty period question. "
        "Warranty claim rejected. Need warranty card. Product needs to be repaired. "
        "Sent the unit in for repair weeks ago. Have heard nothing about the repair."
    ),
    "Account & Login": (
        "Cannot login to account. Forgot password. Password reset not working. Account locked. "
        "OTP not received. Email not recognized. Cannot access account. Sign-in issue. "
        "Profile update failing. Account not found. Login error. Locked out of account."
    ),
    "Product Enquiry & General": (
        "Before I buy. Pre-purchase question. Which product should I buy. Product recommendation. "
        "Is it compatible with my device. What are the specifications. "
        "Difference between two models. Will this work with my phone. Will this talk to. "
        "Will this be compatible with. Is this compatible with Samsung or Nokia. "
        "General feedback. Suggestion for improvement. Product information request."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF keyword fallback (no model required)
# ─────────────────────────────────────────────────────────────────────────────

TFIDF_KEYWORDS: Dict[str, List[str]] = {
    "Delivery & Shipping": [
        "not delivered", "not arrived", "not received", "still waiting", "never came",
        "awaiting delivery", "delivery pending", "where is my order", "tracking",
        "courier", "shipment", "awb", "lost in transit", "delayed delivery",
        "delivery address", "wrong address", "wrong pincode", "change delivery",
        "out for delivery", "delivery failed", "order not",
    ],
    "Billing & Payments": [
        "payment failed", "amount deducted", "amount debited", "double charge",
        "duplicate payment", "charged twice", "money deducted", "payment declined",
        "invoice", "gstin", "gst invoice", "emi", "payment issue", "coupon",
        "price is", "cart says", "ad said", "paid but", "transaction failed",
        "page failed after i paid",
    ],
    "Returns & Refunds": [
        "want to return", "return the product", "return request", "send it back",
        "return pickup", "no one showed up", "pickup not", "refund status",
        "refund not received", "refund pending", "cancel the order", "cancel my order",
        "money back", "return approved", "pickup scheduled but",
    ],
    "Charging & Battery": [
        "not charging", "won't charge", "wont charge", "battery", "charging issue",
        "charging case", "charger", "battery drain", "draining fast",
        "not holding charge", "plugged in", "charging port", "charge twice a day",
        "full to empty", "drains quickly",
    ],
    "Connectivity & Pairing": [
        "bluetooth", "pairing", "not connecting", "cannot connect", "disconnects",
        "connection drops", "keeps dropping", "not found", "not visible",
        "won't pair", "wont pair", "can't pair", "not syncing",
        "phone doesn't see", "phone just doesn't see",
    ],
    "Audio Quality & Hardware Defect": [
        "one side", "one earbud", "crackling", "distorted", "no sound",
        "sound quality", "audio quality", "left earbud", "right earbud",
        "silent", "dead on arrival", "physical damage", "hardware defect",
        "microphone", "mic not", "rattling", "sounds like", "underwater",
        "badly tuned", "screen crack", "not responding to touch", "touch not",
        "screen lights up but", "does nothing", "everyone on my calls",
        "strap broke", "band broke", "arrived damaged", "box was crushed",
    ],
    "App, Firmware & Software": [
        "app ", "firmware", "software update", "companion app", "app crash",
        "app not working", "app not loading", "update failed", "update stuck",
        "feature not working", "vireo app", "app permission", "equalizer",
    ],
    "Warranty & Repair": [
        "warranty", " rma", "service center", " repair", "warranty claim",
        "warranty period", "warranty card", "warranty replacement",
        "sent the unit in", "heard nothing about repair",
    ],
    "Account & Login": [
        "login", "log in", "sign in", "password", "forgot password",
        "account locked", "cannot access account", "otp", "account not found",
        "reset password", "cant login", "locked out",
    ],
    "Product Enquiry & General": [
        "which product", "recommend", "compatible", "difference between",
        "specifications", "pre-purchase", "before i buy", "will this work",
        "general feedback", "suggestion",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter (shared across threads)
# ─────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple token-bucket rate limiter — thread-safe."""

    def __init__(self, rps: float = 0.1):
        self._lock     = threading.Lock()
        self._interval = (1.0 / rps) if rps > 0 else 0.0
        self._last     = 0.0

    def wait(self) -> None:
        with self._lock:
            now  = time.monotonic()
            wait = self._interval - (now - self._last)
            self._last = max(now, self._last + self._interval)
        if wait > 0:
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# TicketClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TicketClassifier:
    """
    Three-tier classifier: Groq LLM → Sentence Transformers → TF-IDF fallback.

    Cache key = ticket_id :: prompt_version :: model_name.
    Changing prompt version or model invalidates existing cache entries for that
    combination without deleting results from other versions.

    Groq uses the OpenAI-compatible API (base_url configurable via LLM_BASE_URL).
    """

    MAX_CONCURRENT = 1   # Keep 1 to prevent TPM bursts against Groq's 8k TPM limit
    BATCH_LOG_EVERY = 1

    def __init__(
        self,
        cache_dir: Path,
        offline: bool = False,
        bypass_cache: bool = False,
        strict_llm: bool = False,
    ) -> None:
        self.cache_dir    = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path   = self.cache_dir / "llm_cache.json"

        self.provider     = os.getenv("LLM_PROVIDER",   "groq")
        self.model_name   = os.getenv("LLM_MODEL",      "openai/gpt-oss-20b")
        self.base_url     = os.getenv("LLM_BASE_URL",   "https://api.groq.com/openai/v1")
        self.api_key      = os.getenv("LLM_API_KEY",    "").strip()
        self.bypass_cache = bypass_cache
        self.strict_llm   = strict_llm

        if self.strict_llm and not offline and not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not set in environment or .env, but strict Groq LLM mode was requested."
            )

        self.offline      = offline or not bool(self.api_key)

        self._cache: Dict[str, dict] = self._load_cache()
        self._llm_client             = None
        self._embed_model            = None
        self._cat_embeddings         = None
        self._rate_limiter           = _RateLimiter(rps=0.12)
        self._cache_lock             = threading.RLock()

        self.stats = {
            "api_calls":          0,
            "cache_hits":         0,
            "cache_misses":       0,
            "llm_successes":      0,
            "failures":           0,
            "input_tokens":       0,
            "output_tokens":      0,
            "estimated_cost_usd": 0.0,
        }

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Cache unreadable ({exc}); starting fresh.")
        return {}

    def _save_cache(self) -> None:
        try:
            with self._cache_lock:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to write cache: {exc}")

    def _cache_key(
        self,
        ticket_id: str,
        provider: Optional[str] = None,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Robust cache key incorporating ticket_id, provider, mode, model, and prompt_version.
        Prevents collisions between offline embeddings and Groq LLM classifications.
        """
        prov = provider or ("offline" if self.offline else self.provider)
        md   = mode or ("offline" if self.offline else "llm")
        mod  = model or ("sentence-transformers" if self.offline else self.model_name)
        pv   = prompt_version or PROMPT_VERSION
        return f"{ticket_id}::{prov}::{md}::{mod}::{pv}"

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        try:
            from openai import OpenAI
            import httpx
            # Works for Groq (and any OpenAI-compatible endpoint) via base_url
            http_client = httpx.Client(
                timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, write=10.0),
            )
            self._llm_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                max_retries=1,
            )
            logger.info(f"LLM client initialised: provider={self.provider} "
                        f"model={self.model_name} base_url={self.base_url}")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            self._llm_client = None
        return self._llm_client

    def _call_llm_raw(self, text: str) -> Tuple[str, int, int, float]:
        """
        Make one Groq API call.
        Returns (raw_text, input_tokens, output_tokens, cost_usd).
        Raises on failure (caller handles retries).
        """
        client = self._get_llm_client()
        if client is None:
            raise RuntimeError("LLM client unavailable")

        price_in  = float(os.getenv("LLM_PRICE_PER_1M_INPUT",  "0.075"))
        price_out = float(os.getenv("LLM_PRICE_PER_1M_OUTPUT", "0.30"))

        self._rate_limiter.wait()

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1024,
        )

        raw   = resp.choices[0].message.content.strip()
        in_t  = resp.usage.prompt_tokens     if resp.usage else 0
        out_t = resp.usage.completion_tokens if resp.usage else 0
        cost  = (in_t / 1_000_000 * price_in) + (out_t / 1_000_000 * price_out)
        return raw, in_t, out_t, cost

    def _parse_llm_json(self, raw: str) -> Optional[dict]:
        """Parse and validate LLM JSON. Returns None on any schema violation."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        if not isinstance(obj, dict) or "category" not in obj:
            return None

        cat = obj["category"]
        if cat not in CATEGORIES:
            # Case-insensitive exact match
            lower_map = {c.lower(): c for c in CATEGORIES}
            cat = lower_map.get(str(cat).lower())
            if cat is None:
                logger.warning(f"LLM returned unknown category: {obj['category']!r}")
                return None
            obj["category"] = cat

        obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.8))))
        return obj

    def _call_single_llm(self, ticket_id: str, text: str) -> dict:
        """
        Call LLM with retry/back-off. Falls back to offline on all failures (unless strict_llm=True).
        Thread-safe: called from ThreadPoolExecutor workers.
        """
        max_attempts = 3  # initial attempt + max 2 retries
        last_error = None
        for attempt in range(max_attempts):
            try:
                raw, in_t, out_t, cost = self._call_llm_raw(text)

                # Thread-safe stats update
                with self._cache_lock:
                    self.stats["api_calls"]          += 1
                    self.stats["input_tokens"]       += in_t
                    self.stats["output_tokens"]      += out_t
                    self.stats["estimated_cost_usd"] += cost

                parsed = self._parse_llm_json(raw)
                if parsed:
                    with self._cache_lock:
                        self.stats["llm_successes"] += 1
                    return {**parsed, "mode": "llm"}

                last_error = f"Bad JSON on attempt {attempt+1}: {raw[:80]!r}"
                logger.warning(f"[{ticket_id}] {last_error}")

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                is_rate_limit = "rate limit" in err_str or "429" in err_str or "too many" in err_str
                if is_rate_limit:
                    m_min = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_str)
                    if m_min:
                        mins = float(m_min.group(1)) if m_min.group(1) else 0.0
                        secs = float(m_min.group(2))
                        wait_s = mins * 60.0 + secs + 2.0
                    else:
                        wait_s = max(8.0, 5.0 * (attempt + 1))
                elif "failed to validate json" in err_str or "json_validate_failed" in err_str or "max completion tokens" in err_str:
                    wait_s = 2.0 * (attempt + 1)
                else:
                    wait_s = 2.0 ** attempt

                if attempt < max_attempts - 1:
                    logger.warning(f"[{ticket_id}] LLM attempt {attempt+1} failed: {exc} — retry in {wait_s:.1f}s")
                    time.sleep(wait_s)
                else:
                    logger.error(f"[{ticket_id}] All LLM retries exhausted: {exc}")

        with self._cache_lock:
            self.stats["failures"] += 1

        if self.strict_llm:
            raise RuntimeError(f"Groq API classification failed for ticket {ticket_id}: {last_error}")

        return {**self._classify_tfidf(text), "mode": "offline-tfidf-llm-fallback"}

    # ── Offline: sentence transformers ────────────────────────────────────────

    def _get_embed_model(self):
        if self._embed_model is not None:
            return self._embed_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading all-MiniLM-L6-v2 (first run may download ~90 MB)...")
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model ready.")
        except Exception as exc:
            logger.warning(f"Sentence transformer unavailable: {exc}")
            self._embed_model = None
        return self._embed_model

    def _get_category_embeddings(self):
        if self._cat_embeddings is not None:
            return self._cat_embeddings
        model = self._get_embed_model()
        if model is None:
            return None
        descs = [CATEGORY_DESCRIPTIONS[c] for c in CATEGORIES]
        self._cat_embeddings = model.encode(descs, normalize_embeddings=True)
        return self._cat_embeddings

    def _classify_batch_offline(
        self, rows: List[pd.Series], texts: List[str]
    ) -> Dict[str, dict]:
        """Batch offline classification. Returns {ticket_id: result_dict}."""
        ticket_ids = [str(r["ticket_id"]) for r in rows]
        results: Dict[str, dict] = {}

        model    = self._get_embed_model()
        cat_embs = self._get_category_embeddings()

        if model is not None and cat_embs is not None:
            logger.info(f"Batch-embedding {len(texts):,} tickets with sentence-transformer...")
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=(len(texts) > 500),
            )
            sims = embeddings @ cat_embs.T   # (N, 10)
            for i, tid in enumerate(ticket_ids):
                best = int(np.argmax(sims[i]))
                results[tid] = {
                    "category":   CATEGORIES[best],
                    "confidence": round(float(sims[i][best]), 4),
                    "mode":       "offline-embedding",
                }
        else:
            logger.info("Sentence transformer unavailable — using TF-IDF keyword fallback.")
            for tid, text in zip(ticket_ids, texts):
                results[tid] = self._classify_tfidf(text)

        return results

    def _classify_tfidf(self, text: str) -> dict:
        """Keyword-count fallback. Requires no model or library."""
        tl = text.lower()
        scores = {cat: sum(1 for kw in kws if kw in tl)
                  for cat, kws in TFIDF_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "Product Enquiry & General"
        return {
            "category":   best,
            "confidence": round(min(scores[best] / 5.0, 1.0), 4),
            "mode":       "offline-tfidf",
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all tickets. Returns df with new columns:
          ai_category, ai_confidence, classification_mode, cache_hit
        """
        # Split into cached / uncached
        cached_results: Dict[str, dict] = {}
        uncached_rows:  List[pd.Series] = []
        uncached_texts: List[str]       = []

        for _, row in df.iterrows():
            tid = str(row["ticket_id"])
            key = self._cache_key(tid)

            cached_entry = None
            if not self.bypass_cache:
                if key in self._cache:
                    candidate = self._cache[key]
                    # Verify cache mode: if in LLM mode, entry MUST have mode == "llm"
                    if not self.offline and candidate.get("mode") != "llm":
                        logger.warning(
                            f"[{tid}] Cache entry found for key {key} with non-LLM mode '{candidate.get('mode')}'. "
                            "Rejecting cached entry to prevent offline contamination."
                        )
                    else:
                        cached_entry = candidate
                elif self.offline:
                    # Backward compatibility for offline cache only: check legacy keys
                    legacy_key = f"{tid}::{PROMPT_VERSION}::{self.model_name}"
                    if legacy_key in self._cache:
                        cached_entry = self._cache[legacy_key]
                    else:
                        legacy_v12 = f"{tid}::v1.2::gpt-4o-mini"
                        if legacy_v12 in self._cache:
                            cached_entry = self._cache[legacy_v12]

            if cached_entry is not None:
                self.stats["cache_hits"] += 1
                cached_results[tid] = {**cached_entry, "cached": True}
            else:
                self.stats["cache_misses"] += 1
                uncached_rows.append(row)
                uncached_texts.append(build_classification_text(row))

        logger.info(
            f"Cache: {len(cached_results):,} hits  |  {len(uncached_rows):,} misses"
        )

        # Process uncached tickets
        new_results: Dict[str, dict] = {}

        if uncached_rows:
            if not self.offline:
                # ── LLM mode: concurrent requests ─────────────────────────
                logger.info(
                    f"Calling Groq API for {len(uncached_rows):,} tickets "
                    f"(model={self.model_name}, concurrency={self.MAX_CONCURRENT})"
                )
                pairs = list(zip(
                    [str(r["ticket_id"]) for r in uncached_rows],
                    uncached_texts,
                ))
                processed = 0
                if self.MAX_CONCURRENT > 1:
                    with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT) as pool:
                        fut_to_tid = {
                            pool.submit(self._call_single_llm, tid, text): tid
                            for tid, text in pairs
                        }
                        for fut in as_completed(fut_to_tid):
                            tid = fut_to_tid[fut]
                            try:
                                result = fut.result()
                            except Exception as exc:
                                if self.strict_llm:
                                    logger.error(f"[{tid}] Strict LLM mode error: {exc}")
                                    raise
                                logger.error(f"[{tid}] Unexpected error: {exc}")
                                result = {**self._classify_tfidf(""), "mode": "error-fallback"}
                            new_results[tid] = result
                            with self._cache_lock:
                                self._cache[self._cache_key(tid)] = {
                                    k: v for k, v in result.items() if k != "cached"
                                }
                                self._save_cache()
                            processed += 1
                            logger.info(
                                f"  [LLM {processed}/{len(pairs)}] {tid} -> {result.get('category')} "
                                f"(conf={result.get('confidence')}) [cost: ${self.stats['estimated_cost_usd']:.4f}]"
                            )
                else:
                    for processed, (tid, text) in enumerate(pairs, 1):
                        try:
                            result = self._call_single_llm(tid, text)
                        except Exception as exc:
                            if self.strict_llm:
                                logger.error(f"[{tid}] Strict LLM mode error: {exc}")
                                raise
                            logger.error(f"[{tid}] Unexpected error: {exc}")
                            result = {**self._classify_tfidf(""), "mode": "error-fallback"}
                        new_results[tid] = result
                        with self._cache_lock:
                            self._cache[self._cache_key(tid)] = {
                                k: v for k, v in result.items() if k != "cached"
                            }
                            self._save_cache()
                        logger.info(
                            f"  [LLM {processed}/{len(pairs)}] {tid} -> {result.get('category')} "
                            f"(conf={result.get('confidence')}) [cost: ${self.stats['estimated_cost_usd']:.4f}]"
                        )
                        sys.stdout.flush()
                # Final cache write
                self._save_cache()
            else:
                # ── Offline batch ──────────────────────────────────────────
                batch_out = self._classify_batch_offline(uncached_rows, uncached_texts)
                new_results.update(batch_out)
                # Write to cache under robust key
                with self._cache_lock:
                    for tid, res in new_results.items():
                        self._cache[self._cache_key(tid)] = res
                self._save_cache()

        # Merge and annotate
        all_results = {
            **cached_results,
            **{tid: {**r, "cached": False} for tid, r in new_results.items()},
        }

        df = df.copy()
        ai_cats, ai_confs, modes, hits = [], [], [], []

        for _, row in df.iterrows():
            tid = str(row["ticket_id"])
            res = all_results.get(tid, {
                "category":   "Product Enquiry & General",
                "confidence": 0.0,
                "mode":       "error-fallback",
                "cached":     False,
            })
            ai_cats.append(res.get("category",   "Product Enquiry & General"))
            ai_confs.append(res.get("confidence", 0.0))
            modes.append(res.get("mode", "unknown"))
            hits.append(bool(res.get("cached", False)))

        df["ai_category"]         = ai_cats
        df["ai_confidence"]       = ai_confs
        df["classification_mode"] = modes
        df["cache_hit"]           = hits
        return df
