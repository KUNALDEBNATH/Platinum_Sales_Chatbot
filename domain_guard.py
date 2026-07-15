"""
domain_guard.py
────────────────────────────────────────────────────────────────────────────
Restricts the chatbot to sales / customer / appointment / feedback /
uploaded-document / uploaded-image / company-related questions.

Any query that has no overlap with the allowed domain — and is not
accompanied by (or referring to) an uploaded file — is rejected with a
polite, randomly chosen refusal message, WITHOUT ever reaching the LLM.

This module now does THREE layers of checking, each one a bit smarter
(and a bit more expensive) than the last, stopping at the first hit:

  1. Exact keyword overlap        — fast, same as before.
  2. Fuzzy / stemmed overlap      — catches word forms the exact list
                                     misses ("enquired" vs "enquiry"),
                                     and known dataset entities (customer
                                     names, cities, vehicle models) that
                                     get registered at startup by api.py.
  3. LLM classification           — last resort only. Re-uses the SAME
                                     chat model test.py already loaded
                                     (no extra download / model), asked a
                                     tightly-constrained yes/no question.
                                     Only invoked when 1 & 2 both miss, so
                                     it does not slow down the common case.

This module is still dependency-light at import time (pure stdlib + re).
The LLM layer lazily imports `test` only when actually needed, so plain
keyword/entity queries never pay that cost.
"""

import random
import re

# ── Words that indicate the query is about the sales domain ──────────────────
SALES_DOMAIN_WORDS = {
    "enquiry", "enquiries", "inquiry", "lead", "leads", "record", "records",
    "customer", "customers", "client", "clients", "profile", "case", "ticket",
    "person", "persons", "people", "individual", "individuals",
    "feedback", "review", "reviews", "rating", "ratings", "comment", "opinion",
    "experience", "satisfaction", "complaint", "complaints", "sentiment",
    "appointment", "appointments", "meeting", "visit", "booking", "booked",
    "schedule", "scheduled", "slot", "session", "confirmed", "cancelled",
    "cancel", "cancellation", "completed", "status", "state", "progress",
    "pending", "closed", "open", "active", "contact", "phone", "mobile",
    "email", "reach", "call", "vehicle", "vehicles", "car", "cars", "bike",
    "bikes", "model", "automobile", "product", "purchase", "buying",
    "payment", "paid", "pay", "loan", "cash", "emi", "finance", "amount",
    "test ride", "test drive", "trial", "demo", "city", "location", "region",
    "area", "new lead", "returning", "existing", "repeat", "sales", "sale",
    "dataset", "data", "database", "company", "showroom", "dealer",
    "dealership", "revenue", "target", "conversion", "quotation", "invoice",
    "enq", "eq",
}

# ── Words that indicate the query is about an uploaded file ──────────────────
DOCUMENT_DOMAIN_WORDS = {
    "document", "documents", "doc", "docs", "file", "files", "pdf", "docx",
    "excel", "xlsx", "csv", "sheet", "spreadsheet", "image", "images",
    "picture", "pictures", "photo", "photos", "png", "jpg", "jpeg", "webp",
    "upload", "uploaded", "attach", "attached", "attachment", "scan",
    "scanned", "ocr", "extract", "invoice", "chart", "graph", "table",
    "board", "screenshot", "read this", "describe this", "analyze this",
    "summarize this", "summarise this", "what does this say",
}

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "ok", "okay", "bye", "goodbye",
}

REFUSAL_MESSAGES = [
    "I'm designed to answer only sales-related questions or analyze uploaded files. Please ask a relevant question.",
    "Please ask questions related to sales, customer data, appointments, feedback, or uploaded documents.",
    "I can only assist with sales information and uploaded files.",
]

# Tracks whether the MOST RECENT answered turn was a sales-domain question
# ("sales") or something else ("other"/None). A short reply that clarifies
# or continues the previous sales question — "I mean 3/28/2026", "for
# Meera", "the second one" — often has NO sales vocabulary of its own, so
# without this it gets wrongly rejected as out-of-domain.
_last_turn_sales: bool = False

_HAS_DIGIT_RE = re.compile(r"\d")
_CAP_WORD_RE  = re.compile(r"\b[A-Z][a-z]+\b")

# ── Dynamic entity registry (populated by api.py at startup) ────────────────
# Real values pulled straight from the loaded CSVs — customer names, cities
# / states, vehicle models — so a query mentioning any of them is recognised
# as in-domain even if it uses none of the generic SALES_DOMAIN_WORDS
# ("list the persons from Chennai", "anything about Kia Seltos?").
_KNOWN_ENTITY_WORDS: set = set()

_STOPWORDS_FOR_ENTITIES = {
    "the", "and", "of", "for", "a", "an", "in", "on", "at", "to", "is",
}


def register_known_entities(*value_lists) -> None:
    """
    Register free-text values (names, "City, ST" strings, vehicle models,
    etc.) so their individual words become recognised in-domain vocabulary.

    Call this once at startup with every column you want the guard to be
    aware of, e.g.:

        domain_guard.register_known_entities(
            df_enquiry["Customer Name"], df_enquiry["City / State"],
            df_enquiry["Vehicle Name / Model"], df_appt["Vehicle"],
        )

    Accepts pandas Series, lists, sets, or plain strings — anything
    iterable of strings (or a single string).
    """
    global _KNOWN_ENTITY_WORDS
    for values in value_lists:
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for v in values:
            if v is None:
                continue
            sval = str(v).strip()
            if not sval or sval.lower() in ("nan", "none"):
                continue
            for w in re.findall(r"[a-zA-Z]+", sval.lower()):
                if len(w) >= 3 and w not in _STOPWORDS_FOR_ENTITIES:
                    _KNOWN_ENTITY_WORDS.add(w)


def mark_sales_turn() -> None:
    """Call after successfully answering a sales-domain question."""
    global _last_turn_sales
    _last_turn_sales = True


def mark_other_turn() -> None:
    """Call after any turn that was NOT a sales-domain answer (document
    turn, refusal, or /api/reset/), so stale context doesn't linger."""
    global _last_turn_sales
    _last_turn_sales = False


def _normalize(text: str) -> tuple:
    """Return (set of lowercase word tokens, lowercased full text)."""
    text_low = text.lower()
    tokens = set(re.findall(r"[a-zA-Z]+", text_low))
    return tokens, text_low


def _stem_prefix(word: str) -> str:
    """
    Crude, dependency-free "stem": strip a common suffix, then take a short
    prefix. Good enough to line up word FORMS ("enquired"/"enquiry"/
    "enquiries"/"enquiring" all → "enqui...") without needing nltk/spacy.
    """
    for suf in ("ations", "ation", "ing", "ies", "ied", "ers", "er", "es", "ed", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    return word[:5]


_SALES_PREFIXES = {
    _stem_prefix(w) for w in SALES_DOMAIN_WORDS if len(w) >= 5
}


def _fuzzy_sales_hit(tokens: set) -> bool:
    """True if any token's stem/prefix matches a sales-domain word's stem,
    e.g. 'enquired' matches 'enquiry', 'cancelling' matches 'cancelled'."""
    for tok in tokens:
        if len(tok) < 5:
            continue
        if _stem_prefix(tok) in _SALES_PREFIXES:
            return True
    return False


def has_sales_keywords(query: str) -> bool:
    """
    True if `query` contains vocabulary clearly about the sales domain
    (enquiries, appointments, feedback, customers, vehicles, payments…),
    including fuzzy word-form matches and known dataset entities (names /
    cities / vehicle models registered via register_known_entities).

    Exposed separately so other modules (e.g. attachment_handler's
    conversation-continuity heuristic) can check "is this clearly a sales
    question?" without duplicating the logic.
    """
    tokens, text_low = _normalize(query)
    if tokens & SALES_DOMAIN_WORDS:
        return True
    for phrase in ("test ride", "test drive", "new lead"):
        if phrase in text_low:
            return True
    if _fuzzy_sales_hit(tokens):
        return True
    if _KNOWN_ENTITY_WORDS and (tokens & _KNOWN_ENTITY_WORDS):
        return True
    return False


def _llm_classify(query: str):
    """
    Last-resort check: ask the SAME chat model test.py already loaded
    (no extra download) a tightly-scoped yes/no question. Returns True,
    False, or None (couldn't get a confident answer / model unavailable —
    caller should treat None as "no").

    Kept lazy and isolated so a plain keyword/entity match never pays this
    cost, and so domain_guard.py stays importable without torch/transformers
    when the LLM path is never exercised.
    """
    try:
        import test as _t
    except Exception:
        return None

    try:
        if not getattr(_t, "_llm_ready", False):
            if not _t._load_llm():
                return None

        import torch

        prompt = (
            "You are a strict binary classifier gating a car-dealership "
            "sales chatbot. The chatbot may ONLY answer questions about: "
            "sales enquiries/leads, customers, appointments, feedback or "
            "ratings, vehicles, payments, or a file the user uploaded. "
            "Everything else (general knowledge, coding, math, small talk "
            "about unrelated topics, etc.) is OUT of scope.\n\n"
            f"Question: {query}\n\n"
            "Could this question plausibly be answered using the "
            "dealership's customer/sales data described above? "
            "Reply with exactly one word: YES or NO."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            text = _t._llm_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = _t._llm_tok(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(_t.LLM_DEVICE)

        with torch.no_grad():
            output = _t._llm_mdl.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=_t._llm_tok.pad_token_id,
                eos_token_id=_t._llm_tok.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = _t._llm_tok.decode(new_tokens, skip_special_tokens=True).strip().lower()

        if answer.startswith("y"):
            return True
        if answer.startswith("n"):
            return False
        return None
    except Exception as e:
        print(f"  [domain_guard] LLM classification error: {e}")
        return None


def is_in_domain(query: str, has_attachment: bool = False) -> bool:
    """
    Decide whether `query` is allowed to reach the LLM / dataset pipeline.

    A query is IN-DOMAIN if any of the following hold (checked cheapest
    first, stopping at the first match):
      1. A file is attached to this very request.
      2. It's a short greeting / pleasantry.
      3. The query text contains sales-domain vocabulary (exact, fuzzy, or
         a known dataset entity — customer name / city / vehicle model).
      4. The query text contains document/image vocabulary.
      5. It's a short clarification of the previous sales turn (date,
         number, proper noun) — see the `_last_turn_sales` fallback below.
      6. Nothing else matched, so — as a final check — the already-loaded
         chat model is asked directly whether the question fits the
         dealership sales domain.

    NOTE: conversational continuations that reference a previously
    uploaded file using bare pronouns ("this", "it", "from this take the
    top two") are handled separately in api.py via
    attachment_handler.looks_like_context_followup, BEFORE this function
    is even called.
    """
    if has_attachment:
        return True

    tokens, text_low = _normalize(query)

    if tokens & {w for phrase in _GREETING_WORDS for w in phrase.split()}:
        if len(tokens) <= 3:
            return True

    if has_sales_keywords(query):
        return True

    doc_hit = bool(tokens & DOCUMENT_DOMAIN_WORDS)
    for phrase in ("read this", "describe this", "analyze this",
                   "analyse this", "summarize this", "summarise this",
                   "what does this say", "what does this document say"):
        if phrase in text_low:
            doc_hit = True

    if doc_hit:
        return True

    # Fallback #1: nothing above matched, BUT the last thing we answered was
    # a sales question, and this message looks like a clarification of it
    # rather than a brand-new unrelated topic — it's short, or it names a
    # specific date/number/proper noun ("3/28/2026", "for Meera", "the
    # second one"). Treat it as still part of that sales conversation
    # instead of rejecting it outright.
    if _last_turn_sales and tokens and (
        len(tokens) <= 4
        or _HAS_DIGIT_RE.search(query)
        or _CAP_WORD_RE.search(query)
    ):
        return True

    # Fallback #2 (last resort): ask the chat model itself. Only reached
    # for genuinely ambiguous queries that matched none of the vocabulary,
    # entity, or continuation heuristics above.
    llm_verdict = _llm_classify(query)
    if llm_verdict:
        return True

    return False


def refusal_message() -> str:
    """Return one of the three fixed refusal messages, chosen at random."""
    return random.choice(REFUSAL_MESSAGES)
