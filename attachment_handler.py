"""
attachment_handler.py
────────────────────────────────────────────────────────────────────────────
Orchestrates everything needed to answer a user query that comes with an
uploaded file attachment:

    1. Validate the file (size / extension).
    2. Save it to a temporary location.
    3. Route to document_parser.py (PDF/DOCX/TXT/CSV/XLSX) or
       vision_parser.py (PNG/JPG/JPEG/WEBP).
    4. Retrieve only the relevant chunks/rows for the question (never the
       whole document — see rag_utils.chunk_text / SimpleTfidfRetriever).
    5. Optionally pull in relevant sales-database context via the existing
       IntelligentRetriever, plus recent conversation history.
    6. Ask the already-loaded text LLM (reused from test.py) to produce the
       final natural-language answer, grounded only in the retrieved
       context.

This module intentionally reuses the LLM already loaded by test.py
(`_llm_tok` / `_llm_mdl`) instead of loading a second copy, and reuses the
existing IntelligentRetriever for sales-database context — per the "reuse
existing architecture wherever possible" requirement.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from typing import Optional

import document_parser
import vision_parser
import domain_guard

# ── Configuration ─────────────────────────────────────────────────────────
MAX_DOCUMENT_SIZE_MB = 15
MAX_IMAGE_SIZE_MB = 8

DOCUMENT_EXTS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_EXTS = DOCUMENT_EXTS | IMAGE_EXTS

_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "sales_chatbot_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

# ── Last-uploaded-document memory ────────────────────────────────────────────
# The parsed document (text/table + its retriever) is kept in memory so that
# a FOLLOW-UP text-only question like "explain the pdf" or "summarize it"
# can still be answered correctly, without requiring the user to re-attach
# the file. This is intentionally simple (single global slot) to match the
# rest of the project's single-session architecture (one shared _chatbot,
# one shared history — see api.py).
_last_document = None            # document_parser.ParsedDocument | None
_last_document_filename = None   # str | None

_FILE_FOLLOWUP_KEYWORDS = {
    "pdf", "document", "documents", "doc", "docs", "file", "files",
    "excel", "xlsx", "csv", "sheet", "spreadsheet", "attachment",
    "attached", "upload", "uploaded", "invoice", "report", "table",
}

# Natural pronouns / references people use when continuing to talk about
# something already on screen ("from THIS take the top two", "summarize
# IT again", "give me more from THAT"). On their own these words are too
# generic to mean anything — they only signal a document follow-up when
# combined with the fact that a document IS currently in focus (see
# looks_like_context_followup below).
_CONTINUATION_WORDS = {
    "this", "it", "that", "these", "those", "above", "previous", "prev",
    "again", "further", "same", "there",
    # Elaboration / expansion requests — "in detail", "explain more",
    # "elaborate", "go deeper", "why", "how" — these have no pronoun at
    # all but are still obviously continuations of whatever was just
    # discussed.
    "detail", "details", "detailed", "more", "elaborate", "expand",
    "explain", "clarify", "briefly", "deeper", "why", "how",
}

# Tracks whether the MOST RECENT answered turn was about the uploaded
# document ("document") or something else ("other"/None). Used as a
# fallback below so short replies that use no keyword we explicitly know
# about (new phrasings we haven't anticipated) still get routed back to
# the document instead of being rejected as out-of-domain.
_last_turn_context: Optional[str] = None


def mark_document_turn() -> None:
    """Call after successfully answering a question about the document."""
    global _last_turn_context
    _last_turn_context = "document"


def mark_other_turn() -> None:
    """Call after answering (or refusing) a turn that was NOT about the
    document, so a stale 'document' context doesn't linger indefinitely."""
    global _last_turn_context
    _last_turn_context = "other"


def has_stored_document() -> bool:
    """True if a document is currently held in memory from a prior upload."""
    return _last_document is not None


def looks_like_file_followup(query: str) -> bool:
    """
    Narrow heuristic: does this plain-text query explicitly name a file
    type or the word "document"/"file" etc (e.g. "explain the pdf")?
    """
    tokens = set(re.findall(r"[a-zA-Z]+", (query or "").lower()))
    return bool(tokens & _FILE_FOLLOWUP_KEYWORDS)


def looks_like_context_followup(query: str) -> bool:
    """
    Broader, "intelligent" heuristic used to keep the conversation grounded
    in the currently-uploaded document. Covers natural follow-ups such as:

        "now from this take the most two important"
        "summarize it again"
        "give me more detail on that"
        "explain the pdf"

    Sales-domain vocabulary always takes priority — e.g. "cancel this
    appointment" is a sales command, not a document follow-up, even though
    it contains the pronoun "this". Only call this when
    has_stored_document() is already True.
    """
    if domain_guard.has_sales_keywords(query):
        return False

    tokens = set(re.findall(r"[a-zA-Z]+", (query or "").lower()))

    if tokens & (_FILE_FOLLOWUP_KEYWORDS | _CONTINUATION_WORDS):
        return True

    # Fallback: no known keyword matched, BUT the very last thing we
    # discussed was this document, and the new message is short (the
    # kind of terse reply people give when continuing a thread: "in
    # detail", "why?", "and the totals?"). Treat it as still about the
    # document rather than rejecting it outright. Longer, unrelated-
    # looking sentences are left alone so we don't hijack a genuinely new
    # topic just because a document happens to still be in memory.
    if _last_turn_context == "document" and 0 < len(tokens) <= 6:
        return True

    return False


def clear_stored_document():
    """Called on /api/reset/ so a new conversation starts with no file memory."""
    global _last_document, _last_document_filename, _last_turn_context
    _last_document = None
    _last_document_filename = None
    _last_turn_context = None


class AttachmentError(Exception):
    """Raised for any user-facing attachment problem (bad type, too big, …)."""


# ─────────────────────────────────────────────────────────── VALIDATION

def validate_file(filename: str, size_bytes: int) -> str:
    """
    Validate extension + size. Returns the lowercase extension on success,
    raises AttachmentError with a friendly message on failure.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        supported = ", ".join(sorted(ALLOWED_EXTS))
        raise AttachmentError(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Supported types are: {supported}"
        )

    limit_mb = MAX_IMAGE_SIZE_MB if ext in IMAGE_EXTS else MAX_DOCUMENT_SIZE_MB
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > limit_mb:
        raise AttachmentError(
            f"File is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {limit_mb} MB for this file type."
        )
    return ext


def save_upload(django_file) -> str:
    """Persist an in-memory Django UploadedFile to a temp path, return path."""
    ext = os.path.splitext(django_file.name)[1].lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(_UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as out:
        for chunk in django_file.chunks():
            out.write(chunk)
    return dest_path


def cleanup_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ─────────────────────────────────────────────────────────── LLM GENERATION

_DOC_SYSTEM_PROMPT = (
    "You are a sales-team assistant analyzing an uploaded file or image on "
    "behalf of the user. Answer the user's question using ONLY the "
    "retrieved context provided below, which comes from the file the user "
    "JUST uploaded. Do not reference or reuse information from any other "
    "document, file, or previous topic. If the answer is not present in "
    "the context, say so plainly. Be concise and factual. Do not invent "
    "information that is not supported by the context."
)

_GEN_ARTIFACTS = [
    "<|end|>", "<|assistant|>", "<|im_end|>", "[/INST]", "<<SYS>>",
    "<</SYS>>", "</s>", "<|endoftext|>", "<|system|>", "<|user|>",
]


def generate_document_answer(query: str, context_blocks: list,
                              sales_context: str = "") -> Optional[str]:
    """
    Compose a grounded prompt from retrieved context + optional sales data
    context, and generate an answer using the text LLM already loaded by
    test.py.

    NOTE: conversation history is deliberately NOT included here. Each
    uploaded file is analyzed strictly on its own retrieved content —
    mixing in prior chat turns (which may describe a *different* uploaded
    file) causes small LLMs to bleed topics from earlier answers into the
    new one. History is still used for the normal (no-attachment) chat
    path in test.py, which is unaffected by this.

    Returns None if the LLM is unavailable (caller should fall back to a
    structured, non-LLM answer in that case).
    """
    import test as _test_mod
    from test import _load_llm  # reuse the already-loaded text LLM

    if not _load_llm():
        return None

    tok = _test_mod._llm_tok
    mdl = _test_mod._llm_mdl
    device = _test_mod.LLM_DEVICE

    context_text = "\n\n".join(f"- {c}" for c in context_blocks[:6]) or "(no matching content found)"

    prompt_parts = [f"Retrieved document context (from the file just uploaded):\n{context_text}"]
    if sales_context:
        prompt_parts.append(f"Relevant sales database context:\n{sales_context}")
    prompt_parts.append(f"Question: {query}")
    user_content = "\n\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": _DOC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"<s>[INST] <<SYS>>\n{_DOC_SYSTEM_PROMPT}\n<</SYS>>\n\n"
                f"{user_content} [/INST]"
            )

        inputs = tok(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        import torch
        with torch.no_grad():
            output = mdl.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        raw = tok.decode(new_tokens, skip_special_tokens=True).strip()
        for art in _GEN_ARTIFACTS:
            raw = raw.replace(art, "").strip()

        raw = raw.strip()
        return raw if len(raw) >= 3 else None

    except Exception as e:
        print(f"  [DocLLM] generation error: {e}")
        return None


# ─────────────────────────────────────────────────────────── ORCHESTRATION

def handle_attachment(django_file, query: str, chatbot=None) -> dict:
    """
    Main entry point called from api.py.

    Returns a dict: {"answer": str, "intent": str, "elapsed": float,
                      "filename": str}
    Raises AttachmentError for user-facing problems (bad file, etc).
    """
    t0 = time.time()
    filename = django_file.name
    size_bytes = django_file.size

    ext = validate_file(filename, size_bytes)
    saved_path = save_upload(django_file)

    try:
        if ext in IMAGE_EXTS:
            answer, intent = _handle_image(saved_path, query)
        else:
            answer, intent = _handle_document(saved_path, filename, ext, query, chatbot)
    except AttachmentError:
        raise
    except Exception as e:
        raise AttachmentError(
            f"Something went wrong while processing '{filename}': {e}"
        )
    finally:
        cleanup_file(saved_path)

    elapsed = time.time() - t0
    return {
        "answer": answer,
        "intent": intent,
        "elapsed": round(elapsed, 3),
        "filename": filename,
    }


def _handle_image(saved_path: str, query: str) -> tuple:
    effective_query = query.strip() or "Describe what is in this image."
    answer = vision_parser.analyze_image(saved_path, effective_query)
    return answer, "image_analysis"


def _handle_document(saved_path: str, filename: str, ext: str,
                      query: str, chatbot) -> tuple:
    global _last_document, _last_document_filename

    if not query.strip():
        query = "Summarize this document."

    doc = document_parser.parse_document(saved_path, filename)

    # Remember this document so a later plain-text follow-up question
    # ("explain the pdf", "summarize the file") can still be answered
    # correctly without requiring re-upload.
    _last_document = doc
    _last_document_filename = filename

    answer, intent = _answer_from_document(doc, query, chatbot)
    mark_document_turn()
    return answer, intent


def answer_from_stored_document(query: str, chatbot=None) -> Optional[dict]:
    """
    Answer a plain-text follow-up question using the document that was
    uploaded earlier in the conversation (see _last_document).

    Returns None if no document has been uploaded yet in this session —
    the caller (api.py) should fall back to the normal chat path in that
    case.
    """
    if _last_document is None:
        return None

    if not query.strip():
        query = "Summarize this document."

    t0 = time.time()
    answer, intent = _answer_from_document(_last_document, query, chatbot)
    mark_document_turn()
    return {
        "answer": answer,
        "intent": intent,
        "elapsed": round(time.time() - t0, 3),
        "filename": _last_document_filename,
    }


def _answer_from_document(doc, query: str, chatbot) -> tuple:
    """Shared RAG + generation logic for a single ParsedDocument."""
    # 1. Try a direct, accurate pandas computation for tabular files.
    quick_fact = document_parser.compute_quick_stat(doc, query)

    # 2. Retrieve only the relevant chunks / rows (never the whole file).
    relevant = document_parser.retrieve_relevant_rows_or_chunks(doc, query, top_k=5)
    context_blocks = ([quick_fact] if quick_fact else []) + relevant
    if doc.kind == "table" and not relevant:
        context_blocks.append(doc.summary)

    # 3. Optionally pull relevant existing sales-database context.
    sales_context = _get_sales_context(chatbot, query)

    # 4. Ask the LLM, grounded only in the retrieved context from THIS file.
    #    (Conversation history is intentionally excluded — see the note on
    #    generate_document_answer for why.)
    generated = generate_document_answer(query, context_blocks, sales_context)

    if generated:
        answer = generated
    else:
        # LLM unavailable — fall back to a structured, still-useful answer.
        answer = _structured_fallback(doc, quick_fact, relevant)

    intent = "table_analysis" if doc.kind == "table" else "document_analysis"
    return answer, intent


def _structured_fallback(doc, quick_fact, relevant) -> str:
    lines = [doc.summary]
    if quick_fact:
        lines.append(quick_fact)
    if relevant:
        lines.append("Most relevant content found:")
        lines.extend(f"- {r[:300]}" for r in relevant[:3])
    return "\n\n".join(lines)


def _get_sales_context(chatbot, query: str, top_k: int = 3) -> str:
    """Pull a few relevant rows from the existing sales retriever, if any."""
    if chatbot is None or not hasattr(chatbot, "retriever"):
        return ""
    try:
        results = chatbot.retriever.retrieve(query, top_k=top_k)
    except Exception:
        return ""
    if not results:
        return ""
    lines = []
    for r in results[:top_k]:
        if r.get("__score__", 0) < 0.05:
            continue
        preview = ", ".join(
            f"{k}: {v}" for k, v in r.items()
            if not str(k).startswith("__") and str(v) not in ("nan", "None", "")
        )
        lines.append(f"[{r.get('__source__', 'record')}] {preview[:250]}")
    return "\n".join(lines)
