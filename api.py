import os
import sys
import json
import threading

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "__main__")

from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="sales-rag-chatbot-dev-secret-key",
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "corsheaders",
            "rest_framework",
        ],
        MIDDLEWARE=[
            "corsheaders.middleware.CorsMiddleware",
            "django.middleware.common.CommonMiddleware",
        ],
        CORS_ALLOW_ALL_ORIGINS=True,
        CORS_ALLOW_HEADERS=[
            "accept",
            "accept-encoding",
            "authorization",
            "content-type",
            "dnt",
            "origin",
            "user-agent",
            "x-csrftoken",
            "x-requested-with",
        ],
        ROOT_URLCONF=__name__,
        DATABASES={},
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )

import django
django.setup()

from django.urls import path
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse, Http404
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

try:
    from test import (
        IntelligentSalesChatbot,
        IntelligentRetriever,
        load_datasets,
        build_docs_per_source,
        _load_llm,
        _llm_ready,
        CSV_FILES,
        _cache_fresh,
        INDEX_CACHE,
        _KNOWN_NAMES_SET,
    )
    IMPORT_OK = True
except ImportError as e:
    print(f"[WARNING] Could not import test.py: {e}")
    IMPORT_OK = False

_chatbot: "IntelligentSalesChatbot | None" = None
_init_error: str = ""
_datasets_loaded: list = []


def _init_chatbot():
    global _chatbot, _init_error, _datasets_loaded, _KNOWN_NAMES_SET

    if not IMPORT_OK:
        _init_error = "test.py could not be imported. Make sure it is in the same directory."
        return

    try:
        print("[API] Loading datasets …")
        dfs = load_datasets()
        _datasets_loaded = list(dfs.keys())

        for src_df in dfs.values():
            for col in src_df.columns:
                if "name" in col.lower() and not col.startswith("__"):
                    _KNOWN_NAMES_SET.update(
                        n for n in src_df[col].dropna().astype(str).unique()
                        if n not in ("nan", "None", "")
                    )

        print("[API] Building retrievers …")
        docs_per_source = build_docs_per_source(dfs)

        if _cache_fresh(INDEX_CACHE, dfs):
            print(f"[API] Loading index from cache: {INDEX_CACHE}")
            retriever = IntelligentRetriever.load(INDEX_CACHE)
        else:
            retriever = IntelligentRetriever(dfs, docs_per_source)
            retriever.save(INDEX_CACHE)

        _chatbot = IntelligentSalesChatbot(retriever)

        threading.Thread(target=_load_llm, daemon=True).start()
        print("[API] Chatbot ready. LLM loading in background …")

    except Exception as exc:
        _init_error = str(exc)
        print(f"[API] Initialisation error: {exc}")

_init_chatbot()


def _json_response(data: dict, status: int = 200) -> JsonResponse:
    return JsonResponse(data, status=status, json_dumps_params={"ensure_ascii": False})


def _parse_body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}

@method_decorator(csrf_exempt, name="dispatch")
class ChatView(View):
    """
    POST /api/chat/
    Body:  {"query": "Show ENQ001 details"}
    Returns: {"answer": "...", "intent": "...", "elapsed": 1.23, "history_length": 4}
    """

    def post(self, request):
        if _chatbot is None:
            return _json_response(
                {"error": f"Chatbot not initialised. {_init_error}"}, status=503
            )

        body = _parse_body(request)
        query = (body.get("query") or "").strip()

        if not query:
            return _json_response({"error": "Missing 'query' in request body."}, status=400)

        try:
            answer, elapsed, intent = _chatbot.chat(query)
            return _json_response(
                {
                    "answer": answer,
                    "intent": intent,
                    "elapsed": round(elapsed, 3),
                    "history_length": len(_chatbot.history),
                }
            )
        except Exception as exc:
            return _json_response({"error": str(exc)}, status=500)


@method_decorator(csrf_exempt, name="dispatch")
class ResetView(View):
    """
    POST /api/reset/
    Clears conversation history.
    Returns: {"status": "ok"}
    """

    def post(self, request):
        if _chatbot is None:
            return _json_response({"error": "Chatbot not initialised."}, status=503)
        _chatbot.history.clear()
        return _json_response({"status": "ok"})


class HealthView(View):
    """
    GET /api/health/
    Returns current system status.
    """

    def get(self, request):
        import test as _test_mod  

        llm_ready = getattr(_test_mod, "_llm_ready", False)

        return _json_response(
            {
                "status": "ok" if _chatbot is not None else "error",
                "chatbot_ready": _chatbot is not None,
                "llm_ready": llm_ready,
                "datasets": _datasets_loaded,
                "init_error": _init_error or None,
                "index_cache": os.path.abspath(INDEX_CACHE) if os.path.exists(INDEX_CACHE) else None,
            }
        )


class SuggestionsView(View):
    """
    GET /api/suggestions/
    Returns example queries to display in the UI.
    """

    SUGGESTIONS = [
        "Show all enquiries",
        "Who gave bad feedback?",
        "All cancelled appointments",
        "Show ENQ001 full details",
        "Returning customers",
        "Customers from Chennai",
        "Who hasn't taken a test ride?",
        "Show me new leads",
        "What's Divya's feedback?",
        "Is Arjun's appointment confirmed?",
        "What car did Sneha enquire about?",
        "Show good feedback customers",
        "All completed appointments",
        "Payment type breakdown",
    ]

    def get(self, request):
        return _json_response({"suggestions": self.SUGGESTIONS})


class IndexView(View):
    """
    GET /
    Serves index.html from the same directory as api.py.
    """

    def get(self, request):
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        if not os.path.exists(html_path):
            return HttpResponse(
                "<h2>index.html not found.</h2>"
                "<p>Place <code>index.html</code> in the same directory as <code>api.py</code>.</p>",
                content_type="text/html",
                status=404,
            )
        with open(html_path, "r", encoding="utf-8") as f:
            return HttpResponse(f.read(), content_type="text/html")


def favicon_view(request):
    """Suppress 404 noise for favicon requests."""
    return HttpResponse(status=204)


urlpatterns = [
    path("",                 IndexView.as_view()),
    path("api/chat/",        ChatView.as_view()),
    path("api/reset/",       ResetView.as_view()),
    path("api/health/",      HealthView.as_view()),
    path("api/suggestions/", SuggestionsView.as_view()),
    path("favicon.ico",      favicon_view),
]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    # Default: runserver 0.0.0.0:8000
    args = sys.argv if len(sys.argv) > 1 else [sys.argv[0], "runserver", "0.0.0.0:8000"]
    execute_from_command_line(args)
