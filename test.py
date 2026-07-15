import os, sys, time, pickle, re, warnings
from difflib import SequenceMatcher
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL_DIR   = "sales_llm_model"
LLM_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 0.3
LLM_TOP_P       = 0.9

CHAT_MODELS = [
<<<<<<< HEAD
    LLM_MODEL_DIR,                   # your fine-tuned model (train.py output), tried first
    "Qwen/Qwen2.5-1.5B-Instruct",    # fallback if sales_llm_model/ is missing or fails to load
    "Qwen/Qwen2.5-0.5B-Instruct",
=======
    "Qwen/Qwen2.5-1.5B-Instruct",   
    "Qwen/Qwen2.5-0.5B-Instruct",   
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
]

_llm_tok   = None
_llm_mdl   = None
_llm_ready = False


def _load_llm() -> bool:
    
    global _llm_tok, _llm_mdl, _llm_ready
    if _llm_ready:
        return True
    for model_path in CHAT_MODELS:
<<<<<<< HEAD
        if model_path == LLM_MODEL_DIR and not os.path.isdir(model_path):
            print(f"  [LLM] Local fine-tuned model '{model_path}' not found — skipping.")
            continue
=======
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
        try:
            print(f"  [LLM] Loading '{model_path}' on {LLM_DEVICE} …")
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if LLM_DEVICE == "cuda" else torch.float32
            mdl   = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=LLM_DEVICE,
                trust_remote_code=True,
            )
            mdl.eval()
            _llm_tok, _llm_mdl, _llm_ready = tok, mdl, True
            print(f"  [LLM] Ready: {model_path}")
            return True
        except Exception as e:
            print(f"  [LLM] '{model_path}' failed: {e}")
    print("  [LLM] No model available — using structured answers only.")
    return False


_LLM_ARTIFACTS = [
    "<|end|>", "<|assistant|>", "<|im_end|>", "[/INST]", "<<SYS>>",
    "<</SYS>>", "</s>", "<|endoftext|>", "<|system|>", "<|user|>",
    "Output:", "Instruct:",
]

_SALES_SYSTEM_PROMPT = (
    "Report the database facts below in plain English. "
    "One or two sentences only. "
    "Use only the facts given. "
    "Do not add opinions, advice, or extra information. "
    "Do not greet. Do not introduce yourself. Just state the facts."
)


def _generate_phrase(query: str, structured_facts: str, intent: str) -> str:
    if not _load_llm():
        return None

    list_intents = {"list_all", "bad_feedback", "good_feedback", "cancelled",
                    "completed", "new_lead", "city", "returning"}
    task = ("State the total count and name up to 3 examples in one sentence."
            if intent in list_intents else
            "State the answer in one sentence using only the data above.")

    if intent in list_intents:
        lines     = structured_facts.strip().split("\n")
        trimmed   = lines[:6]
        remaining = len(lines) - len(trimmed)
        facts_for_llm = "\n".join(trimmed)
        if remaining > 0:
            facts_for_llm += f"\n… and {remaining} more entries."
    else:
        facts_for_llm = structured_facts.strip()

    user_content = (
        f"Data:\n{facts_for_llm}\n\n"
        f"Question: {query}\n\n"
        f"{task}"
    )
    messages = [
        {"role": "system", "content": _SALES_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        try:
            prompt = _llm_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = (
                f"<s>[INST] <<SYS>>\n{_SALES_SYSTEM_PROMPT}\n<</SYS>>\n\n"
                f"{user_content} [/INST]"
            )

        inputs = _llm_tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        ).to(LLM_DEVICE)

        if hasattr(_llm_mdl, "generation_config"):
            _llm_mdl.generation_config.max_length = None

        with torch.no_grad():
            output = _llm_mdl.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                do_sample=True,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                repetition_penalty=1.2,
                pad_token_id=_llm_tok.pad_token_id,
                eos_token_id=_llm_tok.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        raw = _llm_tok.decode(new_tokens, skip_special_tokens=True).strip()

        for art in _LLM_ARTIFACTS:
            raw = raw.replace(art, "").strip()

        SKIP_PATTERNS = [
            r"^(You|User|Human|Farmer|Sales Manager|Question|Q)\s*:",
            r"^here['\s]s (my |the )?(response|answer|reply|result)",
            r"^(in response|as requested|based on the|according to the)",
            r"^(total enquires?|examples?\s*:)",
            r"^(data|retrieved data|sales manager)\s*:",
        ]
        cleaned_lines = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if any(re.match(pat, stripped, re.I) for pat in SKIP_PATTERNS):
                continue
            if re.match(r"^(You|User|Human|Farmer|Sales Manager|Question|Q)\s*:", stripped, re.I):
                break
            cleaned_lines.append(line)
        raw = " ".join(l.strip() for l in cleaned_lines if l.strip()).strip()

        if raw[:80].count(":") > 3:
            return None

        sentences     = re.split(r"(?<=[.!?])\s+", raw.strip())
        good_sentences = [s for s in sentences if re.search(r"[.!?]$", s.strip())]
        if good_sentences:
            raw = " ".join(good_sentences)
        elif raw and raw[-1] not in ".!?":
            raw = raw.rstrip(",;:") + "."

        if len(raw.strip()) < 20:
            return None

        BAD_PHRASES = [
            "i don't have information", "i cannot answer", "as an ai",
            "i was trained", "my knowledge cutoff", "i do not know",
            "i'm not sure", "assistant: yes", "assistant: no",
            "dear customer", "yupv", "nop:", "good day all",
            "sure, here's", "here's how", "congratulations", "thank you for reaching",
            "i am confident", "do not hesitate", "please feel free",
            "based on my knowledge", "i can only provide", "your request",
            "how you could respond", "i am glad", "that sounds great",
            "here's my response", "here is my response", "in response to",
            "arriving 10 minutes", "10 minutes early", "ensure his meeting",
            "sales representative:", "as requested", "total enquires",
        ]
        raw_lower = raw.lower()
        if any(p in raw_lower for p in BAD_PHRASES):
            return None

        words = raw.split()
        if len(words) < 4:
            return None
        if sum(1 for w in words if w.endswith(":")) / max(len(words), 1) > 0.25:
            return None

        final_sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        raw = " ".join(final_sentences[:3]).strip()

        return raw if len(raw) >= 20 else None

    except Exception as e:
        print(f"  [LLM] generation error: {e}")
        return None


warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────── CONFIG
TOP_K       = 8
INDEX_CACHE = "rag_index.pkl"

CSV_FILES = {
    "Enquiry"    : "sales_enquiry_dataset.csv",
    "Appointment": "sales_appointment_dataset.csv",
    "Feedback"   : "sales_feedback_dataset.csv",
}


INTENT_TO_SOURCE = {
    "appointment" : "Appointment",
    "cancelled"   : "Appointment",
    "completed"   : "Appointment",
    "feedback"    : "Feedback",
    "bad_feedback": "Feedback",
    "good_feedback": "Feedback",
    "contact"     : "Enquiry",
    "status"      : "Enquiry",
    "payment"     : "Enquiry",
    "test_ride"   : "Enquiry",
    "vehicle"     : "Enquiry",
    "new_lead"    : "Enquiry",
    "city"        : "Enquiry",
    "returning"   : "Enquiry",
    "summary"     : None,
    "list_all"    : None,
}


_KNOWN_NAMES_SET: set = set()


SYNONYMS = {
    "enquiry":     ["enquiry","inquiry","lead","record","customer","data","details","info",
                    "information","profile","case","file","ticket"],
    "feedback":    ["feedback","review","rating","comment","opinion","experience","satisfaction",
                    "complaint","response","feeling","sentiment","happy","unhappy","satisfied",
                    "dissatisfied","good","bad","poor","excellent","average","reaction"],
    "appointment": ["appointment","meeting","visit","booking","schedule","slot","session",
                    "confirmed","booked","planned","upcoming","timing","time","date"],
    "status":      ["status","state","progress","update","stage","current","situation",
                    "standing","position","pending","closed","open","active"],
    "contact":     ["contact","phone","mobile","email","mail","reach","number","call"],
    "vehicle":     ["vehicle","car","bike","model","automobile","product","item",
                    "interested","buying","purchase","want","enquired about"],
    "payment":     ["payment","paid","pay","loan","cash","emi","finance","amount","mode","method"],
    "test_ride":   ["test ride","test drive","trial","drove","tried","ridden","demo"],
    "bad":         ["bad","poor","negative","low","worst","unhappy","dissatisfied","complaint","below"],
    "good":        ["good","great","excellent","positive","high","happy","satisfied",
                    "wonderful","amazing","best","top"],
    "city":        ["city","location","place","from","region","area","state","where","lives"],
    "new":         ["new","fresh","recent","just","newly","latest"],
    "returning":   ["returning","existing","repeat","old","regular","loyal","again"],
    "all":         ["all","entire","complete","full","every","each","whole","list","show",
                    "give","dataset","records","everyone","dump"],
    "cancelled":   ["cancel","cancelled","cancellation","abort","drop","not coming","no show","withdrew"],
    "completed":   ["completed","done","finished","over","past","visited","successful"],
}

INTENT_CLUSTERS = {
    "list_all":      {"all","entire","complete","full","every","each","whole","list","dataset",
                      "records","give","show","take","display","dump","everybody"},
    "bad_feedback":  {"bad","poor","negative","low","worst","unhappy","dissatisfied","complaint",
                      "complaints","below","rating","feedback","review"},
    "good_feedback": {"good","great","excellent","positive","high","happy","satisfied",
                      "wonderful","amazing","best","top","rating","feedback","review"},
    "feedback":      {"feedback","review","rating","comment","opinion","experience",
                      "satisfaction","sentiment","said","wrote","gave","submitted"},
    "appointment":   {"appointment","meeting","visit","booking","schedule","slot","booked",
                      "planned","upcoming","timing","time","date","session","confirmed"},
    "cancelled":     {"cancel","cancelled","cancellation","abort","drop","no","not"},
    "completed":     {"completed","done","finished","over","past","visited","successful"},
    "contact":       {"contact","phone","mobile","email","mail","reach","number","call"},
    "status":        {"status","state","progress","update","stage","current","situation",
                      "standing","position","pending","closed","open","active"},
    "payment":       {"payment","paid","pay","loan","cash","emi","finance","amount","mode"},
    "test_ride":     {"test","ride","drive","trial","drove","tried","ridden","demo"},
    "vehicle":       {"vehicle","car","bike","model","automobile","product","item",
                      "interested","buying","purchase","want"},
    "city":          {"city","location","place","from","region","area","state","where"},
    "new_lead":      {"new","fresh","recent","newly","latest","lead","enquiry"},
    "returning":     {"returning","existing","repeat","old","regular","loyal"},
    "summary":       {"details","summary","everything","full","complete","about","info",
                      "information","profile","all","tell","know","summarize","summarise"},
}




def expand_query(query: str) -> str:
    q_lower = query.lower()
    extra   = []
    for _, words in SYNONYMS.items():
        if any(w in q_lower for w in words):
            extra.extend(words)
    return query + " " + " ".join(extra)


<<<<<<< HEAD
def detect_intent(query: str, known_name: str | None = None) -> tuple:
=======
def detect_intent(query: str) -> tuple:
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
    expanded    = set(re.findall(r'\w+', expand_query(query).lower()))
    q_lower     = query.lower()
    q_words_raw = set(re.findall(r'\w+', q_lower))

    scores = {intent: len(expanded & cluster) / max(len(cluster), 1)
              for intent, cluster in INTENT_CLUSTERS.items()}
    best = max(scores, key=scores.get)

<<<<<<< HEAD
    # `known_name` lets a caller (e.g. IntelligentSalesChatbot.chat, which
    # may have carried a customer name over from the previous turn for a
    # pronoun-only follow-up like "what is HER enquiry ID?") count as a
    # name being "present" here too, even though this query's own text
    # has none — otherwise the narrowing rules below never fire and the
    # query falls through to an unscoped, full-dataset-style answer.
    has_name = bool(extract_name(query)) or bool(_extract_name_lower(query)) or bool(known_name)
=======
    has_name = bool(extract_name(query)) or bool(_extract_name_lower(query))
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91

    INFO_FIELDS = {"enquiry","id","details","info","information","record","profile",
                   "data","about","for","say","show","tell","give","only","find","get",
                   "take","fetch","pull","of","regarding","related","specific","his","her"}
    if has_name and (q_words_raw & INFO_FIELDS) and best in ("new_lead","list_all","summary"):
        best = "summary"
    if has_name and best == "list_all":
        best = "summary"

    ENQ_ID_WORDS = {"enquiry","id","enq","number","ref","reference","code"}
    if best == "new_lead" and has_name and (q_words_raw & ENQ_ID_WORDS):
        best = "summary"

    POLARITY_POS = {"good","great","excellent","positive","high","happy","satisfied",
                    "wonderful","amazing","best","top"}
    POLARITY_NEG = {"bad","poor","low","unhappy","dissatisfied","worst","complaint","negative"}
    if scores.get("bad_feedback",0) > 0 and scores.get("good_feedback",0) > 0:
        best = "bad_feedback" if q_words_raw & POLARITY_NEG else "good_feedback"

    if best in ("bad_feedback","good_feedback"):
        if has_name and not (q_words_raw & POLARITY_POS) and not (q_words_raw & POLARITY_NEG):
            best = "feedback"

    if best == "appointment":
        if any(w in q_lower for w in ["cancel","cancelled","cancellation","no show","no-show"]):
            best = "cancelled"
        elif any(w in q_lower for w in ["completed","done","finished","past","over","successful"]):
            best = "completed"

    return best, scores[best]


def _extract_name_lower(query: str):
    """
    Extract customer names typed in lowercase using the DYNAMIC name set
    loaded from the actual CSVs (no more hardcoded list).
    """
    words = re.findall(r'\b\w+\b', query.lower())
    for w in words:
        if w.title() in _KNOWN_NAMES_SET:
            return w.title()
    return None


def normalize_query(query: str) -> str:
    query = re.sub(r'\bEQ(\d+)\b',     r'ENQ\1', query, flags=re.IGNORECASE)
    query = re.sub(r'\benq\s*(\d+)\b', r'ENQ\1', query, flags=re.IGNORECASE)
    return query


def extract_enq_id(query: str):
    m = re.search(r'\b(ENQ\d+)\b', query, re.IGNORECASE)
    return m.group(1).upper() if m else None


<<<<<<< HEAD
_DATE_RE = re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b')


def extract_query_date(query: str) -> str | None:
    """
    Pull an explicit date like '3/28/2026' out of the query and normalize
    it to the same M/D/YYYY (no leading zeros) format the CSVs store
    dates in, so it can be compared directly against the Enquiry Date /
    Appointment Date / Date fields.
    """
    m = _DATE_RE.search(query)
    if not m:
        return None
    month, day, year = m.groups()
    try:
        return f"{int(month)}/{int(day)}/{year}"
    except ValueError:
        return None


def _row_matches_date(row: dict, date_str: str) -> bool:
    """True if ANY date-like field on this row exactly equals date_str."""
    for col, val in row.items():
        if col.startswith("__"):
            continue
        if "date" in col.lower() and str(val).strip() == date_str:
            return True
    return False


=======
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
_NAME_STOP = {
    "is","was","are","the","for","when","what","who","has","have","did","does",
    "can","tell","me","my","give","other","details","of","email","phone","show",
    "find","get","contact","id","satisfied","feedback","rating","payment",
    "customer","appointment","status","enquiry","dataset","information","data",
    "all","any","please","about","from","their","his","her","its","this","that",
    "with","and","or","but","in","on","at","to","by","an","a","no","not",
    "take","give","list","entire","complete","every","each","whole","records",
    "summary","summarize","summarise","everything","best","match","assistant",
    "hyderabad","bangalore","bengaluru","chennai","mumbai","delhi","pune",
    "coimbatore","kolkata","ahmedabad","surat","jaipur","lucknow","nagpur",
    "ts","tn","ka","mh","gj","rj","up","wb","ap","telangana","karnataka",
    "tamilnadu","maharashtra","gujarat",
}


def extract_name(query: str):
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    for m in matches:
        parts = [p for p in m.split() if p.lower() not in _NAME_STOP]
        if parts:
            return " ".join(parts)
    return None


def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.75) -> list:
    nl  = name.lower()
    out = []
    for c in candidates:
        cl = c.lower()
        if nl in cl or cl in nl:
            out.append(c)
        elif SequenceMatcher(None, nl, cl).ratio() >= threshold:
            out.append(c)
    return out



def row_to_rich_text(row: dict, source: str) -> str:
    """Convert a row to verbose natural language for richer TF-IDF matching."""
    def v(key): return str(row.get(key, "") or "").strip()

    if source == "Enquiry":
        name   = v("Customer Name")
        eid    = v("ENQUIRY ID")
        phone  = v("Phone Number")
        email  = v("Email")
        gender = v("Gender")
        veh    = v("Vehicle Name / Model")
        src    = v("Enquiry Source")
        edate  = v("Enquiry Date")
        adate  = v("Appointment Date")
        city   = v("City / State")
        ctype  = v("Customer Type")
        pay    = v("Payment Type")
        ride   = v("Test Ride Taken")
        status = v("Status")
        ride_t = "took test ride test drive" if ride.lower() == "yes" else "did not take test ride"
        return (
            f"[Enquiry] Customer {name} enquiry ID {eid} "
            f"is a {gender} {ctype} customer from {city} location city. "
            f"Interested in vehicle car model {veh}. "
            f"Enquiry source channel {src} on date {edate}. "
            f"Appointment scheduled on {adate}. "
            f"Contact phone {phone} email {email}. "
            f"Payment mode method {pay} loan cash emi. "
            f"Status progress stage {status}. "
            f"Test ride test drive {ride} {ride_t}. "
            f"Enquiry record lead data information details profile."
        )

    elif source == "Appointment":
        name   = v("Customer Name")
        eid    = v("Enquiry ID")
        adate  = v("Appointment Date")
        atime  = v("Time")
        veh    = v("Vehicle")
        status = v("Status")
        extra  = ("confirmed scheduled booked planned upcoming" if status == "Scheduled" else
                  "completed done finished successful visited"   if status == "Completed" else
                  "cancelled canceled abort withdrawn no-show"   if status == "Cancelled" else "")
        return (
            f"[Appointment] Customer {name} enquiry ID {eid} "
            f"appointment meeting visit booking on date {adate} at time {atime}. "
            f"Vehicle car model {veh}. Status {status} {extra}. "
            f"Appointment record data details."
        )

    elif source == "Feedback":
        name     = v("Customer Name")
        eid      = v("Enquiry ID")
        feedback = v("Feedback")
        rating   = v("Rating")
        date     = v("Date")
        try:
            r   = int(float(rating))
            sent = ("positive excellent satisfied happy great good" if r >= 4 else
                    "neutral average okay"                           if r == 3 else
                    "negative poor bad unhappy dissatisfied complaint low")
            neg_tag = "complaint negative review low rating bad dissatisfied unhappy" if r <= 2 else ""
            pos_tag = "excellent positive review high rating good satisfied happy"    if r >= 4 else ""
        except Exception:
            sent = neg_tag = pos_tag = ""
        return (
            f"[Feedback] Customer {name} enquiry ID {eid} "
            f"submitted feedback review comment on date {date}: '{feedback}'. "
            f"Rating score {rating} out of 5. "
            f"Sentiment {sent}. {neg_tag} {pos_tag}. "
            f"Feedback review data details."
        )

    else:
        parts = [f"[{source}]"]
        for col, val in row.items():
            if not col.startswith("__") and val and str(val) not in ("nan","None",""):
                parts.append(f"{col} {val}")
        return " ".join(parts)




class IntelligentRetriever:
    """
    Maintains a SEPARATE TF-IDF index for each dataset (Enquiry / Appointment /
    Feedback) so that name searches never bleed across tables.
    A unified index is also kept for cross-source queries (ENQ-ID lookups, list_all).
    """

    def __init__(self, dfs: dict, docs_per_source: dict):
        """
        dfs             : {source_name: pd.DataFrame}   (each df has __source__ col)
        docs_per_source : {source_name: [rich_text, …]}
        """
        self.dfs     = dfs                                   
        self.sources = list(dfs.keys())
        self.df      = pd.concat(dfs.values(), ignore_index=True)  

       
        self._per_source: dict = {}
        for src, src_df in dfs.items():
            docs     = docs_per_source[src]
            vec      = TfidfVectorizer(
                ngram_range=(1, 3), max_features=20_000,
                sublinear_tf=True, min_df=1,
                token_pattern=r'(?u)\b\w[\w\-]*\b',
            )
            expanded = [d + " " + expand_query(d) for d in docs]
            mat      = vec.fit_transform(expanded)
            self._per_source[src] = {"vec": vec, "mat": mat, "df": src_df.reset_index(drop=True)}
            print(f"  OK [{src:<12}] index: {mat.shape[0]} rows × {mat.shape[1]} vocab")

        
        all_docs = [d for src in self.sources for d in docs_per_source[src]]
        vec_all  = TfidfVectorizer(
            ngram_range=(1, 3), max_features=30_000,
            sublinear_tf=True, min_df=1,
            token_pattern=r'(?u)\b\w[\w\-]*\b',
        )
        self.matrix = vec_all.fit_transform([d + " " + expand_query(d) for d in all_docs])
        self.vec    = vec_all
        print(f"  OK [Unified    ] index: {self.matrix.shape[0]} rows × {self.matrix.shape[1]} vocab")

        
        self.name_index: dict = defaultdict(list)
        self.all_names:  list = []
        for src, src_df in dfs.items():
            for _, row in src_df.iterrows():
                for col in src_df.columns:
                    if "name" in col.lower() and not col.startswith("__"):
                        name = str(row[col]).strip()
                        if name and name not in ("nan","None",""):
                            self.name_index[name.lower()].append(
                                {"source": src, "row": row.to_dict()}
                            )
                            self.all_names.append(name)
        self.all_names = sorted(set(self.all_names))

   
    def retrieve_from_source(self, query: str, source: str,
                             top_k: int = TOP_K) -> list:
        """Search ONLY within the given source dataset."""
        if source not in self._per_source:
            return []
        idx  = self._per_source[source]
        vec, mat, src_df = idx["vec"], idx["mat"], idx["df"]

        qvec   = vec.transform([expand_query(query)])
        scores = cosine_similarity(qvec, mat).flatten()
        top_i  = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for i in top_i:
            if scores[i] < 0.01:
                continue
            row               = src_df.iloc[i].to_dict()
            row["__source__"] = source
            row["__score__"]  = round(float(scores[i]), 4)
            results.append(row)
        return results[:top_k]

    
    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """Search across all sources."""
        qvec   = self.vec.transform([expand_query(query)])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        top_i  = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for i in top_i:
            if scores[i] < 0.01:
                continue
            row              = self.df.iloc[i].to_dict()
            row["__score__"] = round(float(scores[i]), 4)
            results.append(row)
        return results[:top_k]

    
    def retrieve_by_name(self, name: str, preferred_source: str = None,
                         top_k: int = TOP_K) -> list:
        """
        Fuzzy name search.  If preferred_source is given, searches ONLY that
        dataset first; falls back to all sources if nothing found there.

        NOTE: Do NOT compare names across datasets — the same ENQ ID has
        different customer names in each table. Routing by preferred_source
        ensures we search the correct table.
        """
        matched_names = fuzzy_match_name(name, self.all_names)

        def rows_for(src: str) -> list:
            found = []
            for mname in matched_names:
                for entry in self.name_index.get(mname.lower(), []):
                    if entry["source"] == src:
                        r               = dict(entry["row"])
                        r["__source__"] = src
                        r["__score__"]  = 1.0
                        found.append(r)
            return found

        if preferred_source:
            rows = rows_for(preferred_source)
        else:
            rows = []

        if not rows:                          # fallback: all sources
            for src in self.sources:
                if src != preferred_source:
                    rows.extend(rows_for(src))

        # Deduplicate
        seen, out = set(), []
        for r in rows:
            key = (r.get("__source__"), str(find_val(r, "enquiry id", "ENQUIRY ID") or ""))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out[:top_k]

    # ── Get all rows for a source ────────────────────────────────────────────
    def get_all_by_source(self, source: str) -> list:
        if source not in self._per_source:
            return []
        rows = []
        for _, row in self._per_source[source]["df"].iterrows():
            r               = row.to_dict()
            r["__source__"] = source
            r["__score__"]  = 1.0
            rows.append(r)
        return rows

    
    def get_by_enq_id(self, enq_id: str) -> dict:
        """
        Return {source: row_dict} for every dataset that contains enq_id.
        Each source may have a DIFFERENT customer for the same ENQ ID — that is
        normal in this dataset and is displayed accordingly.
        """
        result = {}
        for src, idx in self._per_source.items():
            src_df = idx["df"]
            for col in src_df.columns:
                if "enquiry" in col.lower() and "id" in col.lower():
                    mask = src_df[col].astype(str).str.upper() == enq_id.upper()
                    hits = src_df[mask]
                    if not hits.empty:
                        r               = hits.iloc[0].to_dict()
                        r["__source__"] = src
                        result[src]     = r
                    break
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)




def find_val(row: dict, *keys):
    for k in keys:
        for col, val in row.items():
            if col.startswith("__"):
                continue
            if k.lower() in col.lower():
                vs = str(val).strip()
                if vs and vs.lower() not in ("nan","none","","n/a"):
                    return vs
    return None


def _rating(row: dict):
    val = find_val(row, "rating")
    try:    return int(float(val))
    except: return None


def get_source(row: dict) -> str:
    return row.get("__source__", "record")



def build_answer(query: str, results: list, intent: str,
                 retriever: IntelligentRetriever) -> str:

    q_low = query.lower()

    if not results:
        return ("I could not find any relevant records. "
                "Try rephrasing or check the name / ID.")


    eid = extract_enq_id(query)
    if eid:
        by_source = retriever.get_by_enq_id(eid)
        if not by_source:
            return f"No records found for enquiry ID '{eid}'."

        lines = [f"Records for {eid} across all datasets:"]
        for src in ("Enquiry", "Appointment", "Feedback"):
            row = by_source.get(src)
            if not row:
                continue
            lines.append(f"\n── {src} {'─'*(50-len(src))}")
            for col, val in row.items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        return "\n".join(lines)

    # ── LIST / DATASET REQUEST ────────────────────────────────────────────────
    LIST_TRIGGERS = {
        "all records","entire dataset","full list","every record","all customers",
        "all enquiries","show dataset","take enquiry","give dataset","all data",
        "entire database","list all","show all","give all","dump","all feedback",
        "all appointments","show enquiry","show feedback","show appointment",
        "take feedback","take appointment","give enquiry","give feedback",
        "enquiry data","appointment data","feedback data","enquiry dataset",
        "appointment dataset","feedback dataset",
    }
    _list_name = extract_name(query) or _extract_name_lower(query)
    if (intent == "list_all" or any(t in q_low for t in LIST_TRIGGERS)) and not _list_name:
        target_src = ("Feedback"    if "feedback"    in q_low else
                      "Appointment" if "appointment"  in q_low else
                      "Enquiry")
        all_rows = retriever.get_all_by_source(target_src)
        if not all_rows:
            return f"No {target_src} records found."
        limit = min(10, len(all_rows))
        lines = [f"Showing {limit} of {len(all_rows)} {target_src} records:\n"]
        for i, r in enumerate(all_rows[:limit], 1):
            name   = find_val(r,"customer","name") or "?"
            eid_v  = find_val(r,"enquiry id","ENQUIRY ID") or "?"
            status = find_val(r,"status") or "?"
            if target_src == "Enquiry":
                veh = find_val(r,"vehicle","model","car") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} - {veh} - Status: {status}")
            elif target_src == "Appointment":
                adate = find_val(r,"appointment date","date") or "?"
                atime = find_val(r,"time") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} - {adate} {atime} - Status: {status}")
            else:
                rating = find_val(r,"rating") or "?"
                fb     = (find_val(r,"feedback") or "?")[:40]
                lines.append(f"  {i:>2}. [{eid_v}] {name} - Rating: {rating} - \"{fb}\"")
        if len(all_rows) > 10:
            lines.append(f"\n  … and {len(all_rows)-10} more. Ask for a specific customer or filter.")
        return "\n".join(lines)

    # ── BAD FEEDBACK ──────────────────────────────────────────────────────────
    if intent == "bad_feedback":
        bad = [r for r in retriever.get_all_by_source("Feedback")
               if _rating(r) is not None and _rating(r) <= 2]
        if bad:
            lines = [f"Customers with low ratings (≤2/5) — {len(bad)} found:\n"]
            for r in bad:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— Rating: {find_val(r,'rating')}/5 "
                    f"— \"{find_val(r,'feedback') or '?'}\"")
            return "\n".join(lines)
        return "No customers with very low ratings (≤2) found."

    # ── GOOD FEEDBACK ─────────────────────────────────────────────────────────
    if intent == "good_feedback":
        good = [r for r in retriever.get_all_by_source("Feedback")
                if _rating(r) is not None and _rating(r) >= 4]
        if good:
            lines = [f"Customers with high ratings (≥4/5) — {len(good)} found:\n"]
            for r in good:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— Rating: {find_val(r,'rating')}/5 "
                    f"— \"{find_val(r,'feedback') or '?'}\"")
            return "\n".join(lines)
        return "No customers with high ratings (≥4) found."

    # ── CANCELLED ─────────────────────────────────────────────────────────────
    if intent == "cancelled":
        rows = [r for r in retriever.get_all_by_source("Appointment")
                if "cancel" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"Cancelled appointments — {len(rows)} found:\n"]
            for r in rows:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— {find_val(r,'appointment date','date') or '?'} "
                    f"at {find_val(r,'time') or 'N/A'} "
                    f"— {find_val(r,'vehicle','car','model') or 'N/A'}")
            return "\n".join(lines)
        return "No cancelled appointments found."

    # ── COMPLETED ─────────────────────────────────────────────────────────────
    if intent == "completed":
        rows = [r for r in retriever.get_all_by_source("Appointment")
                if "complet" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"Completed appointments — {len(rows)} found:\n"]
            for r in rows:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— {find_val(r,'appointment date','date') or '?'} "
                    f"at {find_val(r,'time') or 'N/A'}")
            return "\n".join(lines)
        return "No completed appointments found."

    # ── APPOINTMENT ───────────────────────────────────────────────────────────
    if intent == "appointment":
        appt = [r for r in results if get_source(r) == "Appointment"]
        row  = appt[0] if appt else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        adate    = find_val(row,"appointment date","date") or "N/A"
        atime    = find_val(row,"time") or "N/A"
        status   = find_val(row,"status") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        followup = {
            "Scheduled": "→ Confirmed. Please arrive 10 minutes early.",
            "Completed": "→ Already completed successfully.",
            "Cancelled": "→ Cancelled. Please call to reschedule.",
        }.get(status, "")
        return (f"Appointment for {customer} (ID: {rid}):\n"
                f"  Date   : {adate}\n"
                f"  Time   : {atime}\n"
                f"  Vehicle: {veh}\n"
                f"  Status : {status}  {followup}")

    # ── FEEDBACK ──────────────────────────────────────────────────────────────
    if intent == "feedback":
        fb_rows = [r for r in results if get_source(r) == "Feedback"]
        row     = fb_rows[0] if fb_rows else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        fb       = find_val(row,"feedback") or "N/A"
        rating   = find_val(row,"rating") or "N/A"
        date     = find_val(row,"date") or "N/A"
        try:
            r_int = int(float(rating))
            bar   = "★" * r_int + "☆" * (5 - r_int)
        except Exception:
            bar = ""
        return (f"Feedback from {customer} (ID: {rid}):\n"
                f"  Comment : \"{fb}\"\n"
                f"  Rating  : {rating}/5  [{bar}]\n"
                f"  Date    : {date}")

    # ── CONTACT ───────────────────────────────────────────────────────────────
    if intent == "contact":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        email    = find_val(row,"email") or "N/A"
        phone    = find_val(row,"phone","mobile","number") or "N/A"
        return (f"Contact details for {customer}:\n"
                f"  Phone : {phone}\n"
                f"  Email : {email}")

    # ── STATUS ────────────────────────────────────────────────────────────────
    if intent == "status":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        status   = find_val(row,"status") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        return (f"Enquiry status for {customer} (ID: {rid}):\n"
                f"  Vehicle : {veh}\n"
                f"  Status  : {status}")

    # ── PAYMENT ───────────────────────────────────────────────────────────────
    if intent == "payment":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        payment  = find_val(row,"payment") or "N/A"
        return f"Payment preference for {customer}: {payment}"

    # ── TEST RIDE ─────────────────────────────────────────────────────────────
    if intent == "test_ride":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        ride     = find_val(row,"test ride","test_ride") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        return (f"Test ride status for {customer}:\n"
                f"  Vehicle   : {veh}\n"
                f"  Test Ride : {ride}")

    # ── VEHICLE ───────────────────────────────────────────────────────────────
    if intent == "vehicle":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        return f"{customer} (ID: {rid}) is interested in the {veh}."

    # ── CITY FILTER ───────────────────────────────────────────────────────────
    if intent == "city":
        city_re = re.search(
            r'\b(hyderabad|bangalore|bengaluru|chennai|mumbai|delhi|pune|coimbatore|'
            r'kolkata|ahmedabad|surat|jaipur|lucknow|nagpur)\b', q_low)
        if city_re:
            cn   = city_re.group(1).title()
            rows = [r for r in retriever.get_all_by_source("Enquiry")
                    if cn.lower() in str(find_val(r,"city","location","state") or "").lower()]
            if rows:
                lines = [f"Customers from {cn} — {len(rows)} found:\n"]
                for r in rows:
                    lines.append(
                        f"  * {find_val(r,'customer','name') or '?'} "
                        f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                        f"— {find_val(r,'vehicle','model') or '?'} "
                        f"— Status: {find_val(r,'status') or '?'}")
                return "\n".join(lines)

    # ── NEW LEADS ─────────────────────────────────────────────────────────────
    if intent == "new_lead":
        ALL_ENQUIRY_TRIGGERS = {"take","give","show","fetch","get","pull","dataset",
                                "data","records","all","entire","full","list","dump"}
        q_words_set = set(re.findall(r'\w+', q_low))
        if (q_words_set & ALL_ENQUIRY_TRIGGERS) and not (extract_name(query) or _extract_name_lower(query)):
            all_rows = retriever.get_all_by_source("Enquiry")
            limit    = min(10, len(all_rows))
            lines    = [f"Showing {limit} of {len(all_rows)} Enquiry records:\n"]
            for i, r in enumerate(all_rows[:limit], 1):
                name   = find_val(r,"customer","name") or "?"
                eid_v  = find_val(r,"enquiry id","ENQUIRY ID") or "?"
                veh    = find_val(r,"vehicle","model","car") or "?"
                status = find_val(r,"status") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} — {veh} — Status: {status}")
            if len(all_rows) > 10:
                lines.append(f"\n  … and {len(all_rows)-10} more.")
            return "\n".join(lines)

        rows = [r for r in retriever.get_all_by_source("Enquiry")
                if "new" in str(find_val(r,"customer type") or "").lower()
                or "new lead" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"New customer leads — {len(rows)} found:\n"]
            for r in rows[:10]:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                    f"— {find_val(r,'vehicle','model') or '?'} "
                    f"— Status: {find_val(r,'status') or '?'}")
            return "\n".join(lines)

    # ── SUMMARY / FULL DETAILS (default) ─────────────────────────────────────
    enq_rows  = [r for r in results if get_source(r) == "Enquiry"]
    appt_rows = [r for r in results if get_source(r) == "Appointment"]
    fb_rows   = [r for r in results if get_source(r) == "Feedback"]

    if enq_rows or appt_rows or fb_rows:
        lines = []
        if enq_rows:
            lines.append("── Enquiry " + "─"*40)
            for col, val in enq_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        if appt_rows:
            lines.append("── Appointment " + "─"*36)
            for col, val in appt_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        if fb_rows:
            lines.append("── Feedback " + "─"*39)
            for col, val in fb_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        return "\n".join(lines) if lines else "No details found."

    # Last resort
    top   = results[0]
    src   = get_source(top)
    lines = [f"Best match from {src}:"]
    for col, val in top.items():
        if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
            lines.append(f"  {col}: {val}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────── 6. CHATBOT CLASS

class IntelligentSalesChatbot:
    def __init__(self, retriever: IntelligentRetriever):
        self.retriever = retriever
        self.history   = []
        self.show_ctx  = False
        self.top_k     = TOP_K
        self.use_llm   = True

    def chat(self, raw_query: str) -> tuple:
        t0    = time.time()
        query = normalize_query(raw_query.strip())

<<<<<<< HEAD
        queried_name = extract_name(query) or _extract_name_lower(query)

        # ── Conversation continuity ──────────────────────────────────────
        # If THIS message has no name of its own but looks like a short
        # clarification of what we were just discussing — "I mean
        # 3/28/2026", "for the Seltos one", "what is HER enquiry ID" —
        # reuse the customer name from the immediately preceding user turn
        # instead of running a nameless, unscoped search. Bounded to just
        # the last turn (not the whole history) so it can't keep dragging
        # an old customer into an unrelated later question.
        if not queried_name and len(self.history) >= 2:
            q_tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
            CONTINUATION_HINTS = {
                "this", "that", "it", "same", "again", "there",
                "he", "she", "him", "her", "his", "their", "they", "them",
            }
            looks_like_clarification = (
                bool(q_tokens & CONTINUATION_HINTS)
                or bool(re.search(r"\d", query))
                or len(q_tokens) <= 6
            )
            if looks_like_clarification:
                prev_role, prev_text = self.history[-2]
                if prev_role == "You":
                    queried_name = extract_name(prev_text) or _extract_name_lower(prev_text)

        # Feed the resolved name (including anything carried over above)
        # into intent detection too — otherwise a pronoun-only follow-up
        # like "what is HER enquiry ID?" has no name of its own from
        # detect_intent's point of view, and the "has_name -> narrow to
        # a single summary" rules below never fire, so it falls through
        # to a full, unscoped "new_lead" style dump instead of answering
        # about the specific customer.
        intent, conf = detect_intent(query, known_name=queried_name)

=======
        intent, conf = detect_intent(query)
        queried_name = extract_name(query) or _extract_name_lower(query)

>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
        # ── Route to the correct dataset ─────────────────────────────────────
        target_source = INTENT_TO_SOURCE.get(intent)  # None = all datasets

        if target_source:
            # Primary source search
            results = self.retriever.retrieve_from_source(
                query, target_source, top_k=self.top_k)
            # Boost with name match WITHIN the same source
            if queried_name:
                name_rows = self.retriever.retrieve_by_name(
                    queried_name, preferred_source=target_source, top_k=self.top_k)
                results = name_rows + [r for r in results if r not in name_rows]
        else:
            # Cross-source search (summary, list_all, or explicit ENQ-ID)
            results = self.retriever.retrieve(query, top_k=self.top_k)
            if queried_name:
                # For summary, search all sources but separate by source
                name_rows = self.retriever.retrieve_by_name(
                    queried_name, top_k=self.top_k * 2)
                results = name_rows + [r for r in results if r not in name_rows]

        # Supplement for specialised list intents
        if intent in ("feedback", "bad_feedback", "good_feedback"):
            extra = self.retriever.retrieve_from_source(
                "customer feedback rating review satisfied unhappy complaint",
                "Feedback", top_k=self.top_k)
            results += [r for r in extra if r not in results]
        elif intent in ("cancelled", "completed", "appointment"):
            extra = self.retriever.retrieve_from_source(
                "appointment meeting booking scheduled cancelled completed",
                "Appointment", top_k=self.top_k)
            results += [r for r in extra if r not in results]

        # Deduplicate
        seen, deduped = set(), []
        for r in results:
            key = (r.get("__source__"),
                   str(find_val(r, "enquiry id", "ENQUIRY ID") or ""))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped

<<<<<<< HEAD
        # ── Explicit date in the query ────────────────────────────────────
        # A name alone ("Meera") is often ambiguous — this dataset has
        # several different Meera enquiries/appointments. If the user gave
        # a specific date ("...on 3/28/2026"), narrow down to ONLY the
        # record(s) whose Enquiry/Appointment/Feedback date matches it,
        # instead of letting an unrelated same-name record slip through
        # and get blended into the answer. If nothing matches, say so
        # plainly rather than guessing.
        query_date = extract_query_date(query)
        if query_date:
            date_filtered = [r for r in results if _row_matches_date(r, query_date)]
            if date_filtered:
                results = date_filtered
            else:
                elapsed = time.time() - t0
                who = f" for {queried_name}" if queried_name else ""
                answer = (
                    f"I couldn't find any record{who} on {query_date}. "
                    f"Double-check the date, or ask without a specific date "
                    f"to see all matching records."
                )
                self.history.append(("You", raw_query))
                self.history.append(("Assistant", answer))
                return answer, elapsed, "no_match"

=======
>>>>>>> f1f6b1dab05ace388b4c1ad720a4a491947a3b91
        if self.show_ctx:
            print(f"\n  [Intent: '{intent}'  confidence={conf:.3f}  target: {target_source}]")
            print("  [Top retrieved records]")
            for i, r in enumerate(results[:5], 1):
                preview = "  ".join(
                    f"{k}:{str(v)[:20]}" for k, v in r.items()
                    if not str(k).startswith("__") and str(v) not in ("nan","None","")
                )[:120]
                print(f"    {i}. score={r.get('__score__',0):.3f} | "
                      f"src={r.get('__source__')} | {preview}")
            print()
            self.show_ctx = False

        # Build structured answer
        structured_facts = build_answer(query, results, intent, self.retriever)

        # Optionally rephrase with LLM
        generated = (_generate_phrase(query, structured_facts, intent)
                     if self.use_llm else None)
        if generated:
            LIST_INTENTS = {"list_all","bad_feedback","good_feedback","cancelled",
                            "completed","new_lead","city","returning"}
            answer = (generated + "\n\n" + structured_facts
                      if intent in LIST_INTENTS else generated)
        else:
            answer = structured_facts

        elapsed = time.time() - t0
        self.history.append(("You", raw_query))
        self.history.append(("Assistant", answer))
        return answer, elapsed, intent


# ─────────────────────────────────────────────────────────── 7. DATA LOADING

def load_datasets() -> dict:
    """Return {source_name: pd.DataFrame} for each CSV."""
    dfs = {}
    for source, path in CSV_FILES.items():
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found — skipping.")
            continue
        df = pd.read_csv(path)
        df.columns      = [c.strip() for c in df.columns]
        df["__source__"] = source
        dfs[source]      = df
        print(f"  OK {source:<12}: {len(df)} records")
    if not dfs:
        sys.exit("[ERROR] No CSV files found.")
    return dfs


def build_docs_per_source(dfs: dict) -> dict:
    """Return {source_name: [rich_text, …]}."""
    docs = {}
    for src, df in dfs.items():
        docs[src] = [row_to_rich_text(row.to_dict(), src) for _, row in df.iterrows()]
    return docs


def _cache_fresh(path: str, dfs: dict) -> bool:
    if not os.path.exists(path):
        return False
    ct = os.path.getmtime(path)
    return all(
        not os.path.exists(p) or os.path.getmtime(p) <= ct
        for p in CSV_FILES.values()
    )


# ──────────────────────────────────────────────────────────────── 8. CLI

BANNER = """
This is a Intelligent sales chatbot
"""


def run_cli(chatbot: IntelligentSalesChatbot):
    print(BANNER)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not user_input:
            continue
        low = user_input.lower()

        if   low == "/quit":
            print("Goodbye!"); break
        elif low == "/llm":
            chatbot.use_llm = not chatbot.use_llm
            print(f"  [LLM generation: {'ON' if chatbot.use_llm else 'OFF'}]\n")
        elif low == "/reset":
            chatbot.history.clear()
            print("  [History cleared]\n")
        elif low == "/history":
            if not chatbot.history:
                print("  [No history yet]\n")
            else:
                for role, text in chatbot.history:
                    print(f"  {role}: {text}")
                print()
        elif low == "/context":
            chatbot.show_ctx = True
            print("  [Context + intent shown for next query]\n")
        elif low.startswith("/topk"):
            try:
                chatbot.top_k = int(user_input.split()[1])
                print(f"  [Top-K set to {chatbot.top_k}]\n")
            except (IndexError, ValueError):
                print("  Usage: /topk 3\n")
        else:
            answer, elapsed, intent = chatbot.chat(user_input)
            print(f"\nAssistant [{intent}]: {answer}")
            print(f"  ({elapsed:.2f}s)\n")


# ──────────────────────────────────────────────────────────────── MAIN

def main():
    global _KNOWN_NAMES_SET

    print("\n[1/3] Loading datasets …")
    dfs = load_datasets()

    # Populate dynamic name set from all datasets
    for src_df in dfs.values():
        for col in src_df.columns:
            if "name" in col.lower() and not col.startswith("__"):
                _KNOWN_NAMES_SET.update(
                    n for n in src_df[col].dropna().astype(str).unique()
                    if n not in ("nan","None","")
                )
    print(f"  OK Known customer names: {sorted(_KNOWN_NAMES_SET)}")

    print("\n[2/3] Building per-dataset retrievers …")
    docs_per_source = build_docs_per_source(dfs)

    if _cache_fresh(INDEX_CACHE, dfs):
        print(f"  Loading from cache: {INDEX_CACHE}")
        retriever = IntelligentRetriever.load(INDEX_CACHE)
        print("  OK Cache loaded")
    else:
        retriever = IntelligentRetriever(dfs, docs_per_source)
        retriever.save(INDEX_CACHE)
        print(f"  OK Index saved to {INDEX_CACHE}")

    print("\n[3/3] Starting up …")
    print(f"  Device : {LLM_DEVICE}")
    print(f"  Models  : {' → '.join(CHAT_MODELS)}")
    import threading
    threading.Thread(target=_load_llm, daemon=True).start()
    print("  [LLM] Loading in background (ready in ~20-40 s) …\n")

    chatbot = IntelligentSalesChatbot(retriever)
    run_cli(chatbot)


if __name__ == "__main__":
    main()