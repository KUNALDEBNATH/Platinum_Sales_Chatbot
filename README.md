# Sales RAG Chatbot — Platinum Software

A retrieval-augmented generation (RAG) chatbot built for automotive sales teams. It reads three CSV datasets — Enquiry, Appointment, and Feedback — builds a TF-IDF search index, and answers natural-language questions about customers, vehicles, appointments, and feedback. An optional fine-tuned language model (Qwen2.5) can be layered on top to rephrase structured answers into plain English sentences.

---

## Table of Contents

1. Project Overview
2. How It Works
3. File Structure
4. Requirements
5. Installation
6. Preparing the Data
7. Training the Language Model (Optional)
8. Running the Chatbot
9. Running the Web API
10. Using the Web Interface
11. API Endpoints
12. CLI Commands
13. Intent Reference
14. Configuration Reference
15. Troubleshooting

---

## 1. Project Overview

The chatbot is built around two layers.

The first layer is the retrieval engine. When a user asks a question, the system detects the intent (for example, "show cancelled appointments" or "what feedback did Divya give"), selects the relevant dataset, and runs a TF-IDF cosine similarity search to find the most relevant records. The results are then formatted into a structured plain-text answer.

The second layer is an optional language model. If a Qwen2.5 model is available (either downloaded from Hugging Face or fine-tuned locally using `train.py`), the structured answer is passed to the model, which rephrases it into one or two natural English sentences. If no model is available, the structured answer is returned directly.

The web interface is a single HTML page served by a Django backend. It connects to the chatbot via a REST API and provides a sidebar of quick-query suggestions, a typing indicator, and intent tagging on each response.

---

## 2. How It Works

**Step 1 — Data loading.**
The system reads the three CSV files at startup. Each row is converted into a rich text document that concatenates all field values with their column names. This document is what gets searched.

**Step 2 — Index building.**
A TF-IDF vectorizer is fitted on all documents from all three datasets. Separate vectorizers are also fitted per source so that intent-targeted queries can search only the relevant dataset. The fitted index is saved to `rag_index.pkl` so that subsequent startups skip the rebuild step unless the CSVs have changed.

**Step 3 — Query processing.**
Each incoming query goes through the following pipeline:
- Text normalisation (lowercasing, expanding contractions, standardising punctuation).
- Intent detection using keyword matching and a confidence score.
- Name extraction using a known-names set populated from the CSV data, plus fuzzy matching as a fallback.
- Record retrieval from the appropriate dataset using cosine similarity.
- Structured answer building from the retrieved records.
- Optional LLM rephrasing.

**Step 4 — Response delivery.**
In CLI mode the answer is printed to the terminal. In API mode it is returned as a JSON object containing the answer text, the detected intent, and the elapsed time.

---

## 3. File Structure

```
project/
|
|-- api.py                        Django REST API server
|-- test.py                       RAG engine, chatbot class, and CLI entry point
|-- train.py                      Fine-tuning pipeline for the optional LLM
|-- index.html                    Web chat interface (served by api.py)
|-- requirements.txt              Python dependencies
|
|-- sales_enquiry_dataset.csv     Customer enquiry records
|-- sales_appointment_dataset.csv Appointment records
|-- sales_feedback_dataset.csv    Feedback and rating records
|
|-- rag_index.pkl                 Auto-generated TF-IDF index cache (created at runtime)
|-- sales_llm_model/              Fine-tuned model directory (created by train.py)
|-- model.pkl                     Pickle bundle of the fine-tuned model (created by train.py)
```

---

## 4. Requirements

**Python version:** 3.9 or later is recommended.

**Hardware for inference:**
- CPU-only is supported. Responses will be slower on CPU when the LLM is active.
- A CUDA-capable GPU with at least 4 GB of VRAM is recommended for comfortable LLM inference.

**Hardware for training:**
- Training on CPU is possible but will take a very long time.
- A GPU with at least 6 GB of VRAM is recommended. The pipeline was developed on a system with an RTX 4060 (8 GB VRAM).

**Python packages:**

```
torch >= 2.1.0
transformers >= 4.40.0
datasets >= 2.18.0
accelerate >= 0.29.0
pandas >= 2.0.0
scikit-learn
numpy
django
djangorestframework
django-cors-headers
```

The `requirements.txt` file in the repository covers the ML and data dependencies. The Django stack must be installed separately (see Section 5).

---

## 5. Installation

**Step 1 — Clone the repository or copy the project files into a folder.**

```
cd your-project-folder
```

**Step 2 — Create and activate a virtual environment (recommended).**

```
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac / Linux
source venv/bin/activate
```

**Step 3 — Install ML dependencies.**

```
pip install -r requirements.txt
```

**Step 4 — Install the web API dependencies.**

```
pip install django djangorestframework django-cors-headers
```

**Step 5 — Verify the installation.**

```
python -c "import torch; import transformers; import django; print('OK')"
```

If you see `OK`, the environment is ready.

---

## 6. Preparing the Data

The system expects three CSV files in the same directory as `test.py` and `api.py`. The file names must match exactly as shown below.

**sales_enquiry_dataset.csv**

Required columns:

| Column | Description |
|---|---|
| ENQUIRY ID | Unique identifier, e.g. ENQ001 |
| Customer Name | Full name of the customer |
| Phone Number | Contact number |
| Email | Email address |
| Gender | Male / Female / Other |
| Vehicle Name / Model | The vehicle the customer is interested in |
| Enquiry Source | Walk-in, Phone, Online, etc. |
| Enquiry Date | Date of the enquiry |
| Appointment Date | Scheduled appointment date |
| City / State | Customer location |
| Customer Type | New / Returning |
| Payment Type | Cash / Finance / Exchange |
| Test Ride Taken | Yes / No |
| Status | New Lead / In Progress / Closed / etc. |

**sales_appointment_dataset.csv**

Required columns:

| Column | Description |
|---|---|
| Enquiry ID | Links back to the enquiry record |
| Appointment Date | Date of the appointment |
| Time | Time of the appointment |
| Customer Name | Full name |
| Vehicle | Vehicle discussed in the appointment |
| Status | Scheduled / Completed / Cancelled |

**sales_feedback_dataset.csv**

Required columns:

| Column | Description |
|---|---|
| Enquiry ID | Links back to the enquiry record |
| Customer Name | Full name |
| Feedback | Free-text feedback comment |
| Rating | Numeric score, typically 1 to 5 |
| Date | Date the feedback was submitted |

If any CSV file is missing, the system will skip it with a warning and continue with the remaining files. If all three are missing, the system will exit with an error.

---

## 7. Training the Language Model (Optional)

The LLM layer is entirely optional. If you skip this section, the chatbot will still work and return structured answers. The LLM only adds natural-language rephrasing on top.

The training script fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` on:
- Question-answer pairs generated from your own CSV data.
- A sample of publicly available instruction datasets (Dolly-15k, Alpaca, and others) downloaded from Hugging Face to improve general language quality.

**To start training:**

```
python train.py
```

Training will proceed through the following stages:
1. CUDA detection and GPU diagnostics.
2. Tokenizer loading from Hugging Face.
3. Generation of Q-A pairs from the three CSV files.
4. Download and merging of Hugging Face augmentation data.
5. Tokenisation of the combined dataset.
6. Model loading and fine-tuning using the Hugging Face Trainer.
7. Saving the fine-tuned model to `./sales_llm_model/`.
8. Saving a pickle bundle to `./model.pkl`.

**Training configuration (editable at the top of `train.py`):**

| Parameter | Default | Description |
|---|---|---|
| MODEL_NAME | Qwen/Qwen2.5-0.5B-Instruct | Base model to fine-tune |
| OUTPUT_DIR | ./sales_llm_model | Where to save the fine-tuned model |
| MAX_LENGTH | 512 | Maximum token length per sample |
| NUM_EPOCHS | 5 | Number of training epochs |
| BATCH_SIZE | 4 | Per-device training batch size |
| GRAD_ACCUMULATE | 2 | Gradient accumulation steps |
| LR | 2e-5 | Learning rate |
| HF_AUGMENT_N | 3000 | Number of rows to pull from HuggingFace datasets |

If you run out of GPU memory, reduce `BATCH_SIZE` to 2 and increase `GRAD_ACCUMULATE` to 4.

**After training**, open `test.py` and prepend the local model path to `CHAT_MODELS`:

```python
CHAT_MODELS = [
    "./sales_llm_model",          # fine-tuned local model
    "Qwen/Qwen2.5-1.5B-Instruct", # fallback
    "Qwen/Qwen2.5-0.5B-Instruct", # fallback
]
```

The system will try each entry in order and use the first one that loads successfully.

---

## 8. Running the Chatbot

To run the chatbot directly in the terminal without the web interface:

```
python test.py
```

The system will:
1. Load all CSV datasets.
2. Build or load the TF-IDF index.
3. Start loading the LLM in the background.
4. Drop you into an interactive CLI prompt.

You can start asking questions immediately. The LLM will become active in the background after roughly 20 to 40 seconds depending on your hardware. Until then, structured answers are returned.

**Example questions to try in the CLI:**

```
You: Show ENQ001 full details
You: Who gave bad feedback?
You: All cancelled appointments
You: Show customers from Chennai
You: What car did Sneha enquire about?
You: Is Arjun's appointment confirmed?
You: Returning customers
You: Who hasn't taken a test ride?
```

---

## 9. Running the Web API

To start the Django web server:

```
python api.py
```

By default this starts the server on `http://0.0.0.0:8000`.

To use a different port:

```
python api.py runserver 0.0.0.0:9000
```

On startup, the API server will:
1. Import the RAG engine from `test.py`.
2. Load all datasets.
3. Build or load the TF-IDF index from `rag_index.pkl`.
4. Start loading the LLM in a background thread.
5. Begin accepting requests.

You can confirm the server is running by visiting `http://localhost:8000/api/health/` in your browser.

---

## 10. Using the Web Interface

Once the API server is running, open a browser and go to:

```
http://localhost:8000
```

The interface has three sections.

**Status pill (top right):** Shows the current system state. It reads "Initialising" while the server is starting, "RAG ready (LLM loading)" once the index is ready, and "LLM + RAG ready" once the language model is also loaded.

**Sidebar (left):** Lists pre-built quick query buttons. Clicking any button sends that query immediately.

**Chat pane (centre):** The main conversation area. Type a question in the text box at the bottom and press Enter or click the send button. Each response shows the detected intent as a tag and the response time in seconds.

**Reset button (top right):** Clears the conversation history on both the client and the server.

---

## 11. API Endpoints

All endpoints are served by the Django application in `api.py`.

**GET /api/health/**

Returns the current system status.

Response:
```json
{
  "status": "ok",
  "chatbot_ready": true,
  "llm_ready": true,
  "datasets": ["Enquiry", "Appointment", "Feedback"],
  "init_error": null,
  "index_cache": "/absolute/path/to/rag_index.pkl"
}
```

**POST /api/chat/**

Sends a query and returns an answer.

Request body:
```json
{
  "query": "Show ENQ001 details"
}
```

Response:
```json
{
  "answer": "ENQ001 belongs to ...",
  "intent": "lookup",
  "elapsed": 0.312,
  "history_length": 4
}
```

**POST /api/reset/**

Clears the conversation history.

Response:
```json
{
  "status": "ok"
}
```

**GET /api/suggestions/**

Returns the list of example queries shown in the sidebar.

Response:
```json
{
  "suggestions": [
    "Show all enquiries",
    "Who gave bad feedback?",
    ...
  ]
}
```

**GET /**

Serves the `index.html` web interface.

---

## 12. CLI Commands

When running `python test.py` in CLI mode, the following special commands are available in addition to regular questions.

| Command | Description |
|---|---|
| /quit | Exit the chatbot |
| /reset | Clear the conversation history |
| /history | Print all previous turns in the conversation |
| /llm | Toggle LLM rephrasing on or off |
| /context | Show the detected intent and top retrieved records for the next query only |
| /topk N | Set the number of records retrieved per query to N (default: 8) |

---

## 13. Intent Reference

The system classifies every query into one of the following intents before searching. Understanding the intents helps you phrase queries more effectively.

| Intent | Triggered by | Dataset searched |
|---|---|---|
| lookup | ENQ ID, "details", "full info" | All |
| list_all | "show all", "list all" | All |
| appointment | "appointment", "meeting", "booking" | Appointment |
| cancelled | "cancelled" | Appointment |
| completed | "completed" | Appointment |
| feedback | "feedback", "review", "comment" | Feedback |
| bad_feedback | "bad feedback", "unhappy", "negative", "poor" | Feedback |
| good_feedback | "good feedback", "satisfied", "positive" | Feedback |
| contact | "phone", "email", "contact" | Enquiry |
| status | "status", "lead", "in progress" | Enquiry |
| payment | "payment", "finance", "cash", "exchange" | Enquiry |
| test_ride | "test ride" | Enquiry |
| city | "from Chennai", "city", "location" | Enquiry |
| returning | "returning customer", "existing customer" | Enquiry |
| new_lead | "new lead", "fresh enquiry" | Enquiry |
| summary | "summary", "overview", "tell me about" | All |

---

## 14. Configuration Reference

The following settings can be changed directly in `test.py`.

| Variable | Default | Description |
|---|---|---|
| TOP_K | 8 | Number of records to retrieve per query |
| INDEX_CACHE | rag_index.pkl | File path for the saved TF-IDF index |
| LLM_MODEL_DIR | sales_llm_model | Directory for a locally saved model |
| LLM_MAX_TOKENS | 80 | Maximum tokens the LLM generates per response |
| LLM_TEMPERATURE | 0.3 | Sampling temperature (lower = more deterministic) |
| LLM_TOP_P | 0.9 | Nucleus sampling probability |
| CHAT_MODELS | [list of Qwen paths] | Models tried in order; first successful one is used |
| CSV_FILES | dict of three paths | Paths to the three CSV datasets |

---

## 15. Troubleshooting

**The server starts but the chatbot status stays on "Initialising".**
Check the terminal where `api.py` is running for error messages. The most common cause is a missing CSV file. Make sure all three CSV files are in the same directory as `api.py`.

**The LLM never loads / stays on "RAG ready (LLM loading)".**
The LLM loads in a background thread after the index is ready. On CPU this can take several minutes. If it never completes, check that Hugging Face can connect to the internet to download the model, or that the local model path in `CHAT_MODELS` is correct.

**Out of memory during training.**
Reduce `BATCH_SIZE` in `train.py` from 4 to 2, and increase `GRAD_ACCUMULATE` from 2 to 4. The effective batch size remains the same but peak VRAM usage is lower.

**The index is rebuilt every time the server starts.**
This happens when the CSV files have a newer modification timestamp than `rag_index.pkl`. Touch the CSVs only when their content has actually changed. If you want to force a rebuild, delete `rag_index.pkl` manually.

**Answers are returning as structured text instead of natural sentences.**
The LLM has not finished loading yet, or it failed to load. Check the terminal for lines beginning with `[LLM]`. If the model failed, a fallback message such as "No model available — using structured answers only" will appear.

**CORS errors when accessing the API from a different origin.**
The Django settings in `api.py` have `CORS_ALLOW_ALL_ORIGINS = True` by default, so CORS should not be an issue in development. If you deploy behind a reverse proxy, ensure the proxy passes the `Origin` header through.

**The page at / returns "index.html not found".**
The file `index.html` must be in the same directory as `api.py`. If you run `api.py` from a different working directory, use an absolute path or change into the project directory before running the server.

---

## License

This project was developed for internal use at Platinum Software. Please contact the project owner before redistributing or modifying for other purposes.
