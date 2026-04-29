
import os
import sys
import pickle
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)


MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR  = "./sales_llm_model"
PKL_PATH    = "./model.pkl"

MAX_LENGTH      = 512          
NUM_EPOCHS      = 5
BATCH_SIZE      = 4            
GRAD_ACCUMULATE = 2          
LR              = 2e-5         
WARMUP_STEPS    = 100
HF_AUGMENT_N    = 3000
SEED            = 42

CUDNN_BENCHMARK     = True
CUDNN_DETERMINISTIC = False
VRAM_WARN_MB        = 512

ENQUIRY_CSV     = "sales_enquiry_dataset.csv"
APPOINTMENT_CSV = "sales_appointment_dataset.csv"
FEEDBACK_CSV    = "sales_feedback_dataset.csv"


def make_enquiry_pairs(path: str) -> list:
    
    df = pd.read_csv(path).fillna("N/A")
    pairs = []
    for _, r in df.iterrows():
        eid    = r["ENQUIRY ID"]          
        name   = r["Customer Name"]
        phone  = r["Phone Number"]
        email  = r["Email"]
        gender = r["Gender"]
        veh    = r["Vehicle Name / Model"]
        src    = r["Enquiry Source"]
        edate  = r["Enquiry Date"]
        adate  = r["Appointment Date"]
        city   = r["City / State"]
        ctype  = r["Customer Type"]
        pay    = r["Payment Type"]
        ride   = r["Test Ride Taken"]
        status = r["Status"]
        ride_str = "has taken" if str(ride).strip() == "Yes" else "has not taken"

        pairs += [
            {
                "prompt":   f"What is the status of enquiry {eid}?",
                "response": (f"Enquiry {eid} for {name} regarding the {veh} is currently "
                             f"'{status}'. Received on {edate} via {src}. "
                             f"Appointment: {adate}. Payment: {pay}."),
            },
            {
                "prompt":   f"Pull up enquiry details for {name}.",
                "response": (f"{name} ({gender}, {ctype} customer) from {city} enquired "
                             f"about the {veh} on {edate} via {src}. "
                             f"Contact: {phone} / {email}. Payment: {pay}. Status: {status}."),
            },
            {
                "prompt":   f"Has {name} taken a test ride for {eid}?",
                "response": f"{name} {ride_str} a test ride for the {veh} (enquiry {eid}).",
            },
            {
                "prompt":   f"Show enquiries for {veh} from {city}.",
                "response": (f"Found {eid}: {name} from {city} is interested in the {veh}. "
                             f"Source: {src}, Date: {edate}, Status: {status}, Payment: {pay}."),
            },
            {
                "prompt":   f"What do you know about {name} with ID {eid}?",
                "response": (f"{name} is a {ctype} customer from {city} who enquired about the "
                             f"{veh} via {src} on {edate}. Email: {email}, Phone: {phone}. "
                             f"Test ride: {ride}. Payment: {pay}. Status: {status}."),
            },
        ]
    return pairs


def make_appointment_pairs(path: str) -> list:
   
    df = pd.read_csv(path).fillna("N/A")
    pairs = []
    for _, r in df.iterrows():
        eid    = r["Enquiry ID"]          
        adate  = r["Appointment Date"]
        atime  = r["Time"]
        name   = r["Customer Name"]
        veh    = r["Vehicle"]
        status = r["Status"]

        follow_up = {
            "Scheduled": "Please arrive 10 minutes early.",
            "Completed": "This appointment has been successfully completed.",
            "Cancelled": "Appointment was cancelled. Please call to reschedule.",
        }.get(str(status), "Contact the dealership for details.")

        pairs += [
            {
                "prompt":   f"What is the appointment status for {name}?",
                "response": (f"{name}'s appointment for the {veh} (Enquiry {eid}) is {status}. "
                             f"Date: {adate} at {atime}. {follow_up}"),
            },
            {
                "prompt":   f"When is the appointment for enquiry {eid}?",
                "response": (f"Appointment for {eid} ({name}) is on {adate} at {atime}. "
                             f"Vehicle: {veh}. Status: {status}."),
            },
            {
                "prompt":   f"Is {name}'s appointment confirmed?",
                "response": (f"{name}'s appointment is '{status}' for {adate} at {atime} "
                             f"regarding the {veh}. {follow_up}"),
            },
            {
                "prompt":   f"Show appointment details for {veh}.",
                "response": (f"Appointment on {adate} at {atime}: {name} (Enquiry {eid}) "
                             f"for the {veh}. Status: {status}."),
            },
        ]
    return pairs


def make_feedback_pairs(path: str) -> list:
  
    df = pd.read_csv(path).fillna("N/A")
    pairs = []
    for _, r in df.iterrows():
        eid      = r["Enquiry ID"]         # mixed case — correct
        name     = r["Customer Name"]
        feedback = r["Feedback"]
        rating   = r["Rating"]
        date     = r["Date"]

        try:
            rating_int = int(float(rating))
        except (ValueError, TypeError):
            rating_int = 0
        sentiment = ("very positive" if rating_int >= 4 else
                     "neutral"       if rating_int == 3 else
                     "negative")

        pairs += [
            {
                "prompt":   f"What feedback did {name} leave?",
                "response": (f"{name} (Enquiry {eid}) submitted feedback on {date}: "
                             f'"{feedback}". Rating: {rating}/5 ({sentiment}).'),
            },
            {
                "prompt":   f"What is the rating for enquiry {eid}?",
                "response": (f"Enquiry {eid} received a {rating}/5 rating. "
                             f'{name} wrote: "{feedback}" on {date}.'),
            },
            {
                "prompt":   f"How satisfied was {name} with the service?",
                "response": (f"{name} gave {rating}/5, indicating a {sentiment} experience. "
                             f'Comment: "{feedback}".'),
            },
            {
                "prompt":   f"Summarise customer feedback for {eid}.",
                "response": (f"Customer: {name} | Date: {date} | Rating: {rating}/5 | "
                             f'Feedback: "{feedback}" | Sentiment: {sentiment}.'),
            },
        ]
    return pairs


def _safe_load(name: str, loader_fn, n: int) -> list:
    try:
        return loader_fn(n)
    except Exception as e:
        print(f"     [Warning] {name} failed: {e}")
        return []


def load_hf_dolly(n: int = 1200) -> list:
    print("  -> Downloading databricks-dolly-15k …")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    pairs = []
    for row in ds:
        prompt = (row.get("instruction") or "").strip()
        ctx    = (row.get("context")     or "").strip()
        resp   = (row.get("response")    or "").strip()
        if ctx:
            prompt = f"{prompt}\nContext: {ctx}"
        if prompt and resp and len(prompt) > 5 and len(resp) > 5:
            pairs.append({"prompt": prompt, "response": resp})
    print(f"     Loaded {len(pairs)} dolly rows.")
    return pairs


def load_hf_alpaca(n: int = 1000) -> list:
    print("  -> Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    pairs = []
    for row in ds:
        prompt = (row.get("instruction") or "").strip()
        inp    = (row.get("input")       or "").strip()
        resp   = (row.get("output")      or "").strip()
        if inp:
            prompt = f"{prompt}\n{inp}"
        if prompt and resp and len(prompt) > 5 and len(resp) > 5:
            pairs.append({"prompt": prompt, "response": resp})
    print(f"     Loaded {len(pairs)} alpaca rows.")
    return pairs


def load_hf_kopen(n: int = 800) -> list:
    print("  -> Downloading kyujinpy/KOpen-platypus …")
    ds = load_dataset("kyujinpy/KOpen-platypus", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    pairs = []
    for row in ds:
        prompt = (row.get("instruction") or row.get("input") or "").strip()
        resp   = (row.get("output") or "").strip()
        if prompt and resp and len(prompt) > 5 and len(resp) > 5:
            pairs.append({"prompt": prompt, "response": resp})
    print(f"     Loaded {len(pairs)} kopen rows.")
    return pairs


def load_hf_open_platypus(n: int = 800) -> list:
    print("  -> Downloading garage-bAInd/Open-Platypus …")
    ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    pairs = []
    for row in ds:
        prompt = (row.get("instruction") or "").strip()
        resp   = (row.get("output")      or "").strip()
        if prompt and resp and len(prompt) > 5 and len(resp) > 5:
            pairs.append({"prompt": prompt, "response": resp})
    print(f"     Loaded {len(pairs)} open-platypus rows.")
    return pairs


def load_hf_hh_rlhf(n: int = 600) -> list:
    print("  -> Downloading Anthropic/hh-rlhf …")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    pairs = []
    for row in ds:
        text = (row.get("chosen") or "").strip()
        if "\n\nHuman: " in text and "\n\nAssistant: " in text:
            parts = text.split("\n\nAssistant: ")
            if len(parts) >= 2:
                human = parts[-2].replace("\n\nHuman: ", "").strip()
                asst  = parts[-1].strip()
                if "\n\nHuman:" in asst:
                    asst = asst.split("\n\nHuman:")[0].strip()
                if human and asst and len(human) > 5 and len(asst) > 5:
                    pairs.append({"prompt": human, "response": asst})
    print(f"     Loaded {len(pairs)} hh-rlhf rows.")
    return pairs


def load_hf_ultrachat(n: int = 600) -> list:
    print("  -> Downloading stingning/ultrachat …")
    ds = load_dataset("stingning/ultrachat", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n * 3, len(ds))))
    pairs = []
    for row in ds:
        msgs = row.get("data", [])
        if isinstance(msgs, list) and len(msgs) >= 2:
            for i in range(0, len(msgs) - 1, 2):
                q = str(msgs[i]).strip()
                a = str(msgs[i + 1]).strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    pairs.append({"prompt": q, "response": a})
                if len(pairs) >= n:
                    break
        if len(pairs) >= n:
            break
    print(f"     Loaded {len(pairs)} ultrachat rows.")
    return pairs


def augment_with_hf(target_n: int = 3000) -> list:
    loaders = [
        ("dolly",         load_hf_dolly,         1200),
        ("kopen",         load_hf_kopen,           800),
        ("alpaca",        load_hf_alpaca,         1000),
        ("open-platypus", load_hf_open_platypus,   800),
        ("hh-rlhf",       load_hf_hh_rlhf,         600),
        ("ultrachat",     load_hf_ultrachat,        600),
    ]
    all_pairs = []
    for name, fn, n in loaders:
        if len(all_pairs) >= target_n:
            break
        result = _safe_load(name, fn, n)
        all_pairs += result
        print(f"     Running total: {len(all_pairs)} HF pairs")
    return all_pairs



def format_sample(prompt: str, response: str, tokenizer) -> str:
    
    messages = [
        {"role": "system",    "content": "You are a helpful sales assistant with access to customer enquiry, appointment, and feedback records. Answer questions about customers accurately and concisely."},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        # apply_chat_template with add_generation_prompt=False gives full turn including response
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback plain format
        eos = tokenizer.eos_token or ""
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>{eos}"
    return text


def build_hf_dataset(pairs: list, tokenizer) -> Dataset:
    texts = [format_sample(p["prompt"], p["response"], tokenizer) for p in pairs]
    return Dataset.from_dict({"text": texts})


def tokenize_fn(batch, tokenizer):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    out["labels"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in ids]
        for ids in out["input_ids"]
    ]
    return out




def setup_cuda() -> str:
    if not torch.cuda.is_available():
        print("  [Warning] CUDA not available — training on CPU (slow).")
        return "cpu"

    torch.backends.cudnn.benchmark     = CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.empty_cache()

    props      = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / (1024 ** 2)
    free_vram  = (props.total_memory - torch.cuda.memory_reserved(0)) / (1024 ** 2)

    print(f"\n{'─'*65}")
    print(f"  GPU     : {props.name}")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  Compute : {props.major}.{props.minor}")
    print(f"  VRAM    : {total_vram:.0f} MB total  |  {free_vram:.0f} MB free")
    print(f"  TF32    : enabled")
    print(f"{'─'*65}\n")

    if free_vram < VRAM_WARN_MB:
        print(f"  [Warning] Low VRAM ({free_vram:.0f} MB) — reduce BATCH_SIZE if OOM.")
    return "cuda"


def print_vram(tag: str = ""):
    if not torch.cuda.is_available():
        return
    alloc    = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0)  / (1024 ** 2)
    label    = f" [{tag}]" if tag else ""
    print(f"  VRAM{label}: allocated={alloc:.0f} MB  reserved={reserved:.0f} MB")



def save_model_pkl(model, tokenizer, path: str = PKL_PATH) -> None:
    print(f"\n  Saving model bundle → {path}")
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config":     model.config,
        "tokenizer":        tokenizer,
        "max_length":       MAX_LENGTH,
        "base_model":       MODEL_NAME,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  model.pkl saved  ({size_mb:.1f} MB)")



def main():
    print("=" * 65)
    print("  Sales LLM — Training Pipeline  [Qwen2.5 · CUDA-optimised]")
    print("=" * 65)

    # ── A. CUDA ───────────────────────────────────────────────────────────────
    print("\n[0/5] Setting up CUDA …")
    device = setup_cuda()

    # ── B. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Qwen2.5 uses <|endoftext|> as EOS; pad_token must be set separately
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── C. Local CSV pairs ────────────────────────────────────────────────────
    print("\n[2/5] Processing local CSV datasets …")
    pairs = []
    for label, fn, path in [
        ("Enquiry",     make_enquiry_pairs,     ENQUIRY_CSV),
        ("Appointment", make_appointment_pairs, APPOINTMENT_CSV),
        ("Feedback",    make_feedback_pairs,    FEEDBACK_CSV),
    ]:
        if not os.path.exists(path):
            print(f"  [Warning] {path} not found, skipping.")
            continue
        chunk  = fn(path)
        pairs += chunk
        print(f"  {label}: {len(chunk)} pairs")
    print(f"  Local total: {len(pairs)} pairs")

    # ── D. HuggingFace augmentation ───────────────────────────────────────────
    print(f"\n[3/5] Augmenting with HuggingFace data (target ~{HF_AUGMENT_N} rows) …")
    hf_pairs = augment_with_hf(target_n=HF_AUGMENT_N)
    pairs   += hf_pairs
    print(f"  Grand total: {len(pairs)} training pairs\n")

    if not pairs:
        sys.exit("[Error] No training data found. Check CSV paths.")

    print("[4/5] Tokenising dataset …")
    raw_ds = build_hf_dataset(pairs, tokenizer).train_test_split(test_size=0.1, seed=SEED)
    tok_ds = raw_ds.map(
        lambda b: tokenize_fn(b, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    print(f"  Train: {len(tok_ds['train'])}  |  Eval: {len(tok_ds['test'])}")

    print(f"\n[5/5] Loading model: {MODEL_NAME}")
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
             else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
   
    model.gradient_checkpointing_enable()

 
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = model.to(device)
    print_vram("after model load")


    use_bf16 = (torch.cuda.is_available() and
                torch.cuda.get_device_capability()[0] >= 8 and
                dtype == torch.bfloat16)

    training_args = TrainingArguments(
        output_dir                   = OUTPUT_DIR,
        num_train_epochs             = NUM_EPOCHS,

        per_device_train_batch_size  = BATCH_SIZE,
        per_device_eval_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUMULATE,

        learning_rate                = LR,
        warmup_steps                 = WARMUP_STEPS,
        weight_decay                 = 0.01,
        max_grad_norm                = 1.0,
        optim                        = "adamw_torch",

        fp16                         = False,
        bf16                         = use_bf16,     # safe on Ampere (RTX 4060 = 8.9)

        logging_steps                = 50,
        eval_steps                   = 150,
        save_steps                   = 300,
        eval_strategy                = "steps",
        save_strategy                = "steps",
        save_total_limit             = 2,
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        greater_is_better            = False,

        gradient_checkpointing       = True,         # must match model.gc_enable() above
        dataloader_num_workers       = 2,
        dataloader_pin_memory        = True,

        report_to                    = "none",
        seed                         = SEED,
        use_cpu                      = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = tok_ds["train"],
        eval_dataset  = tok_ds["test"],
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n  Starting training …\n" + "─" * 65)
    print_vram("pre-train")
    trainer.train()
    print_vram("post-train")

    # ── H. Save ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  Saving fine-tuned model → {OUTPUT_DIR}/")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    model.to("cpu")
    torch.cuda.empty_cache()
    save_model_pkl(model, tokenizer, PKL_PATH)

    print("\nTraining complete!")
    print(f"  HF model → {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Pickle   → {os.path.abspath(PKL_PATH)}")
    print("\nTo use this model in test.py, prepend its path to CHAT_MODELS:")
    print(f'  CHAT_MODELS = ["{os.path.abspath(OUTPUT_DIR)}", …]')
    print("  Run  python test.py  to start the RAG chatbot.\n")


if __name__ == "__main__":
    main()