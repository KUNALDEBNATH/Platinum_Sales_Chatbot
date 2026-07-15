"""
vision_parser.py
────────────────────────────────────────────────────────────────────────────
Handles image uploads (PNG / JPG / JPEG / WEBP) using a lightweight vision
language model (Qwen2.5-VL). Falls back to plain OCR + the existing text
LLM if the VLM cannot be loaded (e.g. no GPU / model not downloaded),
mirroring the graceful-degradation pattern already used for the text LLM
in test.py (_load_llm iterates candidate models and falls back cleanly).
"""

from __future__ import annotations

import threading

import torch
from PIL import Image

VLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_MAX_NEW_TOKENS = 300

# Smallest-first so low-resource machines still get a working VLM.
VLM_CANDIDATES = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

_vlm_model = None
_vlm_processor = None
_vlm_ready = False
_vlm_lock = threading.Lock()
_vlm_load_attempted = False


def _load_vlm() -> bool:
    """Lazily load a Qwen2.5-VL model. Thread-safe, loads only once."""
    global _vlm_model, _vlm_processor, _vlm_ready, _vlm_load_attempted

    if _vlm_ready:
        return True
    with _vlm_lock:
        if _vlm_ready:
            return True
        if _vlm_load_attempted:
            return _vlm_ready
        _vlm_load_attempted = True

        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
            )
        except ImportError:
            print("  [VLM] transformers version does not support Qwen2.5-VL "
                  "(Qwen2_5_VLForConditionalGeneration not found). "
                  "Falling back to OCR-only mode.")
            return False

        for model_path in VLM_CANDIDATES:
            try:
                print(f"  [VLM] Loading '{model_path}' on {VLM_DEVICE} …")
                dtype = torch.float16 if VLM_DEVICE == "cuda" else torch.float32
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, device_map=VLM_DEVICE,
                    trust_remote_code=True,
                )
                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model.eval()
                _vlm_model, _vlm_processor, _vlm_ready = model, processor, True
                print(f"  [VLM] Ready: {model_path}")
                return True
            except Exception as e:
                print(f"  [VLM] '{model_path}' failed: {e}")

        print("  [VLM] No vision model available — using OCR fallback.")
        return False


def _ocr_fallback(image_path: str) -> str:
    """Extract raw text from an image via Tesseract OCR."""
    try:
        import pytesseract
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img).strip()
        return text
    except Exception as e:
        print(f"  [OCR] failed: {e}")
        return ""


def analyze_image(image_path: str, query: str) -> str:
    """
    Answer a question about an uploaded image.

    1. If Qwen2.5-VL loaded successfully, ask it directly (it can read text,
       describe scenes, read charts, etc. in one pass).
    2. Otherwise, run OCR to extract any visible text and hand that off to
       the caller (attachment_handler) to phrase an answer using the text
       LLM, since we cannot "see" the image without a VLM.
    """
    if _load_vlm():
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            text_prompt = _vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image = Image.open(image_path).convert("RGB")
            inputs = _vlm_processor(
                text=[text_prompt], images=[image],
                padding=True, return_tensors="pt",
            ).to(VLM_DEVICE)

            with torch.no_grad():
                output_ids = _vlm_model.generate(
                    **inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False
                )
            trimmed = [
                out[len(inp):] for inp, out in
                zip(inputs.input_ids, output_ids)
            ]
            answer = _vlm_processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()
            if answer:
                return answer
        except Exception as e:
            print(f"  [VLM] generation error: {e}")

    # ── OCR fallback path ──────────────────────────────────────────────
    ocr_text = _ocr_fallback(image_path)
    if ocr_text:
        return (
            "[Vision model unavailable — showing OCR-extracted text instead]\n"
            f"{ocr_text}"
        )
    return (
        "I could not analyze this image. No vision model is currently "
        "available and no readable text was detected via OCR."
    )


def vlm_status() -> dict:
    return {"vlm_ready": _vlm_ready, "vlm_attempted": _vlm_load_attempted}
