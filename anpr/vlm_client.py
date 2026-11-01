"""OPTIONAL experimental extension: ask a remote vision-language model about an image.

The VLM runs on a separate PC/server (e.g. Ollama + LLaVA), never on the Pi 3.
Its answers are generated text, NOT verified plate readings -- OCR stays the source of truth.
"""
import base64

import cv2
import requests

from . import config


def ask_vlm(image, question: str, ocr_text: str = "", timeout: int = 60) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode image")
    prompt = question if not ocr_text else f"{question}\nOCR output (may contain errors): {ocr_text}"
    payload = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(buf.tobytes()).decode()],
        "stream": False,
    }
    r = requests.post(config.VLM_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()
