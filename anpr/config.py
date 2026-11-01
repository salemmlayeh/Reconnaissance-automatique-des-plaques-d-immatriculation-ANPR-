"""Central configuration. Edit values here instead of hunting through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Detection (YOLOv8) ---------------------------------------------------
MODEL_PATH = ROOT / "models" / "best.pt"   # or an exported folder, e.g. models/best_ncnn_model
DET_CONF = 0.25          # minimum detection confidence
IMG_SIZE = 416           # inference size (use 320 on a Raspberry Pi 3)
DEVICE = "cpu"           # "cpu" or "cuda:0"

# --- OCR (Tesseract) ------------------------------------------------------
# Windows example: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_TARGET_HEIGHT = 120  # plate crop is resized to this height before OCR
LEFT_TRIM = 0.0          # fraction of plate width to cut on the left (e.g. 0.10 for a country/EU band)

# --- Validation -----------------------------------------------------------
# A plate is "valid" if it matches ANY pattern. Adapt to your country.
PLATE_PATTERNS = [
    r"^[A-Z]{2}\d{3,4}[A-Z]{2}$",   # Serbia, e.g. BG224NZ
    # r"^\d{3}TUN\d{1,4}$",         # Tunisia, e.g. 123TUN4567
    # r"^[A-Z0-9]{5,10}$",          # generic fallback
]

# --- Optional VLM service (runs on a separate machine, NOT on the Pi) ------
VLM_URL = "http://192.168.1.50:11434/api/generate"   # e.g. Ollama
VLM_MODEL = "llava"
