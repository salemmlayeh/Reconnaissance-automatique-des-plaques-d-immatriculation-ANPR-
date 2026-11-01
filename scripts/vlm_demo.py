"""Optional: ask a remote VLM about a plate image (needs a VLM server, see anpr/vlm_client.py)."""
import sys

import cv2

from anpr import ANPRPipeline
from anpr.vlm_client import ask_vlm

image = cv2.imread(sys.argv[1])
results = ANPRPipeline().process(image)
ocr_text = results[0].text if results else ""
print("OCR  :", ocr_text)
print("VLM  :", ask_vlm(image, "Describe the visible vehicle and its surroundings.", ocr_text))
