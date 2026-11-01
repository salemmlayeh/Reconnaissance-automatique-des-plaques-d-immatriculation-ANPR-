"""Export trained weights to a faster format for the Raspberry Pi.

Usage:  python scripts/export_model.py --weights models/best.pt --format ncnn --imgsz 320
Then set MODEL_PATH in anpr/config.py to the exported folder (e.g. models/best_ncnn_model).
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/best.pt")
ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "torchscript"])
ap.add_argument("--imgsz", type=int, default=320)
args = ap.parse_args()

print("Exported to:", YOLO(args.weights).export(format=args.format, imgsz=args.imgsz))
