"""Train YOLOv8 for license plate detection (run on a PC/GPU or Colab, not on the Pi).

Usage:  python train_yolo.py --data data.yaml --epochs 50
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="nano model = best for the Pi")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project="runs", name="plates")

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    print("Best weights:", Path("runs/plates/weights/best.pt").resolve())
    print("Copy them to models/best.pt")


if __name__ == "__main__":
    main()
