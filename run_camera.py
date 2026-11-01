"""Live ANPR from the Raspberry Pi camera (picamera2) or any USB/OpenCV camera.

Usage:  python run_camera.py [--show] [--every 5] [--log plates.csv]
"""
import argparse
import csv
import time
from datetime import datetime

import cv2

from anpr import ANPRPipeline


def frame_source():
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        cam.start()
        while True:
            yield cam.capture_array()          # "RGB888" gives BGR order, ready for OpenCV
    except ImportError:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--every", type=int, default=5, help="process every Nth frame (saves CPU)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds before logging the same plate again")
    ap.add_argument("--log", default="plates.csv")
    args = ap.parse_args()

    pipeline = ANPRPipeline(imgsz=320)         # smaller input = faster on a Pi 3
    last_seen = {}
    last_results = []

    with open(args.log, "a", newline="") as f:
        writer = csv.writer(f)
        for i, frame in enumerate(frame_source()):
            if i % args.every == 0:
                last_results = pipeline.process(frame)
                for r in last_results:
                    now = time.time()
                    if r.valid and now - last_seen.get(r.text, 0) > args.cooldown:
                        last_seen[r.text] = now
                        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                         r.text, f"{r.ocr_conf:.0f}", f"{r.det_conf:.2f}"])
                        f.flush()
                        print(f"[{datetime.now():%H:%M:%S}] {r.text}")
            if args.show:
                cv2.imshow("ANPR", pipeline.draw(frame, last_results))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
