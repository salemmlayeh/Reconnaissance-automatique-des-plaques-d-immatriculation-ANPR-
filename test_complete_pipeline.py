"""Quick end-to-end test on one image.

Usage:  python test_complete_pipeline.py data/samples/car.jpg [--no-gui]
"""
import argparse
import sys
from pathlib import Path

import cv2

from anpr import ANPRPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="data/samples/car.jpg")
    ap.add_argument("--no-gui", action="store_true", help="skip OpenCV windows (headless Pi)")
    args = ap.parse_args()

    print("=== COMPLETE ANPR PIPELINE TEST ===")
    if not Path(args.image).is_file():
        sys.exit(f"Image not found: {args.image}")
    image = cv2.imread(args.image)
    print(f"1. Original image loaded: {image.shape}")

    results = ANPRPipeline().process(image)
    print(f"2. Plate detected: {bool(results)}")
    if not results:
        return

    best = results[0]
    print(f"3. License Plate Number: {best.text}  "
          f"(valid={best.valid}, det={best.det_conf:.2f}, ocr={best.ocr_conf:.0f})")

    if not args.no_gui:
        cv2.imshow("Original Car Image", ANPRPipeline.draw(image, results))
        cv2.imshow("Plate", best.crop)
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
