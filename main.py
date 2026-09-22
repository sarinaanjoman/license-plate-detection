#!/usr/bin/env python3
"""
Car License Plate Detection (CLI, merged detection logic)

This version keeps the clean CLI interface AND restores the original detection
behavior from the provided script (contours on a blurred grayscale image),
with a simple fallback to an edge-based method for robustness.

- Detect plate region (bbox)
- Optional OCR via EasyOCR
- Clean terminal output (no BBox line)
- Optional save / display
"""

import argparse
import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np

# Optional display (used only if --no-display is NOT set)
import matplotlib.pyplot as plt

try:
    import easyocr
except ImportError:
    easyocr = None

# Keep console clean for demos
warnings.filterwarnings("ignore")


def preprocess_blur(bgr: np.ndarray) -> np.ndarray:
    """Original-style preprocessing: grayscale + Gaussian blur."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def bbox_from_contours(contours) -> tuple[int, int, int, int] | None:
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 500:
        return None
    return x, y, w, h


def detect_plate_bbox_blur(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Match the original script's behavior as closely as possible:
    findContours is applied directly on the blurred grayscale image.
    """
    blurred = preprocess_blur(bgr)
    contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Original kept only the largest contour; bbox_from_contours does that
    return bbox_from_contours(contours)


def detect_plate_bbox_edges(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Simple fallback: Canny edges -> contours -> largest bbox."""
    blurred = preprocess_blur(bgr)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return bbox_from_contours(contours)


def detect_plate_bbox(bgr: np.ndarray, method: str = "auto") -> tuple[int, int, int, int] | None:
    """
    method:
      - auto  : try original blur-contour method first, then edge fallback
      - blur  : only original blur-contour method
      - edges : only edge-based method
    """
    method = method.lower()
    if method not in {"auto", "blur", "edges"}:
        method = "auto"

    if method in {"auto", "blur"}:
        bbox = detect_plate_bbox_blur(bgr)
        if bbox is not None:
            return bbox
        if method == "blur":
            return None

    return detect_plate_bbox_edges(bgr)


def get_easyocr_reader(langs: list[str]):
    """Create EasyOCR reader while suppressing its startup console messages."""
    if easyocr is None:
        raise RuntimeError("easyocr is not installed. Install it or run with --no-ocr.")
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        reader = easyocr.Reader(langs, gpu=False)
    return reader


def read_text_easyocr(plate_bgr: np.ndarray, langs: list[str]) -> list[str]:
    reader = get_easyocr_reader(langs)
    result = reader.readtext(plate_bgr)
    texts: list[str] = []
    for detection in result:
        text = detection[1]
        if text:
            texts.append(text)
    return texts


def annotate_image(bgr: np.ndarray, bbox: tuple[int, int, int, int] | None, texts: list[str]) -> np.ndarray:
    out = bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if texts:
            label = " ".join(texts)
            cv2.putText(out, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return out


def show_image_matplotlib(bgr: np.ndarray, title: str = "Result") -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Car plate detection + OCR (CLI)")
    p.add_argument("--image", required=True, help="Path to an input image (jpg/png).")

    # Keep optional parameters simple
    p.add_argument("--save", default=None, help="Path to save annotated output (e.g., results/out.png).")
    p.add_argument("--no-display", action="store_true", help="Do not open a window.")
    p.add_argument("--no-ocr", action="store_true", help="Detect plate only; do not run OCR.")
    p.add_argument("--lang", default="en", help="EasyOCR language(s), comma-separated (default: en). Example: en,tr")

    # Added, but still simple: choose detection method if needed
    p.add_argument("--method", default="auto", choices=["auto", "blur", "edges"],
                   help="Plate detection method (default: auto). 'blur' matches original script most closely.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: Could not read image: {args.image}", file=sys.stderr)
        return 2

    bbox = detect_plate_bbox(bgr, method=args.method)

    texts: list[str] = []
    if not args.no_ocr and bbox is not None:
        x, y, w, h = bbox
        plate = bgr[y:y + h, x:x + w]
        langs = [s.strip() for s in args.lang.split(",") if s.strip()]
        try:
            texts = read_text_easyocr(plate, langs=langs)
        except Exception:
            # Keep output clean; OCR failure shouldn't crash detection demo
            texts = []

    # Clean terminal output (no BBox line)
    if bbox is None:
        print("Plate detected: NO")
        print("Detected text: (none)")
    else:
        print("Plate detected: YES")
        print("Detected text:", " | ".join(texts) if texts else "(none)")

    out = annotate_image(bgr, bbox, texts)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), out)
        if ok:
            print(f"Saved annotated image to: {out_path}")

    if not args.no_display:
        title = "Car Plate Detection"
        if texts:
            title += f" — {', '.join(texts)}"
        show_image_matplotlib(out, title=title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
