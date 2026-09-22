# Car License Plate Detection (Computer Vision Project)

## Overview

Classical computer vision pipeline for car license plate detection and recognition.

The program is implemented as a **command-line Python script with optional parameters**, fully compliant with typical computer vision course requirements.

---

## Project Category

**Category:** More Complex Computer Vision Problems

This project belongs to the _more complex problems_ category because it:

- Solves a real-world computer vision task
- Uses standard libraries such as OpenCV and OCR tools
- Combines multiple classical CV steps into a complete application

---

## Features

- Command-line interface (CLI)
- Automatic license plate region detection
- Optical Character Recognition (OCR) for plate text
- Clean terminal output (no debug noise)
- Optional image saving
- Runs entirely on CPU (GPU not required)

---

## Requirements

### System Requirements

- Python **3.9+** (tested on macOS)
- macOS / Linux / Windows

### Python Dependencies

All required Python packages are listed in `requirements.txt`.

Main dependencies:

- numpy
- opencv-python
- matplotlib
- easyocr

Install them with:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
car-plate-detection/
│── main.py
│── requirements.txt
│── README.md
│── images/
│   ├── img1.jpg
│   └── img2.jpg
│── results/
│   └── out.png
```

---

## How to Run

### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the program

```bash
python3 main.py --image images/img1.jpg
```

---

## Optional Parameters

| Parameter      | Description                         |
| -------------- | ----------------------------------- |
| `--image`      | Path to input car image (required)  |
| `--save`       | Save annotated output image         |
| `--no-display` | Run without opening an image window |
| `--no-ocr`     | Detect plate only (disable OCR)     |
| `--lang`       | OCR language(s), e.g. `en`, `en,tr` |

Example:

```bash
python3 main.py --image images/img1.jpg --save out.png --no-display
```

---

## Output

The program prints a clean, human-readable result in the terminal:

```
Plate detected: YES
Detected text: 34 TR4444
```

If saving is enabled, an annotated image with the detected plate region is written to disk.

---

## Methodology (High-Level)

1. Convert image to grayscale
2. Apply Gaussian blur for noise reduction
3. Detect edges using Canny
4. Extract contours and select the most likely plate region
5. Crop detected plate area
6. Apply OCR to recognize characters
7. Display and/or save results

---

## Limitations

- Detection accuracy depends on image quality and lighting
- OCR accuracy may vary for blurred or low-resolution plates
- The detection logic uses simple heuristics and may fail in complex scenes

---

## Notes

- GPU acceleration is optional and not required
- Warning messages from external libraries are suppressed for clean output
- The focus of the project is clarity, reproducibility, and correct CV usage
