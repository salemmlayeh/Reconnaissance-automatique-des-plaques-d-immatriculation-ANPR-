# 🚗 System Description & Project Architecture

## 📌 Project Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** system optimized for edge execution on a **Raspberry Pi 3**. The system processes incoming visual data (images or video streams), detects the presence of vehicle license plates using deep learning, isolates the plate region, and extracts the alphanumeric characters with an accuracy rate exceeding **92%**.

---

## ⚙️ Detailed Execution Pipeline

The processing pipeline is divided into four main sequential stages:

### 1. Image Acquisition & Preprocessing (`OpenCV`)

* **Frame Capture & Resizing:** Downsamples raw input frames from the camera to a standard resolution to minimize CPU computational overhead on the Raspberry Pi 3.


* **Grayscale Conversion:** Converts RGB images to single-channel grayscale to reduce memory footprint and simplify pixel intensity calculations.
* **Noise Reduction & Edge Enhancement:** Applies a bilateral filter to smooth out background noise while preserving sharp boundaries around text and license plate borders.

### 2. License Plate Localization (`PyTorch`)

* **Deep Learning Inference:** Passes the preprocessed frame through a lightweight object detection model (e.g., custom CNN or optimized YOLO variant) running on **PyTorch**.


* **Bounding Box Generation:** The model predicts spatial coordinates bounding the license plate region within the full image frame.
* **Region of Interest (ROI) Extraction:** Crops the detected bounding box containing the license plate for isolated processing.

### 3. ROI Normalization & Character Binarization (`OpenCV`)

* **Perspective & Alignment Correction:** Adjusts skewed or angled plates to obtain a horizontal orientation.
* **Adaptive Thresholding:** Applies Otsu’s binarization to separate foreground text characters from the plate background under uneven lighting or shadows.
* **Morphological Operations:** Performs dilation and erosion to connect broken character contours and eliminate residual background artifacts.

### 4. Optical Character Recognition & Filtering (`Tesseract OCR`)

* **Text Extraction:** Feeds the normalized binary ROI into the **Tesseract OCR** engine.


* **Pattern Validation:** Applies regular expression (Regex) pattern matching to validate the recognized text against standard license plate formats and filter out noise characters.
* **Output Generation:** Displays or logs the identified registration string along with the confidence score.

---

## 📊 Technical Challenges & Hardware Optimizations

* **Memory Management:** Streamlined tensor operations in **PyTorch** and explicitly freed unused buffers in memory to prevent memory saturation on the Raspberry Pi 3's 1 GB RAM.


* **Latency Reduction:** Separated plate localization (Deep Learning) from text recognition (OCR) to maintain low execution latency per frame on a quad-core ARM processor.


* **Environmental Robustness:** Optimized preprocessing filters to handle varying outdoor conditions, including low contrast, headlight glare, and inclined angles.



---

## 🎯 Results & Evaluation

* **Overall Recognition Rate:** > 92% accurate character identification across test datasets.


* **Target Environment:** Embedded Linux (Raspberry Pi OS) on ARM Cortex-A53 architecture.

Inference Optimization: Reduced input tensor sizes and optimized image transforms to maintain low execution latency on CPU.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8f16d411-c0a5-45ee-9d9b-982d7d29f20e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/029b9db8-8344-4672-bbe0-77d6bb6a8d31" />

