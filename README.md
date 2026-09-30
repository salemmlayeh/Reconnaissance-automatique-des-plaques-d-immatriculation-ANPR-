🚗 Embedded ANPR System on Raspberry Pi 3
An embedded Automatic Number Plate Recognition (ANPR) system built for edge deployment on a Raspberry Pi 3[cite: 1]. This project combines classical computer vision with deep learning to achieve real-time plate detection and text extraction with over 92% accuracy[cite: 1].

📌 Overview
This project implements an end-to-end vision pipeline capable of detecting license plates from camera/image inputs and extracting text under varying lighting and environmental conditions.

┌─────────────────┐     ┌───────────────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│  Image Input    │ ──► │  OpenCV Preprocessing │ ──► │ PyTorch Plate Detection │ ──► │ ROI Extraction  │
└─────────────────┘     └───────────────────────┘     └─────────────────────────┘     └────────┬────────┘
                                                                                               │
                                                                                               ▼
┌─────────────────┐     ┌───────────────────────┐                                     ┌─────────────────┐
│ Final Text Result│ ◄── │ Regex Post-Processing │ ◄────────────────────────────────── │  Tesseract OCR  │
└─────────────────┘     └───────────────────────┘                                     └─────────────────┘
✨ Key Features
Embedded Deployment: Designed and optimized specifically for constrained hardware (Raspberry Pi 3 / ARM Cortex-A53)[cite: 1].

Deep Learning Localizer: High-precision plate detection powered by PyTorch[cite: 1].

Image Preprocessing: Advanced noise reduction, adaptive thresholding, and perspective correction via OpenCV[cite: 1].

Robust OCR: Character extraction using Tesseract OCR with custom regex filtering for standard plate formats[cite: 1].

High Performance: Evaluated on test datasets achieving > 92% recognition accuracy[cite: 1].

🛠️ Tech Stack & Dependencies
Hardware: Raspberry Pi 3 Model B (Raspbian / Raspberry Pi OS)[cite: 1]

Languages: Python 3.8+[cite: 1]

Deep Learning Framework: PyTorch[cite: 1]

Computer Vision: OpenCV[cite: 1]

OCR Engine: Tesseract OCR (pytesseract)[cite: 1]

Data Manipulation: NumPy, Pandas

🚀 Getting Started
Prerequisites
Update your System and install system-level dependencies on your Raspberry Pi:

Bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y tesseract-ocr libtesseract-dev
Installation
Clone the Repository:

Bash
git clone https://github.com/salemmlayeh/anpr-raspberry-pi.git
cd anpr-raspberry-pi
Create a Virtual Environment & Activate:

Bash
python3 -m venv venv
source venv/bin/activate
Install Python Packages:

Bash
pip install -r requirements.txt
💻 Usage
1. Run Detection on Single Image / Test Set
Bash
python main.py --input data/sample_car.jpg --save-output
2. Run Real-Time Stream (Pi Camera or USB Webcam)
Bash
python main.py --source webcam
📊 Pipeline Explanation
Preprocessing (src/preprocessing.py): Converts frame to grayscale, applies bilateral filtering to smooth noise while preserving edges, and resizes image for optimal inference speed.

Detection (src/detector.py): Runs PyTorch object detection model to locate the bounding box surrounding the license plate[cite: 1].

Segmentation & Binarization (src/segmentation.py): Crops the region of interest (ROI) and applies Otsu/Adaptive Thresholding to highlight characters.

Text Extraction (src/ocr.py): Tesseract extracts character strings which are validated using regular expressions[cite: 1].

📈 Performance & Results
Overall Accuracy: > 92% across test datasets under diverse lighting conditions[cite: 1].

Target Hardware: Raspberry Pi 3[cite: 1]

Inference Optimization: Reduced input tensor sizes and optimized image transforms to maintain low execution latency on CPU.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8f16d411-c0a5-45ee-9d9b-982d7d29f20e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/029b9db8-8344-4672-bbe0-77d6bb6a8d31" />

