# 🚗 Automatic Number Plate Recognition (ANPR) with YOLOv8, OCR & Vision–Language Models

## 📌 Project Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** system designed for embedded vision applications using Python and a Raspberry Pi 3.

The system detects vehicle license plates and extracts their alphanumeric characters through a computer vision pipeline combining **YOLOv8, PyTorch, OpenCV, and Tesseract OCR**. The project evolved from traditional image-processing techniques to deep-learning-based object detection to improve plate localization under challenging visual conditions.

A further experimental extension explores **Vision–Language Models (VLMs)** for contextual image understanding and natural-language question answering, complementing conventional object detection and OCR.

**Key Features**
- Traditional computer vision techniques for initial plate-localization experiments.
- CNN-based deep-learning approaches for visual detection.
- YOLOv8-based license plate detection using PyTorch.
- Image preprocessing and plate-region extraction with OpenCV.
- Character recognition using Tesseract OCR.
- Text validation and filtering using regular expressions.
- Experimental VLM integration for image understanding and contextual question answering.
- Embedded deployment considerations for Raspberry Pi 3.

---

## ⚙️ System Architecture

The system follows a modular pipeline, with the VLM component designed as an optional extension.

### 1. Image Acquisition and Preprocessing — OpenCV

Input images or video frames are acquired from a camera or loaded from image files.

Preprocessing operations may include:
- Image resizing to reduce computational overhead.
- Grayscale conversion for conventional image-processing operations.
- Noise reduction and contrast enhancement.
- Bilateral filtering to preserve important edges while reducing noise.
- Image normalization before model inference.

The preprocessing strategy depends on the detection method and the conditions of the input images.

### 2. Evolution from Traditional Computer Vision to Deep Learning

The project initially explored conventional computer vision techniques for locating license plates.

**Traditional computer vision approach**

Initial experiments used image-processing operations such as thresholding, edge detection, contour extraction, and geometric filtering to identify candidate plate regions.

These methods provided a baseline but could be sensitive to illumination, background complexity, plate orientation, and image quality.

**CNN-based detection**

Deep-learning-based approaches were subsequently explored to learn visual features directly from image data and improve object localization.

**YOLOv8-based plate detection**

The detection pipeline was then developed using YOLOv8 with PyTorch.

The model predicts bounding boxes corresponding to license plate regions in the input image. The detected regions are extracted as regions of interest (ROIs) for subsequent OCR processing.

This progression provides a practical comparison between handcrafted image-processing methods and learned visual representations.

### 3. License Plate Processing and OCR — OpenCV & Tesseract

After detecting a plate, the system extracts the corresponding image region and prepares it for character recognition.

Processing operations may include:
- Cropping the detected plate region.
- Resizing the crop to improve text readability.
- Correcting orientation or perspective when required.
- Applying grayscale conversion and thresholding.
- Using morphological operations to improve text visibility where appropriate.

The processed plate image is passed to **Tesseract OCR**, which extracts the alphanumeric characters.

Regular expressions and format-validation rules can then be applied to filter invalid outputs and reduce recognition errors.

The resulting registration string and available confidence information can be displayed or logged for further analysis.

### 4. Vision–Language Model Extension — Experimental

An additional research direction extends the original ANPR pipeline with a pretrained vision–language model.

Unlike a conventional object detector, which localizes objects, a VLM can relate visual information to natural-language questions and generate contextual descriptions.

The proposed extension explores the following capabilities:

- **Visual question answering:** Answering questions about the contents of an input image.
- **Contextual image understanding:** Describing visible vehicles and relevant scene elements.
- **OCR-assisted interpretation:** Providing recognized text as additional context when investigating an image.
- **Prompt-based experimentation:** Comparing model responses under different questions and contextual instructions.

Example questions include:
- "Describe the visible vehicle and its surroundings."
- "What visual information can be identified in this image?"
- "Based on the image and the OCR output, summarize the available information."

The VLM is intended to complement, rather than replace, YOLOv8 and Tesseract. The detection and OCR modules remain responsible for plate localization and character extraction, while the VLM provides a natural-language interface for interpreting visual content.

**Embedded deployment consideration:** Due to the limited memory and computational resources of the Raspberry Pi 3, initial VLM experiments can run on a separate computer or inference service. Selected images or cropped regions can be sent to the VLM without requiring the entire multimodal model to run on the embedded device.

This extension is intended to investigate the benefits and limitations of combining conventional computer vision, OCR, and vision–language reasoning.

### 5. Output Generation and Validation

The system can produce:
- Detected license plate regions.
- OCR-extracted registration text.
- Text-validation results.
- Optional VLM-generated descriptions and answers.
- Intermediate images and outputs for debugging and evaluation.

VLM responses should be treated as generated interpretations, not as a source of verified license plate text. The OCR pipeline remains the primary source for recognized characters.

---

## 🧠 Technical Challenges and Optimization

### Embedded Computing Constraints

The Raspberry Pi 3 has limited CPU performance and memory compared with desktop systems. The project therefore considers:
- Reducing image dimensions before inference when appropriate.
- Avoiding unnecessary image copies and intermediate buffers.
- Separating detection and OCR into modular processing stages.
- Measuring CPU inference latency and memory consumption.
- Evaluating lightweight model configurations for embedded deployment.

### Environmental Robustness

License plate detection and OCR can be affected by:
- Poor illumination and shadows.
- Motion blur and low-resolution images.
- Inclined or partially occluded plates.
- Background clutter and visually similar objects.
- Variations in plate appearance and character spacing.

Traditional image processing, deep-learning detection, and OCR provide complementary approaches to addressing these challenges.

### Vision–Language Model Limitations

VLM experiments also require attention to:
- Inaccurate or unsupported descriptions.
- Sensitivity to prompt wording.
- Difficulty interpreting small text in full-scene images.
- Inference latency and computational cost.
- The difference between plausible natural-language answers and objectively correct results.

These limitations should be documented during evaluation rather than assuming that a VLM always improves system performance.

---

## 📊 Results and Evaluation

The original project achieved a reported character-recognition rate exceeding **92%** on its test data.

The evaluation of the system can be organized into three parts:

| Component | Evaluation focus |
|---|---|
| Traditional computer vision | Candidate plate localization and sensitivity to image conditions |
| YOLOv8 + Tesseract OCR | Plate detection, character recognition, and end-to-end recognition performance |
| Vision–language extension | Answer relevance, contextual understanding, response consistency, and inference latency |

The reported recognition rate refers to the original ANPR system and should not be interpreted as a VLM evaluation result.

Future experiments can compare the approaches using consistent test images, clearly defined metrics, and documented experimental conditions.

---

## 🛠️ Technologies

- **Programming:** Python
- **Deep Learning:** PyTorch, YOLOv8, CNN-based approaches
- **Computer Vision:** OpenCV
- **Optical Character Recognition:** Tesseract OCR
- **Multimodal AI:** Pretrained vision–language models (experimental extension)
- **Embedded Platform:** Raspberry Pi 3, Raspberry Pi OS / Linux
- **Image Processing:** Thresholding, filtering, contour analysis, morphological operations

---

## 🚀 Future Improvements

- Benchmark traditional computer vision against YOLOv8 using the same evaluation dataset.
- Optimize detection and OCR latency on Raspberry Pi hardware.
- Evaluate recognition performance across different lighting and viewing conditions.
- Integrate and evaluate a pretrained VLM for contextual image understanding.
- Investigate whether OCR output improves VLM responses when supplied as additional context.
- Develop a lightweight interface for image-based questions and answers.
- Measure VLM response quality and latency separately from ANPR accuracy.

---

## 🎯 Project Objective

The objective is to develop a modular embedded vision system that combines object detection and text recognition, while exploring how vision–language models can extend conventional computer vision with contextual understanding and natural-language interaction.

The project provides a practical foundation for studying the transition from traditional image processing to deep learning and multimodal AI in resource-constrained vision applications.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8f16d411-c0a5-45ee-9d9b-982d7d29f20e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/029b9db8-8344-4672-bbe0-77d6bb6a8d31" />

