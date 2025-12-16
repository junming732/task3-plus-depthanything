# Task 4: Metric Scale Recovery with Dual-Stream ScaleNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oNb5up2jCIBLCqTWCz8wEkXkmWMumnrE?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Junming111/eth3d-scalenet-dual)

## Project Overview
Monocular depth estimation models (like Depth Anything V3) excel at relative depth but fail at **Metric Scale Recovery**. They can tell you a chair is closer than a wall, but not if the wall is 5 meters or 50 meters away.

This project implements a **Dual-Stream ScaleNet** that restores absolute metric scale to single-image depth predictions. It was trained and evaluated on the **ETH3D Dataset**.

---

## Easy Start (No Installation)
You can run the full inference pipeline, including interactive 3D visualization, directly in your browser via Google Colab.

**[Click here to open the Demo Notebook](https://colab.research.google.com/drive/1oNb5up2jCIBLCqTWCz8wEkXkmWMumnrE?usp=sharing)**

---

## Methodology: The Dual-Stream Architecture
We decouple the problem into **Geometry** and **Context** streams to avoid dataset overfitting.

1.  **Stream A (The Eyes):** Uses **Depth Anything V3** (frozen) to extract a high-quality relative depth map.
2.  **Stream B (The Brain):** Uses **ResNet18** (ImageNet-pretrained) to extract a 512-dimensional semantic feature vector. This stream recognizes scene context (e.g., "indoor office" vs "outdoor courtyard").
3.  **The Head (ScaleNet):** An MLP (512 $\to$ 128 $\to$ 64 $\to$ 1) that predicts a single scalar value $s$.

**Final Metric Depth** $= D_{rel} \times s$

---

## The Experiment

### Dataset
- **Source:** ETH3D High-Res Multi-View Stereo Dataset.
- **Scenes:** Diverse indoor (pipes, office) and outdoor (courtyards, facade) environments.
- **Ground Truth:** Sparse LiDAR/SfM point clouds.

### Training Strategy
- **Input:** Single RGB Image.
- **Target:** The ratio between the median Ground Truth depth and the median Relative Predicted depth.
- **Loss Function:** Mean Squared Error (MSE).
- **Optimizer:** Adam (LR=1e-3).

### Quantitative Results
We compared our method against a Naive Baseline (which predicts the mean scale of the training set for every image).

| Method | AbsRel Error | Improvement |
| :--- | :--- | :--- |
| **Naive Baseline** | 0.5950 | - |
| **Ours (Dual-Stream)** | **0.0586** | **90.16%** |

> **Conclusion:** By leveraging semantic features, the model reduced error by an order of magnitude, successfully differentiating between large-scale outdoor scenes and constrained indoor spaces.

---

## Qualitative Results
The difference is visually dramatic in 3D Point Clouds.

### 1. Outdoor Facade (Courtyard)
* **Baseline:** Overestimates scale, blowing up a simple wall into a massive monument (Width ~2.0 units).
* **Ours:** Correctly identifies the limited depth of the facade, shrinking the scale to physical reality (Width ~0.6 units).

### 2. Interactive Visualization
The provided Colab notebook includes a **PyVista Interactive Viewer**, allowing you to rotate and inspect the metric 3D point cloud in real-time.

---

## Repository Structure
```bash
.
├── task4_inference_dual.py   # Main inference script (loads streams A & B)
├── task4_train_dual.py       # Training script for ScaleNet
├── task4_evaluate_dual.py    # Evaluation script (calculates errors)
├── task4_prepare_dual.py     # Data preprocessing (extracts features)
├── task4_gen_report_images.py # Generate quanlitaive results
└── README.md                 # This file