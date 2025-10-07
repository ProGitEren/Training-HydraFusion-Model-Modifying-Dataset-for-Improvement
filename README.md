# 🚗 HydraFusion on RADIATE – Robust Sensor Fusion under Adverse Weather

This repository contains our implementation and experimental extensions of the paper  
**[HydraFusion: Context-Aware Selective Sensor Fusion for Robust and Efficient Autonomous Vehicle Perception](https://arxiv.org/abs/2201.06644)**  
([Original GitHub Repository – AICPS/hydrafusion](https://github.com/AICPS/hydrafusion)).

We adapted and trained the HydraFusion model using the **RADIATE dataset**, focusing on robustness evaluation under **adverse weather conditions**.  
Our implementation includes dataset augmentations, modified camera and radar preprocessing, and an improved training/validation pipeline.

---

## 📚 Reference Paper

> Arnav Vaibhav Malawade, Trier Mortlock, and Mohammad Abdullah Al Faruque.  
> *HydraFusion: Context-Aware Selective Sensor Fusion for Robust and Efficient Autonomous Vehicle Perception.*  
> ACM/IEEE International Conference on Cyber-Physical Systems (Milan ’22).  
> [PDF](https://arxiv.org/pdf/2201.06644.pdf)

---

## 🧩 Project Overview

HydraFusion proposes a **context-aware selective sensor fusion** architecture that dynamically decides *how* and *when* to fuse information from multiple modalities (camera, radar, LiDAR) depending on the current driving context.

In this work, we:

- Implemented and trained the HydraFusion model on the **RADIATE** dataset.  
- Introduced **data augmentations** to simulate fog, snow, and rain for radar and camera images.  
- Evaluated model robustness on both **original** and **augmented** datasets.  
- Conducted validation on **overfitted, good-weather, and bad-weather subsets** to analyze domain generalization.

---

## 🧠 Implementation Summary

Our code builds on the original HydraFusion structure but extends the **data pipeline** and **training routines**.

### 🔹 Key Additions

- **Camera Augmentations:**  
  Extreme blur, color jitter, hue shift, and Gaussian noise simulating vision degradation.

- **Radar Augmentations:**  
  - Clutter insertion using random Gaussian blobs  
  - Radial attenuation for distance-based fading  
  - Noise injection and random attenuation scaling  
  These modifications simulate real radar signal degradation under fog, rain, and snow.

- **Dynamic Dataset Splitting:**  
  Good vs. bad weather sequences automatically categorized via sequence name.

- **Visualization Utilities:**  
  Functions to visualize original vs. augmented inputs and ground-truth bounding boxes for each modality.

- **Enhanced Validation Pipeline:**  
  Validation runs can be filtered by condition (`--condition good|bad|all`) and with or without augmentation.

---

## 🧪 Training Code

Main training script: [`Training_Aug.py`](./Training_Aug.py)

### Example usage

```bash
python Training_Aug.py \
  --data_root /path/to/RADIATE \
  --epochs 50 \
  --batch_size 4 \
  --augment True
