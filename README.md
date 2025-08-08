# 🧠 Brain Tumor Segmentation with U-Net

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> 🎯 A PyTorch & TensorFlow implementation of **U-Net** for precise brain tumor segmentation on 2D MRI slices

---

## 🚀 What is this?

This project implements the **U-Net** convolutional neural network architecture for **medical image segmentation**. The model is specifically trained to detect and segment brain tumors from 2D MRI slices, producing pixel-level segmentation masks that accurately outline tumor regions.

### 🔍 Key Features

- **Encoder-Decoder Architecture**: Contracting path captures context, expanding path produces fine-grained segmentation
- **Skip Connections**: Preserves spatial detail for precise boundary detection  
- **Medical-Grade Accuracy**: Optimized for brain tumor detection in MRI scans
- **Production Ready**: Includes comprehensive training, validation, and evaluation pipelines

---

## 📊 Dataset Overview

### 📁 LGG MRI Segmentation Dataset
**Source**: Brain MRI images together with manual FLAIR abnormality segmentation masks from The Cancer Imaging Archive (TCIA) and The Cancer Genome Atlas (TCGA)

**Download Dataset:**
```python
import kagglehub
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
```

**🔗 Dataset Link**: [Kaggle - LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

| **Metric** | **Value** |
|------------|-----------|
| **Type** | 2D MRI slices with binary tumor masks |
| **Patients** | 110 patients with lower-grade gliomas |
| **Training Samples** | ~2,985 |
| **Validation Samples** | ~747 |
| **Test Samples** | ~197 |
| **Format** | (Image, Mask) pairs |
| **Labels** | Binary: `1` = tumor, `0` = background |

### 📝 Data Preprocessing
- **Normalization**: Z-score or min-max scaling per image
- **Augmentation**: Rotations, flips, intensity variations
- **Class Balance**: Handles high background-to-tumor ratio

---

## 🛠️ Tech Stack

### Core Framework
```
🔥 PyTorch          # Model architecture & training
📊 Albumentations   # Fast image & mask augmentation  
🖼️  OpenCV/Pillow   # Image I/O operations
🔢 NumPy           # Numerical computations
📈 Matplotlib      # Visualization & plotting
⚡ tqdm            # Progress tracking
```

### Optional Enhancements
```
🚀 torch.cuda.amp      # Mixed precision training
☁️  huggingface_hub     # Model hosting & sharing
📦 Git LFS             # Large file version control
```

---

## 🏗️ Model Architecture

### 🔄 U-Net Structure

```
📥 INPUT (MRI Slice)
    ↓
🔽 ENCODER (Contracting Path)
   │ Conv → ReLU → Conv → BatchNorm → ReLU
   │ MaxPool (downsampling)
   │ Repeat with increasing channels
    ↓
🔥 BOTTLENECK (Deepest features)
    ↓
🔼 DECODER (Expanding Path)
   │ ConvTranspose2d (upsampling)
   │ Skip connections from encoder
   │ Conv blocks for refinement
    ↓
📤 OUTPUT (1×1 Conv → Sigmoid)
    ↓
🎯 PROBABILITY MAP (per-pixel tumor probability)
```

### 🎯 Why Skip Connections?
Skip connections preserve **fine spatial details** from the encoder, enabling the decoder to recover precise object boundaries essential for medical segmentation.

---

## 🎯 Training Configuration

### 📉 Loss Function
```python
Combined Loss = 0.1 × Binary Cross-Entropy + 0.9 × Dice Loss
```
- **BCE**: Stabilizes pixel-wise learning early in training
- **Dice**: Optimizes overlap for imbalanced tumor masks

### ⚙️ Optimization Setup
| **Parameter** | **Value** |
|---------------|-----------|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 4-16 (GPU dependent) |
| **Scheduler** | ReduceLROnPlateau |
| **Early Stopping** | Based on validation metrics |

### 🔄 Data Augmentation
- Rotations & spatial shifts
- Horizontal/vertical flips  
- Brightness & contrast adjustment
- Optional elastic deformations

---

## 📏 Evaluation Metrics

### 🎲 Dice Coefficient (Primary Metric)
```
Dice = 2|A ∩ B| / (|A| + |B|)
```
- **Range**: 0 (no overlap) → 1 (perfect overlap)
- **Why**: Robust to class imbalance, directly measures tumor overlap

### 🔗 Intersection over Union (IoU)
```
IoU = |A ∩ B| / |A ∪ B|
```  
- **Range**: 0 → 1
- **Why**: Stricter than Dice, excellent complementary metric

### ✅ Additional Metrics
- **Binary Accuracy**: Pixel-wise classification accuracy
- **Training Loss**: Monitors learning dynamics
- **Validation Loss**: Prevents overfitting

> ⚠️ **Note**: Accuracy can be misleading due to background dominance. Focus on Dice/IoU for segmentation quality.


---

## 📈 Results & Performance

| **Metric** | **Score** |
|------------|-----------|
| **Binary Accuracy** | ~90% |

### 🔗 Pre-trained Models
**Download Models**: [MEGA - Pre-trained Weights](https://mega.nz/folder/119WnKZT#_f_hnmFS1zDjivNvIQUsUQ)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if it helped you!**

Made with ❤️ for medical AI research

</div>
