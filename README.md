# 🧠 Brain Tumor Segmentation with U-Net

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> 🎯 A **PyTorch** & TensorFlow implementation of **U-Net** for precise brain tumor segmentation on MRI slices
> 
> **⭐ Primary Implementation**: PyTorch version with 90% accuracy

---

## 🚀 What is this?

This project implements the **U-Net** convolutional neural network architecture for **medical image segmentation**. The **primary PyTorch implementation** is specifically trained to detect and segment brain tumors from 2D MRI slices, producing pixel-level segmentation masks that accurately outline tumor regions.

> **🎯 Main Focus**: The PyTorch version serves as the primary implementation with comprehensive training, evaluation, and performance metrics. TensorFlow implementation is provided as an alternative framework option.

### 🔍 Key Features

- **Encoder-Decoder Architecture**: Contracting path captures context, expanding path produces fine-grained segmentation
- **Skip Connections**: Preserves spatial detail for precise boundary detection  
- **Medical-Grade Accuracy**: Optimized for brain tumor detection in MRI scans
- **Production Ready**: Includes comprehensive training, validation, and evaluation pipelines

---

## 📊 Dataset Overview

### 📁 Available Datasets

#### **PyTorch Implementation**
**LGG MRI Segmentation Dataset**
- **Source**: Brain MRI images with manual FLAIR abnormality segmentation masks from TCIA and TCGA
- **Download**:
```python
import kagglehub
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
```
**🔗 Dataset Link**: [Kaggle - LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

#### **TensorFlow Implementation** 
**BraTS 2019 Dataset**
- **Source**: MICCAI's Dataset on Brain Tumor Segmentation (Year 2019)
- **Download**:
```python
import kagglehub
path = kagglehub.dataset_download("aryashah2k/brain-tumor-segmentation-brats-2019")
```
**🔗 Dataset Link**: [Kaggle - BraTS 2019](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

### 📊 Dataset Statistics

| **Metric** | **LGG (PyTorch)** | **BraTS 2019 (TensorFlow)** |
|------------|-------------------|------------------------------|
| **Type** | 2D MRI slices with binary masks | Multi-modal 3D MRI volumes |
| **Patients** | 110 patients with lower-grade gliomas | 335+ patients with gliomas |
| **Training Samples** | ~2,985 | Variable (3D volumes) |
| **Validation Samples** | ~747 | Variable (3D volumes) |
| **Test Samples** | ~197 | Variable (3D volumes) |
| **Format** | (Image, Mask) pairs | Multi-modal (T1, T1ce, T2, FLAIR) |
| **Labels** | Binary: `1` = tumor, `0` = background | Multi-class: ED, ET, NET/NCR |

### 📝 Data Preprocessing
- **Normalization**: Z-score or min-max scaling per image
- **Augmentation**: Rotations, flips, intensity variations
- **Class Balance**: Handles high background-to-tumor ratio

---

## 🛠️ Tech Stack

### Primary Framework (PyTorch)
```
🔥 PyTorch          # Model architecture & training
📊 Albumentations   # Fast image & mask augmentation  
🖼️  OpenCV/Pillow   # Image I/O operations
🔢 NumPy           # Numerical computations
📈 Matplotlib      # Visualization & plotting
⚡ tqdm            # Progress tracking
```

### Alternative Framework (TensorFlow)
```
🔥 TensorFlow/Keras  # Alternative implementation
🖼️  tf.image         # Image preprocessing & augmentation
📊 tf.data          # Efficient data pipeline
```


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

## 🚀 Getting Started

### PyTorch (Primary Implementation)
```python
# Load pre-trained model
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('best_model.pth'))

# Segment brain tumor
prediction = model(mri_slice)
tumor_mask = (prediction > 0.5).float()
```

### TensorFlow (Alternative Implementation)
```python
# Load pre-trained model
model = tf.keras.models.load_model('unet_brain_tumor_model')

# Segment brain tumor
prediction = model.predict(mri_slice)
tumor_mask = (prediction > 0.5).astype('float32')
```

---

## 📈 Results & Performance

### 🎯 PyTorch Implementation (Primary)

| **Metric** | **Score** |
|------------|-----------|
| **Binary Accuracy** | **90%** |
| **Framework** | PyTorch |
| **Dataset** | LGG MRI Segmentation |

> **📊 Note**: All performance metrics, training configurations, and evaluation results refer to the PyTorch implementation unless otherwise specified.

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
