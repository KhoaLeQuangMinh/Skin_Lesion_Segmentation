# Skin Lesion Segmentation via Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
[![Competition](https://img.shields.io/badge/Competition-AI_Vietnam_WarmUp-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/warm-up-program-ai-vietnam-skin-segmentation/overview)
![Rank](https://img.shields.io/badge/Rank-13th_Place-gold)
![Metric](https://img.shields.io/badge/Score-0.876-success)

## 📌 Project Overview
Melanoma is the deadliest form of skin cancer, but early detection significantly improves survival rates. This project implements a robust semantic segmentation pipeline to identify skin lesions in dermoscopic images.

The solution addresses key challenges in medical imaging: **hair occlusion, variable skin tones, and extreme class imbalance.** The final ensemble achieved a **score of 0.876**, ranking **13th** in the private leaderboard.

## 🏆 Key Achievements
*   **Rank 13:** Top-tier performance on the private test set.
*   **Hybrid Loss Landscape:** Engineered a custom loss function combining region-based, distribution-based, and boundary-based losses.
*   **Artifact Removal:** Implemented a conditional algorithm to remove hair occlusions while preserving lesion texture.

## ⚙️ Technical Pipeline

The solution follows a 5-stage pipeline designed for robustness and generalizability.

### 1. Advanced Data Augmentation
To prevent overfitting and ensure the model generalizes across different skin tones and lighting conditions:
*   **Artifact Removal:** Custom "Hair Removal" algorithm using morphological filtering (selecting structural elements based on aspect ratio) to inpaint hair pixels.
*   **Color Invariance:** RGB and HSV shifts to handle diverse skin types.
*   **Geometric Transforms:** Horizontal Flip, Random Rotate (90°), Affine, and Elastic Transformations.
*   **Regularization:** Coarse Dropout to force the model to learn context rather than relying on specific features.

### 2. Adaptive Preprocessing
Input images undergo adaptive enhancement before entering the network:
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied strictly to the **L-channel** (LAB color space) to enhance texture contrast without distorting color information.
*   **Conditional Smoothing:** Dynamic application of Median Blur based on image Standard Deviation (removing high-frequency noise only when necessary).

### 3. Hybrid Loss Function Engineering
A composite loss function was designed to stabilize convergence and refine boundaries:

$$ L_{total} = 0.6 \cdot L_{FocalTversky} + 0.2 \cdot L_{DynamicBCE} + 0.2 \cdot L_{Surface} $$

*   **Focal Tversky (60%):** Handles the foreground-background class imbalance and forces the model to focus on hard-to-classify lesion pixels (Recall focus).
*   **Dynamic BCE (20%):** Stabilizes pixel-wise classification probabilities.
*   **Surface Loss (20%):** A boundary-aware loss that penalizes discrepancies between the predicted contour and the ground truth, sharpening the edges.

### 4. Model Architecture & Selection
I benchmarked a wide variety of Encoder-Decoder architectures to find the optimal balance of receptive field and resolution:

| Model Architecture | Strengths in this Domain |
| :--- | :--- |
| **Polar-ResUnet++** | Excellent for radial symmetry often found in lesions. |
| **SegFormer** | Transformer-based attention captures long-range dependencies. |
| **DeepLabv3+** | Atrous convolution effectively handles multi-scale lesions. |
| **Unet++** | Nested skip connections reduce the semantic gap between encoder/decoder. |

*Additional experiments included: EPolar-ResUnet, PSPNet, and standard Unet.*

### 5. Post-Processing
*   **CRFs (Conditional Random Fields):** Applied as a refinement step to smooth segmentation boundaries and remove isolated noise pixels.
*   **Mask-to-RLE:** Efficient encoding for submission.

## 🛠️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/KhoaLeQuangMinh/Skin_Lesion_Segmentation.git