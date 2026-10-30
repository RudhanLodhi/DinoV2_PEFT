# AI4002 — Computer Vision  
## Semester Project Part 1  
**Department of Artificial Intelligence and Data Science**  
**FAST NUCES — Chiniot-Faisalabad Campus**

---

## Section 1: Group Information

| Field | Details |
|---|---|
| **Group No.** | `[GROUP NUMBER]` |
| **Student 1 Name** | `[STUDENT 1 FULL NAME]` |
| **Student 1 Roll No.** | `[ROLL NUMBER]` |
| **Student 2 Name** | `[STUDENT 2 FULL NAME]` |
| **Student 2 Roll No.** | `[ROLL NUMBER]` |
| **Project Title** | Layer-Selective Parameter-Efficient Fine-Tuning of DINOv2 for Multi-Label Radiology Classification |

---

## Section 2: Base Paper Summary

### Bibliographic Information

| Field | Details |
|---|---|
| **Title** | Evaluating General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks |
| **Authors** | Mohammed Baharoon, Waseem Qureshi, Jiahong Ouyang, Yanwu Xu, Abdulrhman Aljouie, Wei Peng |
| **Publication Year** | 2024 (submitted December 2023; revised September 2024) |
| **Venue / Identifier** | arXiv preprint arXiv:2312.02366v4 \[cs.CV\] |
| **SCImago SJR Quartile** | N/A — arXiv preprint (not yet published in a peer-reviewed journal at time of project). Manuscript is under review. |
| **Code Repository** | [https://github.com/MohammedSB/DINOv2ForRadiology](https://github.com/MohammedSB/DINOv2ForRadiology) |

> **Note:** The base paper is a preprint on arXiv. The work has been cited extensively (50+ citations) and is considered a reputable experimental reference in the medical AI community. If a Q1/Q2 journal version is required, the original DINOv2 paper (Oquab et al., 2024, *Transactions on Machine Learning Research*) can serve as an alternative base.

---

### Problem Addressed

Medical image analysis requires models that can generalize across diverse imaging modalities (X-ray, CT, MRI), tasks (classification, segmentation), and anatomical targets. The paper addresses the question: **do general-purpose vision foundation models pre-trained on natural images — specifically DINOv2 — transfer effectively to radiology without task-specific pre-training?** Prior to this work, there was no systematic evaluation of DINOv2 across multiple radiology benchmarks covering different modalities and tasks in a controlled, reproducible setting.

---

### Proposed Model / Approach

The paper is an **experimental study**, not a novel architecture. It evaluates **DINOv2** (Oquab et al., 2023) — a self-supervised Vision Transformer pre-trained on 142 million curated natural images via DINO and iBOT objectives — on nine radiology benchmarks using four adaptation strategies:

| Strategy | Description |
|---|---|
| **k-Nearest Neighbours (kNN)** | No training; L2-normalised features from frozen backbone used directly |
| **Linear Probing** | Frozen backbone; only a single linear classification head is trained |
| **End-to-End Fine-Tuning** | All backbone parameters updated during training |
| **PEFT (LoRA / BitFit)** | Parameter-efficient adapters inserted into frozen ViT; only adapter weights updated |

Segmentation is handled by a U-Net decoder with skip connections from four transformer blocks, trained on top of a frozen DINOv2 backbone. Four DINOv2 scales are evaluated: ViT-S/14, ViT-B/14, ViT-L/14, and ViT-g/14.

---

### Main Reported Results

**Classification (AUROC — Linear Probing)**

| Architecture | NIH Chest X-ray | CheXpert | Brain Tumor MRI |
|---|---|---|---|
| DINOv2 ViT-S/14 | 0.747 | 0.805 | 0.962 |
| **DINOv2 ViT-B/14** | **0.755** | **0.812** | **0.972** |
| DINOv2 ViT-L/14 | 0.763 | 0.821 | 0.974 |
| DINOv2 ViT-g/14 | 0.759 | 0.818 | 0.976 |
| Supervised ViT-L/16 | 0.751 | 0.829 | 0.975 |

**PEFT on ViT-g/14 (CheXpert AUROC)**

| Method | Trainable Params | AUROC |
|---|---|---|
| Linear Probing | ~1,500 | 0.818 |
| BitFit | 0.8M | 0.817 |
| LoRA (r=48) | 8M | 0.823 |
| Full Fine-Tuning | 1,100M | 0.848 |

---

### Key Contributions

1. **First systematic multi-benchmark evaluation** of DINOv2 on radiology — spanning X-ray, CT, and MRI across classification, segmentation, and modality-specific challenges.
2. **Demonstrated strong zero-shot transfer**: DINOv2 linear probing matches or exceeds weakly-supervised CLIP/OpenCLIP models on most benchmarks without any medical pre-training.
3. **Scale vs. efficiency analysis**: ViT-B/14 captures most of ViT-g/14's performance at a fraction of the parameters, making it a practical choice for resource-constrained settings.
4. **PEFT validation**: LoRA achieves 97% of full fine-tuning AUROC (0.823 vs. 0.848 on CheXpert) using only 0.7% of ViT-g/14's parameters.
5. **Open-source benchmark suite**: Released code and reproducibility scripts for all nine benchmarks.

---

## Section 3: Dataset Information

### Dataset Summary Table

| Field | CheXpert | Brain Tumor MRI |
|---|---|---|
| **Dataset Name** | CheXpert (Stanford) | Cheng Brain Tumor MRI |
| **Dataset Link** | [huggingface.co/datasets/danjacobellis/chexpert](https://huggingface.co/datasets/danjacobellis/chexpert) | [platform.ultralytics.com/.../cheng-brain-tumor-mri-detection](https://platform.ultralytics.com/aravinthan-s/datasets/cheng-brain-tumor-mri-detection) |
| **Source / Repository** | HuggingFace Hub (`danjacobellis/chexpert`) | Figshare: DOI 10.6084/m9.figshare.1512427.v8; hosted on Ultralytics Platform |
| **Number of Images** | 224,316 total (frontal + lateral); ~161,792 frontal | 3,064 T1-weighted MRI slices |
| **Number of Patients** | ~65,000 patients | 233 patients |
| **Image Dimensions** | Variable (original); resized to 224×224 for training | 512×512 pixels; resized to 224×224 for training |
| **Image Format** | JPEG (HuggingFace parquet format) | JPEG |
| **Classes / Labels** | 14 pathology labels (multi-label); 5 used for competition benchmark | 3 tumor types: meningioma (0), glioma (1), pituitary (2) |
| **Label Type** | Multi-label binary (positive=1, negative=0, uncertain=−1) | Multi-class (mutually exclusive) |
| **Annotation Type** | NLP-extracted from radiology reports (automated) | Pixel-level tumor masks converted to bounding boxes; image-level class labels |
| **Modality** | Chest X-ray (2D, frontal) | Brain MRI (T1-weighted contrast-enhanced, 2D axial slices) |
| **Train Split** | ~223,414 (frontal) / **10,000 subset used** | ~2,074 images (patient-grouped) |
| **Validation Split** | 234 images (official, fixed) | ~522 images (patient-grouped, no patient overlap) |
| **Test Split** | No official test set (validation used for comparison) | ~468 images (patient-grouped) |
| **Dataset Size** | ~118 GB (full); ~500 MB (HuggingFace parquet) | ~118 MB |
| **License** | Stanford Research Use Agreement (research only) | CC BY 4.0 |

> **Note on Train Subset:** CheXpert's full training set (~223k images) is impractical for Colab without multi-GPU setups. We use a **10,000-sample stratified subset** (seed=42) as the default, with support for 20k and full-data modes via `configs/chexpert.yaml`. Validation and test sets remain at their full fixed sizes.

---

### Dataset Preparation and Basic Preprocessing

#### CheXpert
1. **Loading**: Dataset loaded via HuggingFace `datasets.load_dataset("danjacobellis/chexpert")`.
2. **Label selection**: 5 competition labels extracted: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
3. **Uncertainty handling (U-Zeros policy)**: Uncertain labels (−1) are remapped to 0 (negative). This is the most widely used policy and matches the base paper's approach.
4. **Frontal-only filter**: Lateral views filtered out via the `Frontal/Lateral` column, consistent with the paper's protocol.
5. **Subset sampling**: 10,000 random training examples selected with `seed=42` using HuggingFace `.shuffle().select()`.
6. **Augmentation**: Training — `RandomResizedCrop(224, scale=(0.75, 1.0))` + `RandomHorizontalFlip(p=0.5)`. Validation — `Resize` + `CenterCrop(224)`.
7. **Normalisation**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

#### Brain Tumor MRI
1. **Loading**: NDJSON manifest (`cheng-brain-tumor-mri-detection.ndjson`) parsed to extract image file paths, split assignments, and class labels from YOLO annotation indices.
2. **Patient-grouped splits**: The dataset maintains patient-level separation (no patient appears in more than one split), preventing data leakage.
3. **Image download**: Images downloaded from Ultralytics CDN URLs in the NDJSON manifest, cached to disk.
4. **MRI intensity preprocessing** (base paper standard): Percentile clipping at \[5th, 95th\] percentile, rescaled to \[0, 255\]. This removes MRI acquisition artifacts and extreme intensity outliers.
5. **Channel conversion**: Grayscale MRI slices converted to 3-channel RGB for DINOv2 compatibility.
6. **Augmentation**: Same protocol as CheXpert — `RandomResizedCrop(224, scale=(0.75, 1.0))` + `RandomHorizontalFlip(p=0.5)`.
7. **Bounding box metadata**: YOLO bounding box annotations retained in dataset records as metadata for future ROI-crop experiments; not used during classification training.

---

## Section 4: Experiment 1 — Baseline

### Overview
We implement and evaluate the **linear probing** baseline: the DINOv2 ViT-B/14 backbone is completely frozen, and only a small linear classification head is trained on extracted CLS token features. This replicates the base paper's linear probing protocol using our ViT-B/14 checkpoint.

Two separate baseline notebooks are provided:
- `notebooks/0_Baseline_CheXpert.ipynb` — CheXpert 5-label multi-label classification
- `notebooks/0_Baseline_BrainTumor.ipynb` — Brain Tumor MRI 3-class classification

---

### Hyperparameter Table

| Hyperparameter | CheXpert | Brain Tumor MRI | Source |
|---|---|---|---|
| **Model Architecture** | DINOv2 ViT-B/14 + Linear Head | DINOv2 ViT-B/14 + Linear Head | Base paper |
| **Backbone** | `MHRDYN7/dinov2_vitb14` (86M params, frozen) | `MHRDYN7/dinov2_vitb14` (86M params, frozen) | Custom checkpoint |
| **Head** | Dropout(0.1) → Linear(768→5) | Dropout(0.1) → Linear(768→3) | Task-specific |
| **Trainable Params** | ~3,845 (0.0044%) | ~2,307 (0.0027%) | Measured |
| **Input Image Size** | 224 × 224 | 224 × 224 | Base paper |
| **Batch Size** | 128 | 128 | Base paper |
| **Number of Epochs** | 10 | 20 | Base paper |
| **Learning Rate** | 1e-2 | 1e-2 | Base paper LR grid |
| **Optimizer** | SGD (momentum=0.9) | SGD (momentum=0.9) | Base paper |
| **Loss Function** | BCEWithLogitsLoss | CrossEntropyLoss | Task-specific |
| **LR Scheduler** | None (constant) | None (constant) | Matching paper |
| **Train / Val / Test Split** | 10k / 234 / N/A | ~2,074 / ~522 / ~468 | Dataset splits |
| **Data Augmentation** | RandomResizedCrop(scale=(0.75,1.0)) + HFlip(p=0.5) | RandomResizedCrop(scale=(0.75,1.0)) + HFlip(p=0.5) | Base paper |
| **Uncertainty Policy** | U-Zeros (−1 → 0) | N/A (single label per image) | Base paper |
| **MRI Preprocessing** | N/A | Percentile clip \[5th, 95th\] | Base paper |
| **Hardware / GPU** | Google Colab (T4 or A100) | Google Colab (T4 or A100) | Platform |
| **Random Seed** | 42 | 42 | Fixed for reproducibility |

---

### Results

> **Note:** The tables below are populated after running the notebooks on Google Colab with GPU. Run `0_Baseline_CheXpert.ipynb` and `0_Baseline_BrainTumor.ipynb` first, then fill in values from `experiment_log.csv` and `results.json`.

#### CheXpert — Evaluation Metrics

| Metric | Our Result (ViT-B/14, 10k) | Base Paper (ViT-B/14, full data) | Delta |
|---|---|---|---|
| **Mean AUROC (5 labels)** | `[FILL AFTER RUN]` | 0.812 | `[FILL]` |
| **Atelectasis AUROC** | `[FILL]` | — | — |
| **Cardiomegaly AUROC** | `[FILL]` | — | — |
| **Consolidation AUROC** | `[FILL]` | — | — |
| **Edema AUROC** | `[FILL]` | — | — |
| **Pleural Effusion AUROC** | `[FILL]` | — | — |
| **Macro-F1** | `[FILL]` | — | — |
| **Best Epoch** | `[FILL]` | — | — |

#### Brain Tumor MRI — Evaluation Metrics

| Metric | Our Result (ViT-B/14) | Base Paper (ViT-B/14) | Delta |
|---|---|---|---|
| **Mean AUROC (OvR macro)** | `[FILL AFTER RUN]` | 0.972 | `[FILL]` |
| **Meningioma AUROC** | `[FILL]` | — | — |
| **Glioma AUROC** | `[FILL]` | — | — |
| **Pituitary AUROC** | `[FILL]` | — | — |
| **Accuracy** | `[FILL]` | — | — |
| **Macro-F1** | `[FILL]` | — | — |
| **Best Epoch** | `[FILL]` | — | — |

---

### Training / Validation Graphs

All the following plots are generated and saved by the notebooks to the checkpoint directory:

| Plot | File | Generated by |
|---|---|---|
| Train & Val Loss Curves | `loss_auroc_curves.png` | Cell 17 |
| Val AUROC per Epoch | `loss_auroc_curves.png` | Cell 17 |
| Per-Label AUROC Bar Chart | `per_label_auroc.png` | Cell 18 |
| Full ROC Curves (all labels) | `roc_curves.png` | Cell 19 |
| Confusion Matrix (Brain Tumor) | Generated in notebook | Cell 18 (BrainTumor) |
| Per-Class ROC Curves (Brain Tumor) | Generated in notebook | Cell 19 (BrainTumor) |

> Paste the generated PNG plots here after running the notebooks on Colab.

---

### Comparison with Base Paper Results

#### CheXpert

| Method | Architecture | Training Data | CheXpert AUROC |
|---|---|---|---|
| **Our Experiment (E1)** | ViT-B/14, frozen | 10k subset | `[FILL]` |
| Base Paper — Linear Probe | ViT-B/14, frozen | Full ~161k | **0.812** |
| Base Paper — Linear Probe | ViT-L/14, frozen | Full | 0.821 |
| Base Paper — Linear Probe | ViT-g/14, frozen | Full | 0.818 |
| Base Paper — LoRA PEFT | ViT-g/14 | Full | 0.823 |
| Base Paper — Full Fine-Tune | ViT-g/14 | Full | 0.848 |
| Supervised ViT-L/16 | ViT-L/16 | Full | 0.829 |

**Expected gap vs. base paper:** Our result is expected to be slightly below 0.812 due to the 10k training subset (vs. full ~161k). The primary differences are:
1. Smaller model scale than the paper's ViT-g/14 results (we use ViT-B/14, which the paper shows is slightly below ViT-L and ViT-g).
2. Reduced training data (10k vs. full dataset). This gap is intentional — the preprocessing experiments (E2–E4) test whether enhanced preprocessing can partially compensate.

#### Brain Tumor MRI

| Method | Architecture | Brain Tumor AUROC |
|---|---|---|
| **Our Experiment (E1)** | ViT-B/14, frozen | `[FILL]` |
| Base Paper — Linear Probe | ViT-B/14 | **0.972** |
| Base Paper — Linear Probe | ViT-L/14 | 0.974 |
| Base Paper — Linear Probe | ViT-g/14 | 0.976 |
| Supervised DenseNet201 | DenseNet201 | 0.960 |
| Supervised ViT-L/16 | ViT-L/16 (ImageNet-21k) | 0.975 |

---

## Section 5: Experiment 2 — Preprocessing Technique 1 (Log Transform — CheXpert)

### Notebook
`notebooks/0_Baseline_pre_CheXpert.ipynb`

### Technique Description

**Logarithmic Intensity Transformation** applied to CheXpert chest X-ray images.

**CV Curriculum Reference:** Week 2 — Intensity Transformations

**Mathematical Formula:**

$$s = c \cdot \ln(1 + r) \quad \text{where} \quad c = \frac{255}{\ln(1 + 255)} \approx 36.167$$

Here, $r$ is the input pixel value in $[0, 255]$ and $s$ is the transformed output, also in $[0, 255]$.

**How Applied:**
1. Load chest X-ray as RGB PIL Image.
2. Convert to float32 NumPy array.
3. Apply `np.log1p(arr) * c` channel-wise.
4. Clip to $[0, 255]$ and convert back to uint8 PIL Image.
5. Feed the transformed image into the standard DINOv2 augmentation pipeline.
6. Implemented in the `LogTransform` class using NumPy `log1p` (numerically stable equivalent of $\ln(1 + r)$).

**Why Selected:**
- Chest X-rays typically have **low contrast in lung parenchyma** — the region most critical for detecting subtle opacities (Consolidation, Edema, Atelectasis).
- The log transform **expands the dark pixel range** (lung fields) and **compresses the bright range** (bone, diaphragm), effectively increasing the contrast of diagnostically important regions.
- Computationally efficient — no trainable parameters introduced.
- Directly corresponds to the intensity transformation theory from Week 2 of the CV curriculum.

**Expected Effect:**
- Enhanced visibility of low-intensity lung opacities → potentially higher AUROC for Consolidation, Edema, and Pneumonia labels.
- Possible slight reduction in contrast for bright anatomical structures (ribs, diaphragm) which may reduce Cardiomegaly AUROC marginally.

---

### Settings (Identical to Experiment 1 for Fair Comparison)

| Hyperparameter | Value |
|---|---|
| Model | DINOv2 ViT-B/14 (frozen) |
| Head | Linear(768→5), Dropout(0.1) |
| Batch Size | 128 |
| Optimizer | SGD (mom=0.9, lr=1e-2) |
| Epochs | 10 |
| Loss | BCEWithLogitsLoss |
| Augmentation | Same as E1 (applied AFTER log transform) |
| Dataset | CheXpert 10k subset (same seed=42) |

**Only difference from Experiment 1:** `LogTransform` is inserted as the first step in the preprocessing pipeline before all DINOv2 augmentations.

---

### Results

| Metric | E2: Log Transform | E1: Baseline | Δ (E2 − E1) |
|---|---|---|---|
| **Mean AUROC (5 labels)** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Atelectasis AUROC** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Cardiomegaly AUROC** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Consolidation AUROC** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Edema AUROC** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Pleural Effusion AUROC** | `[FILL]` | `[FILL]` | `[FILL]` |
| **Macro-F1** | `[FILL]` | `[FILL]` | `[FILL]` |

> All graphs (loss curves, AUROC curves, ROC plots, before/after preprocessing visualization, E1 vs E2 comparison bar chart) are saved to `outputs/exp2_pre_log_chexpert/`. Paste plots here after Colab run.

---

## Section 6: Experiment 3 — Preprocessing Technique 2 (CLAHE + Morphological Opening — Brain Tumor)

### Notebook
`notebooks/0_Baseline_pre_BrainTumor.ipynb`

### Technique Description

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** combined with **Morphological Opening** applied to Brain Tumor MRI images.

**CV Curriculum References:**
- Week 3 — Histogram Equalization / Histogram-Based Enhancement
- Week 4 — Morphological Transformations

---

#### Sub-Technique 1: CLAHE

**Mathematical Basis:**  
Standard histogram equalization maps each intensity to a new value based on its CDF:
$$s_k = (L-1) \sum_{j=0}^{k} p_r(r_j)$$

CLAHE improves on this by:
1. Dividing the image into a grid of small tiles (8×8 in our case).
2. Computing and equalizing the histogram of each tile independently.
3. Applying a **clip limit** (2.0) to prevent over-amplification of noise in uniform regions.
4. Bilinearly interpolating tile boundaries to avoid block artifacts.

**Parameters used:** `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`

**Why for MRI:** Brain MRI images contain regions with very different local contrast characteristics (e.g., tumor tissue vs. healthy white matter vs. skull). Global histogram equalization would distort diagnostically relevant local features. CLAHE adapts to each region locally, enhancing tumor boundaries without saturating surrounding tissue.

---

#### Sub-Technique 2: Morphological Opening

**Operation:** Erosion followed by Dilation using an elliptical structuring element.

$$\text{Opening}(I, B) = \text{Dilate}(\text{Erode}(I, B), B)$$

**Parameters used:** Elliptical kernel, `(3, 3)` size (`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))`)

**Effect:** Removes small isolated bright regions (noise artifacts from MRI acquisition, scanner speckle) while preserving larger tumor structures.

**Why for MRI:** T1-weighted contrast-enhanced MRI slices often contain punctate noise artifacts and Gibbs ringing near tissue interfaces. Opening with a small elliptical kernel suppresses these without blurring tumor margins.

---

#### Full Pipeline

```
Raw MRI Image
    │
    ▼
Percentile Clip [5th, 95th]   (removes acquisition outliers — base paper standard)
    │
    ▼
Convert to Grayscale
    │
    ▼
CLAHE (clipLimit=2.0, tileGrid=(8,8))   (local contrast enhancement)
    │
    ▼
Morphological Opening (elliptical 3×3)  (noise removal)
    │
    ▼
Convert to RGB (3-channel)
    │
    ▼
Standard DINOv2 Augmentations (RandomResizedCrop + HFlip + ToTensor + Normalize)
```

---

### Settings (Identical to Brain Tumor Baseline — Experiment 2)

| Hyperparameter | Value |
|---|---|
| Model | DINOv2 ViT-B/14 (frozen) |
| Head | Linear(768→3), Dropout(0.1) |
| Batch Size | 128 |
| Optimizer | SGD (mom=0.9, lr=1e-2) |
| Epochs | 20 |
| Loss | CrossEntropyLoss |
| Augmentation | Same as Brain Tumor Baseline |
| Dataset | Cheng Brain Tumor (all splits, patient-grouped) |

**Only difference from Brain Tumor Baseline:** `MRIEnhancedPreprocessing` replaces the basic `MRIPercentileClip` step, adding CLAHE and Morphological Opening after the clip.

---

### Results

| Metric | E4: CLAHE + Morph | E2: Baseline | Δ (E4 − E2) | Base Paper |
|---|---|---|---|---|
| **Mean AUROC (OvR macro)** | `[FILL]` | `[FILL]` | `[FILL]` | 0.972 |
| **Meningioma AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Glioma AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Pituitary AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Accuracy** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Macro-F1** | `[FILL]` | `[FILL]` | `[FILL]` | — |

> Confusion matrix, per-class ROC curves, 3×4 preprocessing visualization grid, and comparison charts saved to `outputs/exp3_pre_clahe_brain_tumor/`. Paste plots here after Colab run.

---

## Section 7: Comparison and Analysis

### Cross-Experiment Comparison Table

#### CheXpert (Experiments 1 & 3)

| Metric | E1: Baseline | E3: Log Transform | Δ (E3 − E1) | Base Paper ViT-B/14 |
|---|---|---|---|---|
| **Mean AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | 0.812 |
| **Atelectasis AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Cardiomegaly AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Consolidation AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Edema AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Pleural Effusion AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Macro-F1** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Trainable Params** | ~3,845 | ~3,845 | 0 | ~1,500 (paper, ViT-g) |

#### Brain Tumor MRI (Experiments 2 & 4)

| Metric | E2: Baseline | E4: CLAHE + Morph | Δ (E4 − E2) | Base Paper ViT-B/14 |
|---|---|---|---|---|
| **Mean AUROC (macro)** | `[FILL]` | `[FILL]` | `[FILL]` | 0.972 |
| **Meningioma AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Glioma AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Pituitary AUROC** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Accuracy** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Macro-F1** | `[FILL]` | `[FILL]` | `[FILL]` | — |
| **Trainable Params** | ~2,307 | ~2,307 | 0 | ~1,500 (paper, ViT-g) |

---

### Detailed Analysis

> **Instructions:** Complete this section after running all 4 notebooks on Colab and collecting results. The questions below are answered in template form; replace the analysis in `[...]` blocks with your actual observations.

---

#### 1. Which experiment performed better overall?

**CheXpert:** `[Compare E1 vs E3 mean AUROC. Example: "E3 (Log Transform) achieved a mean AUROC of X.XXX vs. X.XXX for the baseline, representing a +0.0XX improvement / a −0.0XX drop."]`

**Brain Tumor:** `[Compare E2 vs E4. Example: "E4 (CLAHE + Morphological Opening) achieved AUROC of X.XXX vs. X.XXX baseline..."]`

---

#### 2. Did Preprocessing Technique 1 (Log Transform) improve the CheXpert baseline?

`[Fill in after results are obtained. Expected reasoning: The log transform is expected to improve AUROC for labels related to parenchymal opacity (Consolidation, Edema, Atelectasis) by enhancing low-intensity regions. It may slightly hurt Cardiomegaly or Pleural Effusion where the bright/dark boundary structure is already clearly visible. The net effect on mean AUROC depends on which labels dominate the validation set.]`

**Hypothesis being tested:** A logarithmic intensity transformation that expands low-intensity detail in lung fields will improve the ability of a frozen DINOv2 encoder to distinguish pathological opacity patterns from healthy lung tissue, resulting in a higher mean AUROC.

---

#### 3. Did Preprocessing Technique 2 (CLAHE + Morphological Opening) improve the Brain Tumor baseline?

`[Fill in after results. Expected reasoning: CLAHE is expected to improve AUROC for meningioma and glioma where tumor tissue has distinct but subtle local contrast differences from surrounding healthy tissue. Morphological opening may help by removing noise that could confuse the linear probe. However, overly aggressive CLAHE or morphological operations could over-enhance background anatomy, potentially reducing performance for pituitary tumors which are small and near bright regions of the brain.]`

**Hypothesis being tested:** CLAHE-based local contrast enhancement combined with morphological noise removal will help the DINOv2 encoder better discriminate between tumor classes by presenting cleaner, higher-contrast features, resulting in a higher AUROC.

---

#### 4. Which preprocessing technique yielded the largest improvement?

`[Fill in after collecting results. Compare the absolute AUROC delta for CheXpert (Log Transform) vs. Brain Tumor (CLAHE + Morphological Opening). Note: direct comparison between the two datasets is not straightforward since they have different tasks (multi-label vs. multi-class) and different class counts.]`

---

#### 5. Which metrics improved and which decreased? Why?

**CheXpert Log Transform:**
- `[Consolidation AUROC: Expected to improve — log transform enhances subtle ground-glass opacities]`
- `[Edema AUROC: Expected to improve — bilateral increased opacity in lower lung fields]`
- `[Cardiomegaly AUROC: May decrease — heart silhouette already high-contrast; log transform compresses bright areas]`
- `[Pleural Effusion AUROC: Uncertain — depends on whether the blunting of costophrenic angles is in the dark or bright range]`

**Brain Tumor CLAHE:**
- `[Glioma AUROC: Expected to improve — gliomas show infiltrative low-contrast boundaries that CLAHE enhances]`
- `[Meningioma AUROC: Expected to improve — well-defined but sometimes subtle contrast enhancement in dural tail]`
- `[Pituitary AUROC: May worsen — small sella region; aggressive local enhancement may confuse nearby bright vessels]`

---

#### 6. Were any undesirable artifacts introduced?

**Log Transform artifacts:**
- `[Expected: None significant for natural dynamic ranges. Possible slight flattening of high-intensity structures (ribs) may make cardiomegaly boundaries less sharp.]`

**CLAHE + Morphological Opening artifacts:**
- `[Possible: CLAHE may over-amplify background noise in nearly uniform regions (CSF, ventricles). Morphological opening may slightly erode very small pituitary tumor signal regions. This would manifest as reduced pituitary AUROC.]`

---

#### 7. Visual Observations

`[Paste side-by-side visualizations here after running notebooks:]`

**CheXpert — Log Transform Effect:**
- `[Insert: Original vs. Log-Transformed chest X-ray images from Cell 6 of 0_Baseline_pre_CheXpert.ipynb]`
- `[Observation: e.g., "The log-transformed images show enhanced visibility of lung field texture, with subtle reticular opacities now clearly visible in the right lower lobe."]`

**Brain Tumor — CLAHE + Morphological Opening Effect:**
- `[Insert: 3×4 visualization grid from Cell 7 of 0_Baseline_pre_BrainTumor.ipynb showing Raw / Clipped / CLAHE / Morph for each tumor class]`
- `[Observation: e.g., "Glioma images show enhanced contrast at tumor margins after CLAHE. Pituitary images show slight over-enhancement of adjacent sella region."]`

---

#### 8. Limitations

1. **Training subset size:** Using 10k of ~161k CheXpert images limits statistical representativeness. Our baseline AUROC is expected to be lower than the paper's full-data result (0.812) by approximately 0.01–0.03.
2. **No hyperparameter tuning for preprocessing:** The log transform constant and CLAHE parameters (clipLimit, tileGridSize) were set to standard defaults. Dataset-specific tuning could yield further improvement.
3. **Fixed preprocessing applied globally:** Log transform is applied uniformly regardless of initial image brightness. Some X-rays may be over-exposed and would benefit from a different transformation (e.g., gamma correction or histogram equalization instead of log).
4. **Linear probing ceiling:** With only 3,845 trainable parameters, the model's expressiveness is fundamentally limited. Preprocessing improvements that require non-linear feature interactions cannot be fully exploited by a linear head.
5. **Expired CDN URLs:** Brain Tumor image CDN URLs in the NDJSON manifest may expire, requiring manual regeneration from the Ultralytics platform.

---

## Section 8: Proposed Advanced Direction for Part 2

### A. Observed Limitation in the Base Model / Current Experiments

The linear probing baseline reveals a fundamental constraint: by freezing the entire DINOv2 backbone, we rely entirely on features that were learned from natural images (LVD-142M dataset). While DINOv2's self-supervised pre-training produces remarkably transferable representations, there remains a **distribution gap** between natural images and medical radiographs:

1. **Domain gap:** Chest X-rays are grayscale, high-contrast, and contain domain-specific structures (pulmonary vasculature, diaphragm silhouettes, cardiac borders) that have no close analogues in natural images.
2. **Frozen features are suboptimal:** The paper itself demonstrates that full fine-tuning (0.848 AUROC) substantially outperforms linear probing (0.812 AUROC) on CheXpert, confirming that backbone features must be adapted to the medical domain to reach peak performance.
3. **Training efficiency:** Full fine-tuning of ViT-B/14 (86M parameters) requires significant GPU memory and is prone to overfitting on limited medical data. This makes it impractical for resource-constrained settings like ours.

**Core limitation:** *How can we adapt the DINOv2 backbone to medical imaging efficiently — improving over linear probing — without the computational and overfitting risks of full fine-tuning?*

---

### B. Proposed Improvement

We propose implementing **Layer-Selective Parameter-Efficient Fine-Tuning (LS-PEFT)** of DINOv2 ViT-B/14:

Instead of applying LoRA adapters uniformly to all 18 transformer layers (as in the base paper's ViT-g/14 setup), we will:
1. **Profile layer informativeness** using validation AUROC of intermediate CLS tokens from different layers.
2. **Apply LoRA adapters selectively** only to the top-k most informative layers (hypothesis: later layers are more task-specific and benefit most from adaptation).
3. **Compare against uniform LoRA** with parameter-matched configurations to isolate the effect of layer selection from adapter capacity.

This extends the base paper's PEFT experiments (Table 9) with a controlled ablation study that could demonstrate comparable AUROC with fewer trainable parameters — a meaningful contribution for resource-efficient medical AI.

---

### C. Advanced Model 1

**Method:** LoRA (Low-Rank Adaptation) on DINOv2 ViT-B/14

**Reference:** Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022

**Why Selected:**
- LoRA is the method validated in the base paper (Table 9) on ViT-g/14, making it the natural next step for our ViT-B/14 experiments.
- By inserting low-rank matrices ($\Delta W = AB$, rank $r \ll d$) into the attention projections (Query, Value), LoRA updates < 1% of parameters.
- Enables controlled ablation: rank $r \in \{4, 8, 16, 32, 48\}$ directly controls the capacity-efficiency trade-off.
- The base paper reports LoRA achieving 97% of full fine-tuning AUROC (0.823 vs 0.848) on CheXpert using ViT-g/14 at 0.7% trainable parameters. We will validate this for ViT-B/14 and test whether layer selection can improve over uniform LoRA.

**Expected Advantage:**
- AUROC improvement over linear probing (0.812 → ~0.82–0.84 range).
- Requires only ~2–5M trainable parameters vs. 86M for full fine-tuning.
- Adapters can be merged into the backbone for zero-latency inference.

---

### D. Advanced Model 2

**Method:** IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Reference:** Liu et al. (2022), *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*, NeurIPS 2022

**Why Selected:**
- IA³ is **even more parameter-efficient than LoRA**: it learns per-channel scaling vectors for keys, values, and FFN activations rather than low-rank weight matrices. Typical IA³ uses only 0.02% of model parameters.
- This makes it ideal for few-shot medical imaging scenarios where labelled data is scarce.
- IA³ provides a different adaptation paradigm (activation scaling vs. weight modification) that complements LoRA in the ablation study.
- The base paper's BitFit (bias-only) achieves 0.817 AUROC vs. LoRA's 0.823 on CheXpert; IA³ is expected to fall between BitFit and LoRA in expressiveness while being far more efficient than LoRA.

**Expected Advantage:**
- Training and inference with IA³ is faster than LoRA due to fewer trainable parameters.
- Particularly strong in data-scarce regimes (our 10k CheXpert subset).
- Provides a third data point in the efficiency-performance trade-off curve alongside linear probing and LoRA.

---

### E. Expected Contribution in Part 2

| Contribution | Description |
|---|---|
| **E0 (done)** | Linear probing baseline on CheXpert and Brain Tumor with exact paper hyperparams |
| **E1 — Rank Sweep** | LoRA with $r \in \{4, 8, 16, 32, 48\}$; characterise AUROC vs. trainable params trade-off |
| **E2 — Layer Study** | Q+V only vs. Q+K+V+O; upper-quarter vs. lower vs. distributed layers |
| **E3 — Layer-Selective LoRA** | Apply LoRA only to top-k layers selected from E2 analysis; parameter-matched comparison |
| **E4 — IA³ Comparison** | IA³ with matched training protocol; compare to LoRA at similar parameter count |
| **Final Contribution** | A layer-selective LoRA recipe for DINOv2 ViT-B/14 that achieves competitive AUROC with fewer trainable parameters than uniform LoRA, validated on CheXpert and Brain Tumor MRI |

The expected finding is that **the top 25% of DINOv2 transformer layers (layers 14–17 of 18) capture most of the adaptation benefit**, reducing trainable parameters by ~75% vs. full LoRA while retaining >95% of the AUROC improvement over linear probing. This would be consistent with the broader finding in NLP that later transformer layers are more task-specific and benefit most from adaptation.

---

## Appendix: Repository Structure

```
DinoV2B/
├── Artifacts/                              ← PDFs (base paper, comp paper, project spec)
├── configs/
│   ├── chexpert.yaml                       ← Dataset and training config
│   ├── brain_tumor.yaml
│   └── peft/
│       ├── lora.yaml                       ← LoRA hyperparams (r=48, alpha=16)
│       ├── ia3.yaml                        ← IA³ scaffold
│       └── bitfit.yaml
├── src/
│   ├── data/                               ← Dataset loaders, transforms
│   ├── models/                             ← Backbone, heads, PEFT wrappers
│   ├── training/                           ← Trainer, losses, schedulers
│   ├── evaluation/                         ← Metrics, evaluator loop
│   └── utils/                              ← Config, logging, visualization
└── notebooks/
    ├── 0_Baseline_CheXpert.ipynb           ← Section 4 (Experiment 1)
    ├── 0_Baseline_BrainTumor.ipynb         ← Section 4 (Experiment 1, Brain Tumor)
    ├── 0_Baseline_pre_CheXpert.ipynb       ← Section 5 (Experiment 2)
    ├── 0_Baseline_pre_BrainTumor.ipynb     ← Section 6 (Experiment 3)
    ├── 04_lora_chexpert.ipynb              ← Part 2: E1 LoRA rank sweep
    ├── 05_lora_layer_study_chexpert.ipynb  ← Part 2: E2–E3 layer study
    ├── 06_lora_brain_tumor.ipynb           ← Part 2: Brain Tumor LoRA
    └── 07_evaluation_and_viz.ipynb         ← Section 7 cross-experiment comparison
```

---

*Report prepared for AI4002 Computer Vision, FAST NUCES Chiniot-Faisalabad.*  
*Base model: DINOv2 ViT-B/14 (`MHRDYN7/dinov2_vitb14`). Base paper: Baharoon et al. (2024), arXiv:2312.02366v4.*
