# DINOv2 PEFT — Medical Image Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a modular, reproducible framework for Parameter-Efficient Fine-Tuning (PEFT) of Meta's DINOv2 vision transformer (`ViT-B/14`) on medical imaging benchmarks. We evaluate parameter efficiency, adaptation capacity, and convergence behavior across two distinct medical imaging modalities and task paradigms:

1. **CheXpert Chest X-ray**: 14-label multi-label classification evaluating thoracic pathologies.
2. **Brain Tumor MRI**: 3-class classification (meningioma, glioma, pituitary) on brain MRI slices.

We systematically compare baseline Linear Probing, Low-Rank Adaptation (LoRA) — including rank sweeps, target-module studies, and layer-selective adaptation — as well as activation scaling via $(IA)^3$ and bias-only tuning (BitFit).

## Repository Structure

```
DinoV2B/
├── configs/           # YAML hyperparameter configs
│   └── peft/          # PEFT method configs
├── src/
│   ├── data/          # Dataset loaders and transforms
│   ├── models/        # Backbone, heads, PEFT wrappers
│   ├── training/      # Trainer, losses, schedulers
│   ├── evaluation/    # Metrics and evaluation loop
│   └── utils/         # Config loading, logging, visualization
└── notebooks/         # Colab-ready experiment notebooks
```

## Datasets

### CheXpert
- **Task**: Multi-label chest radiograph classification across 14 radiological observations.
- **Source**: Hugging Face repository [`danjacobellis/chexpert`](https://huggingface.co/datasets/danjacobellis/chexpert).
- **Subsets**: Default training subset of 10,000 images (configurable to 20,000 or full dataset via configuration).
- **Protocol**: Frontal-view filtering, standardized uncertainty label handling policies (`u_zeros` or `u_ones`), and fixed validation/test splits.

### Brain Tumor MRI
- **Task**: 3-class classification distinguishing meningioma (0), glioma (1), and pituitary (2) tumors.
- **Source**: Cheng et al. Figshare dataset comprising 3,064 T1-weighted contrast-enhanced MRI slices acquired from 233 patients.
- **Metadata**: Bounding box annotations preserved as metadata while training 224×224 image classification models.

## Model

The backbone foundation model is DINOv2 ViT-B/14 loaded via Hugging Face `AutoModel`:
- **Hugging Face Model ID**: `MHRDYN7/dinov2_vitb14`
- **Architecture**: Vision Transformer Base with patch size 14×14 (`embed_dim=768`, 12 layers, 12 attention heads).
- **Features**: Global pooling / CLS token representation extraction coupled with modular task-specific classification heads.

## PEFT Methods

- **Linear Probing (E0 baseline)**: Freezes the entire DINOv2 backbone and trains only the task-specific classification head.
- **LoRA (Low-Rank Adaptation)**: Fully implemented low-rank parameter updates on attention projections.
  - *Rank Sweep*: Systematic evaluation across ranks $r \in [4, 8, 16, 32, 48]$.
  - *Target Module Study*: Comparison across query, value, key, and feed-forward projections.
  - *Layer-Selective Adaptation*: Upper-quarter, lower-quarter, distributed, and custom layer targeting to isolate where medical adaptation matters most.
- **$\text{IA}^3$ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**: Scaffolded for Phase III experiments, scaling keys, values, and intermediate feed-forward activations with learned vectors.
- **BitFit**: Bias-only parameter fine-tuning freezing all backbone weights except bias terms.

## Quick Start — Google Colab

To run experiments in Google Colab or an environment with GPU acceleration:

```bash
git clone https://github.com/YOUR_ORG/DinoV2B.git
cd DinoV2B
pip install -e . -r requirements.txt
```

Then open any notebook in `notebooks/` (e.g., CheXpert or Brain Tumor training pipelines) and execute the self-contained experiment cells.

## Local Development (no GPU)

For development, testing, and debugging on CPU-only machines:

1. Clone and install the repository in editable mode:
   ```bash
   pip install -e . -r requirements.txt
   ```
2. Verify imports, configuration loading, dataset pipelines, and model architecture definitions with smoke tests:
   ```bash
   python -c "import dinov2_peft; print('dinov2_peft installed successfully')"
   ```

## Experiment Tracking

All training runs log runtime metrics, parameter counts, and validation/test scores:
- An `experiment_log.csv` file is automatically appended per run under the configured output directory.
- Logged fields include Run ID, dataset, backbone, PEFT method, rank/alpha, adapted layers, trainable parameters, peak GPU memory, training duration, and mean per-label / multi-class AUROC and Macro-F1.

## Citation

If you use this repository or build upon this work, please cite the underlying models and benchmarks:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{irvin2019chexpert,
  title={CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Blankenberg, Katie and others},
  booktitle={Proceedings of the AAAI Conference on Human Computation and Crowdsourcing},
  volume={33},
  number={01},
  pages={590--597},
  year={2019}
}

@article{cheng2017brain,
  title={Brain tumor dataset},
  author={Cheng, Jun},
  journal={figshare},
  year={2017},
  doi={10.6084/m9.figshare.1512427.v5}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
