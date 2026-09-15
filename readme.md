# Brain Tumor Segmentation (U-Net, PyTorch)

Brain tumor segmentation on the LGG (Low-Grade Glioma) MRI Segmentation Dataset, using a U-Net trained from scratch in PyTorch with Dice loss. Trained on Kaggle GPU.

## Overview

This project segments tumor regions in brain MRI scans using a U-Net architecture. The model was trained end-to-end on Kaggle's GPU environment and evaluated against the benchmark result reported in the original `mateuszbuda/brain-segmentation-pytorch` reference implementation.

**Result:** Dice Similarity Coefficient (DSC) of **0.906**, close to the reference benchmark of 0.91.

## Dataset

LGG MRI Segmentation Dataset — paired MRI slices and tumor masks. See `dataset.py` for the loading and preprocessing pipeline.

## Project Structure

```
├── unet.py              # U-Net model architecture
├── dataset.py            # Dataset loading and preprocessing
├── transform.py           # Image/mask transforms and augmentation
├── train.py              # Training loop
├── loss.py               # Dice loss implementation
├── inference.py            # Inference on new scans
├── logger.py              # Training logging utilities
├── utils.py              # Helper functions
├── weights/               # Saved model weights
├── predictions/            # Sample output predictions
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Inference

```bash
python inference.py
```

## Results

| Metric | This implementation | Reference benchmark |
|---|---|---|
| Dice Similarity Coefficient | 0.906 | 0.91 |


## Issues Encountered & Fixes

A few real issues came up during training and inference, documented here in case they're useful to anyone else working from a similar base:

- **Mask value range mismatch.** Ground-truth masks were being loaded with pixel values in the 0–255 range instead of the expected binary 0–1 range, which silently distorted the loss calculation. Fixed by normalizing mask values during preprocessing before they're passed to the loss function.
- **NumPy compatibility issue at inference time.** A NumPy version mismatch broke array handling during inference (works differently under newer NumPy releases than the codebase originally assumed). Resolved by pinning compatible versions in `requirements.txt` and adjusting the affected array operations in `inference.py`.

## Acknowledgements

Built on the architecture and structure of [mateuszbuda/brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch).
