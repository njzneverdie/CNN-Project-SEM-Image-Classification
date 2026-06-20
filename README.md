# SEM Image Classification

Automated 10-class defect classification of Scanning Electron Microscope (SEM) images using InceptionV3 transfer learning.

## Demo

- **Live notebook**: https://njzneverdie.github.io/CNN-Project-SEM-Image-Classification/notebook.html
- **Presentation**: [`CNN_project_presentation.pdf`](CNN_project_presentation.pdf)

## Dataset

[SEM Images 299x299](https://www.kaggle.com/datasets/sharumaan/semimages-299x299) — 18,577 images across 10 material classes.

| Class | Images |
|---|---|
| MEMS devices & electrodes | 4,158 |
| Nanowires | 3,656 |
| Particles | 3,412 |
| Patterned surface | 3,310 |
| Tips | 1,561 |
| Biological | 953 |
| Powder | 895 |
| Films / Coated Surface | 308 |
| Porous Sponge | 171 |
| Fibres | 153 |

## Approach

### Phase 1 — Transfer Learning (frozen backbone)
- Base model: **InceptionV3** pretrained on ImageNet (weights frozen)
- Added head: `GlobalAveragePooling2D → Dropout(0.3) → Dense(256) → Dropout(0.3) → Softmax(10)`
- Optimizer: Adam `lr=1e-3`
- Class imbalance handled via `compute_class_weight('balanced')`

### Phase 2 — Fine-tuning
- Unfreeze last 50 layers of InceptionV3
- Optimizer: Adam `lr=1e-5` (100× smaller to avoid destroying pretrained weights)

### Data Augmentation
`rotation ±10°`, `width/height shift 5%`, `zoom 10%`, `horizontal flip`

## Results (Phase 1, 5 epochs)

| Metric | Value |
|---|---|
| Test Accuracy | **82.2%** |
| Weighted F1 | 0.81 |

Confusion matrix and per-class report available in the [live notebook](https://njzneverdie.github.io/CNN-Project-SEM-Image-Classification/notebook.html).

## Setup

```bash
pip install tensorflow scikit-learn imbalanced-learn matplotlib seaborn
```

Open `sem_images_classification.ipynb` in Google Colab or Jupyter.

## Project Structure

```
├── sem_images_classification.ipynb        # Clean notebook (no outputs)
├── sem_images_classification_with_outputs.ipynb  # Notebook with full outputs
├── CNN_project_presentation.pdf           # Project slides
└── docs/                                  # GitHub Pages
    ├── index.html
    └── notebook.html
```
