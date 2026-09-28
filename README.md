# Weapon Detection System

A modular computer-vision application for detecting and reviewing weapons from **images, uploaded videos, and live CCTV streams**.

The project uses a YOLO-based five-class weapon detector with a Streamlit interface. The final deployment candidate is **V3.pt**, trained with **YOLO26m at 1280px** and evaluated on both the original held-out test set and a separate held-out aerial test set.

## Project Overview

The system was developed as an end-to-end ML project covering:

- Multi-source dataset preparation and normalization
- Video-level train/validation/test splitting to avoid temporal leakage
- CVAT-based annotation and quality control
- Hard-example mining and targeted dataset refinement
- V1, V2, V2.1, V2.2 and V3 model development
- Small-object and resolution diagnostics
- Original-domain and held-out aerial evaluation
- Deployment through a Streamlit application
- Image, video and live CCTV inference
- Incident grouping and evidence snapshots for CCTV monitoring

## Detection Classes

| ID | Class |
|---:|---|
| 0 | Knife |
| 1 | Machette |
| 2 | Handgun |
| 3 | Longgun |
| 4 | Other |

## Final Model

**V3.pt**

- Architecture: YOLO26m
- Inference size: 1280 × 1280
- Training dataset: V2.2
- V2.2 training set: 13,989 images
- Initialized from the retained V2.1 model
- Final completed training epoch: 14
- Partial Epoch 15 was excluded from reporting

The V2.1 model is retained separately as the historical comparison/baseline model.

## Model Development

```text
V1 baseline
   ↓
First V2 hard-example-heavy experiment
   ↓
Rejected after severe regression
   ↓
V2.1
Full original training distribution + targeted hard examples
   ↓
V2.2
V2.1 + 354 aerial adaptation images
   ↓
V3
YOLO26m at 1280px
   ↓
Final deployment candidate
```

The first V2 experiment demonstrated that increasing the proportion of hard examples while removing too much of the original distribution could severely reduce generalization. The later V2.1/V2.2 strategy therefore preserved the broad base distribution and added targeted domain-specific examples.

## Dataset

The original surveillance collection contained:

- 85 source videos
- 3,197 extracted frames
- Video-level split: 59 Train / 13 Validation / 13 Test videos
- Train: 2,236 frames
- Validation: 438 frames
- Test: 523 frames

The broader merged dataset was validated as:

- 15,288 images
- 14,129 annotated objects
- 3,784 empty-label images
- Train: 12,199
- Validation: 2,305
- Test: 784

For V2.1, the training set was expanded to 13,635 images using targeted hard examples while keeping the original validation and test sets unchanged.

For V2.2, **354 aerial adaptation images** were added:

- 184 weapon-source images
- 170 no-weapon hard negatives

This produced the final V2.2 training set of **13,989 images**.

The 354 aerial adaptation images were selected at the clip level so that the final aerial evaluation clips remained independent.

## Final Evaluation

### Original locked test set

The final V2.1 and V3 candidates were evaluated at 1280px on the same locked 784-image original test set.

| Metric | V2.1 | V3 |
|---|---:|---:|
| Precision | 0.678 | 0.658 |
| Recall | 0.628 | 0.544 |
| mAP50 | 0.601 | 0.538 |
| mAP50-95 | 0.413 | 0.356 |

This comparison is retained as an original-domain benchmark. V3 was not treated as a universal replacement for V2.1.

### Held-out aerial test

A separate aerial test was kept completely outside V2.2 training:

- 380 frames
- 38 clips
- 19 weapon clips
- 19 no-weapon clips

Full-frame V3 at 1280px produced:

| Metric | V2.1 | V3 |
|---|---:|---:|
| True positives | 45 | 83 |
| True negatives | 124 | 173 |
| False positives | 66 | 17 |
| False negatives | 145 | 107 |
| Precision | 0.405 | 0.830 |
| Recall | 0.237 | 0.437 |
| F1 | 0.299 | 0.572 |
| Specificity | 0.653 | 0.911 |
| False-positive rate | 0.347 | 0.089 |
| Longgun recall | 0.221 | 0.437 |

At clip level, V3 achieved:

- Precision: 0.765
- Recall: 0.684
- F1: 0.722
- Specificity: 0.789

These aerial results are reported separately from the original-domain benchmark because they represent a different evaluation domain.

## Application Features

### Image

- Upload JPG, JPEG or PNG images
- Direct weapon detection
- Bounding-box visualization
- Confidence and top-class information
- Optional person-aware analysis for difficult scenes

### Video

- Upload MP4, MOV, AVI or MKV files
- Sampled-frame inference
- Detection timeline
- Grouped detection windows
- Clickable detection ranges
- Exact timestamp navigation
- Paused evidence review
- Detection-frame gallery

### Live CCTV

- Webcam index or RTSP/HTTP/RTMP source
- Low-latency latest-frame capture
- Default full-frame V3 weapon detection
- Optional person-aware analysis
- Grouped weapon incidents
- Peak-confidence evidence snapshots
- Session incident tiles
- Stop monitoring without losing the current session
- Clear current session
- Saved incident evidence directory

## System Architecture

```text
                 ┌──────────────────┐
                 │ Image / Video /  │
                 │    Live CCTV     │
                 └────────┬─────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Weapon Detector │
                │     V3.pt        │
                └────────┬─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Class + Confidence  │
              │ + Bounding Boxes    │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             │                       │
          Image/Video               CCTV
             │                       │
       Evidence review        Incident grouping
                                     │
                                     ▼
                              Peak-confidence
                                snapshot
```

### Advanced Analysis

The person detector is **not part of the default detection path**.

When enabled:

```text
Person detection
      ↓
Person ROI
      ↓
V3 weapon detector
      ↓
Optional tracking / temporal confirmation
```

The final aerial experiments showed that direct full-frame V3 inference at 1280px was preferable to the person-ROI path for the tested aerial benchmark.

## Repository Structure

```text
Weapon-Detection/
│
├── app.py
├── cctv.py
├── config.py
├── events.py
├── inference.py
├── models.py
├── pipeline.py
├── roi_processor.py
├── temporal_analysis.py
├── tracking.py
├── ui.py
├── video_processor.py
│
├── models/
│   └── README.md
│
├── preprocessing/
│   ├── README.md
│   ├── 01_source_conversion/
│   ├── 02_dataset_preparation/
│   ├── 03_dataset_splitting/
│   ├── 04_hard_example_processing/
│   └── 05_validation/
│
├── training/
│   ├── README.md
│   └── curated training/evaluation code
│
├── dataset/
│   └── README.md
│
├── incidents/
│   └── .gitkeep
│
├── docs/
│   └── Weapon_Detection_Project_Final_Report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

The preprocessing directory is intentionally organized as a **numbered end-to-end pipeline** rather than as a collection of individual experimental scripts:

```text
01_source_conversion
        ↓
02_dataset_preparation
        ↓
03_dataset_splitting
        ↓
04_hard_example_processing
        ↓
05_validation
```

### Preprocessing Pipeline

#### `01_source_conversion/`

Converts source annotations and datasets into the project's YOLO-compatible format, including the source-to-YOLO workflows used for CVAT and Mock Attack data.

#### `02_dataset_preparation/`

Builds and prepares the working datasets. This stage covers frame extraction, annotation workspace creation, dataset merging, class normalization, SOHAS conversion and preparation of selected annotation sets.

#### `03_dataset_splitting/`

Creates the final train/validation/test organization while preserving the video- or source-level separation required to reduce temporal leakage.

#### `04_hard_example_processing/`

Handles targeted hard-example selection and reconstruction, including the V2.1 and V2.2 aerial hard-example workflow.

#### `05_validation/`

Contains the final dataset and source-data auditing tools, including annotation checks, dataset validation, Kaggle inspection and aerial evaluation utilities.

The preprocessing folders contain the **curated versions of the scripts used for the final project workflow**. Earlier experimental scripts, temporary debugging files and superseded iterations are intentionally kept outside the public repository.

### Why the preprocessing code is included

The preprocessing pipeline is an important part of the project rather than a collection of unrelated utility scripts. The final detector depended on substantial dataset engineering, including source conversion, annotation normalization, video-level splitting, hard-example mining and validation.

Including the curated preprocessing pipeline allows the repository to show how the final training data was produced without requiring the large raw datasets to be distributed with the project.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Weapon-Detection
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights

Place the trained weights inside the `models/` directory:

```text
models/
├── V3.pt
└── yolo26n.pt
```

`V3.pt` is the final weapon detector. `yolo26n.pt` is used only for optional person-aware analysis.

The large model weight files are kept outside the main Git history when repository-size limits make direct GitHub storage impractical.

### 5. Run the application

Use Streamlit:

```bash
python -m streamlit run app.py
```

Do **not** run:

```bash
python app.py
```

## CCTV Notes

The `incidents/` directory is used for CCTV evidence snapshots.

Keep the directory present even when it is empty. The application creates the required dated subdirectories when incidents are saved.

Generated incident images should generally remain out of Git history.

## Reproducibility

The project deliberately preserved the following safeguards:

- Train/Validation/Test splits were separated at the source video/clip level where temporal leakage was a concern.
- The original source datasets were preserved separately from working copies.
- The five-class taxonomy was frozen before final annotation and training.
- The original locked Test set was kept separate from the aerial adaptation data.
- Model-development experiments and negative results were documented rather than silently discarded.
- Final original-domain and aerial-domain results are reported separately.

## Limitations

This system is a research and portfolio proof of concept, not a production or safety-critical security product.

Known limitations include:

- Small and heavily occluded weapons can still be missed.
- Detection quality varies across domains and camera viewpoints.
- Aerial evaluation was performed on a limited held-out benchmark.
- Results depend on the selected confidence threshold and inference resolution.
- Frame-level detection is not equivalent to guaranteed real-world threat identification.
- Human verification remains appropriate for high-consequence decisions.

## Documentation

The full project report is included under `docs/` and documents the complete development history, dataset preparation, annotation process, experiments, model evaluation, aerial adaptation and final system design.

## Project Status

**Development status: Complete / Frozen**

The final development phase is closed. The repository is intended to preserve the finished application, final model configuration, evaluation evidence and selected reproducibility materials.

Further model training was intentionally stopped after the V3 evaluation and frontend integration were completed.
