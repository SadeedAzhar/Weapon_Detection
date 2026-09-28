"""Weapon Detection — V2 Dataset Construction & Training

Cleaned project-facing version of the V2 workflow. The final deployed model is from V3.
"""


# ====================================================================
# CODE SECTION 1
# ====================================================================

# ================================================================
# V2 DATASET BUILDER
# V1 BASE SUBSET + CONSOLIDATED HARD EXAMPLES
# ================================================================

!pip -q install ultralytics

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import zipfile
import shutil
import random
import hashlib
import json
import csv
from collections import Counter, defaultdict

print("=" * 70)
print("V2 DATASET BUILDER")
print("=" * 70)

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

DRIVE_ROOT = Path("/content/drive/MyDrive")

BASE_ZIP = DRIVE_ROOT / "Final_Dataset_Split" / "Final_Dataset_Split.zip"

HARD_ZIP = (
    DRIVE_ROOT
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Hard_Examples_Consolidated.zip"
)

MODEL_PATH = (
    DRIVE_ROOT
    / "Weapon_Detection_Training"
    / "FINAL_CONTINUATION"
    / "checkpoints"
    / "best.pt"
)

WORK_ROOT = Path("/content/V2_Build")

BASE_EXTRACT = WORK_ROOT / "base_extracted"
HARD_EXTRACT = WORK_ROOT / "hard_extracted"

FINAL_DATASET = WORK_ROOT / "V2_Training"

TARGET_BASE_IMAGES = 5000
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

random.seed(RANDOM_SEED)

# ------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------

print("\n" + "=" * 70)
print("PATH CHECK")
print("=" * 70)

for label, path in [
    ("Base dataset ZIP", BASE_ZIP),
    ("Hard-example ZIP", HARD_ZIP),
    ("V1 best.pt", MODEL_PATH),
]:
    print(f"{label:<25}: {path}")
    print(f"Exists{'':<20}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

print("\nAll required Drive artifacts found.")

# ------------------------------------------------
# CLEAN PREVIOUS COLAB BUILD
# ------------------------------------------------

if WORK_ROOT.exists():
    print(f"\nRemoving previous build: {WORK_ROOT}")
    shutil.rmtree(WORK_ROOT)

BASE_EXTRACT.mkdir(parents=True)
HARD_EXTRACT.mkdir(parents=True)

# ------------------------------------------------
# ZIP EXTRACTION
# ------------------------------------------------

def extract_zip(zip_path, output_dir, label):
    print("\n" + "=" * 70)
    print(f"EXTRACTING {label}")
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        print(f"Files in archive: {len(members)}")
        z.extractall(output_dir)

    print(f"Extracted to: {output_dir}")


extract_zip(BASE_ZIP, BASE_EXTRACT, "BASE DATASET")
extract_zip(HARD_ZIP, HARD_EXTRACT, "HARD EXAMPLES")

# ------------------------------------------------
# FIND DATA DIRECTORIES
# ------------------------------------------------

def find_named_dirs(root, dirname):
    return [
        p for p in root.rglob(dirname)
        if p.is_dir()
    ]


def find_train_structure(root):
    candidates = []

    for images_dir in root.rglob("images"):
        if not images_dir.is_dir():
            continue

        labels_dir = images_dir.parent / "labels"

        if labels_dir.exists() and labels_dir.is_dir():
            candidates.append((images_dir, labels_dir))

    return candidates


base_candidates = find_train_structure(BASE_EXTRACT)
hard_candidates = find_train_structure(HARD_EXTRACT)

print("\n" + "=" * 70)
print("DISCOVERED DATA STRUCTURES")
print("=" * 70)

print("\nBase candidates:")
for img, lab in base_candidates:
    print("Images:", img)
    print("Labels:", lab)

print("\nHard-example candidates:")
for img, lab in hard_candidates:
    print("Images:", img)
    print("Labels:", lab)

if not base_candidates:
    raise RuntimeError("Could not find base images/labels structure.")

if not hard_candidates:
    raise RuntimeError("Could not find hard-example images/labels structure.")

# Prefer the structure containing a train directory
base_train = None

for img, lab in base_candidates:
    if img.parent.name.lower() == "train":
        base_train = (img, lab)
        break

if base_train is None:
    base_train = base_candidates[0]

hard_images_dir, hard_labels_dir = hard_candidates[0]
base_images_dir, base_labels_dir = base_train

print("\nSelected base training images:", base_images_dir)
print("Selected base training labels:", base_labels_dir)

print("Selected hard-example images:", hard_images_dir)
print("Selected hard-example labels:", hard_labels_dir)

# ------------------------------------------------
# COLLECT BASE TRAINING RECORDS
# ------------------------------------------------

print("\n" + "=" * 70)
print("ANALYZING ORIGINAL V1 TRAINING SET")
print("=" * 70)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

base_records = []

for image_path in sorted(base_images_dir.rglob("*")):

    if image_path.suffix.lower() not in image_extensions:
        continue

    rel = image_path.relative_to(base_images_dir)
    label_path = base_labels_dir / rel.with_suffix(".txt")

    if not label_path.exists():
        raise RuntimeError(f"Missing label:\n{label_path}")

    classes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            cls = int(parts[0])

            if cls not in CLASS_NAMES:
                raise RuntimeError(
                    f"Unexpected class {cls} in {label_path}"
                )

            classes.append(cls)

    base_records.append({
        "image": image_path,
        "label": label_path,
        "classes": classes,
        "relative": rel,
    })

print(f"Base training images found: {len(base_records)}")

if len(base_records) != 12199:
    print(
        "\nWARNING: Expected 12,199 training images "
        f"according to our verified dataset record, but found {len(base_records)}."
    )
    print("The script will continue, but this should be reviewed.")

# ------------------------------------------------
# BASE CLASS DISTRIBUTION
# ------------------------------------------------

base_class_images = Counter()
base_empty = 0

for r in base_records:

    unique_classes = set(r["classes"])

    if not unique_classes:
        base_empty += 1

    for cls in unique_classes:
        base_class_images[cls] += 1

print("\nBase image distribution:")

for cls, name in CLASS_NAMES.items():
    print(
        f"{name:<12}: "
        f"{base_class_images[cls]:>6} images"
    )

print(f"{'Negatives':<12}: {base_empty:>6} images")

# ------------------------------------------------
# CREATE CLASS BUCKETS
# ------------------------------------------------

class_buckets = defaultdict(list)
negative_bucket = []

for record in base_records:

    classes = set(record["classes"])

    if not classes:
        negative_bucket.append(record)

    for cls in classes:
        class_buckets[cls].append(record)

# ------------------------------------------------
# STRATIFIED BASE SELECTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("SELECTING REPRESENTATIVE BASE SUBSET")
print("=" * 70)

# Target approximately the original class proportions.
# Images can belong to multiple classes, so selection uses
# iterative weighted coverage rather than independent sampling.

selected = {}
remaining = {id(r): r for r in base_records}

# Desired class targets based on original proportions.
class_targets = {}

for cls in CLASS_NAMES:

    proportion = (
        base_class_images[cls]
        / max(1, len(base_records))
    )

    class_targets[cls] = max(
        1,
        round(TARGET_BASE_IMAGES * proportion)
    )

# Negative target preserves the original negative proportion.
negative_target = round(
    TARGET_BASE_IMAGES
    * (base_empty / max(1, len(base_records)))
)

print("Initial target distribution:")

for cls, name in CLASS_NAMES.items():
    print(f"{name:<12}: {class_targets[cls]}")

print(f"{'Negatives':<12}: {negative_target}")

# We use a greedy coverage strategy:
# each candidate gets a score based on how much it helps
# classes that are still below target.

candidate_pool = list(base_records)
random.shuffle(candidate_pool)

current_class_counts = Counter()
current_negative_count = 0

while len(selected) < TARGET_BASE_IMAGES and candidate_pool:

    best_record = None
    best_score = -1

    # Evaluate a bounded random sample for efficiency.
    sample_size = min(500, len(candidate_pool))
    sample = random.sample(candidate_pool, sample_size)

    for record in sample:

        if id(record) in selected:
            continue

        classes = set(record["classes"])

        score = 0.0

        for cls in classes:

            target = class_targets[cls]

            if current_class_counts[cls] < target:
                deficit = target - current_class_counts[cls]
                score += deficit / max(1, target)

        if not classes and current_negative_count < negative_target:
            deficit = negative_target - current_negative_count
            score += deficit / max(1, negative_target)

        # Small preference for multi-class examples because they
        # preserve the original multi-object character of the dataset.
        if len(classes) > 1:
            score += 0.05

        if score > best_score:
            best_score = score
            best_record = record

    if best_record is None:
        break

    selected[id(best_record)] = best_record

    classes = set(best_record["classes"])

    for cls in classes:
        current_class_counts[cls] += 1

    if not classes:
        current_negative_count += 1

    candidate_pool.remove(best_record)

selected_base = list(selected.values())

print(f"\nSelected base images: {len(selected_base)}")

# ------------------------------------------------
# HARD EXAMPLE INVENTORY
# ------------------------------------------------

print("\n" + "=" * 70)
print("ANALYZING FINALIZED HARD-EXAMPLE POOL")
print("=" * 70)

hard_records = []

for image_path in sorted(hard_images_dir.rglob("*")):

    if image_path.suffix.lower() not in image_extensions:
        continue

    rel = image_path.relative_to(hard_images_dir)
    label_path = hard_labels_dir / rel.with_suffix(".txt")

    if not label_path.exists():
        raise RuntimeError(f"Missing hard-example label:\n{label_path}")

    classes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            cls = int(parts[0])

            if cls not in CLASS_NAMES:
                raise RuntimeError(
                    f"Unexpected class {cls} in {label_path}"
                )

            classes.append(cls)

    hard_records.append({
        "image": image_path,
        "label": label_path,
        "classes": classes,
        "relative": rel,
    })

print(f"Hard-example images found: {len(hard_records)}")

if len(hard_records) != 2851:
    raise RuntimeError(
        f"Expected 2,851 hard examples, found {len(hard_records)}."
    )

# ------------------------------------------------
# SHA256 DUPLICATE CHECK
# ------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING BASE ↔ HARD-EXAMPLE DUPLICATES")
print("=" * 70)

def sha256_file(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


hard_hashes = {}

for i, record in enumerate(hard_records, start=1):

    digest = sha256_file(record["image"])
    hard_hashes[digest] = record["image"]

    if i % 500 == 0 or i == len(hard_records):
        print(f"Hashed hard examples: {i}/{len(hard_records)}")

duplicate_base_records = []

for i, record in enumerate(selected_base, start=1):

    digest = sha256_file(record["image"])

    if digest in hard_hashes:
        duplicate_base_records.append(record)

    if i % 500 == 0 or i == len(selected_base):
        print(f"Hashed base subset: {i}/{len(selected_base)}")

print(
    f"\nPhysical duplicates between base subset and hard pool: "
    f"{len(duplicate_base_records)}"
)

# Remove any duplicate base images.
duplicate_ids = {
    id(r) for r in duplicate_base_records
}

selected_base = [
    r for r in selected_base
    if id(r) not in duplicate_ids
]

print(
    f"Base images remaining after duplicate removal: "
    f"{len(selected_base)}"
)

# ------------------------------------------------
# BUILD FINAL V2 TRAINING DATASET
# ------------------------------------------------

print("\n" + "=" * 70)
print("BUILDING FINAL V2 TRAINING DATASET")
print("=" * 70)

FINAL_IMAGES = FINAL_DATASET / "images" / "train"
FINAL_LABELS = FINAL_DATASET / "labels" / "train"

FINAL_IMAGES.mkdir(parents=True, exist_ok=True)
FINAL_LABELS.mkdir(parents=True, exist_ok=True)

manifest_path = FINAL_DATASET / "v2_training_manifest.csv"

manifest_rows = []

used_names = set()

def make_unique_name(original_name, source, used_names):

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()

    candidate = f"{source}_{stem}{suffix}"

    counter = 1

    while candidate in used_names:
        candidate = f"{source}_{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate)

    return candidate


# Copy base subset
print("\nCopying base subset...")

for i, record in enumerate(selected_base, start=1):

    new_name = make_unique_name(
        record["image"].name,
        "base",
        used_names
    )

    destination_image = FINAL_IMAGES / new_name
    destination_label = FINAL_LABELS / (
        Path(new_name).with_suffix(".txt")
    )

    shutil.copy2(record["image"], destination_image)
    shutil.copy2(record["label"], destination_label)

    manifest_rows.append({
        "final_image": new_name,
        "source": "V1_BASE",
        "original_path": str(record["image"]),
        "original_label": str(record["label"]),
    })

    if i % 500 == 0 or i == len(selected_base):
        print(f"Base copied: {i}/{len(selected_base)}")


# Copy hard examples
print("\nCopying hard examples...")

for i, record in enumerate(hard_records, start=1):

    new_name = make_unique_name(
        record["image"].name,
        "hard",
        used_names
    )

    destination_image = FINAL_IMAGES / new_name
    destination_label = FINAL_LABELS / (
        Path(new_name).with_suffix(".txt")
    )

    shutil.copy2(record["image"], destination_image)
    shutil.copy2(record["label"], destination_label)

    manifest_rows.append({
        "final_image": new_name,
        "source": "V2_HARD_EXAMPLES",
        "original_path": str(record["image"]),
        "original_label": str(record["label"]),
    })

    if i % 500 == 0 or i == len(hard_records):
        print(f"Hard examples copied: {i}/{len(hard_records)}")

# ------------------------------------------------
# WRITE MANIFEST
# ------------------------------------------------

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "final_image",
            "source",
            "original_path",
            "original_label",
        ]
    )

    writer.writeheader()
    writer.writerows(manifest_rows)

# ------------------------------------------------
# CREATE DATASET YAML
# ------------------------------------------------

yaml_path = FINAL_DATASET / "data.yaml"

yaml_content = f"""path: {FINAL_DATASET}
train: images/train
val: images/train

nc: 5

names:
  0: Knife
  1: Machette
  2: Handgun
  3: Longgun
  4: Other
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

# ------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL V2 DATASET VALIDATION")
print("=" * 70)

final_images = [
    p for p in FINAL_IMAGES.iterdir()
    if p.suffix.lower() in image_extensions
]

final_labels = list(FINAL_LABELS.glob("*.txt"))

image_stems = {p.stem for p in final_images}
label_stems = {p.stem for p in final_labels}

missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems

print(f"Final images:       {len(final_images)}")
print(f"Final labels:       {len(final_labels)}")
print(f"Missing labels:     {len(missing_labels)}")
print(f"Missing images:     {len(missing_images)}")

if missing_labels or missing_images:
    raise RuntimeError("Final image/label pairing validation FAILED.")

expected_total = len(selected_base) + len(hard_records)

print(f"\nExpected total:     {expected_total}")
print(f"Actual total:       {len(final_images)}")

if len(final_images) != expected_total:
    raise RuntimeError("Final image count does not match expected count.")

# ------------------------------------------------
# OBJECT DISTRIBUTION
# ------------------------------------------------

object_counts = Counter()
image_counts = Counter()
negative_count = 0

for label_path in final_labels:

    classes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            cls = int(parts[0])

            if cls not in CLASS_NAMES:
                raise RuntimeError(
                    f"Invalid class {cls} in {label_path}"
                )

            classes.append(cls)
            object_counts[cls] += 1

    for cls in set(classes):
        image_counts[cls] += 1

    if not classes:
        negative_count += 1

print("\n" + "=" * 70)
print("FINAL CLASS DISTRIBUTION")
print("=" * 70)

for cls, name in CLASS_NAMES.items():

    print(
        f"{name:<12} "
        f"images={image_counts[cls]:>5} "
        f"objects={object_counts[cls]:>5}"
    )

print(
    f"{'Negatives':<12} "
    f"images={negative_count:>5}"
)

# ------------------------------------------------
# SOURCE DISTRIBUTION
# ------------------------------------------------

source_counts = Counter(
    row["source"]
    for row in manifest_rows
)

print("\n" + "=" * 70)
print("SOURCE DISTRIBUTION")
print("=" * 70)

for source, count in source_counts.items():
    print(f"{source:<20}: {count}")

# ------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------

summary = {
    "v2_training_images": len(final_images),
    "base_images_selected": len(selected_base),
    "hard_examples": len(hard_records),
    "physical_duplicates_removed_from_base": len(
        duplicate_base_records
    ),
    "final_labels": len(final_labels),
    "missing_labels": len(missing_labels),
    "missing_images": len(missing_images),
    "negative_images": negative_count,
    "object_counts": {
        CLASS_NAMES[k]: v
        for k, v in object_counts.items()
    },
    "image_counts": {
        CLASS_NAMES[k]: v
        for k, v in image_counts.items()
    },
    "source_counts": dict(source_counts),
    "taxonomy": CLASS_NAMES,
    "v1_model": str(MODEL_PATH),
    "base_dataset": str(BASE_ZIP),
    "hard_examples": str(HARD_ZIP),
    "augmentation": "ON_THE_FLY_DURING_TRAINING",
}

summary_path = FINAL_DATASET / "v2_dataset_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------
# FINAL STATUS
# ------------------------------------------------

print("\n" + "=" * 70)
print("V2 DATASET BUILD COMPLETE")
print("=" * 70)

print(f"""
Base images selected:       {len(selected_base)}
Hard examples:              {len(hard_records)}
Final training images:      {len(final_images)}

Final dataset:
{FINAL_DATASET}

Dataset YAML:
{yaml_path}

Manifest:
{manifest_path}

Summary:
{summary_path}

V1 starting checkpoint:
{MODEL_PATH}

Physical augmentation:
NOT CREATED

Training:
NOT STARTED

Original V1 dataset:
UNMODIFIED

Original hard-example dataset:
UNMODIFIED
""")

print("=" * 70)
print("READY FOR V2 TRAINING")
print("=" * 70)


# ====================================================================
# CODE SECTION 2
# ====================================================================

# ================================================================
# V2 — VALIDATION SET CONFIGURATION
# ================================================================

from pathlib import Path
import shutil
import yaml

print("=" * 70)
print("V2 — VALIDATION CONFIGURATION")
print("=" * 70)

# ------------------------------------------------
# EXISTING V2 TRAINING DATA
# ------------------------------------------------

V2_ROOT = Path("/content/V2_Build/V2_Training")

TRAIN_IMAGES = V2_ROOT / "images" / "train"
TRAIN_LABELS = V2_ROOT / "labels" / "train"

if not TRAIN_IMAGES.exists():
    raise FileNotFoundError(f"V2 training images not found:\n{TRAIN_IMAGES}")

if not TRAIN_LABELS.exists():
    raise FileNotFoundError(f"V2 training labels not found:\n{TRAIN_LABELS}")

# ------------------------------------------------
# FIND ORIGINAL V1 VALIDATION SPLIT
# ------------------------------------------------

BASE_ROOT = Path("/content/V2_Build/base_extracted")

print("\nSearching for original V1 validation split...")

validation_candidates = []

for images_dir in BASE_ROOT.rglob("images"):

    if not images_dir.is_dir():
        continue

    parent_name = images_dir.parent.name.lower()

    if parent_name in {"val", "valid", "validation"}:

        labels_dir = images_dir.parent / "labels"

        if labels_dir.exists() and labels_dir.is_dir():
            validation_candidates.append(
                (images_dir, labels_dir)
            )

if not validation_candidates:
    raise RuntimeError(
        "Could not locate the original V1 validation "
        "images/labels directories."
    )

# Prefer an explicitly named "val" directory
VAL_IMAGES = None
VAL_LABELS = None

for img_dir, lab_dir in validation_candidates:

    if img_dir.parent.name.lower() == "val":
        VAL_IMAGES = img_dir
        VAL_LABELS = lab_dir
        break

if VAL_IMAGES is None:
    VAL_IMAGES, VAL_LABELS = validation_candidates[0]

print("\nOriginal V1 validation set:")
print("Images:", VAL_IMAGES)
print("Labels:", VAL_LABELS)

# ------------------------------------------------
# COUNT VALIDATION IMAGES
# ------------------------------------------------

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

val_images = [
    p for p in VAL_IMAGES.rglob("*")
    if p.is_file()
    and p.suffix.lower() in image_extensions
]

val_labels = [
    p for p in VAL_LABELS.rglob("*.txt")
    if p.is_file()
]

print("\nValidation counts:")
print(f"Images: {len(val_images)}")
print(f"Labels: {len(val_labels)}")

# Our verified V1 record says validation = 2,305 images.
if len(val_images) != 2305:
    print(
        "\nWARNING:"
        f" Expected 2,305 validation images, found {len(val_images)}."
    )
    print("We will NOT silently proceed with a mismatched split.")
    raise RuntimeError(
        "Original V1 validation count does not match "
        "the verified project record."
    )

# ------------------------------------------------
# VALIDATE IMAGE / LABEL PAIRING
# ------------------------------------------------

val_image_stems = {p.stem for p in val_images}
val_label_stems = {p.stem for p in val_labels}

missing_val_labels = val_image_stems - val_label_stems
missing_val_images = val_label_stems - val_image_stems

print("\nValidation pairing:")
print("Missing labels:", len(missing_val_labels))
print("Missing images:", len(missing_val_images))

if missing_val_labels or missing_val_images:
    raise RuntimeError(
        "Validation image/label pairing failed."
    )

# ------------------------------------------------
# CREATE FINAL V2 YAML
# ------------------------------------------------

yaml_path = V2_ROOT / "data_v2.yaml"

yaml_data = {
    "path": str(V2_ROOT),
    "train": str(TRAIN_IMAGES),
    "val": str(VAL_IMAGES),

    "nc": 5,

    "names": [
        "Knife",
        "Machette",
        "Handgun",
        "Longgun",
        "Other",
    ],
}

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        yaml_data,
        f,
        sort_keys=False
    )

# ------------------------------------------------
# DISPLAY CONFIGURATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL V2 DATA CONFIGURATION")
print("=" * 70)

print(f"""
TRAIN:
{TRAIN_IMAGES}

TRAIN IMAGES:
{len(list(TRAIN_IMAGES.iterdir()))}

VALIDATION:
{VAL_IMAGES}

VALIDATION IMAGES:
{len(val_images)}

YAML:
{yaml_path}

CLASSES:
0: Knife
1: Machette
2: Handgun
3: Longgun
4: Other

AUGMENTATION:
ON-THE-FLY DURING TRAINING

HARD EXAMPLES:
TRAINING ONLY

ORIGINAL V1 VALIDATION:
VALIDATION ONLY

TRAINING:
NOT STARTED
""")

print("=" * 70)
print("V2 VALIDATION CONFIGURATION PASS")
print("=" * 70)


# ====================================================================
# CODE SECTION 3
# ====================================================================

# ================================================================
# V2 TRAINING — V1 BEST.PT → V2
# ================================================================
# Initial V2 fine-tuning run.
# The V2 dataset and validation configuration are prepared above.


from ultralytics import YOLO
from pathlib import Path
import shutil
import json
import gc
import torch

print("=" * 70)
print("V2 TRAINING — V1 BEST.PT → V2")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

V2_ROOT = Path("/content/V2_Build/V2_Training")

DATA_YAML = V2_ROOT / "data_v2.yaml"

V1_MODEL = Path(
    "/content/drive/MyDrive/Weapon_Detection_Training/"
    "FINAL_CONTINUATION/checkpoints/best.pt"
)

DRIVE_RUN_ROOT = Path(
    "/content/drive/MyDrive/Weapon_Detection_Training/V2/"
    "V2_Training"
)

DRIVE_RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------
# CHECKS
# ------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING INPUT CHECK")
print("=" * 70)

required_paths = {
    "V2 dataset": V2_ROOT,
    "V2 YAML": DATA_YAML,
    "V1 best.pt": V1_MODEL,
}

for name, path in required_paths.items():

    print(f"{name:<20}: {path}")
    print(f"Exists{'':<15}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Required training input missing:\n{path}"
        )

# ------------------------------------------------
# DATASET COUNTS
# ------------------------------------------------

train_images = list(
    (V2_ROOT / "images" / "train").glob("*")
)

train_images = [
    p for p in train_images
    if p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
]

print(f"\nTraining images:    {len(train_images)}")

if len(train_images) != 6958:
    raise RuntimeError(
        f"Expected 6,958 V2 training images, "
        f"found {len(train_images)}."
    )

# ------------------------------------------------
# GPU CHECK
# ------------------------------------------------

print("\n" + "=" * 70)
print("DEVICE CHECK")
print("=" * 70)

if torch.cuda.is_available():

    device = 0

    print("CUDA available:     True")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    device = "cpu"

    print("CUDA available:     False")
    print("WARNING: Training will use CPU.")

# ------------------------------------------------
# LOAD V1 MODEL
# ------------------------------------------------

print("\n" + "=" * 70)
print("LOADING V1 MODEL")
print("=" * 70)

model = YOLO(str(V1_MODEL))

print("Loaded V1 checkpoint:")
print(V1_MODEL)

# ------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------

EPOCHS = 20

# Conservative batch size for the T4/RAM situation.
BATCH = 8

IMG_SIZE = 1280

# Early stopping
PATIENCE = 6

# Save training output directly to Drive.
PROJECT = str(DRIVE_RUN_ROOT)
NAME = "v2_finetune"

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"""
Starting model:       V1 best.pt
Training images:      6,958
Validation images:    2,305

Epochs:               {EPOCHS}
Batch size:           {BATCH}
Image size:           {IMG_SIZE}
Early stopping:       {PATIENCE} epochs
Device:               {device}

Training output:
{PROJECT}/{NAME}

Augmentation:
ON-THE-FLY

Aerial images:
NOT INCLUDED

Starting from scratch:
NO
""")

# ------------------------------------------------
# TRAIN
# ------------------------------------------------

print("=" * 70)
print("STARTING V2 TRAINING")
print("=" * 70)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

results = model.train(

    # Dataset
    data=str(DATA_YAML),

    # Training duration
    epochs=EPOCHS,

    # Hardware
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=device,
    workers=2,

    # Fine-tuning
    pretrained=False,

    # Optimization
    optimizer="auto",

    # Early stopping
    patience=PATIENCE,

    # Output
    project=PROJECT,
    name=NAME,

    # Keep best/last checkpoints
    save=True,

    # Save validation/training plots
    plots=True,

    # Existing model is already trained.
    # Use conservative augmentation.
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,

    degrees=5.0,
    translate=0.05,
    scale=0.30,
    shear=2.0,

    perspective=0.0,

    flipud=0.0,
    fliplr=0.5,

    mosaic=0.5,
    mixup=0.0,

    # Close mosaic near end of training
    close_mosaic=3,

    # Validation
    val=True,

    # Reproducibility
    seed=42,

    # Verbose logging
    verbose=True,
)

print("\n" + "=" * 70)
print("V2 TRAINING PROCESS FINISHED")
print("=" * 70)

# ------------------------------------------------
# LOCATE OUTPUTS
# ------------------------------------------------

RUN_DIR = DRIVE_RUN_ROOT / NAME
WEIGHTS_DIR = RUN_DIR / "weights"

BEST_MODEL = WEIGHTS_DIR / "best.pt"
LAST_MODEL = WEIGHTS_DIR / "last.pt"
RESULTS_CSV = RUN_DIR / "results.csv"

print("\nTraining output directory:")
print(RUN_DIR)

print("\nBest model:")
print(BEST_MODEL)
print("Exists:", BEST_MODEL.exists())

print("\nLast model:")
print(LAST_MODEL)
print("Exists:", LAST_MODEL.exists())

print("\nResults:")
print(RESULTS_CSV)
print("Exists:", RESULTS_CSV.exists())

# ------------------------------------------------
# SAVE TRAINING SUMMARY
# ------------------------------------------------

summary = {
    "model_initialization": str(V1_MODEL),
    "dataset": str(DATA_YAML),
    "training_images": 6958,
    "validation_images": 2305,
    "epochs_requested": EPOCHS,
    "batch_size": BATCH,
    "image_size": IMG_SIZE,
    "patience": PATIENCE,
    "device": str(device),
    "augmentation": "on_the_fly",
    "aerial_images": False,
    "best_model": str(BEST_MODEL),
    "last_model": str(LAST_MODEL),
    "results_csv": str(RESULTS_CSV),
}

SUMMARY_PATH = RUN_DIR / "v2_training_summary.json"

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nTraining summary:")
print(SUMMARY_PATH)

# ------------------------------------------------
# FINAL STATUS
# ------------------------------------------------

print("\n" + "=" * 70)
print("V2 TRAINING OUTPUT STATUS")
print("=" * 70)

if BEST_MODEL.exists():
    print("BEST.PT:  SAVED TO DRIVE")
else:
    print("BEST.PT:  NOT FOUND")

if LAST_MODEL.exists():
    print("LAST.PT:  SAVED TO DRIVE")
else:
    print("LAST.PT:  NOT FOUND")

if RESULTS_CSV.exists():
    print("RESULTS:  SAVED TO DRIVE")
else:
    print("RESULTS:  NOT FOUND")

print("\nV2 TRAINING CELL COMPLETE.")


# ====================================================================
# CODE SECTION 4
# ====================================================================

# ================================================================
# V2 — CHECKPOINT RECOVERY / RESUME TRAINING
# ================================================================
# This section intentionally does not rebuild the dataset again.
# It uses the V2 dataset and checkpoint already produced above.
#
# ================================================================
# V2 — REBUILD / COMPRESS DATASET + RESUME TRAINING
# ================================================================
from collections import Counter
from ultralytics import YOLO
from pathlib import Path
import shutil
import zipfile
import csv
import json
import hashlib
import random
import gc
import torch

print("=" * 70)
print("V2 — DATASET RECOVERY + RESUME TRAINING")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

DRIVE = Path("/content/drive/MyDrive")

BASE_ZIP = DRIVE / "Final_Dataset_Split" / "Final_Dataset_Split.zip"

HARD_ZIP = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Hard_Examples_Consolidated.zip"
)

V1_MODEL = (
    DRIVE
    / "Weapon_Detection_Training"
    / "FINAL_CONTINUATION"
    / "checkpoints"
    / "best.pt"
)

V2_RUN = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Training"
    / "v2_finetune"
)

LAST_PT = V2_RUN / "weights" / "last.pt"
BEST_PT = V2_RUN / "weights" / "best.pt"

# Permanent compressed V2 dataset
V2_DATASET_ZIP = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Training_Dataset.zip"
)

# Permanent recovery checkpoints
RECOVERY = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Checkpoints"
)

RECOVERY.mkdir(parents=True, exist_ok=True)

# Temporary Colab storage
WORK_ROOT = Path("/content/V2_Build")
V2_ROOT = WORK_ROOT / "V2_Training"

DATA_YAML = V2_ROOT / "data_v2.yaml"

TARGET_BASE_IMAGES = 5000
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ------------------------------------------------
# CHECK CHECKPOINTS
# ------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING V2 CHECKPOINTS")
print("=" * 70)

if not LAST_PT.exists():
    raise FileNotFoundError(
        f"V2 last.pt not found:\n{LAST_PT}"
    )

if not BEST_PT.exists():
    raise FileNotFoundError(
        f"V2 best.pt not found:\n{BEST_PT}"
    )

print("last.pt: FOUND")
print("best.pt: FOUND")

print(
    f"last.pt: {LAST_PT.stat().st_size / (1024**2):.2f} MB"
)

print(
    f"best.pt: {BEST_PT.stat().st_size / (1024**2):.2f} MB"
)

# ------------------------------------------------

# ------------------------------------------------
# RESUME TRAINING FROM THE SAVED V2 CHECKPOINT
# ------------------------------------------------
# GPU
# ------------------------------------------------

print("\n" + "=" * 70)
print("GPU CHECK")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

# ------------------------------------------------
# LOAD LAST CHECKPOINT
# ------------------------------------------------

print("\n" + "=" * 70)
print("LOADING SAVED V2 CHECKPOINT")
print("=" * 70)

model = YOLO(str(LAST_PT))

print("Loaded:")
print(LAST_PT)

# ------------------------------------------------
# RESUME TRAINING
# ------------------------------------------------

TARGET_EPOCH = 26

print("\n" + "=" * 70)
print("RESUMING V2 TRAINING")
print("=" * 70)

print("""
Previous completed epoch: 15
Interrupted during:       Epoch 16
Target epoch:             26

Starting checkpoint:
V2 last.pt
""")

gc.collect()
torch.cuda.empty_cache()

results = model.train(
    data=str(DATA_YAML),

    # Resume full training state
    resume=True,

    # Continue toward Epoch 26
    epochs=TARGET_EPOCH,

    # Hardware
    imgsz=1280,
    batch=8,
    device=0,
    workers=2,

    # Checkpointing
    save=True,

    # Validation
    val=True,

    # Early stopping
    patience=8,

    # Reproducibility
    seed=42,

    # On-the-fly augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,

    degrees=5.0,
    translate=0.05,
    scale=0.30,
    shear=2.0,

    perspective=0.0,

    flipud=0.0,
    fliplr=0.5,

    mosaic=0.5,
    mixup=0.0,

    close_mosaic=3,

    plots=True,
    verbose=True,
)

# ------------------------------------------------
# VERIFY CHECKPOINTS
# ------------------------------------------------

print("\n" + "=" * 70)
print("CHECKPOINT VERIFICATION")
print("=" * 70)

print("best.pt:", BEST_PT.exists())
print("last.pt:", LAST_PT.exists())

if not BEST_PT.exists():
    raise RuntimeError(
        "best.pt was not saved."
    )

if not LAST_PT.exists():
    raise RuntimeError(
        "last.pt was not saved."
    )

# ------------------------------------------------
# PERMANENT RECOVERY COPIES
# ------------------------------------------------

RECOVERY_BEST = (
    RECOVERY / "v2_best.pt"
)

RECOVERY_LAST = (
    RECOVERY / "v2_last.pt"
)

shutil.copy2(
    BEST_PT,
    RECOVERY_BEST
)

shutil.copy2(
    LAST_PT,
    RECOVERY_LAST
)

print("\nPermanent recovery copies:")

print(
    "Best:",
    RECOVERY_BEST
)

print(
    "Last:",
    RECOVERY_LAST
)

# ------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------

summary = {
    "training_images": 6958,
    "validation_images": 2305,
    "previous_completed_epoch": 15,
    "target_epoch": TARGET_EPOCH,
    "starting_checkpoint": str(LAST_PT),
    "best_checkpoint": str(BEST_PT),
    "last_checkpoint": str(LAST_PT),
    "permanent_best": str(RECOVERY_BEST),
    "permanent_last": str(RECOVERY_LAST),
    "compressed_dataset": str(V2_DATASET_ZIP),
    "dataset_yaml": str(DATA_YAML),
    "augmentation": "on_the_fly",
    "aerial_images": False,
}

summary_path = (
    RECOVERY / "v2_training_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("V2 RESUME COMPLETE")
print("=" * 70)


# ====================================================================
# CODE SECTION 5
# ====================================================================

# ================================================================
# V1 vs V2 — IDENTICAL VALIDATION SET COMPARISON
# ================================================================
# Compare both model versions on the same 2,305-image validation set.
#

from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
import torch

print("=" * 70)
print("V1 vs V2 — VALIDATION COMPARISON")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

DRIVE = Path("/content/drive/MyDrive")

V1_MODEL = (
    DRIVE
    / "Weapon_Detection_Training"
    / "FINAL_CONTINUATION"
    / "checkpoints"
    / "best.pt"
)

V2_MODEL = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Training"
    / "v2_finetune"
    / "weights"
    / "best.pt"
)

VAL_IMAGES = (
    Path("/content/V2_Build")
    / "base_extracted"
    / "Final_Dataset_Split"
    / "val"
    / "images"
)

DATA_YAML = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Training"
    / "data_v2.yaml"
)

OUTPUT_DIR = (
    DRIVE
    / "Weapon_Detection_Training"
    / "V2"
    / "V2_Validation_Comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------
# CHECK FILES
# ------------------------------------------------

print("\nChecking files...")

for name, path in {
    "V1 model": V1_MODEL,
    "V2 model": V2_MODEL,
    "Validation images": VAL_IMAGES,
}.items():

    print(
        f"{name}: "
        f"{'FOUND' if path.exists() else 'MISSING'}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

# ------------------------------------------------
# VALIDATION COUNT
# ------------------------------------------------

val_images = [
    p for p in VAL_IMAGES.iterdir()
    if p.is_file()
    and p.suffix.lower()
    in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
]

print("\nValidation images:", len(val_images))

if len(val_images) != 2305:
    raise RuntimeError(
        f"Expected 2,305 validation images, "
        f"found {len(val_images)}."
    )

# ------------------------------------------------
# GPU
# ------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable.")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

v1 = YOLO(str(V1_MODEL))
v2 = YOLO(str(V2_MODEL))

print("V1 loaded.")
print("V2 loaded.")

# ------------------------------------------------
# VALIDATE V1
# ------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING V1")
print("=" * 70)

v1_results = v1.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=1280,
    batch=8,
    device=0,
    workers=2,
    plots=True,
    verbose=True,
)

# ------------------------------------------------
# VALIDATE V2
# ------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING V2")
print("=" * 70)

v2_results = v2.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=1280,
    batch=8,
    device=0,
    workers=2,
    plots=True,
    verbose=True,
)

# ------------------------------------------------
# EXTRACT METRICS
# ------------------------------------------------

def extract_metrics(results):

    return {
        "Precision": float(results.box.mp),
        "Recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }


v1_metrics = extract_metrics(v1_results)
v2_metrics = extract_metrics(v2_results)

# ------------------------------------------------
# COMPARISON
# ------------------------------------------------

comparison = pd.DataFrame(
    {
        "Metric": [
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95",
        ],
        "V1": [
            v1_metrics["Precision"],
            v1_metrics["Recall"],
            v1_metrics["mAP50"],
            v1_metrics["mAP50-95"],
        ],
        "V2": [
            v2_metrics["Precision"],
            v2_metrics["Recall"],
            v2_metrics["mAP50"],
            v2_metrics["mAP50-95"],
        ],
    }
)

comparison["Absolute Change"] = (
    comparison["V2"] - comparison["V1"]
)

comparison["Relative Change %"] = (
    (
        comparison["V2"]
        - comparison["V1"]
    )
    / comparison["V1"]
    * 100
)

print("\n" + "=" * 70)
print("V1 vs V2 RESULTS")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------
# CLASS-WISE METRICS
# ------------------------------------------------

print("\n" + "=" * 70)
print("CLASS-WISE COMPARISON")
print("=" * 70)

class_names = [
    "Knife",
    "Machette",
    "Handgun",
    "Longgun",
    "Other",
]

class_rows = []

for i, name in enumerate(class_names):

    v1_class = v1_results.box.class_result(i)
    v2_class = v2_results.box.class_result(i)

    class_rows.append({
        "Class": name,
        "V1_Precision": float(v1_class[0]),
        "V1_Recall": float(v1_class[1]),
        "V1_mAP50": float(v1_class[2]),
        "V1_mAP50-95": float(v1_class[3]),
        "V2_Precision": float(v2_class[0]),
        "V2_Recall": float(v2_class[1]),
        "V2_mAP50": float(v2_class[2]),
        "V2_mAP50-95": float(v2_class[3]),
    })

class_comparison = pd.DataFrame(
    class_rows
)

print(
    class_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

comparison_csv = (
    OUTPUT_DIR
    / "v1_vs_v2_metrics.csv"
)

class_csv = (
    OUTPUT_DIR
    / "v1_vs_v2_class_metrics.csv"
)

summary_json = (
    OUTPUT_DIR
    / "v1_vs_v2_validation_summary.json"
)

comparison.to_csv(
    comparison_csv,
    index=False
)

class_comparison.to_csv(
    class_csv,
    index=False
)

summary = {
    "validation_images": 2305,
    "imgsz": 1280,
    "v1_model": str(V1_MODEL),
    "v2_model": str(V2_MODEL),
    "validation_dataset": str(VAL_IMAGES),
    "v1_metrics": v1_metrics,
    "v2_metrics": v2_metrics,
}

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("VALIDATION COMPARISON COMPLETE")
print("=" * 70)

print("Metrics:")
print(comparison_csv)

print("Class metrics:")
print(class_csv)

print("Summary:")
print(summary_json)