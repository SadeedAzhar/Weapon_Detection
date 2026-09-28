from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import csv
import sys

V1_ROOT = Path(r"D:\WD Dataset\Processed\Final_Dataset_Split")
HARD_ROOT = Path(
    r"D:\WD Dataset\Processed\V2\V2_Hard_Examples_Consolidated"
    r"\V2_Hard_Examples_Consolidated"
)
OUTPUT_ROOT = Path(
    r"D:\WD Dataset\Processed\V2\V2_Training_Dataset_V2.1"
)
AUDIT_DIR = OUTPUT_ROOT / "composition_audit"

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_V1_TRAIN = 12199
EXPECTED_HARD = 2851
EXPECTED_V1_VAL = 2305
EXPECTED_V1_TEST = 784


def fail(message):
    raise RuntimeError(message)


def image_files(root):
    if not root.exists():
        fail(f"Missing directory: {root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def label_path_for(image_path, image_root):
    rel = image_path.relative_to(image_root)
    return image_root.parent / "labels" / rel.with_suffix(".txt")


def read_yolo_classes(label_path):
    if not label_path.exists():
        return []

    classes = []

    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) != 5:
            fail(
                f"Invalid YOLO label format: "
                f"{label_path} line {line_no}"
            )

        try:
            class_id = int(parts[0])
        except ValueError:
            fail(
                f"Invalid class ID: {label_path} line {line_no}"
            )

        if class_id not in CLASS_NAMES:
            fail(
                f"Unexpected class {class_id}: "
                f"{label_path} line {line_no}"
            )

        classes.append(class_id)

    return classes


def copy_pair(image_src, label_src, image_dst, label_dst):
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_src, image_dst)

    if label_src.exists():
        shutil.copy2(label_src, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def audit_training_root(root, split):
    image_root = root / split / "images"
    label_root = root / split / "labels"
    images = image_files(image_root)

    counts = Counter()
    empty = 0
    missing = []

    for image in images:
        label = label_path_for(image, image_root)

        if not label.exists():
            missing.append(str(image))
            continue

        classes = read_yolo_classes(label)

        if classes:
            counts.update(classes)
        else:
            empty += 1

    return len(images), empty, counts, missing


def build_dataset():
    print("=" * 72)
    print("V2.1 DATASET BUILD + AUDIT")
    print("=" * 72)

    if OUTPUT_ROOT.exists():
        fail(
            "Output already exists. Review or remove it manually first:\n"
            f"{OUTPUT_ROOT}"
        )

    if not V1_ROOT.exists():
        fail(f"V1 dataset not found: {V1_ROOT}")

    if not HARD_ROOT.exists():
        fail(f"Hard-example dataset not found: {HARD_ROOT}")

    v1_train_root = V1_ROOT / "train"
    hard_image_root = HARD_ROOT / "images"

    v1_train_images = image_files(v1_train_root / "images")

    if len(v1_train_images) != EXPECTED_V1_TRAIN:
        fail(
            f"Expected {EXPECTED_V1_TRAIN:,} V1 train images, "
            f"found {len(v1_train_images):,}."
        )

    v1_stats = Counter()
    v1_empty = 0

    for image in v1_train_images:
        label = label_path_for(image, v1_train_root / "images")
        if not label.exists():
            fail(f"Missing V1 label: {label}")

        classes = read_yolo_classes(label)
        if classes:
            v1_stats.update(classes)
        else:
            v1_empty += 1

    hard_images = image_files(hard_image_root)

    if len(hard_images) != EXPECTED_HARD:
        fail(
            f"Expected {EXPECTED_HARD:,} hard-example images, "
            f"found {len(hard_images):,}."
        )

    hard_stats = Counter()
    hard_empty = 0

    for image in hard_images:
        label = HARD_ROOT / "labels" / image.relative_to(hard_image_root)
        label = label.with_suffix(".txt")

        if not label.exists():
            fail(f"Missing hard-example label: {label}")

        classes = read_yolo_classes(label)

        if classes:
            hard_stats.update(classes)
        else:
            hard_empty += 1

    print(f"\nV1 train images       : {len(v1_train_images):,}")
    print(f"V1 train empty labels : {v1_empty:,}")
    print(f"Hard examples         : {len(hard_images):,}")
    print(f"Hard empty labels     : {hard_empty:,}")

    v1_hashes = {}
    for index, image in enumerate(v1_train_images, 1):
        v1_hashes[sha256_file(image)] = image
        if index % 1000 == 0:
            print(f"Hashed V1 images: {index:,}/{len(v1_train_images):,}")

    duplicate_hard = []
    unique_hard = []

    for index, image in enumerate(hard_images, 1):
        digest = sha256_file(image)

        if digest in v1_hashes:
            duplicate_hard.append(
                (image, v1_hashes[digest], digest)
            )
        else:
            unique_hard.append((image, digest))

        if index % 250 == 0:
            print(f"Checked hard images: {index:,}/{len(hard_images):,}")

    print(f"\nHard duplicates skipped : {len(duplicate_hard):,}")
    print(f"Hard examples added     : {len(unique_hard):,}")

    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / split / "images").mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT_ROOT / split / "labels").mkdir(
            parents=True, exist_ok=True
        )

    manifest = []

    for index, image in enumerate(v1_train_images, 1):
        rel = image.relative_to(v1_train_root / "images")
        safe_name = f"v1_{rel.as_posix().replace('/', '__')}"

        copy_pair(
            image,
            v1_train_root / "labels" / rel.with_suffix(".txt"),
            OUTPUT_ROOT / "train" / "images" / safe_name,
            OUTPUT_ROOT / "train" / "labels" / Path(safe_name).with_suffix(".txt"),
        )

        manifest.append({
            "output_filename": safe_name,
            "source": "V1_train",
            "source_path": str(image),
            "original_label_path": str(
                v1_train_root / "labels" / rel.with_suffix(".txt")
            ),
            "sha256": sha256_file(image),
            "duplicate_with_v1": False,
        })

        if index % 1000 == 0:
            print(f"Copied V1 train: {index:,}/{len(v1_train_images):,}")

    for index, (image, digest) in enumerate(unique_hard, 1):
        rel = image.relative_to(hard_image_root)
        safe_name = f"hard_{rel.as_posix().replace('/', '__')}"

        copy_pair(
            image,
            HARD_ROOT / "labels" / rel.with_suffix(".txt"),
            OUTPUT_ROOT / "train" / "images" / safe_name,
            OUTPUT_ROOT / "train" / "labels" / Path(safe_name).with_suffix(".txt"),
        )

        manifest.append({
            "output_filename": safe_name,
            "source": "V2_hard_examples",
            "source_path": str(image),
            "original_label_path": str(
                HARD_ROOT / "labels" / rel.with_suffix(".txt")
            ),
            "sha256": digest,
            "duplicate_with_v1": False,
        })

        if index % 250 == 0:
            print(f"Added hard examples: {index:,}/{len(unique_hard):,}")

    for split, expected in (
        ("val", EXPECTED_V1_VAL),
        ("test", EXPECTED_V1_TEST),
    ):
        images = image_files(V1_ROOT / split / "images")

        if len(images) != expected:
            fail(
                f"Expected {expected:,} {split} images, "
                f"found {len(images):,}."
            )

        for image in images:
            rel = image.relative_to(V1_ROOT / split / "images")

            copy_pair(
                image,
                V1_ROOT / split / "labels" / rel.with_suffix(".txt"),
                OUTPUT_ROOT / split / "images" / rel,
                OUTPUT_ROOT / split / "labels" / rel.with_suffix(".txt"),
            )

    dataset_yaml = """path: .
train: train/images
val: val/images
test: test/images

names:
  0: Knife
  1: Machette
  2: Handgun
  3: Longgun
  4: Other
"""

    (OUTPUT_ROOT / "dataset.yaml").write_text(
        dataset_yaml,
        encoding="utf-8",
    )

    with (OUTPUT_ROOT / "v2.1_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_filename",
                "source",
                "source_path",
                "original_label_path",
                "sha256",
                "duplicate_with_v1",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    with (OUTPUT_ROOT / "hard_examples_duplicate_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "hard_image",
            "matching_v1_image",
            "sha256",
        ])
        writer.writerows(duplicate_hard)

    train_count = len(image_files(OUTPUT_ROOT / "train" / "images"))
    val_count = len(image_files(OUTPUT_ROOT / "val" / "images"))
    test_count = len(image_files(OUTPUT_ROOT / "test" / "images"))

    summary = {
        "experiment": "V2.1",
        "v1_train_images": len(v1_train_images),
        "hard_examples_total": len(hard_images),
        "hard_examples_exact_duplicates_with_v1": len(duplicate_hard),
        "hard_examples_added": len(unique_hard),
        "v2_1_train_images": train_count,
        "v1_validation_images": val_count,
        "v1_test_images": test_count,
        "hard_examples_empty_label_images": hard_empty,
        "taxonomy": CLASS_NAMES,
    }

    (OUTPUT_ROOT / "v2.1_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def audit_dataset():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    v1_train_images, v1_empty, v1_counts, v1_missing = audit_training_root(
        V1_ROOT, "train"
    )
    v21_train_images, v21_empty, v21_counts, v21_missing = audit_training_root(
        OUTPUT_ROOT, "train"
    )

    train_images = image_files(OUTPUT_ROOT / "train" / "images")

    hard_images = [
        p for p in train_images
        if p.name.lower().startswith("hard_")
    ]

    hard_counts = Counter()
    hard_empty = 0

    for image in hard_images:
        label = OUTPUT_ROOT / "train" / "labels" / f"{image.stem}.txt"
        classes = read_yolo_classes(label)

        if classes:
            hard_counts.update(classes)
        else:
            hard_empty += 1

    print("\n" + "=" * 72)
    print("V2.1 COMPOSITION AUDIT")
    print("=" * 72)
    print(f"\nV1 train images   : {v1_train_images:,} (empty: {v1_empty:,})")
    print(f"V2.1 train images : {v21_train_images:,} (empty: {v21_empty:,})")
    print(f"Hard additions    : {len(hard_images):,} (empty: {hard_empty:,})")

    v1_total = sum(v1_counts.values())
    v21_total = sum(v21_counts.values())
    hard_total = sum(hard_counts.values())

    print("\nClass composition:")
    print(
        f"{'Class':<12}{'V1':>10}{'V2.1':>10}"
        f"{'Added':>10}{'V1 %':>10}{'V2.1 %':>10}"
    )

    rows = []

    for class_id, name in CLASS_NAMES.items():
        v1 = v1_counts[class_id]
        v21 = v21_counts[class_id]
        added = hard_counts[class_id]
        v1_pct = 100 * v1 / v1_total if v1_total else 0
        v21_pct = 100 * v21 / v21_total if v21_total else 0

        print(
            f"{name:<12}{v1:>10,}{v21:>10,}{added:>10,}"
            f"{v1_pct:>9.2f}%{v21_pct:>9.2f}%"
        )

        rows.append([
            class_id,
            name,
            v1,
            v21,
            added,
            round(v1_pct, 4),
            round(v21_pct, 4),
            round(v21_pct - v1_pct, 4),
        ])

    with (AUDIT_DIR / "v1_vs_v2_1_class_composition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id",
            "class_name",
            "v1_objects",
            "v21_objects",
            "hard_added_objects",
            "v1_pct",
            "v21_pct",
            "change_pct_points",
        ])
        writer.writerows(rows)

    with (AUDIT_DIR / "composition_audit_summary.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("V2.1 COMPOSITION AUDIT\n\n")
        f.write(f"V1 train images: {v1_train_images:,}\n")
        f.write(f"V2.1 train images: {v21_train_images:,}\n")
        f.write(f"Hard additions: {len(hard_images):,}\n")
        f.write(f"Hard added objects: {hard_total:,}\n\n")

        for class_id, name in CLASS_NAMES.items():
            f.write(
                f"{name}: V1={v1_counts[class_id]:,}, "
                f"V2.1={v21_counts[class_id]:,}, "
                f"added={hard_counts[class_id]:,}\n"
            )

    if v1_missing or v21_missing:
        fail(
            "Image/label alignment failed during audit.\n"
            f"V1 missing: {len(v1_missing)}\n"
            f"V2.1 missing: {len(v21_missing)}"
        )

    if v21_train_images != 13635:
        fail(
            f"Expected 13,635 V2.1 train images, "
            f"found {v21_train_images}."
        )

    print(f"\nAudit reports saved to: {AUDIT_DIR}")
    return {
        "v1_train_images": v1_train_images,
        "v21_train_images": v21_train_images,
        "hard_additions": len(hard_images),
    }


def main():
    summary = build_dataset()
    audit = audit_dataset()

    print("\n" + "=" * 72)
    print("V2.1 BUILD COMPLETE")
    print("=" * 72)
    print(f"V1 train             : {summary['v1_train_images']:,}")
    print(f"Hard examples        : {summary['hard_examples_total']:,}")
    print(f"Duplicates skipped   : {summary['hard_examples_exact_duplicates_with_v1']:,}")
    print(f"Hard examples added  : {summary['hard_examples_added']:,}")
    print(f"V2.1 train           : {audit['v21_train_images']:,}")
    print(f"Validation            : {summary['v1_validation_images']:,}")
    print(f"Test                  : {summary['v1_test_images']:,}")
    print(f"Output                : {OUTPUT_ROOT}")
    print("\nV1 source and original hard-example pool were not modified.")


if __name__ == "__main__":
    main()
