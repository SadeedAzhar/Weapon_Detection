from pathlib import Path
from collections import Counter
import csv
import shutil

ROOT = Path(r"D:\WD Dataset\Processed")
OUTPUT = ROOT / "Final_Merged_Dataset"

SOURCES = {
    "CVAT": ROOT / "CVAT_YOLO_Converted",
    "Kaggle": ROOT / "Kaggle",
    "HGP": ROOT / "HGP",
    "Dangerous_Items": ROOT / "Dangerous_Items_Normalized",
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

EXPECTED = {
    "CVAT": {"images": 2236, "annotations": 967},
    "Kaggle": {"images": 713, "annotations": 1152},
    "HGP": {"images": 2199, "annotations": 2866},
    "Dangerous_Items": {"images": 10140, "annotations": 9144},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(root):
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def find_matching_label(image_path, dataset_root):
    parts = image_path.parts

    for i, part in enumerate(parts):
        if part.lower() == "images":
            relative = Path(*parts[i + 1:]).with_suffix(".txt")
            candidate = Path(*parts[:i]) / "labels" / relative
            if candidate.exists():
                return candidate
            break

    matches = list(dataset_root.rglob(f"{image_path.stem}.txt"))

    if len(matches) == 1:
        return matches[0]

    label_matches = [
        p for p in matches
        if "labels" in {x.lower() for x in p.parts}
    ]
    return label_matches[0] if len(label_matches) == 1 else None


def analyze_label(label_path):
    annotations = 0
    class_counts = Counter()
    malformed = []
    invalid_boxes = []
    unexpected_ids = []

    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                malformed.append((str(label_path), line_no))
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                malformed.append((str(label_path), line_no))
                continue

            if class_id not in CLASS_NAMES:
                unexpected_ids.append(
                    (str(label_path), line_no, class_id)
                )
                continue

            if not (
                0 <= x <= 1 and 0 <= y <= 1
                and 0 < w <= 1 and 0 < h <= 1
            ):
                invalid_boxes.append((str(label_path), line_no))
                continue

            annotations += 1
            class_counts[class_id] += 1

    return annotations, class_counts, malformed, invalid_boxes, unexpected_ids


def validate_sources():
    records = []
    total_images = total_annotations = total_empty = 0
    total_classes = Counter()
    failed = False

    for source_name, source_root in SOURCES.items():
        print(f"\nValidating {source_name}: {source_root}")

        if not source_root.exists():
            print("  ERROR: source not found")
            failed = True
            continue

        images = find_images(source_root)
        labels = find_labels(source_root)
        expected = EXPECTED[source_name]

        print(f"  Images found : {len(images)}")
        print(f"  Labels found : {len(labels)}")

        if len(images) != expected["images"]:
            print(f"  ERROR: expected {expected['images']} images")
            failed = True

        source_annotations = source_empty = 0
        source_classes = Counter()
        missing_labels = []
        malformed = []
        invalid_boxes = []
        unexpected_ids = []

        for image_path in images:
            label_path = find_matching_label(image_path, source_root)
            if label_path is None:
                missing_labels.append(str(image_path))
                continue

            (
                annotations,
                class_counts,
                bad_format,
                bad_boxes,
                bad_ids,
            ) = analyze_label(label_path)

            source_annotations += annotations
            source_classes.update(class_counts)
            malformed.extend(bad_format)
            invalid_boxes.extend(bad_boxes)
            unexpected_ids.extend(bad_ids)

            if annotations == 0:
                source_empty += 1

            records.append({
                "source": source_name,
                "image": image_path,
                "label": label_path,
                "annotations": annotations,
            })

        print(f"  Annotations  : {source_annotations}")
        print(f"  Empty labels : {source_empty}")
        print(f"  Missing labels: {len(missing_labels)}")
        print(f"  Malformed    : {len(malformed)}")
        print(f"  Invalid boxes: {len(invalid_boxes)}")
        print(f"  Invalid IDs  : {len(unexpected_ids)}")

        if source_annotations != expected["annotations"]:
            print(
                f"  ERROR: expected {expected['annotations']} annotations"
            )
            failed = True

        if missing_labels or malformed or invalid_boxes or unexpected_ids:
            failed = True

        total_images += len(images)
        total_annotations += source_annotations
        total_empty += source_empty
        total_classes.update(source_classes)

    print("\nOverall source totals:")
    print(f"  Images       : {total_images}")
    print(f"  Annotations  : {total_annotations}")
    print(f"  Empty labels : {total_empty}")

    for class_id, name in CLASS_NAMES.items():
        print(f"  {name:<10}: {total_classes[class_id]}")

    return records, failed


def write_yaml(path):
    path.write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n\n"
        "names:\n"
        "  0: Knife\n"
        "  1: Machette\n"
        "  2: Handgun\n"
        "  3: Longgun\n"
        "  4: Other\n",
        encoding="utf-8",
    )


def merge(records):
    image_dir = OUTPUT / "images"
    label_dir = OUTPUT / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    class_counts = Counter()
    empty_labels = 0

    for index, record in enumerate(records, 1):
        source = record["source"]
        image_path = record["image"]
        label_path = record["label"]

        stem = f"{source}_{index:06d}"
        image_name = stem + image_path.suffix.lower()
        label_name = stem + ".txt"

        shutil.copy2(image_path, image_dir / image_name)
        shutil.copy2(label_path, label_dir / label_name)

        annotations, counts, _, _, _ = analyze_label(label_path)
        class_counts.update(counts)

        if annotations == 0:
            empty_labels += 1

        manifest.append({
            "merged_image": image_name,
            "merged_label": label_name,
            "source_dataset": source,
            "original_image": str(image_path),
            "original_label": str(label_path),
            "annotations": annotations,
        })

    manifest_path = OUTPUT / "merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "merged_image", "merged_label", "source_dataset",
                "original_image", "original_label", "annotations",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    summary_path = OUTPUT / "merge_class_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "annotations"])
        for class_id, name in CLASS_NAMES.items():
            writer.writerow([class_id, name, class_counts[class_id]])

    write_yaml(OUTPUT / "dataset.yaml")

    print("\nMaster dataset created:")
    print(f"  Images       : {len(records)}")
    print(f"  Annotations  : {sum(r['annotations'] for r in records)}")
    print(f"  Empty labels : {empty_labels}")
    print(f"  Output       : {OUTPUT}")


def main():
    print("=" * 60)
    print("WEAPON DETECTION — MASTER DATASET MERGE")
    print("=" * 60)

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    records, failed = validate_sources()

    if failed:
        print("\nVALIDATION FAILED — no merge performed.")
        return

    print("\nAll source datasets passed validation.")
    merge(records)
    print("\nMerge complete.")
    print("This is the master image pool.")
    print("Train/validation/test splitting is handled separately.")


if __name__ == "__main__":
    main()
