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

VALID_CLASS_IDS = set(CLASS_NAMES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED = {
    "CVAT": {"images": 2236, "annotations": 967},
    "Kaggle": {"images": 713, "annotations": 1152},
    "HGP": {"images": 2199, "annotations": 2866},
    "Dangerous_Items": {"images": 10140, "annotations": 9144},
}

FINAL_EXPECTED = {
    "images": 15288,
    "labels": 15288,
    "annotations": 14129,
    "empty_labels": 3784,
    "classes": {
        0: 2491,
        1: 1756,
        2: 3382,
        3: 4430,
        4: 2070,
    },
}


def find_images(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(root):
    return sorted(
        p for p in root.rglob("*.txt")
        if p.is_file()
    )


def find_matching_label(image_path, dataset_root):
    parts = image_path.parts

    for i, part in enumerate(parts):
        if part.lower() != "images":
            continue

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
        if "labels" in {part.lower() for part in p.parts}
    ]

    if len(label_matches) == 1:
        return label_matches[0]

    return None


def analyze_label(label_path):
    annotations = 0
    class_counts = Counter()
    malformed = []
    invalid_boxes = []
    unexpected_ids = []

    with label_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                malformed.append((str(label_path), line_number))
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                malformed.append((str(label_path), line_number))
                continue

            if class_id not in VALID_CLASS_IDS:
                unexpected_ids.append(
                    (str(label_path), line_number, class_id)
                )
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_boxes.append((str(label_path), line_number))
                continue

            annotations += 1
            class_counts[class_id] += 1

    return (
        annotations,
        class_counts,
        malformed,
        invalid_boxes,
        unexpected_ids,
    )


def validate_sources():
    records = []
    overall = {
        "images": 0,
        "annotations": 0,
        "empty_labels": 0,
        "classes": Counter(),
    }

    for source_name, source_root in SOURCES.items():
        if not source_root.exists():
            raise FileNotFoundError(
                f"{source_name} source not found:\n{source_root}"
            )

        images = find_images(source_root)

        if len(images) != EXPECTED[source_name]["images"]:
            raise RuntimeError(
                f"{source_name}: expected "
                f"{EXPECTED[source_name]['images']} images, "
                f"found {len(images)}"
            )

        source_annotations = 0
        source_empty = 0
        source_classes = Counter()

        for image_path in images:
            label_path = find_matching_label(
                image_path,
                source_root,
            )

            if label_path is None:
                raise RuntimeError(
                    f"{source_name}: missing label for "
                    f"{image_path}"
                )

            (
                annotation_count,
                class_counts,
                malformed,
                invalid_boxes,
                unexpected_ids,
            ) = analyze_label(label_path)

            if malformed or invalid_boxes or unexpected_ids:
                raise RuntimeError(
                    f"{source_name}: invalid annotation in "
                    f"{label_path}"
                )

            source_annotations += annotation_count
            source_classes.update(class_counts)

            if annotation_count == 0:
                source_empty += 1

            records.append(
                {
                    "source": source_name,
                    "image": image_path,
                    "label": label_path,
                    "annotations": annotation_count,
                }
            )

        expected_annotations = EXPECTED[source_name]["annotations"]

        if source_annotations != expected_annotations:
            raise RuntimeError(
                f"{source_name}: expected "
                f"{expected_annotations} annotations, "
                f"found {source_annotations}"
            )

        overall["images"] += len(images)
        overall["annotations"] += source_annotations
        overall["empty_labels"] += source_empty
        overall["classes"].update(source_classes)

        print(
            f"{source_name:<18}"
            f"images={len(images):>6} "
            f"annotations={source_annotations:>6} "
            f"empty={source_empty:>5}"
        )

    return records, overall


def write_dataset_yaml():
    (OUTPUT / "dataset.yaml").write_text(
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
    images_out = OUTPUT / "images"
    labels_out = OUTPUT / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    manifest = []
    merged_classes = Counter()
    empty_labels = 0

    for index, record in enumerate(records, 1):
        source = record["source"]
        image_path = record["image"]
        label_path = record["label"]

        new_stem = f"{source}_{index:06d}"
        image_name = new_stem + image_path.suffix.lower()
        label_name = new_stem + ".txt"

        image_dest = images_out / image_name
        label_dest = labels_out / label_name

        if image_dest.exists() or label_dest.exists():
            raise FileExistsError(
                f"Filename collision: {new_stem}"
            )

        shutil.copy2(image_path, image_dest)
        shutil.copy2(label_path, label_dest)

        (
            annotation_count,
            class_counts,
            _,
            _,
            _,
        ) = analyze_label(label_path)

        merged_classes.update(class_counts)

        if annotation_count == 0:
            empty_labels += 1

        manifest.append(
            {
                "merged_image": image_name,
                "merged_label": label_name,
                "source_dataset": source,
                "original_image": str(image_path),
                "original_label": str(label_path),
                "annotations": annotation_count,
            }
        )

    with (
        OUTPUT / "merge_manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest[0]),
        )
        writer.writeheader()
        writer.writerows(manifest)

    with (
        OUTPUT / "merge_class_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class_id", "class_name", "annotations"]
        )
        for class_id, name in CLASS_NAMES.items():
            writer.writerow(
                [class_id, name, merged_classes[class_id]]
            )

    write_dataset_yaml()

    return {
        "images": len(manifest),
        "annotations": sum(
            row["annotations"] for row in manifest
        ),
        "empty_labels": empty_labels,
        "classes": merged_classes,
    }


def audit():
    images = find_images(OUTPUT / "images")
    labels = find_labels(OUTPUT / "labels")

    image_stems = {}
    label_stems = {}
    duplicate_images = []
    duplicate_labels = []

    for path in images:
        key = path.stem.lower()
        if key in image_stems:
            duplicate_images.append(
                (image_stems[key], path)
            )
        else:
            image_stems[key] = path

    for path in labels:
        key = path.stem.lower()
        if key in label_stems:
            duplicate_labels.append(
                (label_stems[key], path)
            )
        else:
            label_stems[key] = path

    images_without_labels = [
        str(path)
        for key, path in image_stems.items()
        if key not in label_stems
    ]
    labels_without_images = [
        str(path)
        for key, path in label_stems.items()
        if key not in image_stems
    ]

    class_counts = Counter()
    annotations = 0
    empty_labels = 0
    malformed = []
    invalid_boxes = []
    unexpected_ids = []

    for label_path in labels:
        (
            count,
            counts,
            bad_format,
            bad_boxes,
            bad_ids,
        ) = analyze_label(label_path)

        annotations += count
        class_counts.update(counts)
        malformed.extend(bad_format)
        invalid_boxes.extend(bad_boxes)
        unexpected_ids.extend(bad_ids)

        if count == 0:
            empty_labels += 1

    source_counts = Counter()

    for image_path in images:
        for source in SOURCES:
            if image_path.stem.startswith(
                source + "_"
            ):
                source_counts[source] += 1
                break

    problems = []

    if len(images) != FINAL_EXPECTED["images"]:
        problems.append(
            f"images={len(images)}"
        )
    if len(labels) != FINAL_EXPECTED["labels"]:
        problems.append(
            f"labels={len(labels)}"
        )
    if annotations != FINAL_EXPECTED["annotations"]:
        problems.append(
            f"annotations={annotations}"
        )
    if empty_labels != FINAL_EXPECTED["empty_labels"]:
        problems.append(
            f"empty_labels={empty_labels}"
        )
    if images_without_labels:
        problems.append(
            f"images_without_labels={len(images_without_labels)}"
        )
    if labels_without_images:
        problems.append(
            f"labels_without_images={len(labels_without_images)}"
        )
    if duplicate_images:
        problems.append(
            f"duplicate_images={len(duplicate_images)}"
        )
    if duplicate_labels:
        problems.append(
            f"duplicate_labels={len(duplicate_labels)}"
        )
    if malformed:
        problems.append(
            f"malformed={len(malformed)}"
        )
    if invalid_boxes:
        problems.append(
            f"invalid_boxes={len(invalid_boxes)}"
        )
    if unexpected_ids:
        problems.append(
            f"unexpected_ids={len(unexpected_ids)}"
        )

    for class_id, expected in FINAL_EXPECTED["classes"].items():
        if class_counts[class_id] != expected:
            problems.append(
                f"class_{class_id}="
                f"{class_counts[class_id]} "
                f"(expected {expected})"
            )

    for source, expected in EXPECTED.items():
        if source_counts[source] != expected["images"]:
            problems.append(
                f"{source}="
                f"{source_counts[source]} "
                f"(expected {expected['images']})"
            )

    report_rows = []

    for label_path in labels:
        (
            count,
            _,
            bad_format,
            bad_boxes,
            bad_ids,
        ) = analyze_label(label_path)

        report_rows.append(
            {
                "label_file": str(label_path),
                "annotations": count,
                "empty": count == 0,
                "malformed": bool(bad_format),
                "invalid_bbox": bool(bad_boxes),
                "unexpected_class": bool(bad_ids),
            }
        )

    report_path = OUTPUT / "final_merged_dataset_audit_report.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label_file",
                "annotations",
                "empty",
                "malformed",
                "invalid_bbox",
                "unexpected_class",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nMASTER AUDIT")
    print(f"Images       : {len(images)}")
    print(f"Labels       : {len(labels)}")
    print(f"Annotations  : {annotations}")
    print(f"Empty labels : {empty_labels}")
    print(
        "Alignment    : "
        f"{'PASS' if not images_without_labels and not labels_without_images else 'FAIL'}"
    )
    print(
        "Duplicates   : "
        f"{'PASS' if not duplicate_images and not duplicate_labels else 'FAIL'}"
    )
    print(
        "Labels       : "
        f"{'PASS' if not malformed and not invalid_boxes and not unexpected_ids else 'FAIL'}"
    )

    if problems:
        print("\nAUDIT RESULT: FAIL")
        for problem in problems:
            print(" -", problem)
        return False

    print("\nAUDIT RESULT: PASS")
    print("Final_Merged_Dataset is internally consistent.")
    print(f"Audit report: {report_path}")
    return True


def main():
    print("=" * 70)
    print("WEAPON DETECTION — MASTER DATASET MERGE + AUDIT")
    print("=" * 70)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT}\n"
            "Delete or rename it before running again."
        )

    print("\nSOURCE VALIDATION")
    records, source_totals = validate_sources()

    print(
        f"\nTotal validated images      : "
        f"{source_totals['images']}"
    )
    print(
        f"Total validated annotations : "
        f"{source_totals['annotations']}"
    )
    print(
        f"Total validated empty labels: "
        f"{source_totals['empty_labels']}"
    )

    print("\nMERGING")
    merged = merge(records)

    print(
        f"Images merged       : {merged['images']}"
    )
    print(
        f"Annotations merged : {merged['annotations']}"
    )
    print(
        f"Empty labels       : {merged['empty_labels']}"
    )

    print("\nRunning independent post-merge audit...")
    passed = audit()

    if not passed:
        raise SystemExit(1)

    print("\nCOMPLETE")
    print(f"Master dataset: {OUTPUT}")
    print("Original source datasets were not modified.")
    print("This remains the master image pool; splitting is separate.")


if __name__ == "__main__":
    main()
