from pathlib import Path
from collections import Counter
import csv
import json
import re

ROOT = Path(r"D:\WD Dataset\Processed")
OUTPUT = ROOT / "source_dataset_audit"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

EXPECTED_DATASETS = {
    "CVAT": {
        "path": ROOT / "CVAT_YOLO_Converted",
        "images": 2236,
        "annotations": 967,
    },
    "Kaggle": {
        "path": ROOT / "Kaggle",
        "images": 713,
        "annotations": 1152,
    },
    "HGP": {
        "path": ROOT / "HGP",
        "images": 2199,
        "annotations": 2866,
    },
    "Dangerous_Items": {
        "path": ROOT / "Dangerous_Items_Normalized",
        "images": 10140,
        "annotations": 9144,
    },
}

CVAT_EXPECTED = {
    "WD_Pilot_Handgun_Train": {
        "images": 765,
        "annotations": 184,
        "classes": {0: 2, 2: 182},
    },
    "WD_Pilot_Knife_Train": {
        "images": 783,
        "annotations": 124,
        "classes": {0: 123, 2: 1},
    },
    "WD_Pilot_Longgun_Train": {
        "images": 412,
        "annotations": 411,
        "classes": {2: 64, 3: 347},
    },
    "WD_Pilot_Machette_Train": {
        "images": 63,
        "annotations": 28,
        "classes": {1: 28},
    },
    "WD_Pilot_Train": {
        "images": 213,
        "annotations": 220,
        "classes": {0: 74, 1: 14, 2: 75, 3: 57},
    },
}

KNOWN_MAPPINGS = {
    "Dangerous_Items": {
        0: 1,
        1: 0,
        2: 4,
        3: 3,
        4: 2,
    },
    "HGP": {
        2: 2,
    },
    "Kaggle": {
        0: 0,
        2: 2,
        3: 3,
        4: 4,
    },
}

EXPECTED_SOURCE_CLASSES = {
    "Dangerous_Items": {0, 1, 2, 3, 4},
    "HGP": {2},
    "Kaggle": {0, 2, 3, 4},
}


def find_images(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(root):
    return sorted(
        p for p in root.rglob("*.txt")
        if p.is_file()
        and p.name != "CLASS_MAPPING.txt"
    )


def relative_label_map(root):
    mapping = {}

    for label in find_labels(root):
        try:
            relative = label.relative_to(root)
        except ValueError:
            continue

        if "labels" in [part.lower() for part in relative.parts]:
            parts = list(relative.parts)
            index = next(
                i for i, part in enumerate(parts)
                if part.lower() == "labels"
            )
            relative_after_labels = Path(
                *parts[index + 1:]
            )
            mapping[
                str(relative_after_labels.with_suffix("")).lower()
            ] = label
        else:
            mapping[label.stem.lower()] = label

    return mapping


def match_label(image, root, labels):
    try:
        relative = image.relative_to(root)
        parts = list(relative.parts)

        if "images" in [part.lower() for part in parts]:
            index = next(
                i for i, part in enumerate(parts)
                if part.lower() == "images"
            )
            after_images = Path(*parts[index + 1:])
            key = str(after_images.with_suffix("")).lower()

            if key in labels:
                return labels[key]
    except ValueError:
        pass

    candidates = [
        label for label in labels.values()
        if label.stem.lower() == image.stem.lower()
    ]

    return candidates[0] if len(candidates) == 1 else None


def read_label(label_path):
    classes = Counter()
    errors = []

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception as exc:
        return classes, [f"FILE ERROR: {exc}"]

    for line_number, line in enumerate(lines, 1):
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) != 5:
            errors.append(
                f"line {line_number}: expected 5 values"
            )
            continue

        try:
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"line {line_number}: non-numeric value"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"line {line_number}: invalid class {class_id}"
            )
            continue

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            errors.append(
                f"line {line_number}: invalid normalized box"
            )
            continue

        classes[class_id] += 1

    return classes, errors


def audit_tree(name, root):
    images = find_images(root)
    labels = find_labels(root)
    label_map = relative_label_map(root)

    class_counts = Counter()
    malformed = []
    missing_labels = []
    matched_labels = set()
    empty_labels = 0

    image_keys = []

    for image in images:
        try:
            rel = image.relative_to(root)
        except ValueError:
            rel = image

        key = str(rel).lower()

        if "images" in [part.lower() for part in rel.parts]:
            parts = list(rel.parts)
            index = next(
                i for i, part in enumerate(parts)
                if part.lower() == "images"
            )
            key = str(
                Path(*parts[index + 1:]).with_suffix("")
            ).lower()

        image_keys.append(key)

        label = match_label(
            image,
            root,
            label_map,
        )

        if label is None:
            missing_labels.append(str(image))
            continue

        matched_labels.add(label)
        classes, errors = read_label(label)
        malformed.extend(
            (str(label), error)
            for error in errors
        )

        class_counts.update(classes)

        if not classes:
            empty_labels += 1

    labels_without_images = [
        str(label)
        for label in labels
        if label not in matched_labels
    ]

    duplicate_image_keys = [
        key for key, count in Counter(image_keys).items()
        if count > 1
    ]

    return {
        "name": name,
        "root": str(root),
        "images": len(images),
        "labels": len(labels),
        "annotations": sum(class_counts.values()),
        "empty_labels": empty_labels,
        "missing_labels": len(missing_labels),
        "labels_without_images": len(labels_without_images),
        "duplicate_image_stems": len(duplicate_image_keys),
        "malformed": len(malformed),
        "class_counts": dict(sorted(class_counts.items())),
        "missing_label_examples": missing_labels[:20],
        "orphan_label_examples": labels_without_images[:20],
        "malformed_examples": malformed[:20],
    }


def audit_cvat_subsets():
    root = EXPECTED_DATASETS["CVAT"]["path"]
    subset_results = {}

    if not root.exists():
        raise FileNotFoundError(root)

    for subset_name, expected in CVAT_EXPECTED.items():
        subset = root / subset_name

        if not subset.exists():
            subset_results[subset_name] = {
                "error": "subset not found"
            }
            continue

        result = audit_tree(
            subset_name,
            subset,
        )

        problems = []

        if result["images"] != expected["images"]:
            problems.append("image count mismatch")

        if result["annotations"] != expected["annotations"]:
            problems.append("annotation count mismatch")

        actual_classes = {
            int(k): v
            for k, v in result["class_counts"].items()
        }

        if actual_classes != expected["classes"]:
            problems.append("class count mismatch")

        if result["missing_labels"]:
            problems.append("missing labels")

        if result["labels_without_images"]:
            problems.append("orphan labels")

        if result["malformed"]:
            problems.append("malformed annotations")

        result["problems"] = problems
        subset_results[subset_name] = result

    return subset_results


def validate_known_mappings(results):
    mapping_problems = []

    for dataset, expected_classes in EXPECTED_SOURCE_CLASSES.items():
        observed = {
            int(k)
            for k in results[dataset]["class_counts"]
        }

        if observed != expected_classes:
            mapping_problems.append(
                f"{dataset}: expected source IDs "
                f"{sorted(expected_classes)}, "
                f"found {sorted(observed)}"
            )

        for source_id in observed:
            if source_id not in KNOWN_MAPPINGS[dataset]:
                mapping_problems.append(
                    f"{dataset}: no project mapping for "
                    f"source ID {source_id}"
                )

    return mapping_problems


def write_reports(results, cvat_subsets):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "final_project_taxonomy": CLASS_NAMES,
        "known_source_mappings": KNOWN_MAPPINGS,
        "datasets": results,
        "cvat_subsets": cvat_subsets,
    }

    (OUTPUT / "source_dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    rows = []

    for name, result in results.items():
        for class_id, count in result["class_counts"].items():
            source_id = int(class_id)
            mapping = KNOWN_MAPPINGS.get(
                name,
                {},
            )
            target_id = mapping.get(
                source_id,
                "",
            )

            rows.append({
                "dataset": name,
                "source_class_id": source_id,
                "annotation_count": count,
                "final_class_id": target_id,
                "final_class_name": (
                    CLASS_NAMES[target_id]
                    if target_id != ""
                    else ""
                ),
            })

    with (
        OUTPUT / "source_class_mapping_audit.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "source_class_id",
                "annotation_count",
                "final_class_id",
                "final_class_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    cvat_rows = []

    for subset, result in cvat_subsets.items():
        cvat_rows.append({
            "subset": subset,
            "images": result.get("images", 0),
            "annotations": result.get(
                "annotations",
                0,
            ),
            "problems": "; ".join(
                result.get("problems", [])
            ),
        })

    with (
        OUTPUT / "cvat_subset_audit.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "images",
                "annotations",
                "problems",
            ],
        )
        writer.writeheader()
        writer.writerows(cvat_rows)


def main():
    print("=" * 72)
    print("WEAPON DETECTION — SOURCE DATASET AUDIT")
    print("=" * 72)

    results = {}

    for dataset_name, info in EXPECTED_DATASETS.items():
        print(f"\nAuditing {dataset_name}...")
        results[dataset_name] = audit_tree(
            dataset_name,
            info["path"],
        )

        print(
            f"  images={results[dataset_name]['images']:,} "
            f"labels={results[dataset_name]['labels']:,} "
            f"annotations={results[dataset_name]['annotations']:,} "
            f"empty={results[dataset_name]['empty_labels']:,}"
        )

    cvat_subsets = audit_cvat_subsets()

    problems = []

    for dataset_name, info in EXPECTED_DATASETS.items():
        result = results[dataset_name]

        if result["images"] != info["images"]:
            problems.append(
                f"{dataset_name}: image count mismatch"
            )

        if result["annotations"] != info["annotations"]:
            problems.append(
                f"{dataset_name}: annotation count mismatch"
            )

        if result["missing_labels"]:
            problems.append(
                f"{dataset_name}: missing labels"
            )

        if result["labels_without_images"]:
            problems.append(
                f"{dataset_name}: orphan labels"
            )

        if result["malformed"]:
            problems.append(
                f"{dataset_name}: malformed annotations"
            )

        if result["duplicate_image_stems"]:
            problems.append(
                f"{dataset_name}: duplicate image stems"
            )

    problems.extend(
        validate_known_mappings(results)
    )

    for subset, result in cvat_subsets.items():
        for problem in result.get("problems", []):
            problems.append(
                f"CVAT/{subset}: {problem}"
            )

    write_reports(
        results,
        cvat_subsets,
    )

    print("\nCLASS COUNTS")
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}")
        for class_id, count in result["class_counts"].items():
            print(
                f"  {class_id}: "
                f"{CLASS_NAMES[int(class_id)]} "
                f"= {count}"
            )

    if problems:
        print("\nAUDIT RESULT: FAIL")
        for problem in problems:
            print(" -", problem)
    else:
        print("\nAUDIT RESULT: PASS")
        print(
            "Source datasets match the verified project "
            "inventory and frozen taxonomy."
        )

    print(f"\nReports: {OUTPUT}")
    print("READ-ONLY: no dataset files were modified.")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
