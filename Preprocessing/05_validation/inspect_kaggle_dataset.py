from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json

import cv2

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

PROJECT_CLASSES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

EXPECTED_SOURCE_IDS = {0, 2, 3, 4}

TARGET_NAMES = {
    0: "Automatic Rifle",
    1: "Bazooka",
    2: "Grenade Launcher",
    3: "Handgun",
    4: "Knife",
    5: "Shotgun",
    6: "SMG",
    7: "Sniper",
    8: "Sword",
}


def find_images(image_dir):
    if not image_dir.exists():
        return []

    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_label(label_path):
    annotations = []
    errors = []

    if not label_path.exists():
        return annotations, ["missing_label"]

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception as exc:
        return annotations, [f"read_error: {exc}"]

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
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
        except ValueError:
            errors.append(
                f"line {line_number}: non-numeric value"
            )
            continue

        if class_id not in PROJECT_CLASSES:
            errors.append(
                f"line {line_number}: unexpected class {class_id}"
            )
            continue

        if not (
            0 <= coords[0] <= 1
            and 0 <= coords[1] <= 1
            and 0 < coords[2] <= 1
            and 0 < coords[3] <= 1
        ):
            errors.append(
                f"line {line_number}: invalid box"
            )
            continue

        annotations.append(
            (class_id, *coords)
        )

    return annotations, errors


def inspect_dataset(dataset_root):
    dataset_root = Path(dataset_root)

    class_counts = Counter()
    class_images = defaultdict(set)
    split_counts = Counter()
    split_annotations = Counter()
    split_empty = Counter()
    multi_box = Counter()

    missing_labels = []
    orphan_labels = []
    malformed = []
    dimensions = Counter()
    image_count = 0
    label_count = 0

    for split in ["train", "val"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        images = find_images(image_dir)
        labels = sorted(
            p
            for p in label_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".txt"
        ) if label_dir.exists() else []

        image_count += len(images)
        label_count += len(labels)
        split_counts[split] = len(images)

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        for stem in sorted(image_stems - label_stems):
            missing_labels.append(
                (split, stem)
            )

        for stem in sorted(label_stems - image_stems):
            orphan_labels.append(
                (split, stem)
            )

        for image in images:
            img = cv2.imread(str(image))
            if img is not None:
                h, w = img.shape[:2]
                dimensions[(w, h)] += 1

        for label in labels:
            annotations, errors = read_label(label)

            malformed.extend(
                (split, label.name, error)
                for error in errors
                if error != "missing_label"
            )

            split_annotations[split] += len(annotations)
            multi_box[len(annotations)] += 1

            if not annotations:
                split_empty[split] += 1

            for class_id, *_ in annotations:
                class_counts[class_id] += 1
                class_images[class_id].add(
                    f"{split}:{label.stem}"
                )

    summary = {
        "images": image_count,
        "labels": label_count,
        "annotations": sum(split_annotations.values()),
        "empty_labels": sum(split_empty.values()),
        "class_counts": {
            PROJECT_CLASSES[k]: class_counts[k]
            for k in sorted(PROJECT_CLASSES)
        },
        "class_image_counts": {
            PROJECT_CLASSES[k]: len(class_images[k])
            for k in sorted(PROJECT_CLASSES)
        },
        "split_counts": dict(split_counts),
        "split_annotations": dict(split_annotations),
        "split_empty": dict(split_empty),
        "multi_box_distribution": dict(sorted(multi_box.items())),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "malformed": len(malformed),
        "dimensions": {
            f"{w}x{h}": count
            for (w, h), count in dimensions.most_common()
        },
        "observed_source_ids": sorted(class_counts),
    }

    return (
        summary,
        class_images,
        missing_labels,
        orphan_labels,
        malformed,
    )


def make_contact_sheets(
    dataset_root,
    output_dir,
    samples_per_class=12,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires matplotlib."
        ) from exc

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = defaultdict(list)

    for split in ["train", "val"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        for label_path in sorted(
            label_dir.glob("*.txt")
            if label_dir.exists()
            else []
        ):
            image_path = next(
                (
                    image_dir / f"{label_path.stem}{ext}"
                    for ext in IMAGE_EXTENSIONS
                    if (image_dir / f"{label_path.stem}{ext}").exists()
                ),
                None,
            )

            if image_path is None:
                continue

            annotations, _ = read_label(label_path)

            for class_id in {
                annotation[0]
                for annotation in annotations
            }:
                if len(examples[class_id]) < samples_per_class:
                    examples[class_id].append(
                        (image_path, annotations)
                    )

    for class_id, items in sorted(examples.items()):
        fig, axes = plt.subplots(
            3,
            4,
            figsize=(16, 12),
        )
        axes = axes.flatten()

        for axis in axes:
            axis.axis("off")

        for index, (image_path, annotations) in enumerate(items):
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            h, w = image.shape[:2]

            for (
                annotation_class,
                x_center,
                y_center,
                box_width,
                box_height,
            ) in annotations:
                if annotation_class != class_id:
                    continue

                x1 = int(
                    (x_center - box_width / 2) * w
                )
                y1 = int(
                    (y_center - box_height / 2) * h
                )
                x2 = int(
                    (x_center + box_width / 2) * w
                )
                y2 = int(
                    (y_center + box_height / 2) * h
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3,
                )

            axes[index].imshow(image)
            axes[index].set_title(
                f"{image_path.parent.parent.name} "
                f"| {image_path.name}"
            )

        fig.suptitle(
            f"Class {class_id}: "
            f"{PROJECT_CLASSES[class_id]}",
            fontsize=16,
        )
        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"class_{class_id}_"
            f"{PROJECT_CLASSES[class_id].lower()}.png"
        )
        plt.close(fig)

    return {
        class_id: len(items)
        for class_id, items in examples.items()
    }


def write_reports(
    output_dir,
    summary,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "kaggle_inspection_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (
        output_dir / "kaggle_class_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id",
            "class_name",
            "annotations",
            "images",
        ])

        for class_id, class_name in PROJECT_CLASSES.items():
            writer.writerow([
                class_id,
                class_name,
                summary["class_counts"][class_name],
                summary["class_image_counts"][class_name],
            ])


def print_summary(summary):
    print("\n" + "=" * 70)
    print("KAGGLE DATASET INSPECTION")
    print("=" * 70)

    print(
        f"\nImages       : {summary['images']}"
    )
    print(
        f"Labels       : {summary['labels']}"
    )
    print(
        f"Annotations  : {summary['annotations']}"
    )
    print(
        f"Empty labels : {summary['empty_labels']}"
    )

    print("\nSPLITS")
    for split in ["train", "val"]:
        print(
            f"{split:<5}: "
            f"{summary['split_counts'].get(split, 0)} images | "
            f"{summary['split_annotations'].get(split, 0)} annotations | "
            f"{summary['split_empty'].get(split, 0)} empty"
        )

    print("\nCLASS INVENTORY")
    for class_id, class_name in PROJECT_CLASSES.items():
        print(
            f"{class_id}: "
            f"{class_name:<10} "
            f"{summary['class_counts'][class_name]:>5} annotations | "
            f"{summary['class_image_counts'][class_name]:>5} images"
        )

    print("\nDATA QUALITY")
    print(
        f"Missing labels : {summary['missing_labels']}"
    )
    print(
        f"Orphan labels  : {summary['orphan_labels']}"
    )
    print(
        f"Malformed      : {summary['malformed']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Kaggle class inspection, "
            "audit, and optional visualization."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    summary, _, _, _, _ = inspect_dataset(
        Path(args.dataset)
    )

    print_summary(summary)

    problems = []

    if summary["images"] != 713:
        problems.append(
            f"Expected 713 images, found {summary['images']}"
        )

    if summary["labels"] != 713:
        problems.append(
            f"Expected 713 labels, found {summary['labels']}"
        )

    if summary["annotations"] != 1152:
        problems.append(
            f"Expected 1152 annotations, found {summary['annotations']}"
        )

    if set(summary["observed_source_ids"]) != EXPECTED_SOURCE_IDS:
        problems.append(
            "Observed source class IDs do not match "
            "the verified Kaggle inventory."
        )

    if (
        summary["missing_labels"]
        or summary["orphan_labels"]
        or summary["malformed"]
    ):
        problems.append(
            "Image/label or annotation integrity problems detected."
        )

    write_reports(
        Path(args.output),
        summary,
    )

    if args.visualize:
        created = make_contact_sheets(
            args.dataset,
            Path(args.output) / "contact_sheets",
            args.samples_per_class,
        )
        print(
            f"\nContact sheets created for "
            f"{len(created)} classes."
        )

    if problems:
        print("\nAUDIT RESULT: FAIL")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)

    print("\nAUDIT RESULT: PASS")
    print(
        "The Kaggle dataset already uses "
        "the frozen project IDs."
    )
    print(
        "No class remapping is required."
    )
    print(
        f"\nReports: {args.output}"
    )
    print(
        "READ-ONLY: no Kaggle files were modified."
    )


if __name__ == "__main__":
    main()
