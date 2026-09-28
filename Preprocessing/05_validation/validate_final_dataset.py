from pathlib import Path
from collections import Counter, defaultdict
import csv
import math

from PIL import Image

ROOT = Path(r"D:\WD Dataset\Processed")
DATASET = ROOT / "Final_Dataset_Split"
REPORT = DATASET / "final_dataset_validation_report.csv"

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def analyze_label(label_path, image_width, image_height):
    result = {
        "annotations": 0,
        "classes": Counter(),
        "boxes": [],
        "malformed": 0,
        "invalid": 0,
    }

    if not label_path.exists():
        return result

    with label_path.open(
        "r", encoding="utf-8", errors="ignore"
    ) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                result["malformed"] += 1
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                result["malformed"] += 1
                continue

            if class_id not in CLASS_NAMES:
                result["invalid"] += 1
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                result["invalid"] += 1
                continue

            box_width = w * image_width
            box_height = h * image_height
            relative_area = (box_width * box_height) / (
                image_width * image_height
            )

            result["annotations"] += 1
            result["classes"][class_id] += 1
            result["boxes"].append(
                {
                    "class_id": class_id,
                    "width": box_width,
                    "height": box_height,
                    "relative_area": relative_area,
                    "w_norm": w,
                    "h_norm": h,
                }
            )

    return result


def parse_yaml_names(dataset_path):
    yaml_path = dataset_path / "dataset.yaml"

    if not yaml_path.exists():
        return None, "dataset.yaml not found"

    names = {}
    inside_names = False

    for line in yaml_path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        stripped = line.strip()

        if stripped.lower() == "names:":
            inside_names = True
            continue

        if not inside_names:
            continue

        if not line.startswith((" ", "\t")):
            break

        parts = stripped.split(":", 1)
        if len(parts) != 2:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        names[class_id] = parts[1].strip().strip("'\"")

    if not names:
        return None, "No class names found in dataset.yaml"

    return names, None


def find_files(split):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        return [], [], f"Missing images/labels directory for {split}"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )
    return images, labels, None


def validate():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Final_Dataset_Split not found:\n{DATASET}"
        )

    yaml_names, yaml_error = parse_yaml_names(DATASET)

    global_counts = Counter()
    global_boxes = []
    split_stats = {}
    rows = []
    problems = []

    for split in SPLITS:
        images, labels, split_error = find_files(split)

        if split_error:
            problems.append(split_error)
            split_stats[split] = {
                "images": 0,
                "labels": 0,
                "annotations": 0,
                "empty": 0,
            }
            continue

        image_stems = {p.stem.lower() for p in images}
        label_stems = {p.stem.lower() for p in labels}

        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        if missing_labels:
            problems.append(
                f"{split}: {len(missing_labels)} images without labels"
            )
        if orphan_labels:
            problems.append(
                f"{split}: {len(orphan_labels)} labels without images"
            )

        split_annotations = 0
        split_empty = 0
        split_counts = Counter()
        unreadable_images = 0
        malformed = 0
        invalid_boxes = 0

        for image_path in images:
            label_path = DATASET / split / "labels" / f"{image_path.stem}.txt"
            width, height = get_image_size(image_path)

            if width is None or height is None:
                unreadable_images += 1
                problems.append(f"{split}: unreadable image: {image_path.name}")
                continue

            result = analyze_label(label_path, width, height)

            split_annotations += result["annotations"]
            split_counts.update(result["classes"])
            global_counts.update(result["classes"])
            global_boxes.extend(result["boxes"])

            malformed += result["malformed"]
            invalid_boxes += result["invalid"]

            if result["annotations"] == 0:
                split_empty += 1

            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4),
                    "annotations": result["annotations"],
                    "empty": result["annotations"] == 0,
                    "malformed": result["malformed"],
                    "invalid": result["invalid"],
                }
            )

        if malformed:
            problems.append(
                f"{split}: {malformed} malformed label lines"
            )
        if invalid_boxes:
            problems.append(
                f"{split}: {invalid_boxes} invalid annotations"
            )
        if unreadable_images:
            problems.append(
                f"{split}: {unreadable_images} unreadable images"
            )

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "annotations": split_annotations,
            "empty": split_empty,
            "classes": split_counts,
        }

    yaml_ok = (
        yaml_error is None
        and yaml_names == {
            0: "Knife",
            1: "Machette",
            2: "Handgun",
            3: "Longgun",
            4: "Other",
        }
    )

    if not yaml_ok:
        problems.append(
            f"dataset.yaml taxonomy mismatch: {yaml_names or yaml_error}"
        )

    report_summary = {
        "dataset": str(DATASET),
        "yaml_taxonomy_ok": yaml_ok,
        "total_images": sum(v["images"] for v in split_stats.values()),
        "total_labels": sum(v["labels"] for v in split_stats.values()),
        "total_annotations": sum(v["annotations"] for v in split_stats.values()),
        "total_empty_labels": sum(v["empty"] for v in split_stats.values()),
        "class_counts": {
            CLASS_NAMES[k]: global_counts[k]
            for k in CLASS_NAMES
        },
        "split_stats": {
            split: {
                "images": stats["images"],
                "labels": stats["labels"],
                "annotations": stats["annotations"],
                "empty": stats["empty"],
                "classes": {
                    CLASS_NAMES[k]: stats["classes"][k]
                    for k in CLASS_NAMES
                },
            }
            for split, stats in split_stats.items()
        },
        "problem_count": len(problems),
    }

    return report_summary, rows, problems, global_boxes


def write_report(rows):
    with REPORT.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "width",
                "height",
                "aspect_ratio",
                "annotations",
                "empty",
                "malformed",
                "invalid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_preflight_summary(summary, boxes):
    output = DATASET / "training_preflight_summary.json"

    total_boxes = len(boxes)
    small = sum(
        1 for box in boxes
        if box["relative_area"] < 0.01
    )
    medium = sum(
        1 for box in boxes
        if 0.01 <= box["relative_area"] < 0.10
    )
    large = sum(
        1 for box in boxes
        if box["relative_area"] >= 0.10
    )

    summary["object_size"] = {
        "small_lt_1pct": small,
        "medium_1_to_10pct": medium,
        "large_gte_10pct": large,
        "small_pct": round(
            100 * small / total_boxes, 2
        ) if total_boxes else 0,
        "medium_pct": round(
            100 * medium / total_boxes, 2
        ) if total_boxes else 0,
        "large_pct": round(
            100 * large / total_boxes, 2
        ) if total_boxes else 0,
    }

    for imgsz in [640, 768, 896, 1024, 1280]:
        very_tiny = 0
        tiny = 0

        for box in boxes:
            shortest = min(
                box["w_norm"] * imgsz,
                box["h_norm"] * imgsz,
            )
            if shortest < 8:
                very_tiny += 1
            elif shortest < 16:
                tiny += 1

        summary.setdefault("training_size_visibility", {})[
            str(imgsz)
        ] = {
            "shortest_side_lt_8px": very_tiny,
            "shortest_side_lt_16px": tiny,
        }

    (DATASET / "training_preflight_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main():
    print("=" * 70)
    print("WEAPON DETECTION")
    print("FINAL DATASET VALIDATION + TRAINING PREFLIGHT")
    print("=" * 70)

    summary, rows, problems, boxes = validate()

    write_report(rows)
    write_preflight_summary(summary, boxes)

    print("\nDataset:", DATASET)

    print("\nGLOBAL SUMMARY")
    print("Images       :", summary["total_images"])
    print("Label files  :", summary["total_labels"])
    print("Annotations  :", summary["total_annotations"])
    print("Empty labels :", summary["total_empty_labels"])

    print("\nCLASS DISTRIBUTION")
    for class_id, name in CLASS_NAMES.items():
        print(
            f"{class_id} = {name:<10}: "
            f"{summary['class_counts'][name]}"
        )

    print("\nSPLITS")
    for split in SPLITS:
        stats = summary["split_stats"][split]
        print(
            f"{split:<5} images={stats['images']:<6} "
            f"labels={stats['labels']:<6} "
            f"annotations={stats['annotations']:<6} "
            f"empty={stats['empty']}"
        )

    print("\nTRAINING PREFLIGHT")
    size = summary["object_size"]
    print(
        f"Small objects (<1%): {size['small_lt_1pct']} "
        f"({size['small_pct']}%)"
    )
    print(
        f"Medium objects      : {size['medium_1_to_10pct']} "
        f"({size['medium_pct']}%)"
    )
    print(
        f"Large objects       : {size['large_gte_10pct']} "
        f"({size['large_pct']}%)"
    )

    if problems:
        print("\nVALIDATION RESULT: FAIL")
        print("Problems:")
        for problem in problems[:30]:
            print(" -", problem)
        if len(problems) > 30:
            print(f" ... and {len(problems) - 30} more")
    else:
        print("\nVALIDATION RESULT: PASS")
        print("Final dataset is internally consistent.")

    print("\nReports:")
    print(REPORT)
    print(DATASET / "training_preflight_summary.json")
    print("\nREAD-ONLY: no dataset files were modified.")


if __name__ == "__main__":
    main()
