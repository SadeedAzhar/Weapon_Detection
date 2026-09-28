"""
CVAT XML → YOLO converter

Converts CVAT image annotations into the five-class YOLO
taxonomy used by the weapon-detection project.

The source dataset is never modified.
"""

from pathlib import Path
import csv
import math
import shutil
import xml.etree.ElementTree as ET


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("path/to/cvat_export")
OUTPUT_DIR = Path("path/to/cvat_yolo")

ANNOTATIONS_FILE = SOURCE_DIR / "annotations.xml"
IMAGE_DIR = SOURCE_DIR / "images"

CLASS_MAP = {
    "knife": 0,
    "machette": 1,
    "handgun": 2,
    "longgun": 3,
    "long gun": 3,
    "other": 4,
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_label(label):
    return (
        label.strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def rotated_box_to_axis_aligned(xtl, ytl, xbr, ybr, rotation=0):
    rotation = float(rotation or 0)

    if abs(rotation) < 1e-9:
        return xtl, ytl, xbr, ybr

    cx = (xtl + xbr) / 2
    cy = (ytl + ybr) / 2
    half_w = (xbr - xtl) / 2
    half_h = (ybr - ytl) / 2
    theta = math.radians(rotation)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    rotated = []

    for x, y in corners:
        rx = cx + x * cos_t - y * sin_t
        ry = cy + x * sin_t + y * cos_t
        rotated.append((rx, ry))

    xs = [x for x, _ in rotated]
    ys = [y for _, y in rotated]

    return min(xs), min(ys), max(xs), max(ys)


def convert_box(xtl, ytl, xbr, ybr, width, height, rotation=0):
    xtl, ytl, xbr, ybr = rotated_box_to_axis_aligned(
        xtl, ytl, xbr, ybr, rotation
    )

    xtl = max(0.0, min(xtl, width))
    xbr = max(0.0, min(xbr, width))
    ytl = max(0.0, min(ytl, height))
    ybr = max(0.0, min(ybr, height))

    box_width = xbr - xtl
    box_height = ybr - ytl

    if width <= 0 or height <= 0:
        return None

    if box_width <= 0 or box_height <= 0:
        return None

    center_x = (xtl + xbr) / 2
    center_y = (ytl + ybr) / 2

    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def find_source_image(filename):
    name = Path(filename).name
    direct = IMAGE_DIR / name

    if direct.exists():
        return direct

    matches = [
        path
        for path in IMAGE_DIR.rglob(name)
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return matches[0] if matches else None


def parse_annotations():
    tree = ET.parse(ANNOTATIONS_FILE)
    root = tree.getroot()

    records = []
    unknown_labels = set()

    for image in root.findall(".//image"):
        filename = image.get("name", "")
        width = int(float(image.get("width", 0)))
        height = int(float(image.get("height", 0)))

        objects = []

        for box in image.findall("box"):
            raw_label = box.get("label", "")
            label = normalize_label(raw_label)

            if label not in CLASS_MAP:
                unknown_labels.add(raw_label)
                continue

            objects.append({
                "class_id": CLASS_MAP[label],
                "class_name": CLASS_NAMES[CLASS_MAP[label]],
                "xtl": float(box.get("xtl", 0)),
                "ytl": float(box.get("ytl", 0)),
                "xbr": float(box.get("xbr", 0)),
                "ybr": float(box.get("ybr", 0)),
                "rotation": float(box.get("rotation", 0)),
            })

        records.append({
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        })

    if unknown_labels:
        labels = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Unknown CVAT labels: {labels}")

    return records


# ============================================================
# CONVERSION
# ============================================================

def convert():
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing CVAT annotations:\n{ANNOTATIONS_FILE}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing image directory:\n{IMAGE_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT_DIR}\n"
            "Remove or rename it before running the conversion."
        )

    records = parse_annotations()

    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"

    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)

    report_rows = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    missing_images = 0
    empty_labels = 0

    for record in records:
        source_image = find_source_image(record["filename"])

        if source_image is None:
            missing_images += 1
            continue

        destination_image = output_images / source_image.name

        if destination_image.exists():
            stem = destination_image.stem
            suffix = destination_image.suffix
            index = 2

            while destination_image.exists():
                destination_image = (
                    output_images / f"{stem}_{index}{suffix}"
                )
                index += 1

        shutil.copy2(source_image, destination_image)

        label_path = output_labels / f"{destination_image.stem}.txt"
        yolo_lines = []

        for obj in record["objects"]:
            converted = convert_box(
                obj["xtl"],
                obj["ytl"],
                obj["xbr"],
                obj["ybr"],
                record["width"],
                record["height"],
                obj["rotation"],
            )

            if converted is None:
                continue

            cx, cy, bw, bh = converted
            class_id = obj["class_id"]

            yolo_lines.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

            class_counts[obj["class_name"]] += 1

        label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8",
        )

        if not yolo_lines:
            empty_labels += 1

        report_rows.append({
            "image": destination_image.name,
            "source_image": str(source_image),
            "annotation_count": len(yolo_lines),
        })

    report_path = OUTPUT_DIR / "conversion_report.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "source_image",
                "annotation_count",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("=" * 60)
    print("CVAT → YOLO CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Images in XML : {len(records):,}")
    print(f"Images copied : {len(report_rows):,}")
    print(f"Missing images: {missing_images:,}")
    print(f"Empty labels  : {empty_labels:,}")

    print("\nClass counts:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id}: {name:<10} {class_counts[name]:,}")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert()
