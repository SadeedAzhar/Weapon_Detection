"""
Mock Attack XML → YOLO conversion

Converts the Mock Attack dataset into the project's five-class
YOLO taxonomy for hard-example mining.

Original source files are never modified.
"""

from pathlib import Path
from collections import Counter
import shutil
import xml.etree.ElementTree as ET


# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = Path(r"D:\WD Dataset\Sources\V2\weapons_images_2fps")
SOURCE_IMAGES = SOURCE_ROOT / "Images"

OUTPUT_ROOT = Path(r"D:\WD Dataset\Sources\V2\Mock_Attack_V1")
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"

CLASS_MAP = {
    "knife": 0,
    "handgun": 2,
    "short_rifle": 3,
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

def get_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_box(obj, width, height):
    bbox = obj.find("bndbox")

    if bbox is None:
        return None, "Missing bounding box"

    try:
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
    except (TypeError, ValueError):
        return None, "Non-numeric bounding box"

    if (
        xmin < 0 or ymin < 0
        or xmax <= xmin or ymax <= ymin
        or xmax > width or ymax > height
    ):
        return None, (
            f"Invalid bbox: {xmin}, {ymin}, {xmax}, {ymax}"
        )

    box_width = xmax - xmin
    box_height = ymax - ymin

    center_x = xmin + box_width / 2
    center_y = ymin + box_height / 2

    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    ), None


def write_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"

    lines = [
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: images",
        "val: images",
        "test: images",
        "names:",
    ]

    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")

    lines.append("nc: 5")

    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return yaml_path


# ============================================================
# CONVERSION
# ============================================================

def convert():
    if not SOURCE_IMAGES.exists():
        raise FileNotFoundError(
            f"Source image directory not found:\n{SOURCE_IMAGES}"
        )

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    images = get_images(SOURCE_IMAGES)

    source_classes = Counter()
    converted_classes = Counter()
    problems = Counter()

    unknown_classes = Counter()
    positive_images = 0
    negative_images = 0
    converted_images = 0

    for image_path in images:
        xml_path = image_path.with_suffix(".xml")

        if not xml_path.exists():
            problems["missing_xml"] += 1
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            problems["invalid_xml"] += 1
            continue

        size = root.find("size")

        try:
            width = float(size.findtext("width"))
            height = float(size.findtext("height"))

            if width <= 0 or height <= 0:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            problems["invalid_dimensions"] += 1
            continue

        yolo_lines = []
        unknown_in_image = False

        for index, obj in enumerate(root.findall("object")):
            source_name = obj.findtext("name")

            if not source_name:
                problems["missing_class"] += 1
                continue

            source_name = source_name.strip().lower()
            source_classes[source_name] += 1

            if source_name not in CLASS_MAP:
                unknown_classes[source_name] += 1
                unknown_in_image = True
                continue

            converted_box, error = parse_box(obj, width, height)

            if converted_box is None:
                problems[error] += 1
                continue

            class_id = CLASS_MAP[source_name]
            cx, cy, bw, bh = converted_box

            yolo_lines.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

            converted_classes[class_id] += 1

        destination_image = OUTPUT_IMAGES / image_path.name
        destination_label = OUTPUT_LABELS / f"{image_path.stem}.txt"

        shutil.copy2(image_path, destination_image)

        destination_label.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8",
        )

        converted_images += 1

        if yolo_lines:
            positive_images += 1
        else:
            negative_images += 1

        if unknown_in_image:
            problems["images_with_unknown_classes"] += 1

    yaml_path = write_yaml()

    unknown_report = OUTPUT_ROOT / "unknown_classes.txt"

    if unknown_classes:
        unknown_lines = [
            f"{name}: {count}"
            for name, count in sorted(unknown_classes.items())
        ]
    else:
        unknown_lines = ["No unknown classes found."]

    unknown_report.write_text(
        "\n".join(unknown_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("MOCK ATTACK → YOLO CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Source images      : {len(images):,}")
    print(f"Images converted   : {converted_images:,}")
    print(f"Positive images    : {positive_images:,}")
    print(f"Negative images    : {negative_images:,}")

    print("\nConverted objects:")
    for class_id, name in CLASS_NAMES.items():
        print(
            f"  {class_id}: {name:<10} "
            f"{converted_classes[class_id]:,}"
        )

    print("\nSource classes:")
    for name, count in sorted(source_classes.items()):
        print(f"  {name:<15} {count:,}")

    print("\nUnknown classes:")
    if unknown_classes:
        for name, count in sorted(unknown_classes.items()):
            print(f"  {name:<15} {count:,}")
    else:
        print("  None")

    print("\nProblems:")
    for name, count in sorted(problems.items()):
        print(f"  {name:<28} {count:,}")

    print(f"\nOutput: {OUTPUT_ROOT}")
    print(f"YAML  : {yaml_path}")
    print(f"Report: {unknown_report}")


if __name__ == "__main__":
    convert()
