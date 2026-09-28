from pathlib import Path
import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_MAP = {
    "knife": 0,
    "machette": 1,
    "handgun": 2,
    "longgun": 3,
    "long gun": 3,
    "long_gun": 3,
    "other": 4,
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}


def normalize_label(label):
    return (
        label.strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def rotated_box_to_axis_aligned(
    xtl,
    ytl,
    xbr,
    ybr,
    rotation=0,
):
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
        rotated.append(
            (
                cx + x * cos_t - y * sin_t,
                cy + x * sin_t + y * cos_t,
            )
        )

    xs = [x for x, _ in rotated]
    ys = [y for _, y in rotated]

    return min(xs), min(ys), max(xs), max(ys)


def convert_box(
    xtl,
    ytl,
    xbr,
    ybr,
    width,
    height,
    rotation=0,
):
    if width <= 0 or height <= 0:
        return None

    xtl, ytl, xbr, ybr = rotated_box_to_axis_aligned(
        xtl,
        ytl,
        xbr,
        ybr,
        rotation,
    )

    xtl = max(0.0, min(xtl, width))
    xbr = max(0.0, min(xbr, width))
    ytl = max(0.0, min(ytl, height))
    ybr = max(0.0, min(ybr, height))

    box_width = xbr - xtl
    box_height = ybr - ytl

    if box_width <= 0 or box_height <= 0:
        return None

    return (
        (xtl + xbr) / 2 / width,
        (ytl + ybr) / 2 / height,
        box_width / width,
        box_height / height,
    )


def discover_xml_files(source_dir):
    source_dir = Path(source_dir)

    files = sorted(
        p for p in source_dir.rglob("annotations.xml")
        if p.is_file()
    )

    if not files:
        direct = source_dir / "annotations.xml"
        if direct.exists():
            files = [direct]

    return files


def find_source_image(
    source_dir,
    xml_path,
    filename,
):
    name = Path(filename).name

    roots = [
        xml_path.parent,
        xml_path.parent.parent,
        Path(source_dir),
    ]

    for root in roots:
        direct = root / name
        if (
            direct.exists()
            and direct.is_file()
            and direct.suffix.lower() in IMAGE_EXTENSIONS
        ):
            return direct

    matches = []

    for root in roots:
        if not root.exists():
            continue

        matches.extend(
            p for p in root.rglob(name)
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    unique = list(dict.fromkeys(matches))
    return unique[0] if unique else None


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    unknown_labels = set()
    malformed_boxes = []

    for image in root.findall(".//image"):
        filename = image.get("name", "")
        width = int(float(image.get("width", 0)))
        height = int(float(image.get("height", 0)))

        objects = []

        for index, box in enumerate(image.findall("box")):
            raw_label = box.get("label", "")
            label = normalize_label(raw_label)

            if label not in CLASS_MAP:
                unknown_labels.add(raw_label)
                continue

            try:
                xtl = float(box.get("xtl", 0))
                ytl = float(box.get("ytl", 0))
                xbr = float(box.get("xbr", 0))
                ybr = float(box.get("ybr", 0))
                rotation = float(box.get("rotation", 0))
            except ValueError:
                malformed_boxes.append(
                    (filename, index, "non-numeric box")
                )
                continue

            objects.append({
                "class_id": CLASS_MAP[label],
                "class_name": CLASS_NAMES[CLASS_MAP[label]],
                "xtl": xtl,
                "ytl": ytl,
                "xbr": xbr,
                "ybr": ybr,
                "rotation": rotation,
            })

        records.append({
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        })

    return records, unknown_labels, malformed_boxes


def convert(
    source_dir,
    output_dir,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}"
        )

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    xml_files = discover_xml_files(source_dir)

    if not xml_files:
        raise FileNotFoundError(
            f"No annotations.xml files found under: {source_dir}"
        )

    image_output = output_dir / "images"
    label_output = output_dir / "labels"

    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    report = []
    class_counts = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    unknown_labels = set()
    malformed_boxes = []
    missing_images = []
    invalid_boxes = 0
    empty_labels = 0
    converted_images = 0

    used_names = set()

    for xml_path in xml_files:
        records, unknown, malformed = parse_xml(
            xml_path
        )

        unknown_labels.update(unknown)
        malformed_boxes.extend(malformed)

        for record in records:
            source_image = find_source_image(
                source_dir,
                xml_path,
                record["filename"],
            )

            if source_image is None:
                missing_images.append(
                    {
                        "xml": str(xml_path),
                        "image": record["filename"],
                    }
                )
                continue

            filename = source_image.name
            stem = source_image.stem
            suffix = source_image.suffix.lower()

            destination = image_output / filename
            counter = 2

            while destination.name.lower() in used_names:
                destination = (
                    image_output
                    / f"{stem}_{counter}{suffix}"
                )
                counter += 1

            used_names.add(
                destination.name.lower()
            )

            shutil.copy2(
                source_image,
                destination,
            )

            label_lines = []

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
                    invalid_boxes += 1
                    continue

                cx, cy, bw, bh = converted
                class_id = obj["class_id"]

                label_lines.append(
                    f"{class_id} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}"
                )

                class_counts[class_id] += 1

            (label_output / f"{destination.stem}.txt").write_text(
                "\n".join(label_lines),
                encoding="utf-8",
            )

            if not label_lines:
                empty_labels += 1

            converted_images += 1

            report.append({
                "image": destination.name,
                "source_xml": str(xml_path),
                "source_image": str(source_image),
                "annotation_count": len(label_lines),
            })

    yaml = (
        "path: .\n"
        "train: images\n"
        "val: images\n"
        "test: images\n\n"
        "names:\n"
        "  0: Knife\n"
        "  1: Machette\n"
        "  2: Handgun\n"
        "  3: Longgun\n"
        "  4: Other\n"
        "nc: 5\n"
    )

    (output_dir / "dataset.yaml").write_text(
        yaml,
        encoding="utf-8",
    )

    with (
        output_dir / "conversion_report.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "source_xml",
                "source_image",
                "annotation_count",
            ],
        )
        writer.writeheader()
        writer.writerows(report)

    with (
        output_dir / "conversion_issues.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "source",
                "detail",
            ],
        )
        writer.writeheader()

        for item in missing_images:
            writer.writerow({
                "type": "missing_image",
                "source": item["xml"],
                "detail": item["image"],
            })

        for item in malformed_boxes:
            writer.writerow({
                "type": "malformed_box",
                "source": item[0],
                "detail": (
                    f"object={item[1]} "
                    f"{item[2]}"
                ),
            })

        for label in sorted(unknown_labels):
            writer.writerow({
                "type": "unknown_label",
                "source": "",
                "detail": label,
            })

        if invalid_boxes:
            writer.writerow({
                "type": "invalid_box_count",
                "source": "",
                "detail": str(invalid_boxes),
            })

    summary = {
        "xml_files": len(xml_files),
        "images_converted": converted_images,
        "empty_labels": empty_labels,
        "missing_images": len(missing_images),
        "malformed_boxes": len(malformed_boxes),
        "invalid_boxes": invalid_boxes,
        "unknown_labels": sorted(unknown_labels),
        "class_counts": {
            CLASS_NAMES[class_id]: count
            for class_id, count in class_counts.items()
        },
    }

    (output_dir / "conversion_summary.json").write_text(
        json_dumps(summary),
        encoding="utf-8",
    )

    return summary


def json_dumps(data):
    import json
    return json.dumps(
        data,
        indent=2,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert CVAT XML annotations to the "
            "project's five-class YOLO format."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CVAT export directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YOLO dataset directory.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WEAPON DETECTION — CVAT XML → YOLO")
    print("=" * 70)

    summary = convert(
        args.input,
        args.output,
    )

    print("\nCONVERSION SUMMARY")
    print(f"XML files       : {summary['xml_files']}")
    print(
        f"Images converted: "
        f"{summary['images_converted']}"
    )
    print(
        f"Empty labels   : "
        f"{summary['empty_labels']}"
    )
    print(
        f"Missing images  : "
        f"{summary['missing_images']}"
    )
    print(
        f"Invalid boxes   : "
        f"{summary['invalid_boxes']}"
    )

    print("\nCLASS COUNTS")
    for class_name, count in summary[
        "class_counts"
    ].items():
        print(
            f"{class_name:<10}: {count}"
        )

    if summary["unknown_labels"]:
        print(
            "\nUnknown labels:"
        )
        for label in summary[
            "unknown_labels"
        ]:
            print(f" - {label}")

    print(
        f"\nOutput: {args.output}"
    )
    print("Original CVAT data was not modified.")


if __name__ == "__main__":
    main()
