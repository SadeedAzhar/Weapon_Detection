from pathlib import Path
import csv
import json
import shutil
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAPPING = {
    "knife": 0,
    "pistol": 2,
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}

NON_WEAPON_CLASSES = {
    "smartphone",
    "monedero",
    "billete",
    "tarjeta",
}

EXPECTED_SOURCE_CLASSES = set(CLASS_MAPPING) | NON_WEAPON_CLASSES
SPLITS = ["train", "test"]


def convert_bbox(x1, y1, x2, y2, width, height):
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / width, yc / height, bw / width, bh / height


def convert_sohas(dataset_root, output_root):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "Delete or rename it before regenerating."
        )

    stats = {
        "images": Counter(),
        "source_objects": Counter(),
        "converted_objects": Counter(),
        "nonweapon_objects": Counter(),
        "weapon_images": Counter(),
        "negative_images": Counter(),
        "missing_annotations": [],
        "invalid_json": [],
        "invalid_geometry": [],
        "invalid_bbox": [],
        "unknown_classes": [],
    }
    manifest = []

    for split in SPLITS:
        image_dir = dataset_root / split / "img"
        annotation_dir = dataset_root / split / "ann"
        out_image_dir = output_root / split / "images"
        out_label_dir = output_root / split / "labels"

        if not image_dir.exists() or not annotation_dir.exists():
            continue

        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_files:
            stats["images"][split] += 1
            annotation_path = annotation_dir / f"{image_path.name}.json"

            if not annotation_path.exists():
                stats["missing_annotations"].append(
                    (split, image_path.name, annotation_path.name)
                )
                continue

            try:
                with annotation_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                width = float(data["size"]["width"])
                height = float(data["size"]["height"])
                if width <= 0 or height <= 0:
                    raise ValueError("Non-positive image dimensions")
            except Exception as e:
                stats["invalid_json"].append(
                    (split, image_path.name, str(e))
                )
                continue

            source_classes = set()
            converted_sources = set()
            nonweapon_sources = set()
            labels = []

            for obj in data.get("objects", []):
                source_class = str(obj.get("classTitle", "")).strip().lower()
                if not source_class:
                    stats["invalid_geometry"].append(
                        (split, image_path.name, "Missing classTitle")
                    )
                    continue

                source_classes.add(source_class)
                stats["source_objects"][source_class] += 1

                if source_class not in EXPECTED_SOURCE_CLASSES:
                    stats["unknown_classes"].append(
                        (split, image_path.name, source_class)
                    )
                    continue

                if source_class in NON_WEAPON_CLASSES:
                    stats["nonweapon_objects"][source_class] += 1
                    nonweapon_sources.add(source_class)
                    continue

                if obj.get("geometryType") != "rectangle":
                    stats["invalid_geometry"].append(
                        (
                            split,
                            image_path.name,
                            f"Expected rectangle, got {obj.get('geometryType')}",
                        )
                    )
                    continue

                exterior = obj.get("points", {}).get("exterior", [])
                if (
                    len(exterior) != 2
                    or any(len(point) != 2 for point in exterior)
                ):
                    stats["invalid_bbox"].append(
                        (split, image_path.name, source_class, "Invalid exterior points")
                    )
                    continue

                try:
                    x1, y1 = map(float, exterior[0])
                    x2, y2 = map(float, exterior[1])
                except Exception:
                    stats["invalid_bbox"].append(
                        (split, image_path.name, source_class, "Non-numeric coordinates")
                    )
                    continue

                if x2 <= x1 or y2 <= y1:
                    stats["invalid_bbox"].append(
                        (split, image_path.name, source_class, "Zero/negative area")
                    )
                    continue

                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    stats["invalid_bbox"].append(
                        (split, image_path.name, source_class, "Outside image")
                    )
                    continue

                xc, yc, bw, bh = convert_bbox(
                    x1, y1, x2, y2, width, height
                )
                class_id = CLASS_MAPPING[source_class]

                labels.append((class_id, xc, yc, bw, bh))
                stats["converted_objects"][class_id] += 1
                converted_sources.add(source_class)

            shutil.copy2(image_path, out_image_dir / image_path.name)

            label_path = out_label_dir / f"{image_path.stem}.txt"
            with label_path.open("w", encoding="utf-8") as f:
                for class_id, xc, yc, bw, bh in labels:
                    f.write(
                        f"{class_id} {xc:.6f} {yc:.6f} "
                        f"{bw:.6f} {bh:.6f}\n"
                    )

            has_weapon = bool(labels)
            if has_weapon:
                stats["weapon_images"][split] += 1
            else:
                stats["negative_images"][split] += 1

            manifest.append({
                "split": split,
                "image": image_path.name,
                "source_annotation": annotation_path.name,
                "source_classes": ",".join(sorted(source_classes)),
                "converted_classes": ",".join(sorted(converted_sources)),
                "nonweapon_classes": ",".join(sorted(nonweapon_sources)),
                "converted_object_count": len(labels),
                "has_weapon": has_weapon,
            })

    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "data.yaml").write_text(
        """path: .
train: train/images
test: test/images

names:
  0: Knife
  1: Machette
  2: Handgun
  3: Longgun
  4: Other

nc: 5
""",
        encoding="utf-8",
    )

    with (output_root / "conversion_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "split",
            "image",
            "source_annotation",
            "source_classes",
            "converted_classes",
            "nonweapon_classes",
            "converted_object_count",
            "has_weapon",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "source_taxonomy": sorted(EXPECTED_SOURCE_CLASSES),
        "v1_taxonomy": CLASS_NAMES,
        "class_mapping": {
            "knife": {"v1_id": 0, "v1_name": "Knife"},
            "pistol": {"v1_id": 2, "v1_name": "Handgun"},
        },
        "nonweapon_classes_not_converted": sorted(NON_WEAPON_CLASSES),
        "image_counts": dict(stats["images"]),
        "source_object_counts": dict(stats["source_objects"]),
        "converted_object_counts": {
            CLASS_NAMES[k]: v
            for k, v in stats["converted_objects"].items()
        },
        "nonweapon_object_counts": dict(stats["nonweapon_objects"]),
        "weapon_images": dict(stats["weapon_images"]),
        "negative_images": dict(stats["negative_images"]),
        "missing_annotations": len(stats["missing_annotations"]),
        "invalid_json": len(stats["invalid_json"]),
        "invalid_geometry": len(stats["invalid_geometry"]),
        "invalid_bbox": len(stats["invalid_bbox"]),
        "unknown_classes": len(stats["unknown_classes"]),
        "manifest_rows": len(manifest),
    }

    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("SOHAS → V1 CONVERSION COMPLETE")
    print("=" * 60)

    for split in SPLITS:
        print(f"{split:>5} images: {stats['images'][split]}")
        print(f"{split:>5} weapon images: {stats['weapon_images'][split]}")
        print(f"{split:>5} negative images: {stats['negative_images'][split]}")

    print("\nV1 objects created:")
    for class_id in range(5):
        print(
            f"{class_id}: {CLASS_NAMES[class_id]:<10} "
            f"{stats['converted_objects'].get(class_id, 0)}"
        )

    print("\nValidation:")
    print(f"Missing annotations: {len(stats['missing_annotations'])}")
    print(f"Invalid JSON:        {len(stats['invalid_json'])}")
    print(f"Invalid geometry:    {len(stats['invalid_geometry'])}")
    print(f"Invalid boxes:       {len(stats['invalid_bbox'])}")
    print(f"Unknown classes:     {len(stats['unknown_classes'])}")
    print(f"\nOutput: {output_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Sohas DetectionNinja annotations to the V1 YOLO taxonomy."
    )
    parser.add_argument("--input", required=True, help="Sohas dataset root")
    parser.add_argument("--output", required=True, help="Output conversion directory")
    args = parser.parse_args()

    convert_sohas(args.input, args.output)
