from pathlib import Path
import shutil

# Dangerous Items -> unified Weapon Detection classes
# 0 = Knife, 1 = Machette, 2 = Handgun, 3 = Longgun, 4 = Other

SOURCE = Path(r"D:\WD Dataset\Processed\Dangerous_Items")
OUTPUT = Path(r"D:\WD Dataset\Processed\Dangerous_Items_Normalized")

CLASS_MAPPING = {
    0: 1,   # Machete -> Machette
    1: 0,   # Knife -> Knife
    2: 4,   # Baseball bat / rod -> Other
    3: 3,   # Rifle / possible drill noise -> Longgun
    4: 2    # Pistol -> Handgun
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_LABELS = 10140
EXPECTED_IMAGES = 10140
EXPECTED_ANNOTATIONS = 9144
EXPECTED_SOURCE_COUNTS = {
    0: 1714,
    1: 2134,
    2: 1755,
    3: 3506,
    4: 35
}


def find_label_dirs():
    return sorted(p for p in SOURCE.rglob("labels") if p.is_dir())


def build_image_index(label_dirs):
    indexes = {}

    for labels_dir in label_dirs:
        images_dir = labels_dir.parent / "images"
        index = {}

        if images_dir.exists():
            for image_path in images_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    key = image_path.stem.lower()
                    if key in index:
                        raise RuntimeError(
                            f"Duplicate image stem in {images_dir}:\n"
                            f"{index[key]}\n{image_path}"
                        )
                    index[key] = image_path

        indexes[labels_dir] = index

    return indexes


def read_and_remap(label_path):
    output = []
    counts = {}

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Malformed YOLO label: {label_path} line {line_no}"
                )

            try:
                source_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric YOLO value: {label_path} line {line_no}"
                ) from e

            if source_id not in CLASS_MAPPING:
                raise ValueError(
                    f"Unexpected source class {source_id}: "
                    f"{label_path} line {line_no}"
                )

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(
                    f"Invalid YOLO coordinates: {label_path} line {line_no}"
                )

            final_id = CLASS_MAPPING[source_id]
            output.append(f"{final_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            counts[source_id] = counts.get(source_id, 0) + 1

    return output, counts


def main():
    print("=" * 70)
    print("DANGEROUS ITEMS NORMALIZATION")
    print("=" * 70)
    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT}")

    if not SOURCE.exists():
        print("\nERROR: Source folder does not exist.")
        return

    if OUTPUT.exists():
        print("\nERROR: Output folder already exists.")
        print("Delete it before running this script.")
        return

    label_dirs = find_label_dirs()
    if not label_dirs:
        print("\nERROR: No labels directories found.")
        return

    image_indexes = build_image_index(label_dirs)
    records = []

    for labels_dir in label_dirs:
        for label_path in sorted(labels_dir.rglob("*.txt")):
            image_path = image_indexes[labels_dir].get(label_path.stem.lower())
            records.append((labels_dir, label_path, image_path))

    print(f"\nLabel files found: {len(records)}")
    if len(records) != EXPECTED_LABELS:
        print(f"ERROR: Expected {EXPECTED_LABELS} label files.")
        return

    total_images = 0
    total_annotations = 0
    source_counts = {i: 0 for i in CLASS_MAPPING}
    prepared = []
    missing_images = []

    print("\nValidating labels...")

    for i, (labels_dir, label_path, image_path) in enumerate(records, start=1):
        output_lines, counts = read_and_remap(label_path)

        if image_path is None:
            missing_images.append(str(label_path))
        else:
            total_images += 1

        total_annotations += len(output_lines)
        for class_id, count in counts.items():
            source_counts[class_id] += count

        prepared.append((labels_dir, label_path, image_path, output_lines))

        if i % 1000 == 0:
            print(f"  {i}/{len(records)}")

    counts_ok = source_counts == EXPECTED_SOURCE_COUNTS
    validation_ok = (
        total_images == EXPECTED_IMAGES
        and total_annotations == EXPECTED_ANNOTATIONS
        and not missing_images
        and counts_ok
    )

    print("\nValidation results:")
    print(f"  Labels       : {len(records)} / {EXPECTED_LABELS}")
    print(f"  Images       : {total_images} / {EXPECTED_IMAGES}")
    print(f"  Annotations  : {total_annotations} / {EXPECTED_ANNOTATIONS}")
    print(f"  Missing imgs : {len(missing_images)}")

    print("\nSource class counts:")
    for source_id in sorted(CLASS_MAPPING):
        final_id = CLASS_MAPPING[source_id]
        print(
            f"  {source_id} -> {final_id} ({CLASS_NAMES[final_id]}): "
            f"{source_counts[source_id]} / {EXPECTED_SOURCE_COUNTS[source_id]}"
        )

    if not validation_ok:
        print("\nVALIDATION FAILED. No output created.")
        if missing_images:
            print("First missing images:")
            for path in missing_images[:10]:
                print(f"  {path}")
        return

    print("\nValidation passed. Creating normalized dataset...")

    for labels_dir, source_label, source_image, output_lines in prepared:
        relative_label = source_label.relative_to(SOURCE)
        destination_label = OUTPUT / relative_label
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text("\n".join(output_lines), encoding="utf-8")

        relative_image = source_image.relative_to(SOURCE)
        destination_image = OUTPUT / relative_image
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)

    print("\nNormalization complete.")
    print(f"Output: {OUTPUT}")
    print("Original source was not modified.")


if __name__ == "__main__":
    main()
