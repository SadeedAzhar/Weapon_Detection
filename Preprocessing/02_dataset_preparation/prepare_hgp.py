from pathlib import Path
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# HGP: 0 = Hand, 1 = Phone, 2 = Gun
GUN_CLASS = 2
OUR_HANDGUN_CLASS = 2


def prepare_hgp(hgp_root, output_root):
    hgp_root = Path(hgp_root)
    output_root = Path(output_root)

    stats = {
        "images_processed": 0,
        "images_with_gun": 0,
        "negative_images": 0,
        "gun_annotations_kept": 0,
        "non_gun_annotations_removed": 0,
        "failed_images": 0,
        "images_without_labels": 0,
        "labels_without_images": 0,
    }

    for split in ["train", "val"]:
        input_image_dir = hgp_root / "images" / f"{split}2017"
        input_label_dir = hgp_root / "labels" / f"{split}2017"

        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in input_image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_files:
            label_path = input_label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                stats["failed_images"] += 1
                stats["images_without_labels"] += 1
                continue

            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                stats["failed_images"] += 1
                continue

            new_annotations = []

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                if class_id == GUN_CLASS:
                    new_annotations.append(
                        f"{OUR_HANDGUN_CLASS} {' '.join(parts[1:])}"
                    )
                    stats["gun_annotations_kept"] += 1
                else:
                    stats["non_gun_annotations_removed"] += 1

            shutil.copy2(image_path, output_image_dir / image_path.name)

            (output_label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(new_annotations),
                encoding="utf-8"
            )

            stats["images_processed"] += 1

            if new_annotations:
                stats["images_with_gun"] += 1
            else:
                stats["negative_images"] += 1

        if input_label_dir.exists():
            image_stems = {p.stem for p in image_files}
            for label_path in input_label_dir.iterdir():
                if (
                    label_path.is_file()
                    and label_path.suffix.lower() == ".txt"
                    and label_path.stem not in image_stems
                ):
                    stats["labels_without_images"] += 1

    print("=" * 60)
    print("HGP PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Images processed               : {stats['images_processed']}")
    print(f"Images containing handguns     : {stats['images_with_gun']}")
    print(f"Negative images                : {stats['negative_images']}")
    print(f"Handgun annotations kept       : {stats['gun_annotations_kept']}")
    print(f"Hand/phone annotations removed : {stats['non_gun_annotations_removed']}")
    print(f"Failed images                  : {stats['failed_images']}")
    print(f"Images without labels          : {stats['images_without_labels']}")
    print(f"Labels without images          : {stats['labels_without_images']}")
    print(f"Output                         : {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HGP for the unified weapon dataset.")
    parser.add_argument("--input", required=True, help="Path to the HGP dataset root.")
    parser.add_argument("--output", required=True, help="Path for the processed HGP dataset.")
    args = parser.parse_args()

    prepare_hgp(args.input, args.output)
