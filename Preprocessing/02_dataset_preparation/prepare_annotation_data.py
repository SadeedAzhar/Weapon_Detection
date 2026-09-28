from pathlib import Path
from collections import defaultdict
import argparse
import csv
import shutil

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CATEGORIES = [
    "Handgun",
    "Knife",
    "Longgun",
    "Machette",
]

SPLITS = ["train", "val", "test"]

EXPECTED_WORKSPACE = {
    "train": {"videos": 59, "frames": 2236},
    "val": {"videos": 13, "frames": 438},
    "test": {"videos": 13, "frames": 523},
}

EXPECTED_FINAL = 890
EXPECTED_FINAL_SPLITS = {
    "train": 589,
    "val": 117,
    "test": 184,
}
EXPECTED_VIDEOS = 85
EXPECTED_TOTAL_FRAMES = 3197

CLASS_MAPPING = """WEAPON DETECTION PROJECT
FINAL UNIFIED YOLO CLASS MAPPING
========================================

Class ID    Class
----------------------------------------
0           Knife
1           Machette
2           Handgun
3           Long Gun
4           Other

========================================

This mapping is frozen for the current
weapon detection dataset.

Do not change class IDs without updating
all annotations and dataset configuration.
"""


def image_files(root):
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_workspace(source, workspace):
    if not source.exists():
        raise FileNotFoundError(
            f"Frames_Split not found: {source}"
        )

    if workspace.exists():
        raise FileExistsError(
            f"Frames_Annotation already exists: {workspace}"
        )

    split_counts = {
        split: {
            "videos": 0,
            "frames": 0,
        }
        for split in SPLITS
    }

    for split in SPLITS:
        for category in CATEGORIES:
            source_category = source / split / category

            if not source_category.exists():
                raise FileNotFoundError(
                    f"Missing directory: {source_category}"
                )

            for video_folder in sorted(
                p
                for p in source_category.iterdir()
                if p.is_dir()
            ):
                output_video = (
                    workspace
                    / split
                    / category
                    / video_folder.name
                )

                output_images = output_video / "images"
                output_labels = output_video / "labels"

                output_images.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                output_labels.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                frames = image_files(video_folder)

                for frame in frames:
                    shutil.copy2(
                        frame,
                        output_images / frame.name,
                    )

                split_counts[split]["videos"] += 1
                split_counts[split]["frames"] += len(frames)

    (workspace / "CLASS_MAPPING.txt").write_text(
        CLASS_MAPPING,
        encoding="utf-8",
    )

    total_videos = sum(
        value["videos"]
        for value in split_counts.values()
    )
    total_frames = sum(
        value["frames"]
        for value in split_counts.values()
    )

    for split in SPLITS:
        expected = EXPECTED_WORKSPACE[split]
        actual = split_counts[split]

        if actual != expected:
            raise RuntimeError(
                f"{split} mismatch: {actual} != {expected}"
            )

    if total_videos != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEOS} videos, "
            f"found {total_videos}"
        )

    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(
            f"Expected 3197 frames, "
            f"found {total_frames}"
        )

    print("\nWORKSPACE")
    for split in SPLITS:
        stats = split_counts[split]
        print(
            f"{split:<5}: "
            f"{stats['videos']} videos | "
            f"{stats['frames']} frames"
        )

    print(
        f"TOTAL: {total_videos} videos | "
        f"{total_frames} frames"
    )


def read_manifest(manifest):
    if not manifest.exists():
        raise FileNotFoundError(
            f"Selection manifest not found: {manifest}"
        )

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_FINAL:
        raise ValueError(
            f"Expected {EXPECTED_FINAL} selected frames, "
            f"found {len(rows)}"
        )

    required = {
        "split",
        "category",
        "video_id",
        "frame_name",
    }

    if not rows:
        raise ValueError("Selection manifest is empty.")

    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    keys = [
        (
            row["split"],
            row["category"],
            row["video_id"],
            row["frame_name"],
        )
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate frame selections detected."
        )

    split_counts = {
        split: sum(
            row["split"] == split
            for row in rows
        )
        for split in SPLITS
    }

    if split_counts != EXPECTED_FINAL_SPLITS:
        raise ValueError(
            f"Final split mismatch: {split_counts}"
        )

    videos = {
        (
            row["split"],
            row["category"],
            row["video_id"],
        )
        for row in rows
    }

    if len(videos) != EXPECTED_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_VIDEOS} videos, "
            f"found {len(videos)}"
        )

    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(
                f"Unexpected split: {row['split']}"
            )
        if row["category"] not in CATEGORIES:
            raise ValueError(
                f"Unexpected category: {row['category']}"
            )

    return rows


def create_annotation_set(
    workspace,
    manifest,
    output,
):
    if not output:
        raise ValueError("Output directory is required.")

    if output.exists():
        raise FileExistsError(
            f"Annotation_Set already exists: {output}"
        )

    rows = read_manifest(manifest)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output / "CLASS_MAPPING.txt").write_text(
        CLASS_MAPPING,
        encoding="utf-8",
    )

    missing = []
    split_counts = defaultdict(int)
    video_keys = set()

    for row in rows:
        source_image = (
            workspace
            / row["split"]
            / row["category"]
            / row["video_id"]
            / "images"
            / row["frame_name"]
        )

        if not source_image.exists():
            missing.append(str(source_image))
            continue

        destination_video = (
            output
            / row["split"]
            / row["category"]
            / row["video_id"]
        )

        destination_images = destination_video / "images"
        destination_labels = destination_video / "labels"

        destination_images.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_image = (
            destination_images
            / row["frame_name"]
        )
        destination_label = (
            destination_labels
            / f"{Path(row['frame_name']).stem}.txt"
        )

        if (
            destination_image.exists()
            or destination_label.exists()
        ):
            raise FileExistsError(
                f"Destination collision: "
                f"{destination_image}"
            )

        shutil.copy2(
            source_image,
            destination_image,
        )

        destination_label.touch(
            exist_ok=False,
        )

        split_counts[row["split"]] += 1
        video_keys.add(
            (
                row["split"],
                row["category"],
                row["video_id"],
            )
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected images were not found."
        )

    actual_images = image_files(output)
    actual_labels = [
        p
        for p in output.rglob("*.txt")
        if p.name != "CLASS_MAPPING.txt"
    ]

    if len(actual_images) != EXPECTED_FINAL:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL} images, "
            f"found {len(actual_images)}"
        )

    if len(actual_labels) != EXPECTED_FINAL:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL} labels, "
            f"found {len(actual_labels)}"
        )

    actual_splits = {
        split: split_counts[split]
        for split in SPLITS
    }

    if actual_splits != EXPECTED_FINAL_SPLITS:
        raise RuntimeError(
            f"Final split mismatch: {actual_splits}"
        )

    if len(video_keys) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEOS} videos, "
            f"found {len(video_keys)}"
        )

    with (
        output / "annotation_selection_manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nANNOTATION SET")
    for split in SPLITS:
        print(
            f"{split:<5}: "
            f"{actual_splits[split]}"
        )
    print(
        f"TOTAL: {len(actual_images)} images | "
        f"{len(actual_labels)} labels"
    )


def prepare_remaining(
    source_train,
    annotated_images,
    output,
):
    if not source_train.exists():
        raise FileNotFoundError(
            f"Source training folder not found: {source_train}"
        )

    if not annotated_images.exists():
        raise FileNotFoundError(
            f"Annotated images folder not found: "
            f"{annotated_images}"
        )

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    annotated = {
        path.name.lower()
        for path in annotated_images.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    source_by_class = defaultdict(list)

    for category in CATEGORIES:
        category_dir = source_train / category

        if not category_dir.exists():
            continue

        source_by_class[category].extend(
            image_files(category_dir)
        )

    seen = {}

    for category, files in source_by_class.items():
        for path in files:
            key = path.name.lower()

            if key in seen:
                raise RuntimeError(
                    "Duplicate filename detected:\n"
                    f"{seen[key]}\n{path}"
                )

            seen[key] = path

    remaining = defaultdict(list)
    completed = defaultdict(list)

    for category, files in source_by_class.items():
        for path in files:
            if path.name.lower() in annotated:
                completed[category].append(path)
            else:
                remaining[category].append(path)

    total = sum(
        len(files)
        for files in source_by_class.values()
    )
    done = sum(
        len(files)
        for files in completed.values()
    )
    left = sum(
        len(files)
        for files in remaining.values()
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for category in CATEGORIES:
        class_output = output / category
        class_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for source_file in remaining[category]:
            destination_file = (
                class_output
                / source_file.name
            )

            if destination_file.exists():
                raise FileExistsError(
                    f"Destination collision: "
                    f"{destination_file}"
                )

            shutil.copy2(
                source_file,
                destination_file,
            )
            copied += 1

    report = output / "remaining_annotation_report.csv"

    with report.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class",
            "Source Total",
            "Already Annotated",
            "Remaining",
        ])

        for category in CATEGORIES:
            writer.writerow([
                category,
                len(source_by_class[category]),
                len(completed[category]),
                len(remaining[category]),
            ])

        writer.writerow([
            "TOTAL",
            total,
            done,
            left,
        ])

    actual = len(
        image_files(output)
    )

    if actual != left or copied != left:
        raise RuntimeError(
            f"Remaining copy mismatch: "
            f"expected {left}, "
            f"copied {copied}, "
            f"found {actual}"
        )

    print("\nREMAINING ANNOTATION SET")
    print(f"Source total      : {total}")
    print(f"Already annotated : {done}")
    print(f"Remaining         : {left}")
    print(f"Copied            : {copied}")
    print(f"Report            : {report}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the surveillance annotation workspace, "
            "build the locked 890-frame annotation set, "
            "or prepare remaining unannotated training images."
        )
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    workspace_parser = subparsers.add_parser(
        "workspace",
        help="Copy the complete Frames_Split workspace.",
    )
    workspace_parser.add_argument(
        "--source",
        required=True,
    )
    workspace_parser.add_argument(
        "--output",
        required=True,
    )

    selected_parser = subparsers.add_parser(
        "selected",
        help="Create the locked 890-frame annotation set.",
    )
    selected_parser.add_argument(
        "--workspace",
        required=True,
    )
    selected_parser.add_argument(
        "--manifest",
        required=True,
    )
    selected_parser.add_argument(
        "--output",
        required=True,
    )

    remaining_parser = subparsers.add_parser(
        "remaining",
        help="Prepare images not yet annotated in training.",
    )
    remaining_parser.add_argument(
        "--source-train",
        required=True,
    )
    remaining_parser.add_argument(
        "--annotated-images",
        required=True,
    )
    remaining_parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WEAPON DETECTION — ANNOTATION DATA PREPARATION")
    print("=" * 70)

    if args.mode == "workspace":
        create_workspace(
            Path(args.source),
            Path(args.output),
        )
        print("\nWorkspace creation complete.")

    elif args.mode == "selected":
        create_annotation_set(
            Path(args.workspace),
            Path(args.manifest),
            Path(args.output),
        )
        print("\nSelected annotation set creation complete.")

    elif args.mode == "remaining":
        prepare_remaining(
            Path(args.source_train),
            Path(args.annotated_images),
            Path(args.output),
        )
        print("\nRemaining annotation preparation complete.")


if __name__ == "__main__":
    main()
