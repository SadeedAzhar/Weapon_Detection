from pathlib import Path
from collections import defaultdict
import csv
import shutil

PROCESSED = Path(r"D:\WD Dataset\Processed")
SOURCE = PROCESSED / "Frames_Split"
WORKSPACE = PROCESSED / "Frames_Annotation"
FINAL_SET = PROCESSED / "Annotation_Set"
MANIFEST = PROCESSED / "frame_selection_final_proposal.csv"

SPLITS = ["train", "val", "test"]
CATEGORIES = ["Handgun", "Knife", "Longgun", "Machette"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_WORKSPACE = {
    "train": {"videos": 59, "frames": 2236},
    "val": {"videos": 13, "frames": 438},
    "test": {"videos": 13, "frames": 523},
}

EXPECTED_FINAL = 890
EXPECTED_FINAL_SPLITS = {"train": 589, "val": 117, "test": 184}
EXPECTED_VIDEOS = 85

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


def copy_full_workspace():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Frames_Split not found:\n{SOURCE}")

    if WORKSPACE.exists():
        raise FileExistsError(
            f"Frames_Annotation already exists:\n{WORKSPACE}\n"
            "Delete or rename it before recreating."
        )

    total_videos = 0
    total_frames = 0
    split_counts = {
        split: {"videos": 0, "frames": 0}
        for split in SPLITS
    }

    for split in SPLITS:
        for category in CATEGORIES:
            source_category = SOURCE / split / category

            if not source_category.exists():
                raise FileNotFoundError(
                    f"Missing directory:\n{source_category}"
                )

            for video_folder in sorted(
                p for p in source_category.iterdir()
                if p.is_dir()
            ):
                destination_video = (
                    WORKSPACE / split / category / video_folder.name
                )
                destination_images = destination_video / "images"
                destination_labels = destination_video / "labels"

                destination_images.mkdir(
                    parents=True, exist_ok=True
                )
                destination_labels.mkdir(
                    parents=True, exist_ok=True
                )

                frames = sorted(
                    p for p in video_folder.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                )

                for frame in frames:
                    shutil.copy2(
                        frame,
                        destination_images / frame.name,
                    )

                total_videos += 1
                total_frames += len(frames)
                split_counts[split]["videos"] += 1
                split_counts[split]["frames"] += len(frames)

    (WORKSPACE / "CLASS_MAPPING.txt").write_text(
        CLASS_MAPPING,
        encoding="utf-8",
    )

    print("\nANNOTATION WORKSPACE")
    for split in SPLITS:
        actual = split_counts[split]
        expected = EXPECTED_WORKSPACE[split]
        print(
            f"{split:<5} videos={actual['videos']}/{expected['videos']} "
            f"frames={actual['frames']}/{expected['frames']}"
        )

        if actual != expected:
            raise RuntimeError(
                f"Workspace mismatch for {split}: "
                f"{actual} != {expected}"
            )

    if total_videos != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEOS} videos, found {total_videos}"
        )

    if total_frames != 3197:
        raise RuntimeError(
            f"Expected 3197 frames, found {total_frames}"
        )

    print(f"TOTAL : {total_videos} videos / {total_frames} frames")
    print("Workspace creation: PASS")


def read_final_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Final selection manifest not found:\n{MANIFEST}"
        )

    with MANIFEST.open(
        "r", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_FINAL:
        raise ValueError(
            f"Expected {EXPECTED_FINAL} frames, "
            f"found {len(rows)}"
        )

    required = {
        "split",
        "category",
        "video_id",
        "frame_name",
    }

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Manifest missing columns: "
            f"{required - set(rows[0].keys())}"
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
        raise ValueError("Duplicate frames found in final manifest.")

    split_counts = defaultdict(int)
    video_keys = set()

    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(
                f"Unexpected split: {row['split']}"
            )
        if row["category"] not in CATEGORIES:
            raise ValueError(
                f"Unexpected category: {row['category']}"
            )

        split_counts[row["split"]] += 1
        video_keys.add(
            (
                row["split"],
                row["category"],
                row["video_id"],
            )
        )

    actual_splits = {
        split: split_counts[split]
        for split in SPLITS
    }

    if actual_splits != EXPECTED_FINAL_SPLITS:
        raise ValueError(
            f"Final split mismatch: "
            f"{actual_splits}"
        )

    if len(video_keys) != EXPECTED_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_VIDEOS} videos in manifest, "
            f"found {len(video_keys)}"
        )

    return rows


def create_final_annotation_set(rows):
    if FINAL_SET.exists():
        raise FileExistsError(
            f"Annotation_Set already exists:\n{FINAL_SET}\n"
            "Delete or rename it before recreating."
        )

    FINAL_SET.mkdir(
        parents=True,
        exist_ok=True,
    )

    (FINAL_SET / "CLASS_MAPPING.txt").write_text(
        CLASS_MAPPING,
        encoding="utf-8",
    )

    copied = 0
    missing = []
    split_counts = defaultdict(int)
    category_counts = defaultdict(int)
    video_counts = set()

    for row in rows:
        source_image = (
            WORKSPACE
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
            FINAL_SET
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
            destination_images / row["frame_name"]
        )
        destination_label = (
            destination_labels
            / f"{Path(row['frame_name']).stem}.txt"
        )

        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(
                f"Destination collision:\n{destination_image}"
            )

        shutil.copy2(
            source_image,
            destination_image,
        )
        destination_label.touch(
            exist_ok=False,
        )

        copied += 1
        split_counts[row["split"]] += 1
        category_counts[
            (row["split"], row["category"])
        ] += 1
        video_counts.add(
            (
                row["split"],
                row["category"],
                row["video_id"],
            )
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest images were not found."
        )

    if copied != EXPECTED_FINAL:
        raise RuntimeError(
            f"Copied {copied} frames, expected {EXPECTED_FINAL}"
        )

    actual_images = [
        p for p in FINAL_SET.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    actual_labels = [
        p for p in FINAL_SET.rglob("*.txt")
        if p.is_file()
        and p.name != "CLASS_MAPPING.txt"
    ]

    if len(actual_images) != EXPECTED_FINAL:
        raise RuntimeError(
            f"Final image count mismatch: "
            f"{len(actual_images)}"
        )

    if len(actual_labels) != EXPECTED_FINAL:
        raise RuntimeError(
            f"Final label count mismatch: "
            f"{len(actual_labels)}"
        )

    for split in SPLITS:
        expected = EXPECTED_FINAL_SPLITS[split]
        actual = split_counts[split]

        if actual != expected:
            raise RuntimeError(
                f"{split} mismatch: {actual} != {expected}"
            )

    if len(video_counts) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Final video coverage mismatch: "
            f"{len(video_counts)}"
        )

    with (
        FINAL_SET / "annotation_selection_manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nFINAL ANNOTATION SET")
    print(f"Images : {len(actual_images)}")
    print(f"Labels : {len(actual_labels)}")
    for split in SPLITS:
        print(
            f"{split:<5}: "
            f"{split_counts[split]} / "
            f"{EXPECTED_FINAL_SPLITS[split]}"
        )
    print(f"Videos : {len(video_counts)} / {EXPECTED_VIDEOS}")
    print("Final annotation set creation: PASS")


def main():
    print("=" * 70)
    print("WEAPON DETECTION — ANNOTATION WORKSPACE PREPARATION")
    print("=" * 70)

    copy_full_workspace()

    rows = read_final_manifest()
    create_final_annotation_set(rows)

    print("\nCOMPLETE")
    print(f"Full workspace : {WORKSPACE}")
    print(f"890-frame set  : {FINAL_SET}")
    print("\nOriginal Frames_Split was not modified.")


if __name__ == "__main__":
    main()
