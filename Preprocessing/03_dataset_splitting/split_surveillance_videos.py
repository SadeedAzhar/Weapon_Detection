from pathlib import Path
import argparse
import csv
import random
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
SEED = 42

EXPECTED_VIDEOS = 85
EXPECTED_FRAMES = 3197

EXPECTED_SPLIT_VIDEOS = {
    "train": 59,
    "val": 13,
    "test": 13,
}

EXPECTED_SPLIT_FRAMES = {
    "train": 2236,
    "val": 438,
    "test": 523,
}

MACHETTE_SPLIT = {
    "M001": "test",
    "M002": "train",
    "M003": "train",
    "M004": "train",
    "M005": "val",
}


def find_frames(video_dir):
    return [
        path
        for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_videos(root):
    video_data = []

    for category in CATEGORIES:
        category_path = root / category

        if not category_path.exists():
            raise FileNotFoundError(
                f"Missing category directory: {category_path}"
            )

        for video_folder in sorted(
            p for p in category_path.iterdir()
            if p.is_dir()
        ):
            frames = find_frames(video_folder)

            video_data.append({
                "category": category,
                "video_id": video_folder.name,
                "frames": len(frames),
            })

    return video_data


def validate_frame_inventory(video_data):
    total_videos = len(video_data)
    total_frames = sum(item["frames"] for item in video_data)

    if total_videos != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEOS} videos, "
            f"found {total_videos}"
        )

    if total_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, "
            f"found {total_frames}"
        )


def create_initial_proposal(video_data):
    random.seed(SEED)
    rows = []

    for category in CATEGORIES:
        category_videos = [
            item.copy()
            for item in video_data
            if item["category"] == category
        ]

        random.shuffle(category_videos)

        total = len(category_videos)
        train_count = round(total * 0.70)
        remaining = total - train_count
        val_count = round(remaining * 0.50)

        for item in category_videos[:train_count]:
            item["split"] = "train"
            rows.append(item)

        for item in category_videos[
            train_count:train_count + val_count
        ]:
            item["split"] = "val"
            rows.append(item)

        for item in category_videos[
            train_count + val_count:
        ]:
            item["split"] = "test"
            rows.append(item)

    return sorted(
        rows,
        key=lambda item: (
            CATEGORIES.index(item["category"]),
            item["video_id"],
        ),
    )


def correct_machette_split(rows):
    corrected = []

    for row in rows:
        row = dict(row)

        if row["category"] == "Machette":
            video_id = row["video_id"]

            if video_id not in MACHETTE_SPLIT:
                raise ValueError(
                    f"Unexpected Machette video: {video_id}"
                )

            row["split"] = MACHETTE_SPLIT[video_id]

        corrected.append(row)

    return corrected


def validate_proposal(rows):
    if len(rows) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_VIDEOS} split rows, "
            f"found {len(rows)}"
        )

    keys = [
        (row["category"], row["video_id"])
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate video assignments detected."
        )

    total_frames = sum(
        int(row["frames"])
        for row in rows
    )

    if total_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, "
            f"found {total_frames}"
        )

    for category in CATEGORIES:
        for split in SPLITS:
            matches = [
                row for row in rows
                if row["category"] == category
                and row["split"] == split
            ]

            if not matches:
                raise RuntimeError(
                    f"{category} has no {split} videos."
                )

    split_videos = {
        split: sum(
            row["split"] == split
            for row in rows
        )
        for split in SPLITS
    }

    split_frames = {
        split: sum(
            int(row["frames"])
            for row in rows
            if row["split"] == split
        )
        for split in SPLITS
    }

    if split_videos != EXPECTED_SPLIT_VIDEOS:
        raise RuntimeError(
            f"Unexpected video counts: {split_videos}"
        )

    if split_frames != EXPECTED_SPLIT_FRAMES:
        raise RuntimeError(
            f"Unexpected frame counts: {split_frames}"
        )

    return split_videos, split_frames


def save_proposal(rows, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "video_id",
                "split",
                "frames",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_split_dataset(source_root, proposal_file, destination):
    source_root = Path(source_root)
    proposal_file = Path(proposal_file)
    destination = Path(destination)

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source Frames directory not found: {source_root}"
        )

    if not proposal_file.exists():
        raise FileNotFoundError(
            f"Split proposal not found: {proposal_file}"
        )

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}"
        )

    with proposal_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    validate_proposal(rows)

    copied_frames = 0
    copied_videos = 0

    for row in rows:
        category = row["category"]
        video_id = row["video_id"]
        split = row["split"]

        source_video = (
            source_root
            / category
            / video_id
        )

        if not source_video.exists():
            raise FileNotFoundError(
                f"Source video folder not found: {source_video}"
            )

        destination_video = (
            destination
            / split
            / category
            / video_id
        )

        shutil.copytree(
            source_video,
            destination_video,
        )

        actual_frames = len(
            find_frames(destination_video)
        )
        expected_frames = int(row["frames"])

        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Frame count mismatch for "
                f"{category}/{video_id}: "
                f"{actual_frames} != {expected_frames}"
            )

        copied_videos += 1
        copied_frames += actual_frames

    if copied_videos != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Copied {copied_videos} videos, "
            f"expected {EXPECTED_VIDEOS}"
        )

    if copied_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Copied {copied_frames} frames, "
            f"expected {EXPECTED_FRAMES}"
        )

    source_frames_after = sum(
        len(find_frames(video))
        for category in source_root.iterdir()
        if category.is_dir()
        for video in category.iterdir()
        if video.is_dir()
    )

    if source_frames_after != EXPECTED_FRAMES:
        raise RuntimeError(
            "Original Frames dataset changed."
        )

    return {
        "videos": copied_videos,
        "frames": copied_frames,
    }


def print_summary(rows, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for category in CATEGORIES:
        print(f"\n{category}")

        for split in SPLITS:
            matches = [
                row for row in rows
                if row["category"] == category
                and row["split"] == split
            ]

            frames = sum(
                int(row["frames"])
                for row in matches
            )

            print(
                f"{split.upper():<5}: "
                f"{len(matches):>2} videos | "
                f"{frames:>4} frames"
            )

    split_videos = {
        split: sum(
            row["split"] == split
            for row in rows
        )
        for split in SPLITS
    }

    split_frames = {
        split: sum(
            int(row["frames"])
            for row in rows
            if row["split"] == split
        )
        for split in SPLITS
    }

    print("\nOVERALL")
    for split in SPLITS:
        print(
            f"{split.upper():<5}: "
            f"{split_videos[split]:>2} videos | "
            f"{split_frames[split]:>4} frames"
        )

    print(
        f"TOTAL : {len(rows):>2} videos | "
        f"{sum(split_frames.values()):>4} frames"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible video-level Train/Val/Test "
            "split and optionally copy the split dataset."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Frames/ directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for split dataset.",
    )
    parser.add_argument(
        "--proposal",
        default=None,
        help="CSV path for the corrected split proposal.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Create/validate the proposal without copying frames.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    proposal_path = (
        Path(args.proposal)
        if args.proposal
        else source.parent
        / "video_split_proposal_corrected.csv"
    )

    video_data = collect_videos(source)
    validate_frame_inventory(video_data)

    initial_rows = create_initial_proposal(video_data)
    corrected_rows = correct_machette_split(initial_rows)

    validate_proposal(corrected_rows)

    save_proposal(
        corrected_rows,
        proposal_path,
    )

    print_summary(
        corrected_rows,
        "CORRECTED VIDEO-LEVEL SPLIT",
    )

    print(
        f"\nProposal saved to: {proposal_path}"
    )

    if args.no_copy:
        print(
            "\nNo frame files were copied or moved."
        )
        return

    result = create_split_dataset(
        source,
        proposal_path,
        output,
    )

    print("\nCOPY COMPLETE")
    print(f"Videos : {result['videos']}")
    print(f"Frames : {result['frames']}")
    print(f"Output : {output}")
    print("Original Frames dataset remains untouched.")


if __name__ == "__main__":
    main()
