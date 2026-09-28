from pathlib import Path
from collections import defaultdict
import csv
import re

import numpy as np
from PIL import Image

PROCESSED = Path(r"D:\WD Dataset\Processed")
FRAME_ROOT = PROCESSED / "Frames"
ANNOTATION_ROOT = PROCESSED / "Frames_Annotation"

CANDIDATE_FILE = PROCESSED / "frame_selection_proposal.csv"
FINAL_FILE = PROCESSED / "frame_selection_final_proposal.csv"

SPLITS = ["train", "val", "test"]
CATEGORIES = ["Handgun", "Knife", "Longgun", "Machette"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_FPS = 3
DHASH_THRESHOLD = 8
PIXEL_DIFF_THRESHOLD = 0.08
IMAGE_SIZE = (160, 120)

EXPECTED_VIDEOS = 85
EXPECTED_FRAMES = 3197
EXPECTED_CANDIDATES = 1150
EXPECTED_FINAL = 890
EXPECTED_SPLITS = {"train": 589, "val": 117, "test": 184}

FRAME_PATTERN = re.compile(
    r"^(H\d{3}|K\d{3}|L\d{3}|M\d{3})_frame_"
    r"(\d{6})\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)


def check_frame_integrity():
    if not FRAME_ROOT.exists():
        raise FileNotFoundError(f"Frames not found:\n{FRAME_ROOT}")

    issues = []
    video_count = 0
    frame_count = 0

    for category in CATEGORIES:
        category_root = FRAME_ROOT / category

        if not category_root.exists():
            issues.append(f"Missing category: {category_root}")
            continue

        for video_dir in sorted(p for p in category_root.iterdir() if p.is_dir()):
            video_count += 1
            numbers = []

            for frame in sorted(video_dir.iterdir()):
                if not frame.is_file():
                    continue

                if frame.suffix.lower() not in IMAGE_EXTENSIONS:
                    issues.append(f"{video_dir.name}: unexpected file {frame.name}")
                    continue

                match = FRAME_PATTERN.match(frame.name)
                if not match:
                    issues.append(f"{video_dir.name}: invalid filename {frame.name}")
                    continue

                filename_video_id = match.group(1).upper()
                frame_number = int(match.group(2))

                if filename_video_id != video_dir.name.upper():
                    issues.append(
                        f"{video_dir.name}: wrong video ID in {frame.name}"
                    )

                numbers.append(frame_number)

            if len(numbers) != len(set(numbers)):
                issues.append(f"{video_dir.name}: duplicate frame numbers")

            expected = list(range(len(numbers)))
            if sorted(numbers) != expected:
                issues.append(f"{video_dir.name}: non-contiguous frame sequence")

            frame_count += len(numbers)

    print("\nFRAME INTEGRITY")
    print(f"Videos : {video_count} / {EXPECTED_VIDEOS}")
    print(f"Frames : {frame_count} / {EXPECTED_FRAMES}")

    if video_count != EXPECTED_VIDEOS:
        issues.append(
            f"Expected {EXPECTED_VIDEOS} videos, found {video_count}"
        )

    if frame_count != EXPECTED_FRAMES:
        issues.append(
            f"Expected {EXPECTED_FRAMES} frames, found {frame_count}"
        )

    if issues:
        print("STATUS: REVIEW REQUIRED")
        for issue in issues[:30]:
            print(" -", issue)
        if len(issues) > 30:
            print(f" ... and {len(issues) - 30} more")
        return False

    print("STATUS: PASS")
    return True


def target_frames(total_frames):
    if total_frames <= 10:
        return total_frames
    if total_frames <= 30:
        return min(total_frames, 10)
    if total_frames <= 60:
        return min(total_frames, 15)
    if total_frames <= 100:
        return min(total_frames, 20)
    return min(total_frames, 25)


def select_evenly(total_frames, target):
    if target >= total_frames:
        return list(range(total_frames))
    if target == 1:
        return [total_frames // 2]

    positions = [
        round(i * (total_frames - 1) / (target - 1))
        for i in range(target)
    ]
    return sorted(set(positions))


def build_candidate_proposal():
    if not ANNOTATION_ROOT.exists():
        raise FileNotFoundError(
            f"Frames_Annotation not found:\n{ANNOTATION_ROOT}"
        )

    rows = []

    for split in SPLITS:
        for category in CATEGORIES:
            category_root = ANNOTATION_ROOT / split / category
            if not category_root.exists():
                raise FileNotFoundError(
                    f"Missing category directory:\n{category_root}"
                )

            for video_dir in sorted(p for p in category_root.iterdir() if p.is_dir()):
                image_dir = video_dir / "images"
                frames = sorted(
                    p for p in image_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                )

                target = target_frames(len(frames))
                indices = select_evenly(len(frames), target)

                for index in indices:
                    rows.append({
                        "split": split,
                        "category": category,
                        "video_id": video_dir.name,
                        "frame_index": index,
                        "frame_name": frames[index].name,
                        "total_video_frames": len(frames),
                        "target_frames": target,
                    })

    if len(rows) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES} candidates, found {len(rows)}"
        )

    with CANDIDATE_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "category",
                "video_id",
                "frame_index",
                "frame_name",
                "total_video_frames",
                "target_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nTEMPORAL CANDIDATE SELECTION")
    print(f"Candidates : {len(rows)}")
    print(f"Saved to   : {CANDIDATE_FILE}")
    return rows


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def calculate_dhash(image, hash_size=8):
    gray = image.convert("L")
    resized = gray.resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(resized)
    return (
        pixels[:, 1:] > pixels[:, :-1]
    ).flatten()


def hamming_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def normalized_pixel_difference(image_a, image_b):
    a = np.asarray(
        image_a.convert("L").resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0
    b = np.asarray(
        image_b.convert("L").resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0

    return float(np.mean(np.abs(a - b)))


def frames_are_similar(frame_a, frame_b):
    return (
        hamming_distance(frame_a["hash"], frame_b["hash"])
        <= DHASH_THRESHOLD
        and normalized_pixel_difference(
            frame_a["image"], frame_b["image"]
        )
        <= PIXEL_DIFF_THRESHOLD
    )


def reduce_video(video_rows, image_dir):
    candidates = []

    for row in video_rows:
        image_path = image_dir / row["frame_name"]

        if not image_path.exists():
            continue

        try:
            image = load_image(image_path)
            candidates.append({
                "row": row,
                "image": image,
                "hash": calculate_dhash(image),
            })
        except Exception:
            continue

    if not candidates:
        return []

    runs = []
    current_run = [candidates[0]]

    for current in candidates[1:]:
        if frames_are_similar(current_run[-1], current):
            current_run.append(current)
        else:
            runs.append(current_run)
            current_run = [current]

    runs.append(current_run)

    selected = []

    for run in runs:
        if len(run) == 1:
            selected.append(run[0])
        elif len(run) == 2:
            selected.extend(run)
        else:
            selected.append(run[0])
            selected.append(run[len(run) // 2])
            selected.append(run[-1])

    unique = {
        item["row"]["frame_name"]: item
        for item in selected
    }

    return sorted(
        unique.values(),
        key=lambda item: int(item["row"]["frame_index"]),
    )


def build_final_proposal(rows):
    video_groups = defaultdict(list)

    for row in rows:
        video_groups[
            (row["split"], row["category"], row["video_id"])
        ].append(row)

    final_rows = []

    for key in sorted(video_groups):
        split, category, video_id = key
        video_rows = sorted(
            video_groups[key],
            key=lambda row: int(row["frame_index"]),
        )

        if split == "test":
            for row in video_rows:
                new_row = dict(row)
                new_row["selection_stage"] = "test_protected"
                final_rows.append(new_row)
            continue

        image_dir = (
            ANNOTATION_ROOT
            / split
            / category
            / video_id
            / "images"
        )

        for item in reduce_video(video_rows, image_dir):
            new_row = dict(item["row"])
            new_row["selection_stage"] = "adaptive_redundancy"
            final_rows.append(new_row)

    final_rows.sort(
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["category"],
            row["video_id"],
            int(row["frame_index"]),
        )
    )

    split_counts = {
        split: sum(row["split"] == split for row in final_rows)
        for split in SPLITS
    }

    if len(final_rows) != EXPECTED_FINAL:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL} final frames, found {len(final_rows)}"
        )

    if split_counts != EXPECTED_SPLITS:
        raise RuntimeError(
            f"Final split mismatch: {split_counts}"
        )

    original_keys = {
        (
            row["split"],
            row["category"],
            row["video_id"],
            row["frame_name"],
        )
        for row in rows
    }
    final_keys = {
        (
            row["split"],
            row["category"],
            row["video_id"],
            row["frame_name"],
        )
        for row in final_rows
    }

    if not final_keys.issubset(original_keys):
        raise RuntimeError("Final proposal contains frames outside the candidate set.")

    test_original = {
        (
            row["category"],
            row["video_id"],
            row["frame_name"],
        )
        for row in rows
        if row["split"] == "test"
    }
    test_final = {
        (
            row["category"],
            row["video_id"],
            row["frame_name"],
        )
        for row in final_rows
        if row["split"] == "test"
    }

    if test_original != test_final:
        raise RuntimeError("Protected test candidates were changed.")

    if len(final_keys) != len(final_rows):
        raise RuntimeError("Duplicate frame selections detected.")

    with FINAL_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "category",
                "video_id",
                "frame_index",
                "frame_name",
                "total_video_frames",
                "target_frames",
                "selection_stage",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print("\nFINAL FRAME SELECTION")
    print(f"Final frames : {len(final_rows)}")
    for split in SPLITS:
        print(f"{split:<5}: {split_counts[split]}")
    print(f"Saved to     : {FINAL_FILE}")

    return final_rows


def main():
    print("=" * 70)
    print("WEAPON DETECTION — SURVEILLANCE FRAME SELECTION")
    print("=" * 70)

    if not check_frame_integrity():
        raise SystemExit("Frame integrity failed. Stopping.")

    rows = build_candidate_proposal()
    build_final_proposal(rows)

    print("\nDONE")
    print("No images were copied, moved, renamed, or modified.")


if __name__ == "__main__":
    main()
