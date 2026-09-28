from pathlib import Path
from collections import defaultdict
import argparse
import cv2
import json

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TARGET_FPS = 3


def extract_frames(input_dir, output_dir, target_fps=TARGET_FPS):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than 0")

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    failed_videos = 0
    total_frames = 0

    category_dirs = sorted(
        p for p in input_dir.iterdir()
        if p.is_dir()
    )

    for category_dir in category_dirs:
        output_category = output_dir / category_dir.name
        output_category.mkdir(
            parents=True,
            exist_ok=True,
        )

        videos = sorted(
            p for p in category_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        for video_index, video_path in enumerate(videos, 1):
            total_videos += 1

            print(
                f"\n[{category_dir.name}] "
                f"{video_index}/{len(videos)}: "
                f"{video_path.stem}"
            )

            output_video = (
                output_category / video_path.stem
            )
            output_video.mkdir(
                parents=True,
                exist_ok=True,
            )

            existing_frames = [
                p for p in output_video.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".jpg"
            ]

            if existing_frames:
                print(
                    f"  Skipped: "
                    f"{len(existing_frames)} existing frames"
                )
                skipped_videos += 1
                continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("  ERROR: Could not open video")
                failed_videos += 1
                continue

            original_fps = cap.get(
                cv2.CAP_PROP_FPS
            )
            source_frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if original_fps <= 0:
                print("  ERROR: Invalid source FPS")
                cap.release()
                failed_videos += 1
                continue

            frame_interval = max(
                1,
                round(original_fps / target_fps),
            )

            print(
                f"  FPS={original_fps:.2f} "
                f"frames={source_frame_count} "
                f"interval={frame_interval}"
            )

            frame_number = 0
            saved_frames = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    output_path = (
                        output_video
                        / f"{video_path.stem}_frame_"
                        f"{saved_frames:06d}.jpg"
                    )

                    success = cv2.imwrite(
                        str(output_path),
                        frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            95,
                        ],
                    )

                    if success:
                        saved_frames += 1

                frame_number += 1

            cap.release()

            if saved_frames == 0:
                print("  ERROR: No frames extracted")
                failed_videos += 1
                continue

            print(f"  Extracted: {saved_frames}")
            processed_videos += 1
            total_frames += saved_frames

    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "target_fps": target_fps,
        "total_videos": total_videos,
        "processed_videos": processed_videos,
        "skipped_videos": skipped_videos,
        "failed_videos": failed_videos,
        "total_frames_extracted": total_frames,
    }

    return summary


def inspect_frames(frames_dir):
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}"
        )

    category_stats = {}
    dimensions = defaultdict(int)
    corrupted_images = []
    total_images = 0
    readable_images = 0
    total_width = 0
    total_height = 0

    for category_dir in sorted(
        p for p in frames_dir.iterdir()
        if p.is_dir()
    ):
        video_stats = {}
        category_total = 0

        for video_dir in sorted(
            p for p in category_dir.iterdir()
            if p.is_dir()
        ):
            valid_count = 0

            for image_path in sorted(
                p for p in video_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ):
                total_images += 1

                image = cv2.imread(str(image_path))

                if image is None:
                    corrupted_images.append(
                        str(image_path)
                    )
                    continue

                height, width = image.shape[:2]

                dimensions[
                    (width, height)
                ] += 1

                total_width += width
                total_height += height

                readable_images += 1
                valid_count += 1
                category_total += 1

            video_stats[video_dir.name] = valid_count

        category_stats[category_dir.name] = {
            "videos": len(video_stats),
            "frames": category_total,
            "min_frames_per_video": (
                min(video_stats.values())
                if video_stats
                else 0
            ),
            "max_frames_per_video": (
                max(video_stats.values())
                if video_stats
                else 0
            ),
            "average_frames_per_video": (
                sum(video_stats.values())
                / len(video_stats)
                if video_stats
                else 0
            ),
            "videos_detail": video_stats,
        }

    summary = {
        "frames_directory": str(frames_dir),
        "total_images": total_images,
        "readable_images": readable_images,
        "corrupted_images": len(corrupted_images),
        "average_width": (
            total_width / readable_images
            if readable_images
            else 0
        ),
        "average_height": (
            total_height / readable_images
            if readable_images
            else 0
        ),
        "categories": category_stats,
        "dimensions": {
            f"{width}x{height}": count
            for (width, height), count in sorted(
                dimensions.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "corrupted_paths": corrupted_images,
    }

    return summary


def print_inspection(summary):
    print("\n" + "=" * 70)
    print("FRAME DATASET INSPECTION")
    print("=" * 70)

    for category, stats in summary["categories"].items():
        print(
            f"\n{category}: "
            f"{stats['videos']} videos, "
            f"{stats['frames']} readable frames"
        )
        print(
            f"  min/video={stats['min_frames_per_video']} "
            f"max/video={stats['max_frames_per_video']} "
            f"avg/video={stats['average_frames_per_video']:.2f}"
        )

    print("\nFINAL STATISTICS")
    print(
        f"Total images       : {summary['total_images']}"
    )
    print(
        f"Readable images    : {summary['readable_images']}"
    )
    print(
        f"Corrupted images   : {summary['corrupted_images']}"
    )
    print(
        f"Average dimensions : "
        f"{summary['average_width']:.1f} x "
        f"{summary['average_height']:.1f}"
    )

    print("\nIMAGE DIMENSIONS")
    for dimension, count in summary["dimensions"].items():
        print(f"{dimension}: {count}")


def save_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract surveillance frames and inspect "
            "the resulting frame hierarchy."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing category folders with videos.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where frames will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=TARGET_FPS,
        help="Target extraction FPS.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Skip extraction and inspect an existing frame directory.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON path for the inspection summary.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WEAPON DETECTION — FRAME EXTRACTION + INSPECTION")
    print("=" * 70)

    if not args.inspect_only:
        extraction = extract_frames(
            args.input,
            args.output,
            args.fps,
        )
        print("\nEXTRACTION SUMMARY")
        for key, value in extraction.items():
            print(
                f"{key:<24}: {value}"
            )

    inspection = inspect_frames(
        args.output
    )

    print_inspection(inspection)

    if args.summary:
        save_summary(
            inspection,
            args.summary,
        )
        print(
            f"\nInspection summary saved to: "
            f"{args.summary}"
        )


if __name__ == "__main__":
    main()
