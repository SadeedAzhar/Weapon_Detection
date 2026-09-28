from pathlib import Path
import json
import random
import re
import shutil

import pandas as pd

AERIAL_ROOT = Path(r"D:\WD Dataset\Sources\aerial\samples\samples")
CSV_PATH = Path(
    r"D:\WD Dataset\Aerial_V2.1_Person_ROI_Evaluation\frame_level_results.csv"
)
OUTPUT_ROOT = Path(
    r"D:\WD Dataset\Aerial_V2.2_Hard_Example_Selection"
)

SEED = 42
FRAMES_PER_CLIP = 2
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPECTED_TOTAL = 354
EXPECTED_WEAPON = 184
EXPECTED_NO_WEAPON = 170


def load_results():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found:\n{CSV_PATH}")

    df = pd.read_excel(CSV_PATH) if CSV_PATH.suffix.lower() == ".xlsx" else pd.read_csv(CSV_PATH)

    required = [
        "clip",
        "clip_name",
        "frame",
        "ground_truth",
        "weapon_detected",
        "longgun_detected",
        "predicted_classes",
        "max_weapon_confidence",
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["ground_truth"] = (
        df["ground_truth"].astype(str).str.strip().str.lower()
    )

    for column in ["weapon_detected", "longgun_detected"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
        )

    df["max_weapon_confidence"] = pd.to_numeric(
        df["max_weapon_confidence"], errors="coerce"
    ).fillna(0.0)

    df["frame_number"] = (
        df["frame"].astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    expected = {"weapon", "no_weapon"}
    actual = set(df["ground_truth"].unique())

    if not expected.issubset(actual):
        raise ValueError(
            f"Expected ground-truth values {expected}, found {actual}"
        )

    return df


def build_clip_split(df):
    clip_df = (
        df.groupby("clip")
        .agg(
            clip_name=("clip_name", "first"),
            ground_truth=("ground_truth", "first"),
            frame_count=("frame", "count"),
        )
        .reset_index()
    )

    mixed = df.groupby("clip")["ground_truth"].nunique()
    mixed = mixed[mixed > 1]

    if not mixed.empty:
        raise ValueError(
            f"Clips with mixed ground truth: {mixed.index.tolist()}"
        )

    random.seed(SEED)
    records = []

    for category in ["weapon", "no_weapon"]:
        clips = clip_df.loc[
            clip_df["ground_truth"] == category, "clip"
        ].tolist()
        random.shuffle(clips)

        n = len(clips)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)

        for clip in clips[:n_train]:
            records.append(
                {"clip": clip, "ground_truth": category, "split": "train"}
            )
        for clip in clips[n_train:n_train + n_val]:
            records.append(
                {"clip": clip, "ground_truth": category, "split": "validation"}
            )
        for clip in clips[n_train + n_val:]:
            records.append(
                {"clip": clip, "ground_truth": category, "split": "test"}
            )

    split_df = pd.DataFrame(records)

    split_counts = (
        pd.crosstab(split_df["split"], split_df["ground_truth"])
        if not split_df.empty
        else pd.DataFrame()
    )

    return clip_df, split_df, split_counts


def select_spread_candidates(candidates, n):
    if len(candidates) <= n:
        return candidates

    candidates = (
        candidates.sort_values("frame_number")
        .reset_index(drop=True)
    )

    if n == 1:
        return candidates.iloc[[len(candidates) // 2]]

    positions = [
        round(i * (len(candidates) - 1) / (n - 1))
        for i in range(n)
    ]
    return candidates.iloc[positions]


def select_training_examples(df, clip_split_df):
    train_clip_ids = set(
        clip_split_df.loc[
            clip_split_df["split"] == "train", "clip"
        ]
    )

    train_df = df[df["clip"].isin(train_clip_ids)].copy()
    selected = []

    weapon_df = train_df[train_df["ground_truth"] == "weapon"]

    for clip, group in weapon_df.groupby("clip"):
        missed = group[~group["weapon_detected"]].copy()

        weak = group[
            group["weapon_detected"]
            & (
                ~group["longgun_detected"]
                | (group["max_weapon_confidence"] < 0.40)
            )
        ].copy()

        normal = group[
            ~group.index.isin(missed.index)
            & ~group.index.isin(weak.index)
        ].copy()

        candidates = pd.concat([missed, weak, normal])
        candidates = candidates[~candidates.index.duplicated()]
        chosen = select_spread_candidates(
            candidates, FRAMES_PER_CLIP
        )

        for _, row in chosen.iterrows():
            if not row["weapon_detected"]:
                reason = "missed_rifle"
            elif (
                not row["longgun_detected"]
                or row["max_weapon_confidence"] < 0.40
            ):
                reason = "weak_longgun_detection"
            else:
                reason = "representative_rifle"

            selected.append(
                {
                    "split": "train",
                    "category": "weapon",
                    "ground_truth": "rifle",
                    "clip": row["clip"],
                    "clip_name": row["clip_name"],
                    "frame": row["frame"],
                    "frame_number": int(row["frame_number"]),
                    "weapon_detected_by_v21": bool(row["weapon_detected"]),
                    "longgun_detected_by_v21": bool(row["longgun_detected"]),
                    "max_weapon_confidence": float(
                        row["max_weapon_confidence"]
                    ),
                    "predicted_classes": str(row["predicted_classes"]),
                    "selection_reason": reason,
                }
            )

    no_weapon_df = train_df[
        train_df["ground_truth"] == "no_weapon"
    ]

    for clip, group in no_weapon_df.groupby("clip"):
        false_positives = group[group["weapon_detected"]].copy()
        false_positives = false_positives.sort_values(
            "max_weapon_confidence", ascending=False
        )

        remaining = group[
            ~group.index.isin(false_positives.index)
        ].copy()

        candidates = pd.concat([false_positives, remaining])
        candidates = candidates[~candidates.index.duplicated()]
        chosen = select_spread_candidates(
            candidates, FRAMES_PER_CLIP
        )

        for _, row in chosen.iterrows():
            reason = (
                "high_confidence_weapon_false_positive"
                if row["weapon_detected"]
                else "representative_umbrella_negative"
            )

            selected.append(
                {
                    "split": "train",
                    "category": "no_weapon",
                    "ground_truth": "umbrella",
                    "clip": row["clip"],
                    "clip_name": row["clip_name"],
                    "frame": row["frame"],
                    "frame_number": int(row["frame_number"]),
                    "weapon_detected_by_v21": bool(row["weapon_detected"]),
                    "longgun_detected_by_v21": bool(row["longgun_detected"]),
                    "max_weapon_confidence": float(
                        row["max_weapon_confidence"]
                    ),
                    "predicted_classes": str(row["predicted_classes"]),
                    "selection_reason": reason,
                }
            )

    selection_df = pd.DataFrame(selected)

    if len(selection_df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} selected images, "
            f"found {len(selection_df)}."
        )

    weapon_count = int(
        (selection_df["category"] == "weapon").sum()
    )
    no_weapon_count = int(
        (selection_df["category"] == "no_weapon").sum()
    )

    if weapon_count != EXPECTED_WEAPON:
        raise RuntimeError(
            f"Expected {EXPECTED_WEAPON} weapon selections, found {weapon_count}."
        )

    if no_weapon_count != EXPECTED_NO_WEAPON:
        raise RuntimeError(
            f"Expected {EXPECTED_NO_WEAPON} no-weapon selections, "
            f"found {no_weapon_count}."
        )

    return selection_df


def locate_image(row):
    clip_value = str(row["clip"]).replace("\\", "/")
    candidate = (
        AERIAL_ROOT
        / Path(clip_value)
        / str(row["frame"])
    )

    if candidate.exists():
        return candidate

    frame_name = str(row["frame"])
    clip_name = str(row["clip_name"])

    matches = list(AERIAL_ROOT.rglob(frame_name))

    for match in matches:
        if clip_name in str(match.parent):
            return match

    return matches[0] if matches else None


def make_unique_name(source_path):
    try:
        relative = source_path.relative_to(AERIAL_ROOT)
        parts = list(relative.parts)
    except ValueError:
        parts = list(source_path.parts[-4:])

    safe_parts = []
    for part in parts:
        safe = str(part)
        safe = re.sub(r'[:/\\*?"<>|]', "_", safe)
        safe_parts.append(safe)

    return "__".join(safe_parts)


def reconstruct_selection(selection_df, clip_split_df):
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT_ROOT}\n"
            "Delete or rename it before regenerating."
        )

    weapon_out = OUTPUT_ROOT / "train" / "weapon"
    no_weapon_out = OUTPUT_ROOT / "train" / "no_weapon"
    locked_val = OUTPUT_ROOT / "locked_validation"
    locked_test = OUTPUT_ROOT / "locked_test"

    for folder in [weapon_out, no_weapon_out, locked_val, locked_test]:
        folder.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    destination_names = set()

    for _, row in selection_df.iterrows():
        source = locate_image(row)

        if source is None:
            missing.append({
                "clip": row["clip"],
                "frame": row["frame"],
            })
            continue

        output_dir = (
            weapon_out
            if row["category"] == "weapon"
            else no_weapon_out
        )

        destination = output_dir / make_unique_name(source)

        if destination.name.lower() in destination_names:
            raise RuntimeError(
                f"Destination collision detected: {destination}"
            )

        shutil.copy2(source, destination)
        destination_names.add(destination.name.lower())

        copied.append(
            {
                "category": row["category"],
                "clip": row["clip"],
                "clip_name": row["clip_name"],
                "frame": row["frame"],
                "source_path": str(source),
                "selected_path": str(destination),
            }
        )

    if missing:
        raise RuntimeError(
            f"{len(missing)} selected source images were missing."
        )

    copied_df = pd.DataFrame(copied)
    output_manifest = selection_df.merge(
        copied_df,
        on=["category", "clip", "clip_name", "frame"],
        how="left",
    )

    output_manifest.to_csv(
        OUTPUT_ROOT / "train_selection_manifest_collision_safe.csv",
        index=False,
    )

    val_clip_ids = set(
        clip_split_df.loc[
            clip_split_df["split"] == "validation", "clip"
        ]
    )
    test_clip_ids = set(
        clip_split_df.loc[
            clip_split_df["split"] == "test", "clip"
        ]
    )

    source_df = pd.read_csv(CSV_PATH)
    source_df["clip"] = source_df["clip"].astype(str)

    source_df[source_df["clip"].isin(val_clip_ids)].to_csv(
        locked_val / "validation_frame_manifest.csv",
        index=False,
    )

    source_df[source_df["clip"].isin(test_clip_ids)].to_csv(
        locked_test / "test_frame_manifest.csv",
        index=False,
    )

    physical_weapon = list(
        weapon_out.glob("*")
    )
    physical_no_weapon = list(
        no_weapon_out.glob("*")
    )

    physical_weapon = [
        p for p in physical_weapon if p.is_file()
    ]
    physical_no_weapon = [
        p for p in physical_no_weapon if p.is_file()
    ]

    if len(physical_weapon) != EXPECTED_WEAPON:
        raise RuntimeError(
            f"Physical weapon count mismatch: {len(physical_weapon)}"
        )

    if len(physical_no_weapon) != EXPECTED_NO_WEAPON:
        raise RuntimeError(
            f"Physical no-weapon count mismatch: {len(physical_no_weapon)}"
        )

    return {
        "copied_images": len(copied),
        "weapon_images": len(physical_weapon),
        "no_weapon_images": len(physical_no_weapon),
        "missing_images": len(missing),
    }


def main():
    print("=" * 70)
    print("AERIAL V2.2 HARD-EXAMPLE SELECTION + RECONSTRUCTION")
    print("=" * 70)

    df = load_results()

    print(f"\nRows loaded: {len(df):,}")

    clip_df, clip_split_df, split_counts = build_clip_split(df)

    print("\nClip-level split:")
    print(split_counts)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clip_split_df.to_csv(
        OUTPUT_ROOT / "clip_split_manifest.csv",
        index=False,
    )

    selection_df = select_training_examples(
        df, clip_split_df
    )

    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True)

    selection_df.to_csv(
        OUTPUT_ROOT / "train_selection_manifest.csv",
        index=False,
    )

    reconstruction = reconstruct_selection(
        selection_df, clip_split_df
    )

    summary = {
        "source_csv": str(CSV_PATH),
        "source_frames": int(len(df)),
        "source_clips": int(len(clip_df)),
        "train_clips": int(
            (clip_split_df["split"] == "train").sum()
        ),
        "validation_clips": int(
            (clip_split_df["split"] == "validation").sum()
        ),
        "test_clips": int(
            (clip_split_df["split"] == "test").sum()
        ),
        "frames_per_training_clip": FRAMES_PER_CLIP,
        "selected_training_images": int(len(selection_df)),
        "selected_weapon_images": int(
            (selection_df["category"] == "weapon").sum()
        ),
        "selected_no_weapon_images": int(
            (selection_df["category"] == "no_weapon").sum()
        ),
        "copied_images": reconstruction["copied_images"],
        "missing_images": reconstruction["missing_images"],
        "validation_test_locked": True,
        "random_seed": SEED,
    }

    (OUTPUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("V2.2 AERIAL HARD-EXAMPLE PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Source frames          : {len(df):,}")
    print(f"Source clips           : {len(clip_df):,}")
    print(f"Selected weapon        : {reconstruction['weapon_images']}")
    print(f"Selected no-weapon     : {reconstruction['no_weapon_images']}")
    print(f"TOTAL                  : {reconstruction['copied_images']}")
    print(f"Missing images         : {reconstruction['missing_images']}")
    print(f"Output                 : {OUTPUT_ROOT}")
    print("\nValidation/Test clips remain locked.")
    print("Only TRAIN clips are mined for V2.2 hard examples.")
    print("Weapon images require manual bounding-box annotation.")
    print("No-weapon images require no boxes.")


if __name__ == "__main__":
    main()
