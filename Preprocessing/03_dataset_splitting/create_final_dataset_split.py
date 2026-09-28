from pathlib import Path
from collections import Counter, defaultdict
import csv
import shutil
import yaml





ROOT = Path(
    r"D:\WD Dataset\Processed"
)

SOURCE_DATASET = (
    ROOT /
    "Final_Merged_Dataset"
)

SPLIT_PLAN = (
    ROOT /
    "proposed_split_plan_v5.csv"
)

OUTPUT_DATASET = (
    ROOT /
    "Final_Dataset_Split"
)

DATASET_YAML = (
    OUTPUT_DATASET /
    "dataset.yaml"
)

VERIFICATION_REPORT = (
    OUTPUT_DATASET /
    "final_split_verification_report.csv"
)



CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other"
}


SPLITS = [
    "train",
    "val",
    "test"
]



TARGET_RATIOS = {
    "train": 0.80,
    "val": 0.15,
    "test": 0.05
}



def read_label(label_path):

    annotations = 0
    classes = Counter()

    with open(
        label_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(
                    float(parts[0])
                )

            except ValueError:

                continue

            if class_id in CLASS_NAMES:

                classes[class_id] += 1
                annotations += 1

    return annotations, classes



def load_split_plan():

    if not SPLIT_PLAN.exists():

        raise FileNotFoundError(
            f"Split plan not found:\n"
            f"{SPLIT_PLAN}"
        )

    records = []

    with open(
        SPLIT_PLAN,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "split",
            "merged_image",
            "merged_label",
            "source",
            "original_image",
            "group",
            "annotations",
            "empty"
        }

        missing = (
            required_columns -
            set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Split plan is missing "
                f"columns: {missing}"
            )

        for row in reader:

            split = row["split"].strip()

            if split not in SPLITS:

                raise ValueError(
                    f"Invalid split "
                    f"'{split}' in manifest."
                )

            records.append(row)

    return records



def create_directories():

    for split in SPLITS:

        (
            OUTPUT_DATASET /
            split /
            "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_DATASET /
            split /
            "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )



def copy_dataset(records):

    copied = Counter()

    processed = 0

    total = len(records)

    for record in records:

        split = record["split"]

        image_name = (
            record["merged_image"]
        )

        label_name = (
            record["merged_label"]
        )

        source_image = (
            SOURCE_DATASET /
            "images" /
            image_name
        )

        source_label = (
            SOURCE_DATASET /
            "labels" /
            label_name
        )

        if not source_image.exists():

            raise FileNotFoundError(
                f"Source image missing:\n"
                f"{source_image}"
            )

        if not source_label.exists():

            raise FileNotFoundError(
                f"Source label missing:\n"
                f"{source_label}"
            )

        destination_image = (
            OUTPUT_DATASET /
            split /
            "images" /
            image_name
        )

        destination_label = (
            OUTPUT_DATASET /
            split /
            "labels" /
            label_name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        shutil.copy2(
            source_label,
            destination_label
        )

        copied[split] += 1

        processed += 1

        if (
            processed % 500 == 0
            or processed == total
        ):

            print(
                f"  Copied "
                f"{processed}/{total}..."
            )

    return copied



def verify_alignment():

    problems = []

    results = {}

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for split in SPLITS:

        image_dir = (
            OUTPUT_DATASET /
            split /
            "images"
        )

        label_dir = (
            OUTPUT_DATASET /
            split /
            "labels"
        )

        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in image_extensions
        }

        labels = {
            path.stem
            for path in label_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".txt"
        }

        images_without_labels = (
            images - labels
        )

        labels_without_images = (
            labels - images
        )

        results[split] = {
            "images": len(images),
            "labels": len(labels),
            "images_without_labels":
                len(images_without_labels),
            "labels_without_images":
                len(labels_without_images)
        }

        if images_without_labels:

            problems.append(
                f"{split}: "
                f"{len(images_without_labels)} "
                f"images without labels"
            )

        if labels_without_images:

            problems.append(
                f"{split}: "
                f"{len(labels_without_images)} "
                f"labels without images"
            )

    return results, problems



def verify_annotations():

    class_counts = {
        split: Counter()
        for split in SPLITS
    }

    annotation_counts = {
        split: 0
        for split in SPLITS
    }

    empty_counts = {
        split: 0
        for split in SPLITS
    }

    malformed_counts = {
        split: 0
        for split in SPLITS
    }

    invalid_class_counts = {
        split: 0
        for split in SPLITS
    }

    invalid_box_counts = {
        split: 0
        for split in SPLITS
    }

    for split in SPLITS:

        label_dir = (
            OUTPUT_DATASET /
            split /
            "labels"
        )

        for label_path in (
            label_dir.glob("*.txt")
        ):

            lines = label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

            valid_annotations = 0

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:

                    malformed_counts[
                        split
                    ] += 1

                    continue

                try:

                    class_id = int(
                        float(parts[0])
                    )

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    malformed_counts[
                        split
                    ] += 1

                    continue

                if class_id not in CLASS_NAMES:

                    invalid_class_counts[
                        split
                    ] += 1

                    continue

                if not (
                    0 <= x <= 1
                    and
                    0 <= y <= 1
                    and
                    0 < w <= 1
                    and
                    0 < h <= 1
                ):

                    invalid_box_counts[
                        split
                    ] += 1

                    continue

                class_counts[
                    split
                ][class_id] += 1

                annotation_counts[
                    split
                ] += 1

                valid_annotations += 1

            if valid_annotations == 0:

                empty_counts[
                    split
                ] += 1

    return (
        class_counts,
        annotation_counts,
        empty_counts,
        malformed_counts,
        invalid_class_counts,
        invalid_box_counts
    )



def verify_against_plan(
    records
):

    expected_images = Counter()

    expected_annotations = Counter()

    expected_empty = Counter()

    expected_classes = {
        split: Counter()
        for split in SPLITS
    }

    for record in records:

        split = record["split"]

        expected_images[
            split
        ] += 1

        expected_annotations[
            split
        ] += int(
            record["annotations"]
        )

        if record["empty"].lower() == "true":

            expected_empty[
                split
            ] += 1

        label_path = (
            SOURCE_DATASET /
            "labels" /
            record["merged_label"]
        )

        _, classes = read_label(
            label_path
        )

        expected_classes[
            split
        ].update(
            classes
        )

    return (
        expected_images,
        expected_annotations,
        expected_empty,
        expected_classes
    )



def verify_sequence_leakage(
    records
):

    group_splits = defaultdict(
        set
    )

    for record in records:

        group_splits[
            record["group"]
        ].add(
            record["split"]
        )

    leakage = []

    for group, splits in (
        group_splits.items()
    ):

        if len(splits) > 1:

            leakage.append(
                {
                    "group": group,
                    "splits": ",".join(
                        sorted(splits)
                    )
                }
            )

    return leakage



def create_dataset_yaml():

    yaml_data = {

        "path": str(
            OUTPUT_DATASET
        ),

        "train": "train/images",

        "val": "val/images",

        "test": "test/images",

        "names": {
            0: "Knife",
            1: "Machette",
            2: "Handgun",
            3: "Longgun",
            4: "Other"
        },

        "nc": 5
    }

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            yaml_data,
            f,
            sort_keys=False
        )



def write_report(
    alignment,
    class_counts,
    annotation_counts,
    empty_counts,
    malformed_counts,
    invalid_class_counts,
    invalid_box_counts,
    expected_images,
    expected_annotations,
    expected_empty,
    expected_classes
):

    rows = []

    for split in SPLITS:

        rows.append(
            {
                "split": split,

                "expected_images":
                    expected_images[split],

                "actual_images":
                    alignment[split][
                        "images"
                    ],

                "expected_labels":
                    expected_images[split],

                "actual_labels":
                    alignment[split][
                        "labels"
                    ],

                "expected_annotations":
                    expected_annotations[
                        split
                    ],

                "actual_annotations":
                    annotation_counts[
                        split
                    ],

                "expected_empty":
                    expected_empty[
                        split
                    ],

                "actual_empty":
                    empty_counts[
                        split
                    ],

                "malformed":
                    malformed_counts[
                        split
                    ],

                "invalid_classes":
                    invalid_class_counts[
                        split
                    ],

                "invalid_boxes":
                    invalid_box_counts[
                        split
                    ],

                "knife":
                    class_counts[
                        split
                    ][0],

                "machette":
                    class_counts[
                        split
                    ][1],

                "handgun":
                    class_counts[
                        split
                    ][2],

                "longgun":
                    class_counts[
                        split
                    ][3],

                "other":
                    class_counts[
                        split
                    ][4]
            }
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        VERIFICATION_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )



def main():

    print("=" * 70)

    print(
        "WEAPON DETECTION"
    )

    print(
        "FINAL DATASET SPLIT CREATION"
    )

    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Final_Merged_Dataset will NOT be modified."
    )

    print(
        "\nSource:"
    )

    print(
        SOURCE_DATASET
    )

    print(
        "\nSplit plan:"
    )

    print(
        SPLIT_PLAN
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DATASET
    )


    if not SOURCE_DATASET.exists():

        raise FileNotFoundError(
            "Final_Merged_Dataset does not exist."
        )

    if not SPLIT_PLAN.exists():

        raise FileNotFoundError(
            "V5 split plan does not exist."
        )

    if OUTPUT_DATASET.exists():

        print(
            "\nWARNING:"
        )

        print(
            "Final_Dataset_Split already exists:"
        )

        print(
            OUTPUT_DATASET
        )

        print(
            "\nDelete/rename the existing output "
            "before running this script."
        )

        return


    print("\n")
    print("=" * 70)

    print(
        "LOADING APPROVED V5 SPLIT PLAN"
    )

    print("=" * 70)

    records = load_split_plan()

    print(
        f"\nRecords in split plan: "
        f"{len(records)}"
    )

    split_counts = Counter(
        record["split"]
        for record in records
    )

    for split in SPLITS:

        print(
            f"  {split:<5}: "
            f"{split_counts[split]}"
        )


    print("\n")
    print("=" * 70)

    print(
        "CREATING DATASET DIRECTORIES"
    )

    print("=" * 70)

    create_directories()

    print(
        "\nDirectories created."
    )


    print("\n")
    print("=" * 70)

    print(
        "COPYING IMAGES AND LABELS"
    )

    print("=" * 70)

    copied = copy_dataset(
        records
    )

    print(
        "\nCopy complete:"
    )

    for split in SPLITS:

        print(
            f"  {split:<5}: "
            f"{copied[split]} images"
        )


    print("\n")
    print("=" * 70)

    print(
        "CREATING DATASET YAML"
    )

    print("=" * 70)

    create_dataset_yaml()

    print(
        f"\nCreated:"
    )

    print(
        DATASET_YAML
    )


    print("\n")
    print("=" * 70)

    print(
        "VERIFYING IMAGE / LABEL ALIGNMENT"
    )

    print("=" * 70)

    (
        alignment,
        alignment_problems
    ) = verify_alignment()

    for split in SPLITS:

        result = alignment[
            split
        ]

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Images : "
            f"{result['images']}"
        )

        print(
            f"  Labels : "
            f"{result['labels']}"
        )

        print(
            f"  Images without labels : "
            f"{result['images_without_labels']}"
        )

        print(
            f"  Labels without images : "
            f"{result['labels_without_images']}"
        )


    print("\n")
    print("=" * 70)

    print(
        "VERIFYING ANNOTATIONS"
    )

    print("=" * 70)

    (
        class_counts,
        annotation_counts,
        empty_counts,
        malformed_counts,
        invalid_class_counts,
        invalid_box_counts
    ) = verify_annotations()

    for split in SPLITS:

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Annotations : "
            f"{annotation_counts[split]}"
        )

        print(
            f"  Empty labels: "
            f"{empty_counts[split]}"
        )

        print(
            f"  Malformed   : "
            f"{malformed_counts[split]}"
        )

        print(
            f"  Invalid IDs : "
            f"{invalid_class_counts[split]}"
        )

        print(
            f"  Invalid boxes: "
            f"{invalid_box_counts[split]}"
        )

        print(
            "  Classes:"
        )

        for class_id in CLASS_NAMES:

            print(
                f"    {class_id} = "
                f"{CLASS_NAMES[class_id]:<10} "
                f"{class_counts[split][class_id]}"
            )


    print("\n")
    print("=" * 70)

    print(
        "VERIFYING AGAINST V5 SPLIT PLAN"
    )

    print("=" * 70)

    (
        expected_images,
        expected_annotations,
        expected_empty,
        expected_classes
    ) = verify_against_plan(
        records
    )

    plan_pass = True

    for split in SPLITS:

        print(
            f"\n{split.upper()}"
        )

        if (
            alignment[split]["images"]
            != expected_images[split]
        ):

            plan_pass = False

        if (
            alignment[split]["labels"]
            != expected_images[split]
        ):

            plan_pass = False

        if (
            annotation_counts[split]
            != expected_annotations[split]
        ):

            plan_pass = False

        if (
            empty_counts[split]
            != expected_empty[split]
        ):

            plan_pass = False

        print(
            f"  Images expected/actual: "
            f"{expected_images[split]} / "
            f"{alignment[split]['images']}"
        )

        print(
            f"  Annotations expected/actual: "
            f"{expected_annotations[split]} / "
            f"{annotation_counts[split]}"
        )

        print(
            f"  Empty labels expected/actual: "
            f"{expected_empty[split]} / "
            f"{empty_counts[split]}"
        )

        for class_id in CLASS_NAMES:

            expected = (
                expected_classes[
                    split
                ][class_id]
            )

            actual = (
                class_counts[
                    split
                ][class_id]
            )

            print(
                f"  {CLASS_NAMES[class_id]:<10}: "
                f"{expected} / {actual}"
            )

            if expected != actual:

                plan_pass = False


    print("\n")
    print("=" * 70)

    print(
        "SEQUENCE LEAKAGE CHECK"
    )

    print("=" * 70)

    leakage = (
        verify_sequence_leakage(
            records
        )
    )

    print(
        f"\nGroups crossing splits: "
        f"{len(leakage)}"
    )

    if leakage:

        plan_pass = False

        print(
            "\nWARNING: Sequence leakage detected!"
        )

        for item in leakage[:20]:

            print(
                f"  {item['group']} "
                f"-> {item['splits']}"
            )

    else:

        print(
            "No sequence group crosses "
            "train/val/test."
        )


    if alignment_problems:

        plan_pass = False

    for split in SPLITS:

        if malformed_counts[
            split
        ] > 0:

            plan_pass = False

        if invalid_class_counts[
            split
        ] > 0:

            plan_pass = False

        if invalid_box_counts[
            split
        ] > 0:

            plan_pass = False


    write_report(
        alignment,
        class_counts,
        annotation_counts,
        empty_counts,
        malformed_counts,
        invalid_class_counts,
        invalid_box_counts,
        expected_images,
        expected_annotations,
        expected_empty,
        expected_classes
    )


    print("\n")
    print("=" * 70)

    print(
        "FINAL SPLIT VERIFICATION"
    )

    print("=" * 70)

    if plan_pass:

        print(
            "\nOVERALL RESULT: PASS"
        )

        print(
            "The physical dataset exactly matches "
            "the approved V5 split plan."
        )

        print(
            "\nTrain / Val / Test dataset is ready."
        )

    else:

        print(
            "\nOVERALL RESULT: FAIL"
        )

        print(
            "One or more verification checks failed."
        )

        print(
            "\nDO NOT use this dataset for training "
            "until the issue is resolved."
        )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DATASET
    )

    print(
        "\nDataset YAML:"
    )

    print(
        DATASET_YAML
    )

    print(
        "\nVerification report:"
    )

    print(
        VERIFICATION_REPORT
    )



if __name__ == "__main__":

    main()
