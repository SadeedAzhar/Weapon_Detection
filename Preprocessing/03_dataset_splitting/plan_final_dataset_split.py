from pathlib import Path
from collections import Counter, defaultdict
import csv
import random
import re





ROOT = Path(
    r"D:\WD Dataset\Processed"
)

DATASET = (
    ROOT /
    "Final_Merged_Dataset"
)

MANIFEST = (
    DATASET /
    "merge_manifest.csv"
)

REPORT = ROOT / "proposed_split_plan.csv"



SPLITS = [
    "train",
    "val",
    "test"
]

RATIOS = {
    "train": 0.80,
    "val": 0.15,
    "test": 0.05
}

RANDOM_SEED = 42

INITIAL_ATTEMPTS = 25

REFINEMENT_ROUNDS = 25

MOVE_CANDIDATES = 600

SWAP_CANDIDATES = 600



CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other"
}



SOURCES = {
    "CVAT",
    "Dangerous_Items",
    "HGP",
    "Kaggle"
}



IMAGE_WEIGHT = 1000

CLASS_WEIGHT = 700

LONGGUN_EXTRA_WEIGHT = 1000

EMPTY_WEIGHT = 250

SOURCE_WEIGHT = 150



def read_label(label_path):

    class_counts = Counter()

    annotations = 0

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

                class_counts[
                    class_id
                ] += 1

                annotations += 1

    return (
        annotations,
        class_counts
    )



def detect_group(
    source,
    original_image
):

    name = Path(
        original_image
    ).stem


    match = re.match(
        r"^(.*?)_frame_\d+$",
        name,
        re.IGNORECASE
    )

    if match:

        sequence = match.group(1)

        return (
            f"{source}::VIDEO::{sequence}"
        )


    match = re.match(
        r"^(.*?)[_-]\d{4,6}$",
        name,
        re.IGNORECASE
    )

    if match:

        sequence = match.group(1)

        if len(sequence) >= 2:

            return (
                f"{source}::SEQUENCE::{sequence}"
            )


    return (
        f"{source}::IMAGE::{name}"
    )



def load_records():

    records = []

    with open(
        MANIFEST,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            image_name = row[
                "merged_image"
            ]

            label_name = row[
                "merged_label"
            ]

            image_path = (
                DATASET /
                "images" /
                image_name
            )

            label_path = (
                DATASET /
                "labels" /
                label_name
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Missing image:\n"
                    f"{image_path}"
                )

            if not label_path.exists():

                raise FileNotFoundError(
                    f"Missing label:\n"
                    f"{label_path}"
                )

            (
                annotations,
                classes
            ) = read_label(
                label_path
            )

            source = row[
                "source_dataset"
            ]

            original_image = row[
                "original_image"
            ]

            group = detect_group(
                source,
                original_image
            )

            records.append(
                {
                    "merged_image":
                        image_name,

                    "merged_label":
                        label_name,

                    "source":
                        source,

                    "original_image":
                        original_image,

                    "group":
                        group,

                    "annotations":
                        annotations,

                    "classes":
                        classes,

                    "empty":
                        annotations == 0
                }
            )

    return records



def build_groups(
    records
):

    groups = defaultdict(list)

    for record in records:

        groups[
            record["group"]
        ].append(
            record
        )

    group_list = []

    for group_id, items in (
        groups.items()
    ):

        class_counts = Counter()

        source_counts = Counter()

        annotations = 0

        empty = 0

        for item in items:

            class_counts.update(
                item["classes"]
            )

            source_counts[
                item["source"]
            ] += 1

            annotations += (
                item["annotations"]
            )

            if item["empty"]:

                empty += 1

        group_list.append(
            {
                "group":
                    group_id,

                "records":
                    items,

                "images":
                    len(items),

                "annotations":
                    annotations,

                "classes":
                    class_counts,

                "sources":
                    source_counts,

                "empty":
                    empty
            }
        )

    return group_list



def get_global_statistics(
    records
):

    classes = Counter()

    sources = Counter()

    empty = 0

    for record in records:

        classes.update(
            record["classes"]
        )

        sources[
            record["source"]
        ] += 1

        if record["empty"]:

            empty += 1

    return (
        classes,
        sources,
        empty
    )



def calculate_targets(
    total_images,
    global_classes,
    global_sources,
    global_empty
):

    image_targets = {}

    class_targets = {
        split: {}
        for split in SPLITS
    }

    source_targets = {
        split: {}
        for split in SPLITS
    }

    empty_targets = {}

    for split in SPLITS:

        ratio = RATIOS[
            split
        ]

        image_targets[
            split
        ] = (
            total_images *
            ratio
        )

        empty_targets[
            split
        ] = (
            global_empty *
            ratio
        )

        for class_id in CLASS_NAMES:

            class_targets[
                split
            ][class_id] = (
                global_classes[
                    class_id
                ] *
                ratio
            )

        for source in SOURCES:

            source_targets[
                split
            ][source] = (
                global_sources[
                    source
                ] *
                ratio
            )

    return (
        image_targets,
        class_targets,
        source_targets,
        empty_targets
    )



def calculate_score(
    images,
    classes,
    sources,
    empty,
    image_targets,
    class_targets,
    source_targets,
    empty_targets
):

    score = 0.0


    for split in SPLITS:

        target = (
            image_targets[
                split
            ]
        )

        actual = (
            images[
                split
            ]
        )

        error = (
            (actual - target)
            / target
        )

        score += (
            IMAGE_WEIGHT *
            error *
            error
        )


    for split in SPLITS:

        for class_id in CLASS_NAMES:

            target = (
                class_targets[
                    split
                ][class_id]
            )

            actual = (
                classes[
                    split
                ][class_id]
            )

            error = (
                (actual - target)
                / target
            )

            weight = CLASS_WEIGHT

            if class_id == 3:

                weight += (
                    LONGGUN_EXTRA_WEIGHT
                )

            score += (
                weight *
                error *
                error
            )


    for split in SPLITS:

        target = (
            empty_targets[
                split
            ]
        )

        actual = (
            empty[
                split
            ]
        )

        error = (
            (actual - target)
            / target
        )

        score += (
            EMPTY_WEIGHT *
            error *
            error
        )


    for split in SPLITS:

        for source in SOURCES:

            target = (
                source_targets[
                    split
                ][source]
            )

            actual = (
                sources[
                    split
                ][source]
            )

            error = (
                (actual - target)
                / target
            )

            score += (
                SOURCE_WEIGHT *
                error *
                error
            )

    return score



def create_state():

    return {

        "records": {
            split: []
            for split in SPLITS
        },

        "images": {
            split: 0
            for split in SPLITS
        },

        "classes": {
            split: Counter()
            for split in SPLITS
        },

        "sources": {
            split: Counter()
            for split in SPLITS
        },

        "empty": {
            split: 0
            for split in SPLITS
        }
    }



def add_group(
    state,
    split,
    group
):

    state[
        "records"
    ][split].extend(
        group["records"]
    )

    state[
        "images"
    ][split] += (
        group["images"]
    )

    state[
        "classes"
    ][split].update(
        group["classes"]
    )

    state[
        "sources"
    ][split].update(
        group["sources"]
    )

    state[
        "empty"
    ][split] += (
        group["empty"]
    )



def remove_group(
    state,
    split,
    group
):

    ids = {
        id(record)
        for record in
        group["records"]
    }

    state[
        "records"
    ][split] = [
        record
        for record in state[
            "records"
        ][split]
        if id(record)
        not in ids
    ]

    state[
        "images"
    ][split] -= (
        group["images"]
    )

    for class_id in CLASS_NAMES:

        state[
            "classes"
        ][split][class_id] -= (
            group[
                "classes"
            ][class_id]
        )

    for source in SOURCES:

        state[
            "sources"
        ][split][source] -= (
            group[
                "sources"
            ][source]
        )

    state[
        "empty"
    ][split] -= (
        group["empty"]
    )



def get_assignments(
    state,
    groups
):

    assignments = {}

    for split in SPLITS:

        record_ids = {
            id(record)
            for record in state[
                "records"
            ][split]
        }

        for group in groups:

            for record in (
                group["records"]
            ):

                if id(record) in record_ids:

                    assignments[
                        group["group"]
                    ] = split

                    break

    return assignments



def create_initial_split(
    groups,
    targets,
    seed
):

    rng = random.Random(
        seed
    )

    ordered = list(
        groups
    )

    rng.shuffle(
        ordered
    )

    ordered.sort(
        key=lambda group:
        group["images"],
        reverse=True
    )

    state = create_state()

    for group in ordered:

        candidates = []

        for split in SPLITS:

            add_group(
                state,
                split,
                group
            )

            score = (
                calculate_score(
                    state[
                        "images"
                    ],
                    state[
                        "classes"
                    ],
                    state[
                        "sources"
                    ],
                    state[
                        "empty"
                    ],
                    targets[
                        "images"
                    ],
                    targets[
                        "classes"
                    ],
                    targets[
                        "sources"
                    ],
                    targets[
                        "empty"
                    ]
                )
            )

            target = targets[
                "images"
            ][split]

            actual = state[
                "images"
            ][split]

            if actual > target:

                overflow = (
                    actual - target
                ) / target

                score += (
                    5000 *
                    overflow *
                    overflow
                )

            remove_group(
                state,
                split,
                group
            )

            score += (
                rng.random() *
                0.01
            )

            candidates.append(
                (
                    score,
                    split
                )
            )

        candidates.sort(
            key=lambda item:
            item[0]
        )

        chosen_split = (
            candidates[0][1]
        )

        add_group(
            state,
            chosen_split,
            group
        )

    return state



def refine_state(
    state,
    groups,
    targets,
    seed
):

    rng = random.Random(
        seed
    )

    group_lookup = {
        group["group"]:
        group
        for group in groups
    }

    current_score = (
        calculate_score(
            state["images"],
            state["classes"],
            state["sources"],
            state["empty"],
            targets["images"],
            targets["classes"],
            targets["sources"],
            targets["empty"]
        )
    )

    print(
        f"\nInitial refinement score: "
        f"{current_score:.2f}"
    )


    for round_number in range(
        REFINEMENT_ROUNDS
    ):

        assignments = (
            get_assignments(
                state,
                groups
            )
        )

        group_ids = list(
            assignments.keys()
        )

        rng.shuffle(
            group_ids
        )

        group_ids = group_ids[
            :MOVE_CANDIDATES
        ]

        improved = False

        for group_id in group_ids:

            group = group_lookup[
                group_id
            ]

            old_split = (
                assignments[
                    group_id
                ]
            )

            possible_splits = [
                split
                for split in SPLITS
                if split != old_split
            ]

            rng.shuffle(
                possible_splits
            )

            for new_split in (
                possible_splits
            ):

                remove_group(
                    state,
                    old_split,
                    group
                )

                add_group(
                    state,
                    new_split,
                    group
                )

                new_score = (
                    calculate_score(
                        state["images"],
                        state["classes"],
                        state["sources"],
                        state["empty"],
                        targets["images"],
                        targets["classes"],
                        targets["sources"],
                        targets["empty"]
                    )
                )

                if new_score < (
                    current_score - 0.001
                ):

                    current_score = (
                        new_score
                    )

                    improved = True

                    break

                remove_group(
                    state,
                    new_split,
                    group
                )

                add_group(
                    state,
                    old_split,
                    group
                )

            if improved:
                break

        print(
            f"  Move round "
            f"{round_number + 1:02d}: "
            f"score "
            f"{current_score:.2f}"
        )

        if not improved:
            break


    print(
        "\nStarting group-swap refinement..."
    )

    for round_number in range(
        REFINEMENT_ROUNDS
    ):

        assignments = (
            get_assignments(
                state,
                groups
            )
        )

        group_ids = list(
            assignments.keys()
        )

        rng.shuffle(
            group_ids
        )

        group_ids = group_ids[
            :SWAP_CANDIDATES
        ]

        improved = False

        for i in range(
            len(group_ids)
        ):

            group_a_id = (
                group_ids[i]
            )

            group_a = group_lookup[
                group_a_id
            ]

            split_a = (
                assignments[
                    group_a_id
                ]
            )

            partner_indices = list(
                range(
                    i + 1,
                    len(group_ids)
                )
            )

            rng.shuffle(
                partner_indices
            )

            for j in (
                partner_indices[:40]
            ):

                group_b_id = (
                    group_ids[j]
                )

                split_b = (
                    assignments[
                        group_b_id
                    ]
                )

                if split_a == split_b:
                    continue

                group_b = group_lookup[
                    group_b_id
                ]

                remove_group(
                    state,
                    split_a,
                    group_a
                )

                remove_group(
                    state,
                    split_b,
                    group_b
                )

                add_group(
                    state,
                    split_b,
                    group_a
                )

                add_group(
                    state,
                    split_a,
                    group_b
                )

                new_score = (
                    calculate_score(
                        state["images"],
                        state["classes"],
                        state["sources"],
                        state["empty"],
                        targets["images"],
                        targets["classes"],
                        targets["sources"],
                        targets["empty"]
                    )
                )

                if new_score < (
                    current_score - 0.001
                ):

                    current_score = (
                        new_score
                    )

                    improved = True

                    break

                remove_group(
                    state,
                    split_b,
                    group_a
                )

                remove_group(
                    state,
                    split_a,
                    group_b
                )

                add_group(
                    state,
                    split_a,
                    group_a
                )

                add_group(
                    state,
                    split_b,
                    group_b
                )

            if improved:
                break

        print(
            f"  Swap round "
            f"{round_number + 1:02d}: "
            f"score "
            f"{current_score:.2f}"
        )

        if not improved:
            break

    return (
        state,
        current_score
    )



def check_sequence_leakage(
    state
):

    group_split = {}

    leakage = []

    for split in SPLITS:

        for record in state[
            "records"
        ][split]:

            group = record[
                "group"
            ]

            if group in group_split:

                previous_split = (
                    group_split[
                        group
                    ]
                )

                if previous_split != split:

                    leakage.append(
                        (
                            group,
                            previous_split,
                            split
                        )
                    )

            else:

                group_split[
                    group
                ] = split

    return leakage



def main():

    print("=" * 70)

    print(
        "WEAPON DETECTION"
    )

    print(
        "FINAL DATASET SPLIT PLANNER - VERSION 5"
    )

    print("=" * 70)

    print(
        "\nDRY RUN ONLY."
    )

    print(
        "No files will be copied or modified."
    )

    print(
        "\nInternal split:"
    )

    print(
        "  TRAIN : 80%"
    )

    print(
        "  VAL   : 15%"
    )

    print(
        "  TEST  : 5%"
    )

    print(
        "\nMaster dataset:"
    )

    print(
        DATASET
    )


    if not DATASET.exists():

        print(
            "\nERROR: Master dataset not found."
        )

        return

    if not MANIFEST.exists():

        print(
            "\nERROR: merge_manifest.csv not found."
        )

        return


    print("\n")
    print("=" * 70)

    print(
        "LOADING MASTER DATASET"
    )

    print("=" * 70)

    records = load_records()

    print(
        f"\nImages loaded: "
        f"{len(records)}"
    )


    (
        global_classes,
        global_sources,
        global_empty
    ) = get_global_statistics(
        records
    )

    print(
        "\nGlobal classes:"
    )

    for class_id in CLASS_NAMES:

        print(
            f"  {class_id} = "
            f"{CLASS_NAMES[class_id]:<10} "
            f"{global_classes[class_id]}"
        )

    print(
        "\nGlobal sources:"
    )

    for source in sorted(
        global_sources
    ):

        print(
            f"  {source:<20} "
            f"{global_sources[source]}"
        )

    print(
        f"\nGlobal empty-label images: "
        f"{global_empty}"
    )


    print("\n")
    print("=" * 70)

    print(
        "BUILDING SEQUENCE GROUPS"
    )

    print("=" * 70)

    groups = build_groups(
        records
    )

    sizes = [
        group["images"]
        for group in groups
    ]

    print(
        f"\nGroups detected: "
        f"{len(groups)}"
    )

    print(
        f"Largest group: "
        f"{max(sizes)} images"
    )

    print(
    f"Single-image groups: "
    f"{sum(1 for size in sizes if size == 1)}")


    (
        image_targets,
        class_targets,
        source_targets,
        empty_targets
    ) = calculate_targets(
        len(records),
        global_classes,
        global_sources,
        global_empty
    )

    targets = {
        "images":
            image_targets,

        "classes":
            class_targets,

        "sources":
            source_targets,

        "empty":
            empty_targets
    }

    print("\n")
    print("=" * 70)

    print(
        "TARGET SPLIT"
    )

    print("=" * 70)

    for split in SPLITS:

        print(
            f"{split.upper():<6}: "
            f"{image_targets[split]:.0f} "
            f"images "
            f"({RATIOS[split]:.0%})"
        )


    print("\n")
    print("=" * 70)

    print(
        "SEARCHING FOR BEST INITIAL CANDIDATE"
    )

    print("=" * 70)

    best_state = None
    best_score = None
    best_seed = None

    for attempt in range(
        INITIAL_ATTEMPTS
    ):

        seed = (
            RANDOM_SEED +
            attempt
        )

        state = (
            create_initial_split(
                groups,
                targets,
                seed
            )
        )

        score = (
            calculate_score(
                state["images"],
                state["classes"],
                state["sources"],
                state["empty"],
                image_targets,
                class_targets,
                source_targets,
                empty_targets
            )
        )

        print(
            f"  Attempt "
            f"{attempt + 1:02d}/"
            f"{INITIAL_ATTEMPTS}"
            f" -> score "
            f"{score:.2f}"
        )

        if (
            best_score is None
            or score < best_score
        ):

            best_state = state

            best_score = score

            best_seed = seed


    print("\n")
    print("=" * 70)

    print(
        "REFINING BEST CANDIDATE"
    )

    print("=" * 70)

    (
        best_state,
        final_score
    ) = refine_state(
        best_state,
        groups,
        targets,
        best_seed + 1000
    )


    print("\n")
    print("=" * 70)

    print(
        "FINAL PROPOSED SPLIT"
    )

    print("=" * 70)

    for split in SPLITS:

        image_count = (
            best_state[
                "images"
            ][split]
        )

        percentage = (
            image_count /
            len(records) *
            100
        )

        annotation_count = sum(
            record["annotations"]
            for record in (
                best_state[
                    "records"
                ][split]
            )
        )

        group_count = len(
            set(
                record["group"]
                for record in (
                    best_state[
                        "records"
                    ][split]
                )
            )
        )

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Images      : "
            f"{image_count}"
        )

        print(
            f"  Percentage  : "
            f"{percentage:.2f}%"
        )

        print(
            f"  Groups      : "
            f"{group_count}"
        )

        print(
            f"  Annotations : "
            f"{annotation_count}"
        )

        print(
            "  Classes:"
        )

        for class_id in CLASS_NAMES:

            count = (
                best_state[
                    "classes"
                ][split][class_id]
            )

            total = (
                global_classes[
                    class_id
                ]
            )

            pct = (
                count /
                total *
                100
                if total
                else 0
            )

            print(
                f"    {class_id} = "
                f"{CLASS_NAMES[class_id]:<10} "
                f"{count:<6} "
                f"({pct:.1f}% of class)"
            )


    print("\n")
    print("=" * 70)

    print(
        "SOURCE DISTRIBUTION BY SPLIT"
    )

    print("=" * 70)

    for split in SPLITS:

        print(
            f"\n{split.upper()}:"
        )

        for source in sorted(
            SOURCES
        ):

            count = (
                best_state[
                    "sources"
                ][split][source]
            )

            total = (
                global_sources[
                    source
                ]
            )

            pct = (
                count /
                total *
                100
                if total
                else 0
            )

            print(
                f"  {source:<20} "
                f"{count:<6} "
                f"({pct:.1f}% of source)"
            )


    print("\n")
    print("=" * 70)

    print(
        "EMPTY LABEL DISTRIBUTION"
    )

    print("=" * 70)

    for split in SPLITS:

        empty = (
            best_state[
                "empty"
            ][split]
        )

        images = (
            best_state[
                "images"
            ][split]
        )

        pct = (
            empty /
            images *
            100
            if images
            else 0
        )

        print(
            f"{split.upper():<6}: "
            f"{empty} "
            f"({pct:.2f}% of split)"
        )


    print("\n")
    print("=" * 70)

    print(
        "SEQUENCE LEAKAGE CHECK"
    )

    print("=" * 70)

    leakage = (
        check_sequence_leakage(
            best_state
        )
    )

    print(
        f"\nGroups crossing splits: "
        f"{len(leakage)}"
    )

    if leakage:

        print(
            "\nWARNING: Sequence leakage detected."
        )

        for item in leakage[:20]:

            print(
                f"  {item}"
            )

    else:

        print(
            "No sequence group crosses "
            "train/val/test."
        )


    print("\n")
    print("=" * 70)

    print(
        "CLASS TARGET DEVIATION"
    )

    print("=" * 70)

    for class_id in CLASS_NAMES:

        print(
            f"\n{class_id} = "
            f"{CLASS_NAMES[class_id]}"
        )

        total = (
            global_classes[
                class_id
            ]
        )

        for split in SPLITS:

            actual = (
                best_state[
                    "classes"
                ][split][class_id]
            )

            target = (
                total *
                RATIOS[split]
            )

            deviation = (
                actual -
                target
            )

            print(
                f"  {split:<5}: "
                f"{actual:<6} "
                f"target "
                f"{target:.1f} "
                f"deviation "
                f"{deviation:+.1f}"
            )


    with open(
        REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "split",
                "merged_image",
                "merged_label",
                "source",
                "original_image",
                "group",
                "annotations",
                "empty"
            ]
        )

        for split in SPLITS:

            for record in (
                best_state[
                    "records"
                ][split]
            ):

                writer.writerow(
                    [
                        split,
                        record[
                            "merged_image"
                        ],
                        record[
                            "merged_label"
                        ],
                        record[
                            "source"
                        ],
                        record[
                            "original_image"
                        ],
                        record[
                            "group"
                        ],
                        record[
                            "annotations"
                        ],
                        record[
                            "empty"
                        ]
                    ]
                )


    total_split_images = sum(
        best_state[
            "images"
        ].values()
    )

    passed = True

    if (
        total_split_images !=
        len(records)
    ):

        passed = False

    if leakage:

        passed = False

    for split in SPLITS:

        if (
            best_state[
                "images"
            ][split] == 0
        ):

            passed = False

        for class_id in CLASS_NAMES:

            if (
                best_state[
                    "classes"
                ][split][class_id]
                == 0
            ):

                passed = False


    print("\n")
    print("=" * 70)

    print(
        "SPLIT PLANNING RESULT"
    )

    print("=" * 70)

    if passed:

        print(
            "\nDRY-RUN RESULT: PASS"
        )

        print(
            "A group-aware 80/15/5 "
            "split was successfully generated."
        )

        print(
            "\nNo files were copied."
        )

        print(
            "No files were modified."
        )

        print(
            "\nReview the distribution above "
            "before creating the actual split."
        )

    else:

        print(
            "\nDRY-RUN RESULT: FAIL"
        )

        print(
            "Do NOT create the actual split."
        )

    print(
        f"\nFinal score: "
        f"{final_score:.2f}"
    )

    print(
        "\nDry-run report:"
    )

    print(
        REPORT
    )



if __name__ == "__main__":

    main()
