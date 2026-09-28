from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json

from ultralytics import YOLO

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_NAMES = {
    0: "Knife",
    1: "Machette",
    2: "Handgun",
    3: "Longgun",
    4: "Other",
}


def safe_divide(a, b):
    return a / b if b else 0.0


def calculate_metrics(tp, tn, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
    }


def find_clip_folders(root):
    root = Path(root)

    return sorted(
        folder
        for folder in root.rglob("*")
        if folder.is_dir()
        and (
            folder.name.lower().startswith("real_")
            or folder.name.lower().startswith("falso_")
        )
    )


def find_images(clip_folder):
    return sorted(
        path
        for path in clip_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_ground_truth(clip_folder):
    name = clip_folder.name.lower()

    if name.startswith("real_"):
        return True

    if name.startswith("falso_"):
        return False

    return None


def expand_box(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    expansion,
):
    width = x2 - x1
    height = y2 - y1

    expand_x = width * expansion / 2.0
    expand_y = height * expansion / 2.0

    return (
        max(
            0,
            min(
                int(round(x1 - expand_x)),
                image_width - 1,
            ),
        ),
        max(
            0,
            min(
                int(round(y1 - expand_y)),
                image_height - 1,
            ),
        ),
        max(
            1,
            min(
                int(round(x2 + expand_x)),
                image_width,
            ),
        ),
        max(
            1,
            min(
                int(round(y2 + expand_y)),
                image_height,
            ),
        ),
    )


def evaluate_full_frame(
    weapon_model,
    aerial_root,
    output_root,
    image_size,
    confidence,
    iou,
    device,
    max_saved_images,
):
    annotated_root = output_root / "annotated_predictions"
    annotated_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_results = []
    clip_results = []

    overall = {
        "real": {
            "images": 0,
            "detected_weapon": 0,
            "detected_longgun": 0,
            "no_detection": 0,
        },
        "false": {
            "images": 0,
            "detected_weapon": 0,
            "detected_longgun": 0,
            "no_detection": 0,
        },
    }

    clips = find_clip_folders(aerial_root)

    for clip_index, clip in enumerate(clips, 1):
        ground_truth = get_ground_truth(clip)
        images = find_images(clip)

        if ground_truth is None or not images:
            continue

        relative_clip = clip.relative_to(aerial_root)
        output_clip = (
            annotated_root / relative_clip
        )
        output_clip.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = 0
        clip_data = {
            "ground_truth": (
                "weapon" if ground_truth else "no_weapon"
            ),
            "images": 0,
            "weapon_detected": 0,
            "longgun_detected": 0,
            "no_detection": 0,
            "max_weapon_conf": 0.0,
        }

        print(
            f"\n[{clip_index}/{len(clips)}] "
            f"{relative_clip}"
        )

        for image_path in images:
            result = weapon_model.predict(
                source=str(image_path),
                imgsz=image_size,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )[0]

            predicted = []
            confidences = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):
                for class_id, conf_value in zip(
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    class_id = int(class_id)
                    conf_value = float(conf_value)

                    predicted.append(
                        CLASS_NAMES.get(
                            class_id,
                            f"class_{class_id}",
                        )
                    )
                    confidences.append(conf_value)

            weapon_detected = bool(predicted)
            longgun_detected = (
                "Longgun" in predicted
            )
            max_conf = (
                max(confidences)
                if confidences
                else 0.0
            )

            label = (
                "TP"
                if ground_truth and weapon_detected
                else "FN"
                if ground_truth
                else "FP"
                if weapon_detected
                else "TN"
            )

            group = (
                "real"
                if ground_truth
                else "false"
            )

            overall[group]["images"] += 1

            if weapon_detected:
                overall[group][
                    "detected_weapon"
                ] += 1
                clip_data[
                    "weapon_detected"
                ] += 1
            else:
                overall[group][
                    "no_detection"
                ] += 1
                clip_data[
                    "no_detection"
                ] += 1

            if longgun_detected:
                overall[group][
                    "detected_longgun"
                ] += 1
                clip_data[
                    "longgun_detected"
                ] += 1

            clip_data["images"] += 1
            clip_data["max_weapon_conf"] = max(
                clip_data["max_weapon_conf"],
                max_conf,
            )

            frame_results.append({
                "clip": str(relative_clip),
                "clip_name": clip.name,
                "image": image_path.name,
                "ground_truth": (
                    "weapon"
                    if ground_truth
                    else "no_weapon"
                ),
                "result": label,
                "weapon_detected": weapon_detected,
                "longgun_detected": longgun_detected,
                "predicted_classes": "|".join(
                    predicted
                ),
                "max_weapon_confidence": round(
                    max_conf,
                    6,
                ),
                "image_path": str(image_path),
            })

            if (
                max_saved_images is None
                or saved < max_saved_images
            ):
                cv2.imwrite(
                    str(
                        output_clip
                        / image_path.name
                    ),
                    result.plot(),
                )
                saved += 1

        clip_results.append({
            "clip": str(relative_clip),
            "ground_truth": clip_data[
                "ground_truth"
            ],
            "frames": clip_data["images"],
            "frames_weapon_detected": clip_data[
                "weapon_detected"
            ],
            "frames_longgun_detected": clip_data[
                "longgun_detected"
            ],
            "detection_rate": safe_divide(
                clip_data["weapon_detected"],
                clip_data["images"],
            ),
            "longgun_detection_rate": safe_divide(
                clip_data["longgun_detected"],
                clip_data["images"],
            ),
            "max_weapon_confidence": round(
                clip_data["max_weapon_conf"],
                6,
            ),
        })

    tp = overall["real"]["detected_weapon"]
    fn = overall["real"]["no_detection"]
    fp = overall["false"]["detected_weapon"]
    tn = overall["false"]["no_detection"]

    return {
        "frame_results": frame_results,
        "clip_results": clip_results,
        "frame_level": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            **calculate_metrics(tp, tn, fp, fn),
        },
        "longgun": {
            "TP": overall["real"]["detected_longgun"],
            "FN": (
                overall["real"]["images"]
                - overall["real"]["detected_longgun"]
            ),
            "FP": overall["false"][
                "detected_longgun"
            ],
        },
    }


def evaluate_person_roi(
    person_model,
    weapon_model,
    aerial_root,
    output_root,
    person_image_size,
    person_confidence,
    person_iou,
    weapon_image_size,
    weapon_confidence,
    weapon_iou,
    device,
    roi_expansion,
    process_all_people,
    min_person_confidence,
    max_saved_images,
    save_roi_images,
):
    annotated_root = (
        output_root / "annotated_predictions"
    )
    roi_root = (
        output_root / "roi_crops"
    )

    annotated_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if save_roi_images:
        roi_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    frame_results = []
    clip_results = []

    global_tp = 0
    global_tn = 0
    global_fp = 0
    global_fn = 0

    global_longgun_tp = 0
    global_longgun_fn = 0
    global_longgun_fp = 0

    clips = find_clip_folders(aerial_root)

    for clip_index, clip in enumerate(clips, 1):
        ground_truth = get_ground_truth(clip)
        images = find_images(clip)

        if ground_truth is None or not images:
            continue

        relative_clip = clip.relative_to(aerial_root)
        output_clip = (
            annotated_root / relative_clip
        )
        output_clip.mkdir(
            parents=True,
            exist_ok=True,
        )

        if save_roi_images:
            roi_clip = (
                roi_root / relative_clip
            )
            roi_clip.mkdir(
                parents=True,
                exist_ok=True,
            )

        clip_weapon_detections = 0
        clip_longgun_detections = 0
        clip_weapon_frames = 0
        clip_longgun_frames = 0
        saved_annotated = 0

        print(
            f"\n[{clip_index}/{len(clips)}] "
            f"{relative_clip}"
        )

        for image_path in images:
            person_result = person_model.predict(
                source=str(image_path),
                imgsz=person_image_size,
                conf=person_confidence,
                iou=person_iou,
                device=device,
                verbose=False,
            )[0]

            original_image = person_result.orig_img
            image_height, image_width = (
                original_image.shape[:2]
            )

            people = []

            if person_result.boxes is not None:
                for box in person_result.boxes:
                    class_id = int(
                        box.cls[0].item()
                    )
                    confidence = float(
                        box.conf[0].item()
                    )

                    if class_id != 0:
                        continue

                    if (
                        confidence
                        < min_person_confidence
                    ):
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    people.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                    })

            frame_weapon_detected = False
            frame_longgun_detected = False
            predictions = []

            for person_index, person in enumerate(
                people
            ):
                x1, y1, x2, y2 = expand_box(
                    person["x1"],
                    person["y1"],
                    person["x2"],
                    person["y2"],
                    image_width,
                    image_height,
                    roi_expansion,
                )

                roi = original_image[
                    y1:y2,
                    x1:x2,
                ]

                if roi.size == 0:
                    continue

                result = weapon_model.predict(
                    source=roi,
                    imgsz=weapon_image_size,
                    conf=weapon_confidence,
                    iou=weapon_iou,
                    device=device,
                    verbose=False,
                )[0]

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(
                            box.cls[0].item()
                        )
                        confidence = float(
                            box.conf[0].item()
                        )

                        class_name = CLASS_NAMES.get(
                            class_id,
                            f"class_{class_id}",
                        )

                        frame_weapon_detected = True

                        if class_id == 3:
                            frame_longgun_detected = True

                        predictions.append({
                            "person_index": person_index,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                        })

                if save_roi_images:
                    cv2.imwrite(
                        str(
                            roi_clip
                            / (
                                f"{image_path.stem}"
                                f"_person_{person_index:02d}.jpg"
                            )
                        ),
                        roi,
                    )

                if not process_all_people:
                    break

            if ground_truth:
                if frame_weapon_detected:
                    global_tp += 1
                else:
                    global_fn += 1

                if frame_longgun_detected:
                    global_longgun_tp += 1
                else:
                    global_longgun_fn += 1

            else:
                if frame_weapon_detected:
                    global_fp += 1
                else:
                    global_tn += 1

                if frame_longgun_detected:
                    global_longgun_fp += 1

            if frame_weapon_detected:
                clip_weapon_detections += 1
            if frame_longgun_detected:
                clip_longgun_detections += 1

            frame_results.append({
                "clip": str(relative_clip),
                "clip_name": clip.name,
                "frame": image_path.name,
                "ground_truth": (
                    "weapon"
                    if ground_truth
                    else "no_weapon"
                ),
                "people_detected": len(people),
                "weapon_detected": (
                    frame_weapon_detected
                ),
                "longgun_detected": (
                    frame_longgun_detected
                ),
                "predicted_classes": "|".join(
                    p["class_name"]
                    for p in predictions
                ),
                "max_weapon_confidence": (
                    max(
                        (
                            p["confidence"]
                            for p in predictions
                        ),
                        default=0.0,
                    )
                ),
            })

            if (
                saved_annotated
                < max_saved_images
            ):
                annotated = original_image.copy()

                for person_index, person in enumerate(
                    people
                ):
                    px1, py1, px2, py2 = (
                        expand_box(
                            person["x1"],
                            person["y1"],
                            person["x2"],
                            person["y2"],
                            image_width,
                            image_height,
                            roi_expansion,
                        )
                    )

                    cv2.rectangle(
                        annotated,
                        (px1, py1),
                        (px2, py2),
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        (
                            f"Person {person_index} "
                            f"{person['confidence']:.2f}"
                        ),
                        (
                            px1,
                            max(20, py1 - 5),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                cv2.putText(
                    annotated,
                    (
                        "WEAPON"
                        if frame_weapon_detected
                        else "NO WEAPON"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0)
                    if frame_weapon_detected
                    else (0, 0, 255),
                    3,
                )

                cv2.imwrite(
                    str(
                        output_clip
                        / (
                            f"{image_path.stem}"
                            "_roi_result.jpg"
                        )
                    ),
                    annotated,
                )

                saved_annotated += 1

        total_frames = len(images)

        clip_results.append({
            "clip": str(relative_clip),
            "clip_name": clip.name,
            "ground_truth": (
                "weapon"
                if ground_truth
                else "no_weapon"
            ),
            "frames": total_frames,
            "frames_with_weapon_detection": (
                clip_weapon_detections
            ),
            "weapon_detection_rate": safe_divide(
                clip_weapon_detections,
                total_frames,
            ),
            "clip_weapon_detected": (
                clip_weapon_detections > 0
            ),
            "frames_with_longgun_detection": (
                clip_longgun_detections
            ),
            "longgun_detection_rate": safe_divide(
                clip_longgun_detections,
                total_frames,
            ),
            "clip_longgun_detected": (
                clip_longgun_detections > 0
            ),
        })

    frame_metrics = calculate_metrics(
        global_tp,
        global_tn,
        global_fp,
        global_fn,
    )

    clip_tp = clip_tn = clip_fp = clip_fn = 0

    for result in clip_results:
        gt = result["ground_truth"] == "weapon"
        predicted = result["clip_weapon_detected"]

        if gt and predicted:
            clip_tp += 1
        elif gt:
            clip_fn += 1
        elif predicted:
            clip_fp += 1
        else:
            clip_tn += 1

    return {
        "frame_results": frame_results,
        "clip_results": clip_results,
        "frame_level": {
            "TP": global_tp,
            "TN": global_tn,
            "FP": global_fp,
            "FN": global_fn,
            **frame_metrics,
        },
        "longgun": {
            "TP": global_longgun_tp,
            "FN": global_longgun_fn,
            "FP": global_longgun_fp,
            "recall": safe_divide(
                global_longgun_tp,
                global_longgun_tp
                + global_longgun_fn,
            ),
        },
        "clip_level": {
            "TP": clip_tp,
            "TN": clip_tn,
            "FP": clip_fp,
            "FN": clip_fn,
            **calculate_metrics(
                clip_tp,
                clip_tn,
                clip_fp,
                clip_fn,
            ),
        },
    }


def save_results(output_root, result):
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_results = result["frame_results"]
    clip_results = result["clip_results"]

    if frame_results:
        frame_path = (
            output_root / "frame_level_results.csv"
        )
        with frame_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=frame_results[0].keys(),
            )
            writer.writeheader()
            writer.writerows(frame_results)

    if clip_results:
        clip_path = (
            output_root / "clip_level_results.csv"
        )
        with clip_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=clip_results[0].keys(),
            )
            writer.writeheader()
            writer.writerows(clip_results)

    summary = {
        "taxonomy": CLASS_NAMES,
        "frame_level": result["frame_level"],
        "longgun": result["longgun"],
    }

    if "clip_level" in result:
        summary["clip_level"] = result["clip_level"]

    (output_root / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the V2.1 weapon detector on the "
            "held-out aerial dataset using full-frame "
            "or person-ROI analysis."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["full-frame", "person-roi"],
        required=True,
    )
    parser.add_argument(
        "--aerial-root",
        required=True,
    )
    parser.add_argument(
        "--weapon-model",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--weapon-imgsz",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--weapon-conf",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--weapon-iou",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--max-saved-images",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--person-model",
        default="yolo26n.pt",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--person-iou",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--roi-expansion",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--min-person-conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--single-person",
        action="store_true",
        help="Process only the first detected person in each frame.",
    )
    parser.add_argument(
        "--no-roi-images",
        action="store_true",
    )

    args = parser.parse_args()

    aerial_root = Path(args.aerial_root)
    output = Path(args.output)

    if not aerial_root.exists():
        raise FileNotFoundError(
            f"Aerial dataset not found: {aerial_root}"
        )

    print("=" * 70)
    print("WEAPON DETECTION — AERIAL EVALUATION")
    print("=" * 70)
    print(f"Mode   : {args.mode}")
    print(f"Dataset: {aerial_root}")

    weapon_model = YOLO(
        str(args.weapon_model)
    )

    if args.mode == "full-frame":
        result = evaluate_full_frame(
            weapon_model=weapon_model,
            aerial_root=aerial_root,
            output_root=output,
            image_size=args.weapon_imgsz,
            confidence=args.weapon_conf,
            iou=args.weapon_iou,
            device=args.device,
            max_saved_images=args.max_saved_images,
        )
    else:
        person_model = YOLO(
            str(args.person_model)
        )

        result = evaluate_person_roi(
            person_model=person_model,
            weapon_model=weapon_model,
            aerial_root=aerial_root,
            output_root=output,
            person_image_size=args.person_imgsz,
            person_confidence=args.person_conf,
            person_iou=args.person_iou,
            weapon_image_size=args.weapon_imgsz,
            weapon_confidence=args.weapon_conf,
            weapon_iou=args.weapon_iou,
            device=args.device,
            roi_expansion=args.roi_expansion,
            process_all_people=not args.single_person,
            min_person_confidence=args.min_person_conf,
            max_saved_images=args.max_saved_images,
            save_roi_images=not args.no_roi_images,
        )

    save_results(
        output,
        result,
    )

    print("\nFRAME-LEVEL")
    print(result["frame_level"])

    if "clip_level" in result:
        print("\nCLIP-LEVEL")
        print(result["clip_level"])

    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
