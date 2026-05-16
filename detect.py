from pathlib import Path
import sys
import shutil
import cv2
import numpy as np
import csv
from ultralytics import YOLO



# =========================
# Basic paths
# =========================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
PROCESSED_DIR = INPUT_DIR / "processed"

FOOTPATH_MODEL = ROOT / "models" / "footpath_best.pt"
ROAD_MODEL = ROOT / "models" / "road_evidence_best.pt"
TRUNK_MODEL = ROOT / "models" / "trunk.pt"
POLE_MODEL = ROOT / "models" / "pole.pt"
TRAFFIC_LIGHT_MODEL = ROOT / "models" / "trafficlight.pt"
ROAD_BARRIER_MODEL = ROOT / "models" / "road_barriers.pt"
BIKE_LANE_MODEL = ROOT / "models" / "bike_lane.pt"
TRAFFIC_SIGN_MODEL = ROOT / "models" / "traffic_sign.pt"


# =========================
# Base confidence settings
# =========================
# Important:
# For models with per-class confidence, use a low base confidence first.
# Then filter by class-specific thresholds later.

FOOTPATH_CONF = 0.10

ROAD_CONF = 0.05

TRUNK_CONF = 0.55
POLE_CONF = 0.25

# YOLO default traffic light model.
# Keep this higher because we only want reliable traffic light boxes.
TRAFFIC_LIGHT_CONF = 0.35

ROAD_BARRIER_CONF = 0.25
BIKE_LANE_CONF = 0.25

# Traffic sign model has many classes.
# Keep base confidence low, then apply TRAFFIC_SIGN_CLASS_CONF below.
TRAFFIC_SIGN_CONF = 0.05


# =========================
# Per-class confidence settings
# =========================

ROAD_CLASS_CONF = {
    "crosswalk_marking": 0.60,
    "lane_marking": 0.60,
    "stop_line": 0.01,
}

TRAFFIC_SIGN_CLASS_CONF = {
    "bicycle_sign": 0.25,
    "curve_ahead": 0.25,
    "give_way": 0.20,
    "information": 0.30,
    "intersection": 0.25,
    "keep_left_right": 0.25,
    "no_entry": 0.30,
    "no_parking": 0.30,
    "no_stopping": 0.30,
    "one_way": 0.30,
    "other_sign": 0.45,
    "parking": 0.30,
    "pedestrian_crossing": 0.25,
    "railroad_crossing": 0.20,
    "road_works": 0.25,
    "roundabout": 0.25,
    "school_zone": 0.20,
    "speed_limit": 0.35,
    "stop_sign": 0.25,
    "traffic_signals": 0.30,
    "turn_restriction": 0.30,
    "vehicle_restriction": 0.30,
    "warning": 0.35,
    "wildlife_crossing": 0.20,
}


# =========================
# Video and file settings
# =========================

VIDEO_SIZE = (640, 360)
SKIP_FRAMES = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# =========================
# Labels
# =========================

ROAD_EVIDENCE_LABELS = {
    "crosswalk_marking",
    "lane_marking",
    "stop_line",
}

TRAFFIC_LIGHT_LABELS = {
    "traffic_light",
}

TRAFFIC_SIGN_LABELS = {
    "bicycle_sign",
    "curve_ahead",
    "give_way",
    "information",
    "intersection",
    "keep_left_right",
    "no_entry",
    "no_parking",
    "no_stopping",
    "one_way",
    "other_sign",
    "parking",
    "pedestrian_crossing",
    "railroad_crossing",
    "road_works",
    "roundabout",
    "school_zone",
    "speed_limit",
    "stop_sign",
    "traffic_signals",
    "turn_restriction",
    "vehicle_restriction",
    "warning",
    "wildlife_crossing",
}


# =========================
# Colors
# =========================
# OpenCV uses BGR color order.

COLOR_MAP = {
    "footpath": (0, 255, 0),              # green
    "crosswalk_marking": (255, 0, 0),     # blue
    "lane_marking": (0, 0, 255),          # red
    "stop_line": (0, 255, 255),           # yellow
    "trunk": (0, 165, 255),               # orange
    "pole": (255, 0, 255),                # purple
    "traffic_light": (255, 255, 0),       # cyan
    "road_barrier": (255, 128, 0),        # light blue
    "bike_lane": (128, 0, 255),           # violet
}

TRAFFIC_SIGN_COLOR = (0, 128, 255)        # orange-like


# =========================
# Model configuration
# =========================

MODEL_CONFIGS = [
    {
        "name": "footpath",
        "path": FOOTPATH_MODEL,
        "conf": FOOTPATH_CONF,
        "force_label": "footpath",
        "allowed_labels": None,
        "class_conf": None,
        "alpha": 0.40,
    },
    {
        "name": "road_evidence",
        "path": ROAD_MODEL,
        "conf": ROAD_CONF,
        "force_label": None,
        "allowed_labels": ROAD_EVIDENCE_LABELS,
        "class_conf": ROAD_CLASS_CONF,
        "alpha": 0.50,
    },
    {
        "name": "trunk",
        "path": TRUNK_MODEL,
        "conf": TRUNK_CONF,
        "force_label": "trunk",
        "allowed_labels": None,
        "class_conf": None,
        "alpha": 0.45,
    },
    {
        "name": "pole",
        "path": POLE_MODEL,
        "conf": POLE_CONF,
        "force_label": "pole",
        "allowed_labels": None,
        "class_conf": None,
        "alpha": 0.45,
    },
    {
        "name": "traffic_light",
        "path": TRAFFIC_LIGHT_MODEL,
        "conf": TRAFFIC_LIGHT_CONF,

        # Do not force all YOLO default detections to traffic_light.
        # Only keep actual traffic light detections.
        "force_label": None,
        "allowed_labels": TRAFFIC_LIGHT_LABELS,

        "class_conf": None,
        "alpha": 0.45,
    },
    {
        "name": "road_barrier",
        "path": ROAD_BARRIER_MODEL,
        "conf": ROAD_BARRIER_CONF,
        "force_label": "road_barrier",
        "allowed_labels": None,
        "class_conf": None,
        "alpha": 0.45,
    },
    {
        "name": "bike_lane",
        "path": BIKE_LANE_MODEL,
        "conf": BIKE_LANE_CONF,
        "force_label": "bike_lane",
        "allowed_labels": None,
        "class_conf": None,
        "alpha": 0.45,
    },
    {
        "name": "traffic_sign",
        "path": TRAFFIC_SIGN_MODEL,
        "conf": TRAFFIC_SIGN_CONF,

        # Keep original traffic sign class names.
        # Example: speed_limit, stop_sign, no_parking, warning.
        "force_label": None,
        "allowed_labels": TRAFFIC_SIGN_LABELS,

        # Apply per-class confidence for each traffic sign type.
        "class_conf": TRAFFIC_SIGN_CLASS_CONF,
        "alpha": 0.45,
    },
]


# =========================
# Helper functions
# =========================

def save_report_csv(file_path, counts):
    """
    Save detection counts into reports folder.
    """

    report_folder = REPORT_DIR / file_path.stem
    report_folder.mkdir(parents=True, exist_ok=True)

    csv_path = report_folder / "detection_report.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Class", "Count"])

        for label, count in counts.items():
            writer.writerow([label, count])

    print(f"CSV report saved: {csv_path}")

def normalize_label_name(label):
    """
    Normalize YOLO class names.

    Examples:
        "traffic light" -> "traffic_light"
        "0_crosswalk_marking" -> "crosswalk_marking"
        "1_lane_marking" -> "lane_marking"
        "2_stop_line" -> "stop_line"
    """
    text = str(label).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    parts = text.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        text = parts[1]

    return text


def normalize_label_set(labels):
    """
    Normalize every label in a set.
    """
    if labels is None:
        return None

    return {normalize_label_name(label) for label in labels}


def normalize_conf_dict(conf_dict):
    """
    Normalize keys in a confidence dictionary.
    """
    if conf_dict is None:
        return None

    return {
        normalize_label_name(label): conf
        for label, conf in conf_dict.items()
    }


def get_label_name(result, class_id):
    """
    Convert YOLO class id to normalized class name.
    """
    try:
        raw_label = result.names[int(class_id)]
    except Exception:
        raw_label = str(class_id)

    return normalize_label_name(raw_label)


def get_color(label):
    """
    Get drawing color for a label.
    """
    label = normalize_label_name(label)

    if label in TRAFFIC_SIGN_LABELS:
        return TRAFFIC_SIGN_COLOR

    return COLOR_MAP.get(label, (255, 255, 255))


def should_keep_detection(label, conf, class_conf=None):
    """
    Apply class-specific confidence threshold.

    If class_conf is None, keep all detections that passed YOLO's base conf.
    If class_conf exists, use the threshold for that class.
    """
    if class_conf is None:
        return True

    label = normalize_label_name(label)
    class_conf = normalize_conf_dict(class_conf)

    threshold = class_conf.get(label, 0.25)
    return conf >= threshold


def put_label(frame, text, x, y, color):
    """
    Draw readable label text with black outline.
    """
    y = max(20, y)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_masks_and_boxes(
    frame,
    result,
    allowed_labels=None,
    force_display_label=None,
    alpha=0.45,
    draw_contours=True,
    draw_boxes=True,
    class_conf=None,
):
    """
    Draw YOLO result on frame.

    Supports:
        1. segmentation models with masks
        2. detection models with boxes only
    """
    output = frame.copy()
    overlay = frame.copy()
    counts = {}

    h, w = frame.shape[:2]

    boxes = result.boxes
    masks = result.masks

    if boxes is None:
        return output, counts

    allowed_labels = normalize_label_set(allowed_labels)
    class_conf = normalize_conf_dict(class_conf)

    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    # =========================
    # Draw segmentation masks
    # =========================

    if masks is not None:
        mask_data = masks.data.cpu().numpy()

        for i, mask_raw in enumerate(mask_data):
            class_id = class_ids[i]
            conf = float(confs[i])

            original_label = get_label_name(result, class_id)

            if force_display_label is not None:
                display_label = normalize_label_name(force_display_label)
            else:
                display_label = original_label

            if allowed_labels is not None and original_label not in allowed_labels:
                continue

            if not should_keep_detection(original_label, conf, class_conf):
                continue

            color = get_color(display_label)

            mask_resized = cv2.resize(mask_raw, (w, h))
            mask = mask_resized > 0.5

            mask_area = int(mask.sum())
            frame_area = h * w
            area_ratio = mask_area / max(frame_area, 1)

            if area_ratio < 0.00003:
                continue

            counts[display_label] = counts.get(display_label, 0) + 1

            if display_label == "lane_marking":
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color, 3)

            elif display_label == "stop_line":
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color, 4)

                overlay[mask] = (
                    0.65 * overlay[mask]
                    + 0.35 * np.array(color, dtype=np.uint8)
                ).astype(np.uint8)

            elif display_label in {
                "trunk",
                "pole",
                "traffic_light",
                "road_barrier",
                "bike_lane",
            } or display_label in TRAFFIC_SIGN_LABELS:
                overlay[mask] = (
                    (1 - alpha) * overlay[mask]
                    + alpha * np.array(color, dtype=np.uint8)
                ).astype(np.uint8)

                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color, 3)

            else:
                overlay[mask] = (
                    (1 - alpha) * overlay[mask]
                    + alpha * np.array(color, dtype=np.uint8)
                ).astype(np.uint8)

                if draw_contours:
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask_uint8,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(overlay, contours, -1, color, 2)

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

            put_label(
                overlay,
                f"{display_label} {conf:.2f}",
                x1,
                y1 - 6,
                color,
            )

    # =========================
    # Draw boxes only
    # =========================

    else:
        for i, b in enumerate(boxes):
            class_id = int(b.cls[0])
            conf = float(b.conf[0])

            original_label = get_label_name(result, class_id)

            if force_display_label is not None:
                display_label = normalize_label_name(force_display_label)
            else:
                display_label = original_label

            if allowed_labels is not None and original_label not in allowed_labels:
                continue

            if not should_keep_detection(original_label, conf, class_conf):
                continue

            counts[display_label] = counts.get(display_label, 0) + 1

            color = get_color(display_label)
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

            if draw_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            put_label(
                overlay,
                f"{display_label} {conf:.2f}",
                x1,
                y1 - 6,
                color,
            )

    output = overlay
    return output, counts


def merge_counts(a, b):
    """
    Merge two count dictionaries.
    """
    merged = dict(a)

    for key, value in b.items():
        merged[key] = merged.get(key, 0) + value

    return merged


def print_counts(title, counts):
    """
    Print detection counts.
    """
    print(title)

    if not counts:
        print("  No detections")
        return

    for label, count in counts.items():
        print(f"  {label}: {count}")


def get_output_path(file_path):
    """
    Generate output path in results folder.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if file_path.suffix.lower() in IMAGE_EXTS:
        return OUTPUT_DIR / f"detected_{file_path.stem}.png"

    return OUTPUT_DIR / f"detected_{file_path.stem}.mp4"


# =========================
# Model loading
# =========================

def load_models():
    """
    Load all available models.

    Missing models are skipped.
    At least one model must exist.
    """
    loaded_models = []

    for config in MODEL_CONFIGS:
        model_path = config["path"]

        if not model_path.exists():
            print(f"{config['name']} model not found, skipped: {model_path}")
            continue

        print(f"Using {config['name']} model: {model_path}")

        model = YOLO(str(model_path))
        print(f"{config['name']} model classes: {model.names}")

        loaded_models.append({
            "config": config,
            "model": model,
        })

    if not loaded_models:
        expected = "\n".join(str(config["path"]) for config in MODEL_CONFIGS)
        raise FileNotFoundError(
            "No detection models found. Please place at least one model in models folder:\n"
            + expected
        )

    return loaded_models


# =========================
# Core processing
# =========================

def process_frame(frame, loaded_models):
    """
    Process one frame using all loaded models.
    """
    result_frame = frame.copy()
    total_counts = {}

    for item in loaded_models:
        config = item["config"]
        model = item["model"]

        result = model.predict(
            source=frame,
            conf=config["conf"],
            save=False,
            verbose=False,
        )[0]

        result_frame, counts = draw_masks_and_boxes(
            frame=result_frame,
            result=result,
            allowed_labels=config["allowed_labels"],
            force_display_label=config["force_label"],
            alpha=config["alpha"],
            draw_contours=True,
            draw_boxes=True,
            class_conf=config["class_conf"],
        )

        total_counts = merge_counts(total_counts, counts)

    return result_frame, total_counts


def process_image(file_path, loaded_models):
    """
    Process one image file.
    """
    frame = cv2.imread(str(file_path))

    if frame is None:
        print(f"Cannot read image: {file_path}")
        return False

    result, counts = process_frame(frame, loaded_models)

    output_path = get_output_path(file_path)
    cv2.imwrite(str(output_path), result)

    print(f"Image: {file_path.name}")
    print_counts("Detections:", counts)
    print(f"Output: {output_path}")

    save_report_csv(file_path, counts)

    return True


def process_video(file_path, loaded_models):
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        print(f"Cannot open video: {file_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    output_path = get_output_path(file_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        VIDEO_SIZE,
    )

    if not writer.isOpened():
        cap.release()
        print(f"Cannot create output video: {output_path}")
        return False

    # label -> set of unique integer track IDs seen so far
    seen_ids = {}

    # (display_label, track_id) -> snapshot frame captured on first detection
    first_seen_frames = {}

    frame_count = 0
    last_result_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, VIDEO_SIZE)

        if frame_count % SKIP_FRAMES == 0 or last_result_frame is None:
            result_frame = frame.copy()

            for item in loaded_models:
                config = item["config"]
                model  = item["model"]

                # Track individual detected elements
                track_results = model.track(
                    source=frame,
                    conf=config["conf"],
                    persist=True,
                    tracker="bytetrack.yaml", # Ultralystics library feature -> keep same ID for same object across frames
                    save=False,
                    verbose=False,
                )

                if not track_results:
                    continue

                result = track_results[0]

                result_frame, _ = draw_masks_and_boxes(
                    frame=result_frame,
                    result=result,
                    allowed_labels=config["allowed_labels"],
                    force_display_label=config["force_label"],
                    alpha=config["alpha"],
                    draw_contours=True,
                    draw_boxes=True,
                    class_conf=config["class_conf"],
                )

                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue

                track_ids = boxes.id.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confs     = boxes.conf.cpu().numpy()

                allowed    = normalize_label_set(config["allowed_labels"])
                class_conf = normalize_conf_dict(config["class_conf"])

                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)

                    label = get_label_name(result, class_ids[i])
                    conf  = float(confs[i])

                    if allowed is not None and label not in allowed:
                        continue
                    if not should_keep_detection(label, conf, class_conf):
                        continue

                    display_label = (
                        normalize_label_name(config["force_label"])
                        if config["force_label"]
                        else label
                    )

                    if display_label not in seen_ids:
                        seen_ids[display_label] = set()

                    # Check BEFORE adding so we can capture the first frame
                    is_new = track_id not in seen_ids[display_label]
                    seen_ids[display_label].add(track_id)

                    if is_new:
                        obj_key = (display_label, track_id)
                        first_seen_frames[obj_key] = frame.copy()

            last_result_frame = result_frame

        writer.write(last_result_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Frames processed: {frame_count}")

    # Release handles before doing anything else
    cap.release()
    writer.release()

    # Save first-seen snapshots for verification
    snapshot_dir = REPORT_DIR / file_path.stem / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for (label, tid), snap_frame in first_seen_frames.items():
        snap_path = snapshot_dir / f"{label}_id{tid}.jpg"
        cv2.imwrite(str(snap_path), snap_frame)

    print(f"Snapshots saved to: {snapshot_dir}")

    # Convert seen ID sets to final counts
    total_counts = {label: len(ids) for label, ids in seen_ids.items()}

    print(f"Video: {file_path.name}")
    print(f"Frames processed: {frame_count}")
    print_counts("Unique detections:", total_counts)
    print(f"Output: {output_path}")

    save_report_csv(file_path, total_counts)
    return True


def move_to_processed(file_path):
    """
    Move processed file to uploads/processed.
    """
    PROCESSED_DIR.mkdir(exist_ok=True)

    target_path = PROCESSED_DIR / file_path.name

    if target_path.exists():
        target_path = PROCESSED_DIR / f"{file_path.stem}_processed{file_path.suffix}"

    shutil.move(str(file_path), str(target_path))
    print(f"Moved original file to: {target_path}")


# =========================
# Public detection function
# =========================

def run_detection(input_path, move_original=False):
    """
    Process one image or video.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    loaded_models = load_models()

    if suffix in IMAGE_EXTS:
        success = process_image(input_path, loaded_models)
    else:
        success = process_video(input_path, loaded_models)

    if success and move_original:
        move_to_processed(input_path)

    return success


def process_all_uploads():
    """
    Process all images/videos in uploads folder.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    files = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file()
        and (
            f.suffix.lower() in IMAGE_EXTS
            or f.suffix.lower() in VIDEO_EXTS
        )
    ]

    if not files:
        print("No image or video files found in uploads.")
        return

    loaded_models = load_models()

    for file_path in files:
        suffix = file_path.suffix.lower()

        success = False

        if suffix in IMAGE_EXTS:
            success = process_image(file_path, loaded_models)

        elif suffix in VIDEO_EXTS:
            success = process_video(file_path, loaded_models)

        if success:
            move_to_processed(file_path)


# =========================
# Main script mode
# =========================

def main():
    """
    Usage:
        python detect.py
        python detect.py uploads/test.mp4
        python detect.py "E:/path/to/image.jpg"
    """
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        run_detection(input_path, move_original=False)
    else:
        process_all_uploads()


if __name__ == "__main__":
    main()