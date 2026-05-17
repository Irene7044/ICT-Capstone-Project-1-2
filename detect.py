from pathlib import Path
import sys
import shutil
import cv2
import numpy as np
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

# Important:
# The model file name must be exactly bike_lane.pt.
# Do not use bike_lane(1).pt in the code.
BIKE_LANE_MODEL = ROOT / "models" / "bike_lane.pt"

TRAFFIC_SIGN_MODEL = ROOT / "models" / "traffic_sign.pt"


# =========================
# Base confidence settings
# =========================

FOOTPATH_CONF = 0.15
ROAD_CONF = 0.05

TRUNK_CONF = 0.55
POLE_CONF = 0.25
TRAFFIC_LIGHT_CONF = 0.35
ROAD_BARRIER_CONF = 0.25

# Modified bike lane model
BIKE_LANE_CONF = 0.25

# Traffic sign model has many classes.
# Keep base confidence low, then apply class-specific thresholds.
TRAFFIC_SIGN_CONF = 0.05


# =========================
# Per-class confidence settings
# =========================

ROAD_CLASS_CONF = {
    "crosswalk_marking": 0.55,
    "lane_marking": 0.40,
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

TRAFFIC_SIGN_COLOR = (0, 128, 255)


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
        # Only keep actual traffic_light detections.
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
        "force_label": None,
        "allowed_labels": TRAFFIC_SIGN_LABELS,

        # Apply per-class confidence thresholds.
        "class_conf": TRAFFIC_SIGN_CLASS_CONF,
        "alpha": 0.45,
    },
]


# =========================
# Helper functions
# =========================

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
    if labels is None:
        return None

    return {normalize_label_name(label) for label in labels}


def normalize_conf_dict(conf_dict):
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

    return True


def process_video(file_path, loaded_models):
    """
    Process one video file.
    """
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

    frame_count = 0
    processed_count = 0
    total_counts = {}

    last_result = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, VIDEO_SIZE)

        if frame_count % SKIP_FRAMES == 0 or last_result is None:
            result, counts = process_frame(frame, loaded_models)
            last_result = result

            total_counts = merge_counts(total_counts, counts)
            processed_count += 1

        else:
            result = last_result

        writer.write(result)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Frames read: {frame_count}, frames detected: {processed_count}")

    cap.release()
    writer.release()

    print(f"Video: {file_path.name}")
    print(f"Frames read: {frame_count}")
    print(f"Frames actually detected: {processed_count}")
    print_counts("Total detections:", total_counts)
    print(f"Output: {output_path}")

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