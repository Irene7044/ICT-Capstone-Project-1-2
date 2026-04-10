from ultralytics import YOLO
import cv2
import os
import csv

MODEL_PATH = "models/pole_best.pt"
CONF_THRESHOLD = 0.50

model = YOLO(MODEL_PATH)


def ensure_folder(path):
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(file_path, fieldnames, rows):
    ensure_folder(os.path.dirname(file_path))

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_and_collect(frame, source_file, source_type, frame_index=None, timestamp_sec=None):
    results = model(frame)

    detections = []
    pole_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = str(model.names[cls]).lower()
            conf = float(box.conf[0])

            if label == "pole" and conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                detections.append({
                    "source_file": source_file,
                    "source_type": source_type,
                    "frame_index": frame_index if frame_index is not None else "",
                    "timestamp_sec": round(timestamp_sec, 3) if timestamp_sec is not None else "",
                    "class_name": label,
                    "confidence": round(conf, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_width": box_width,
                    "box_height": box_height,
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2)
                })

                pole_count += 1

    summary_row = {
        "source_file": source_file,
        "source_type": source_type,
        "frame_index": frame_index if frame_index is not None else "",
        "timestamp_sec": round(timestamp_sec, 3) if timestamp_sec is not None else "",
        "pole_count": pole_count
    }

    return frame, detections, summary_row


def detect_image(input_image, output_image):
    img = cv2.imread(input_image)

    if img is None:
        raise ValueError(f"Could not read image: {input_image}")

    source_file = os.path.basename(input_image)
    source_type = "image"

    result_img, detections, summary_row = annotate_and_collect(
        img,
        source_file=source_file,
        source_type=source_type
    )

    ensure_folder(os.path.dirname(output_image))
    cv2.imwrite(output_image, result_img)

    csv_folder = "results/csv"
    detections_csv = os.path.join(csv_folder, "pole_detections.csv")
    summary_csv = os.path.join(csv_folder, "pole_summary.csv")

    detection_fields = [
        "source_file", "source_type", "frame_index", "timestamp_sec",
        "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "box_width", "box_height", "center_x", "center_y"
    ]

    summary_fields = [
        "source_file", "source_type", "frame_index", "timestamp_sec", "pole_count"
    ]

    write_csv(detections_csv, detection_fields, detections)
    write_csv(summary_csv, summary_fields, [summary_row])

    print(f"Pole image result saved to {output_image}")
    print(f"Detection CSV saved to {detections_csv}")
    print(f"Summary CSV saved to {summary_csv}")


def detect_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    ensure_folder(os.path.dirname(output_video))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    source_file = os.path.basename(input_video)
    source_type = "video"

    all_detections = []
    all_summary = []

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_index / fps

        result_frame, detections, summary_row = annotate_and_collect(
            frame,
            source_file=source_file,
            source_type=source_type,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec
        )

        out.write(result_frame)
        all_detections.extend(detections)
        all_summary.append(summary_row)

        frame_index += 1

    cap.release()
    out.release()

    csv_folder = "results/csv"
    detections_csv = os.path.join(csv_folder, "pole_detections.csv")
    summary_csv = os.path.join(csv_folder, "pole_summary.csv")

    detection_fields = [
        "source_file", "source_type", "frame_index", "timestamp_sec",
        "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "box_width", "box_height", "center_x", "center_y"
    ]

    summary_fields = [
        "source_file", "source_type", "frame_index", "timestamp_sec", "pole_count"
    ]

    write_csv(detections_csv, detection_fields, all_detections)
    write_csv(summary_csv, summary_fields, all_summary)

    print(f"Pole video result saved to {output_video}")
    print(f"Detection CSV saved to {detections_csv}")
    print(f"Summary CSV saved to {summary_csv}")


def detect_pole(input_path, output_path):
    lower_path = input_path.lower()

    if lower_path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        detect_image(input_path, output_path)
    elif lower_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
        detect_video(input_path, output_path)
    else:
        raise ValueError("Unsupported file type")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Change these for your test
    input_path = "testing resource/1mnt_poles_sample.mp4"
    output_path = "results/pole_result_train4.mp4"

    detect_pole(input_path, output_path)