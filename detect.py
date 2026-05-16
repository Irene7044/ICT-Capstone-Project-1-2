from ultralytics import YOLO
import cv2
import struct
import subprocess
import json
import os
import csv
from datetime import datetime

# ── Models ───────────────────────────────────────────────────
model      = YOLO("models/yolov8n.pt")       # traffic lights
sign_model = YOLO("models/traffic_sign.pt")          # traffic signs

TRAFFIC_LIGHT_COLOR = (0, 255, 0)    # green
TRAFFIC_SIGN_COLOR  = (0, 165, 255)  # orange

# ── GPS Extraction ───────────────────────────────────────────

def extract_gps_track(video_path):
    """
    Extracts GPS points from GoPro GPMF metadata using ffmpeg.
    Returns a list of dicts: [{time_s, lat, lon, alt}, ...]
    time_s is seconds from start of video.
    Returns empty list if no GPS data found.
    """
    print(f"\n[GPS] Checking for GPS metadata in: {video_path}")

    # Step 1: check if file has a gpmd stream using ffprobe
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", video_path],
            capture_output=True, text=True, timeout=30
        )
        probe_data = json.loads(probe.stdout)
        streams    = probe_data.get("streams", [])

        gpmd_index = None
        for s in streams:
            tags = s.get("tags", {})
            if "gpmd" in s.get("codec_tag_string", "").lower() or \
               tags.get("handler_name", "").lower() in ["gopro met", "gpmf"]:
                gpmd_index = s["index"]
                print(f"[GPS] ✅ Found GPMF stream at index {gpmd_index}")
                break

        if gpmd_index is None:
            print("[GPS] ❌ No GPMF stream found — video has no GPS metadata")
            print("[GPS]    This may not be a GoPro video, or GPS was disabled")
            return []

    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[GPS] ❌ ffprobe failed: {e}")
        print("[GPS]    Make sure ffmpeg is installed: sudo apt install ffmpeg")
        return []

    # Step 2: extract the raw GPMF binary stream
    bin_path = video_path + ".gpmd.bin"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-map", f"0:{gpmd_index}",
             "-c", "copy", "-f", "rawvideo", bin_path],
            capture_output=True, timeout=120
        )
        print(f"[GPS] Raw GPMF data extracted to: {bin_path}")
    except subprocess.TimeoutExpired:
        print("[GPS] ❌ ffmpeg timed out extracting GPMF stream")
        return []

    # Step 3: parse the binary GPMF data for GPS5 samples
    # GPS5 format: lat, lon, alt, speed2d, speed3d (all scaled)
    gps_points = []
    try:
        with open(bin_path, "rb") as f:
            data = f.read()

        print(f"[GPS] Parsing {len(data)} bytes of GPMF data...")

        i = 0
        current_time = 0.0
        scale        = 1

        while i < len(data) - 8:
            try:
                # GPMF KLV: 4-char key, 1-char type, 1-char size, 2-char repeat
                key    = data[i:i+4].decode("ascii", errors="ignore")
                type_  = chr(data[i+4]) if data[i+4] != 0 else "\x00"
                size   = data[i+5]
                repeat = struct.unpack(">H", data[i+6:i+8])[0]
                length = size * repeat
                # Align to 4 bytes
                padded = (length + 3) & ~3
                payload = data[i+8:i+8+length]

                if key == "SCAL" and type_ == "L" and size == 4:
                    scale = struct.unpack(">i", payload[:4])[0]
                    if scale == 0:
                        scale = 1

                elif key == "GPS5" and type_ == "l" and size == 20:
                    # Each sample: lat, lon, alt, speed2d, speed3d (int32 big endian)
                    num_samples = repeat
                    for s in range(num_samples):
                        offset = s * 20
                        if offset + 20 > len(payload):
                            break
                        lat, lon, alt, spd2, spd3 = struct.unpack(
                            ">iiiii", payload[offset:offset+20]
                        )
                        gps_points.append({
                            "time_s": round(current_time, 3),
                            "lat":    lat / scale,
                            "lon":    lon / scale,
                            "alt":    alt / scale,
                        })
                    # GPS5 samples are roughly 18Hz — increment time accordingly
                    current_time += repeat / 18.0

                elif key == "GPSU":
                    # GPS timestamp — could be used for absolute time if needed
                    pass

                i += 8 + padded

            except (struct.error, UnicodeDecodeError):
                i += 1  # skip bad byte and keep scanning

        print(f"[GPS] ✅ Extracted {len(gps_points)} GPS points")
        if gps_points:
            first = gps_points[0]
            last  = gps_points[-1]
            print(f"[GPS]    First point: {first['lat']:.6f}, {first['lon']:.6f} at t={first['time_s']}s")
            print(f"[GPS]    Last point:  {last['lat']:.6f}, {last['lon']:.6f} at t={last['time_s']}s")

    except Exception as e:
        print(f"[GPS] ❌ Failed to parse GPMF data: {e}")
        gps_points = []
    finally:
        # Clean up temp binary file
        if os.path.exists(bin_path):
            os.remove(bin_path)

    return gps_points


def get_gps_at_time(gps_track, time_s):
    """
    Given a GPS track list and a time in seconds,
    returns the closest GPS point to that time.
    Returns None if track is empty.
    """
    if not gps_track:
        return None
    closest = min(gps_track, key=lambda p: abs(p["time_s"] - time_s))
    return closest


def fix_mp4(video_path):
    """
    Fixes MP4 files where the moov atom is at the end (common with trimmed files).
    Returns path to fixed file, or original path if fix not needed/failed.
    """
    fixed_path = video_path + "_fixed.mp4"
    print(f"[Fix] Attempting to fix MP4 structure: {video_path}")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-c", "copy",
             "-movflags", "faststart",  # moves moov atom to front
             fixed_path],
            capture_output=True, text=True, timeout=120
        )
        if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
            print(f"[Fix] ✅ Fixed MP4 saved to: {fixed_path}")
            return fixed_path
        else:
            print(f"[Fix] ❌ Fix failed, using original")
            return video_path
    except Exception as e:
        print(f"[Fix] ❌ Error: {e}")
        return video_path
    
    
# ── Frame analysis ───────────────────────────────────────────

def _run_models_on_frame(frame):
    """Runs both models on a single frame and returns annotated frame."""

    # Traffic lights
    for r in model(frame):
        for box in r.boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            if label.lower() == "traffic light":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_LIGHT_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRAFFIC_LIGHT_COLOR, 2)

    # Traffic signs
    for r in sign_model.track(frame, persist=True):
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                cls   = int(box.cls[0])
                label = sign_model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_SIGN_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRAFFIC_SIGN_COLOR, 2)

    return frame


def _run_models_on_frame_with_detections(frame):
    """
    Same as _run_models_on_frame but also returns detection results
    for report generation.
    Returns: (annotated_frame, traffic_lights, traffic_signs)
    Each list contains dicts: {label, confidence, track_id}
    """
    traffic_lights = []
    traffic_signs  = []

    # Traffic lights
    for r in model(frame):
        for box in r.boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            if label.lower() == "traffic light":
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_LIGHT_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRAFFIC_LIGHT_COLOR, 2)
                traffic_lights.append({"label": label, "confidence": round(conf, 3)})

    # Traffic signs with tracking
    for r in sign_model.track(frame, persist=True):
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                cls      = int(box.cls[0])
                label    = sign_model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_SIGN_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRAFFIC_SIGN_COLOR, 2)
                traffic_signs.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "track_id":   track_id
                })

    return frame, traffic_lights, traffic_signs


# ── Report generation ────────────────────────────────────────
def generate_report(input_path, report_folder, is_video=True):
    os.makedirs(report_folder, exist_ok=True)
    basename    = os.path.splitext(os.path.basename(input_path))[0]
    report_csv  = os.path.join(report_folder, f"{basename}_report.csv")
    report_json = os.path.join(report_folder, f"{basename}_report.json")

    if not is_video:
        # ── Image report ─────────────────────────────────────
        print(f"\n[Report] Processing image: {input_path}")
        img = cv2.imread(input_path)
        if img is None:
            print("[Report] ❌ Could not read image")
            return None

        _, traffic_lights, traffic_signs = _run_models_on_frame_with_detections(img)

        report_data = {
            "file":           os.path.basename(input_path),
            "type":           "image",
            "processed_at":   datetime.now().isoformat(),
            "traffic_lights": [{"label": d["label"], "confidence": d["confidence"]}
                               for d in traffic_lights],
            "traffic_signs":  [{"label": d["label"], "confidence": d["confidence"]}
                               for d in traffic_signs],
        }

        with open(report_json, "w") as f:
            json.dump(report_data, f, indent=2)

        with open(report_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "type", "model", "label", "confidence",
                             "latitude", "longitude"])
            for d in traffic_lights:
                writer.writerow([basename, "image", "traffic_light",
                                 d["label"], d["confidence"], "", ""])
            for d in traffic_signs:
                writer.writerow([basename, "image", "traffic_sign",
                                 d["label"], d["confidence"], "", ""])

        print(f"[Report] ✅ Image report saved:")
        print(f"         CSV:  {report_csv}")
        print(f"         JSON: {report_json}")
        return report_data

    else:
        # ── Video report ──────────────────────────────────────
        print(f"\n[Report] Processing video: {input_path}")

        # Fix moov atom issue if needed
        cap_test    = cv2.VideoCapture(input_path)
        fps_test    = cap_test.get(cv2.CAP_PROP_FPS)
        frames_test = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_test.release()

        if fps_test <= 0 or frames_test <= 0:
            print(f"[Report] ⚠️  File appears unreadable — attempting MP4 fix...")
            input_path = fix_mp4(input_path)

        # Extract GPS track
        gps_track = extract_gps_track(input_path)

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30
            print(f"[Report] Warning: could not read FPS, assuming {fps}")

        sign_tracks = {}
        light_seen  = {}
        frame_count = 0
        print(f"[Report] Processing {total_frames} frames at {fps:.1f} fps...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_count / fps
            _, traffic_lights, traffic_signs = _run_models_on_frame_with_detections(frame)

            for det in traffic_lights:
                label = det["label"]
                light_seen[label] = {
                    "last_frame":  frame_count,
                    "last_time_s": time_s,
                    "confidence":  det["confidence"]
                }

            for det in traffic_signs:
                tid = det["track_id"]
                if tid is None:
                    tid = f"notag_{det['label']}"
                sign_tracks[tid] = {
                    "label":       det["label"],
                    "last_frame":  frame_count,
                    "last_time_s": time_s,
                    "confidence":  det["confidence"]
                }

            frame_count += 1

        cap.release()
        print(f"[Report] Processed {frame_count} frames")

        traffic_light_rows = []
        for label, info in light_seen.items():
            gps = get_gps_at_time(gps_track, info["last_time_s"])
            traffic_light_rows.append({
                "label":       label,
                "confidence":  info["confidence"],
                "last_frame":  info["last_frame"],
                "last_time_s": round(info["last_time_s"], 3),
                "latitude":    gps["lat"] if gps else None,
                "longitude":   gps["lon"] if gps else None,
                "altitude":    gps["alt"] if gps else None,
            })

        traffic_sign_rows = []
        for tid, info in sign_tracks.items():
            gps = get_gps_at_time(gps_track, info["last_time_s"])
            traffic_sign_rows.append({
                "track_id":    tid,
                "label":       info["label"],
                "confidence":  info["confidence"],
                "last_frame":  info["last_frame"],
                "last_time_s": round(info["last_time_s"], 3),
                "latitude":    gps["lat"] if gps else None,
                "longitude":   gps["lon"] if gps else None,
                "altitude":    gps["alt"] if gps else None,
            })

        report_data = {
            "file":           os.path.basename(input_path),
            "type":           "video",
            "processed_at":   datetime.now().isoformat(),
            "total_frames":   frame_count,
            "fps":            round(fps, 2),
            "gps_available":  len(gps_track) > 0,
            "gps_points":     len(gps_track),
            "traffic_lights": traffic_light_rows,
            "traffic_signs":  traffic_sign_rows,
        }

        with open(report_json, "w") as f:
            json.dump(report_data, f, indent=2)

        with open(report_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "type", "model", "label", "track_id",
                "confidence", "last_frame", "last_time_s",
                "latitude", "longitude", "altitude"
            ])
            for d in traffic_light_rows:
                writer.writerow([
                    basename, "video", "traffic_light",
                    d["label"], "",
                    d["confidence"], d["last_frame"], d["last_time_s"],
                    d["latitude"], d["longitude"], d["altitude"]
                ])
            for d in traffic_sign_rows:
                writer.writerow([
                    basename, "video", "traffic_sign",
                    d["label"], d["track_id"],
                    d["confidence"], d["last_frame"], d["last_time_s"],
                    d["latitude"], d["longitude"], d["altitude"]
                ])

        # Clean up fixed file if one was created
        if input_path.endswith("_fixed.mp4") and os.path.exists(input_path):
            os.remove(input_path)
            print(f"[Fix] Cleaned up temporary fixed file")

        print(f"[Report] ✅ Video report saved:")
        print(f"         CSV:  {report_csv}")
        print(f"         JSON: {report_json}")
        print(f"         Traffic lights detected: {len(traffic_light_rows)}")
        print(f"         Traffic signs detected:  {len(traffic_sign_rows)}")
        return report_data


# ── Legacy functions kept for compatibility ───────────────────

def detect_traffic_light(input_path, output_path):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        detect_image(input_path, output_path)
    else:
        detect_video(input_path, output_path)


def detect_image(input_image, output_image):
    img = cv2.imread(input_image)
    img = _run_models_on_frame(img)
    cv2.imwrite(output_image, img)


def detect_video(input_video, output_video):
    cap    = cv2.VideoCapture(input_video)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    out    = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(_run_models_on_frame(frame))
    cap.release()
    out.release()