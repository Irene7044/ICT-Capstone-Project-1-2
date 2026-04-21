from ultralytics import YOLO
import cv2

# Model 1: traffic lights (existing)
model = YOLO("models/yolov8n.pt")
sign_model = YOLO("models/traffic_sign.pt")

TRAFFIC_LIGHT_COLOR = (0, 255, 0)   # green
TRAFFIC_SIGN_COLOR  = (0, 165, 255) # orange

# Analysis frame by frame (Function to reuse for image & video anaylysis)
def _run_models_on_frame(frame):
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
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
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