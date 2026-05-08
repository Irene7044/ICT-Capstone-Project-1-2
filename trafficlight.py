from ultralytics import YOLO
import cv2

#model = YOLO("models/yolov8n.pt")


def detect_image(input_image, output_image):
    img = cv2.imread(input_image)
    results = model(img)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "traffic light":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imwrite(output_image, img)


def detect_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "traffic light":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()


def detect_traffic_light(input_path, output_path):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        detect_image(input_path, output_path)
    else:
        detect_video(input_path, output_path)