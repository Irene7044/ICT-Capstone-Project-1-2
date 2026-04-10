import cv2
import os

VIDEO_PATH = r"/mnt/d/Uni Adelaide/ICT/videos/1mnt_poles_sample_2.mp4"   # replace with your actual video
OUTPUT_DIR = "datasets/pole/images/train"
FRAME_SKIP = 30   # around 1 frame per second if video is 30 FPS

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % FRAME_SKIP == 0:
        output_path = os.path.join(OUTPUT_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(output_path, frame)
        saved += 1

    count += 1

cap.release()
print(f"Saved {saved} frames")