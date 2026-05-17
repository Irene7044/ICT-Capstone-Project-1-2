import cv2, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "testing resource", "prospect2.mp4")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "frames_for_annotation")
SAVE_EVERY_N_FRAMES = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_num, saved = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % SAVE_EVERY_N_FRAMES == 0:
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved:05d}.jpg", frame)
        saved += 1
    frame_num += 1

cap.release()
print(f"Saved {saved} frames to '{OUTPUT_DIR}/'")