from ultralytics import YOLO
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML  = os.path.join(SCRIPT_DIR, "Bicycle-Lanes-Segmentation-1", "data.yaml")

model = YOLO("yolov8m-seg.pt")  

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,        # lower than Colab's 16 — safer for local memory
    name="bike_lane_model3",
    project=os.path.join(SCRIPT_DIR, "runs")
)

print("Training complete!")
print(f"Best model saved to: {SCRIPT_DIR}/runs/bike_lane_model2/weights/best.pt")