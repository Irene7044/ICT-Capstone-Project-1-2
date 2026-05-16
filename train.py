from ultralytics import YOLO

# Load pretrained base model
model = YOLO('yolov8s.pt')

# Train on your dataset
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,        # stops early if no improvement
    name='hazard_model'
)

print("Training complete!")
print("Best model saved to: runs/detect/hazard_model/weights/best.pt")