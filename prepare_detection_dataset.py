import os
import random
import shutil

# =========================
# USER SETTINGS
# =========================

ELEMENT_NAME = "pole" #Change it to the relevant element name

# Path to exported Roboflow folder that contains:
# train/images and train/labels
SOURCE_IMAGES = r"/mnt/d/Uni Adelaide/ICT/Object Detection_200.yolov11/train/images" # Change this to where us save the exported file
SOURCE_LABELS = r"/mnt/d/Uni Adelaide/ICT/Object Detection_200.yolov11/train/labels" # Change this to where us save the exported file

VAL_RATIO = 0.2
RANDOM_SEED = 42

# =========================
# AUTO PATHS
# =========================

DEST_TRAIN_IMAGES = f"datasets/{ELEMENT_NAME}/images/train"
DEST_VAL_IMAGES = f"datasets/{ELEMENT_NAME}/images/val"
DEST_TRAIN_LABELS = f"datasets/{ELEMENT_NAME}/labels/train"
DEST_VAL_LABELS = f"datasets/{ELEMENT_NAME}/labels/val"


def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path):
            os.remove(path)


def main():
    random.seed(RANDOM_SEED)

    clear_folder(DEST_TRAIN_IMAGES)
    clear_folder(DEST_VAL_IMAGES)
    clear_folder(DEST_TRAIN_LABELS)
    clear_folder(DEST_VAL_LABELS)

    image_files = [
        f for f in os.listdir(SOURCE_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()
    random.shuffle(image_files)

    total = len(image_files)
    val_count = int(total * VAL_RATIO)

    val_files = set(image_files[:val_count])
    train_files = image_files[val_count:]

    for image_file in train_files:
        image_src = os.path.join(SOURCE_IMAGES, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_src = os.path.join(SOURCE_LABELS, label_file)

        shutil.copy2(image_src, os.path.join(DEST_TRAIN_IMAGES, image_file))

        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(DEST_TRAIN_LABELS, label_file))
        else:
            print(f"Warning: missing label for {image_file}")

    for image_file in val_files:
        image_src = os.path.join(SOURCE_IMAGES, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_src = os.path.join(SOURCE_LABELS, label_file)

        shutil.copy2(image_src, os.path.join(DEST_VAL_IMAGES, image_file))

        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(DEST_VAL_LABELS, label_file))
        else:
            print(f"Warning: missing label for {image_file}")

    print("Dataset split complete.")
    print(f"Element: {ELEMENT_NAME}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")


if __name__ == "__main__":
    main()