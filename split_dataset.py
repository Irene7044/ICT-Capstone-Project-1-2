import os
import shutil
import random

# Paths
train_images = 'dataset/train/images'
train_labels = 'dataset/train/labels'
val_images = 'dataset/valid/images'
val_labels = 'dataset/valid/labels'

# Create val folders
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Get all images
images = os.listdir(train_images)
random.shuffle(images)

# Take 20% for validation
split = int(len(images) * 0.2)
val_files = images[:split]

# Move files
for img in val_files:
    # Move image
    shutil.move(f'{train_images}/{img}', f'{val_images}/{img}')
    
    # Move matching label
    label = img.replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(f'{train_labels}/{label}'):
        shutil.move(f'{train_labels}/{label}', f'{val_labels}/{label}')

print(f"Moved {split} images to validation set")
print(f"Training images remaining: {len(images) - split}")