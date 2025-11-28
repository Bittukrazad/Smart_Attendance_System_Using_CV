"""
Updated encode_faces.py
- Handles EXIF rotation
- Resizes images for faster encoding
- Handles corrupted images
- Uses num_jitters=5 for more accurate encodings
- Skips non-image files safely

Dataset structure:
dataset/
   person1/
       image1.jpg
       image2.png
   person2/
       ...

Outputs:
encodings.pickle with keys: 'encodings', 'names'
"""

import face_recognition
import pickle
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

print("[INFO] Starting face encoding...")
dataset_path = 'dataset'

# Initialize lists to store encodings and names
known_encodings = []
known_names = []

# Loop through each person folder in dataset
total_people = len(os.listdir(dataset_path))

for name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, name)

    if not os.path.isdir(person_dir):
        continue

    print(f"\n[INFO] Processing person: {name}")

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)

        # Skip non-image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            print(f"[WARNING] Skipping non-image file: {filename}")
            continue

        # Load image with EXIF rotation handling
        try:
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)  # Fix rotation
            pil_img = pil_img.convert('RGB')
            image = np.array(pil_img)
        except Exception as e:
            print(f"[ERROR] Cannot load image {filename}: {e}")
            continue

        # Optional: Resize image for faster encoding
        try:
            image_small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        except:
            print(f"[WARNING] Could not resize image {filename}, using full size")
            image_small = image

        # Detect face
        boxes = face_recognition.face_locations(image_small, model='hog')

        if len(boxes) == 0:
            print(f"[WARNING] No face found in {filename}, skipping.")
            continue

        # Compute facial embeddings – use multiple jitters for accuracy
        try:
            encoding = face_recognition.face_encodings(image_small, boxes, num_jitters=5)[0]
        except Exception as e:
            print(f"[ERROR] Failed to encode {filename}: {e}")
            continue

        known_encodings.append(encoding)
        known_names.append(name)
        print(f"[INFO] Encoded {filename} for {name}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
print("\n[INFO] Saving encodings to encodings.pickle...")
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding process completed successfully!")
