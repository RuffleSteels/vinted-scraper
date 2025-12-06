import os
import csv
import torch
from PIL import Image
import tempfile
import shutil
import pickle
from io import BytesIO
from torchvision import transforms

from Scraper.config import Y_MEAN, Y_STD


def compile_jpeg_dataset(images_dir, csv_path, X_path="data/X_data.pt", y_path="data/y_data.pt", size=(224, 224)):
    """
    Loads all JPEGs listed in prices.csv (format: item_id,price)
    Converts to tensors, stacks them, and atomically saves to output_path.
    """
    image_tensors = []
    label_tensors = []

    # ---- Load each row from CSV ----
    with open(csv_path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) != 2:
                print(f"[WARN] Skipping malformed row: {row}")
                continue

            item_id, price_str = row
            price = float(price_str)

            image_path = os.path.join(images_dir, f"{item_id}.jpg")

            if not os.path.exists(image_path):
                print(f"[WARN] Missing image: {image_path}")
                continue

            resize = transforms.Resize(size)

            try:
                with open(image_path, "rb") as img_file:
                    img = Image.open(BytesIO(img_file.read())).convert("RGB")
                    tensor = resize(transforms.ToTensor()(img))  # your method
            except Exception as e:
                print(f"[WARN] Could not load {image_path}: {e}")
                continue

            image_tensors.append(tensor)
            print((price - Y_MEAN) / Y_STD)
            label_tensors.append(torch.tensor([price], dtype=torch.float32))

    if not image_tensors:
        print("[ERROR] No valid images found. Aborting.")
        return

    X = torch.stack(image_tensors)  # [N, 3, H, W]
    y = torch.stack(label_tensors)  # [N, 1]

    # Save tensors separately
    torch.save(X, X_path)
    torch.save(y, y_path)

    print(f"[OK] Saved {X.shape[0]} images to {X_path} and labels to {y_path}")

compile_jpeg_dataset(
    images_dir="images",
    csv_path="prices.csv",
    size=(224, 224)
)