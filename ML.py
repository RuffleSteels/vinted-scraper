import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Scraper.config import Y_STD, Y_MEAN
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torchvision.models as models

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using Cuda")
else:
    device = "cpu"
    print("Using CPU")

# 1️⃣ Dataset
class ImagePriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2️⃣ Model
# class CNNRegressor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [32,224,224]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                             # [32,112,112]
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), # [64,112,112]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                             # [64,56,56]
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),# [128,56,56]
#             nn.ReLU(),
#             nn.MaxPool2d(2),                             # [128,28,28]
#         )
#         self.regressor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128*28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)  # single output for regression
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.regressor(x)
#         return x.squeeze(1)  # shape [batch]
#
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Replace final FC with regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)

if __name__ == "__main__":
    # -----------------------------
    # 🔧 CONFIG
    # -----------------------------
    RUN_TEST_ONLY = True   # 👈 set to True to skip training and run test only
    MODEL_PATH = "./model.pt"
    SHOW_IMAGES = True  # 👈 set True to show images during test

    # -----------------------------
    # 🔄 LOAD DATA
    # -----------------------------
    X = torch.load("./data/X_data.pt")
    y = torch.load("./data/y_data.pt")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    train_dataset = ImagePriceDataset(X_train, y_train)
    test_dataset  = ImagePriceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # -----------------------------
    # 🤖 MODEL + OPTIMIZER
    # -----------------------------
    model = CNNRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------------
    # 🚀 TEST-ONLY MODE
    # -----------------------------
    if RUN_TEST_ONLY:
        print("\n⚡ Test-only mode enabled. Loading model and evaluating...\n")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        preds, trues = [], []
        test_images = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).cpu().numpy()

                preds.extend(outputs.flatten().tolist())
                trues.extend(batch_y.numpy().flatten().tolist())
                test_images.extend(batch_X.cpu())

        preds_scaled = [(p * Y_STD + Y_MEAN) for p in preds]
        trues_scaled = [(t * Y_STD + Y_MEAN) for t in trues]

        # Average absolute error
        abs_errors = [abs(p - t) for p, t in zip(preds_scaled, trues_scaled)]
        avg_abs_error = sum(abs_errors) / len(abs_errors)

        print(f"\n📊 Average Absolute Error: £{avg_abs_error:.2f}")

        for i, (p, t_, img_tensor) in enumerate(zip(preds_scaled, trues_scaled, test_images)):
            print(f"Test Sample {i+1}: Pred = {p:.2f}, Actual = {t_:.2f}")
            if SHOW_IMAGES:


                root = tk.Tk()
                root.title("Scrollable Image Viewer")

                canvas = tk.Canvas(root)
                scroll_y = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)

                frame = ttk.Frame(canvas)

                for i, img_tensor in enumerate(test_images):
                    img = img_tensor
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy()

                    img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((150,150))
                    tk_img = ImageTk.PhotoImage(pil_img)

                    lbl = ttk.Label(frame, image=tk_img, text=f"Pred £{(preds_scaled[i]):.2f}\nAcc £{(trues_scaled[i]):.2f}", compound="top")
                    lbl.image = tk_img
                    lbl.grid(row=i//6, column=i%6, padx=5, pady=5)

                canvas.create_window(0,0, anchor="nw", window=frame)
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"), yscrollcommand=scroll_y.set)

                canvas.pack(fill="both", expand=True, side="left")
                root.geometry(f"{frame.winfo_reqwidth() + 20}x600")
                scroll_y.pack(fill="y", side="right")

                root.mainloop()
        print("\n✔ Test-only evaluation complete.")
        exit(0)

    # -----------------------------
    # 🏋️ TRAINING LOOP (interactive)
    # -----------------------------
    epoch = 0

    while True:
        epoch += 1
        model.train()
        running_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).float().squeeze(1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved model checkpoint to {MODEL_PATH}")

