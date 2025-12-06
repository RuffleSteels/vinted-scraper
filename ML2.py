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
from torchvision import transforms

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using Cuda")
else:
    device = "cpu"
    print("Using CPU")
class ImagePriceDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]

        # Convert CHW → HWC if it's a tensor from your saved dataset
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        # Transform: PIL → CHW tensor
        if self.transform:
            img = self.transform(img)

        # label should be float32
        label = float(self.y[idx])

        return img, label
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace classifier head for regression
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

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

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImagePriceDataset(X_train, y_train, transform=train_tf)
    test_dataset = ImagePriceDataset(X_test, y_test, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # -----------------------------
    # 🤖 MODEL + OPTIMIZER
    # -----------------------------
    model = CNNRegressor().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

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
                # Move input to device and run model
                batch_X = batch_X.to(device)

                # Predict log(price) → convert back to raw price
                outputs = model(batch_X).cpu()
                outputs = torch.expm1(outputs).clamp(min=0)
                outputs = outputs.numpy()

                # Store raw predictions and raw truth prices
                preds.extend(outputs.flatten().tolist())
                trues.extend(batch_y.numpy().astype(float).flatten().tolist())
                test_images.extend(batch_X.cpu())

        # Already raw prices — no scaling needed
        preds_scaled = preds
        trues_scaled = trues

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
    criterion = nn.SmoothL1Loss()

    for epoch in range(8):  # 5–10 is good

        model.train()
        running_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            # No Mixup — use raw batch
            inputs = batch_X


            # Train on log(price)
            outputs = model(inputs)
            y_log = torch.log1p(batch_y.clamp(min=0))
            loss = criterion(outputs, y_log)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"[Stage 1] Epoch {epoch + 1} - Loss: {epoch_loss:.4f}")

        scheduler.step()

    print("\n🔓 Stage 2: Fine-tuning EfficientNet...\n")
    for epoch in range(150):  # fine-tune for longer
        model.train()
        running_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)

            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            # No Mixup — use raw batch
            inputs = batch_X

            # Train on log(price)
            outputs = model(inputs)
            y_log = torch.log1p(batch_y.clamp(min=0))
            loss = criterion(outputs, y_log)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"[Stage 2] Epoch {epoch + 1} - Loss: {epoch_loss:.4f}")

        scheduler.step()

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Saved model checkpoint to {MODEL_PATH}")
