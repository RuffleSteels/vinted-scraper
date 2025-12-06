from torchvision import transforms
import requests
from io import BytesIO
import torch.nn as nn
from torchvision.models import resnet34
import random
from PIL import Image
from Scraper.config import *
import csv
import os
import pickle
import torch

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def url_to_tensor(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = resize(img)   # -> [3, 224, 224]
        return tensor

    except Exception as e:
        print("Failed to load image:", e, url)
        return None
def save_jpeg(image_url, item_id, out_dir="images", size=(224, 224), quality=80):
    """Downloads, resizes, and saves an image as JPEG. Returns the saved path or None."""
    try:
        os.makedirs(out_dir, exist_ok=True)

        resp = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(resp.content)).convert("RGB")

        # downscale
        img = img.resize(size, Image.LANCZOS)

        path = os.path.join(out_dir, f"{item_id}.jpg")
        img.save(path, "JPEG", quality=quality)

        return path
    except Exception:
        return None
def random_user_agent():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    ]
    return random.choice(agents)

def get_driver():
    """Launch a headless Chromium browser using Playwright."""
    p = sync_playwright().start()
    
    browser = p.chromium.launch(
        headless=True,  # HEADLESS = HIGH DETECTION on Vinted; keep it visible
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--ignore-certificate-errors",
        ],
    )

    context = browser.new_context(
        user_agent=random_user_agent(),
        locale="en-GB",
        timezone_id="Europe/London",
        viewport={"width": 1366, "height": 768},
        device_scale_factor=1,
        java_script_enabled=True,
    )

    page = context.new_page()
    stealth = Stealth()
    stealth.apply_stealth_sync(page)

    return p, browser, context, page

def download_image(link, name):
    """Downloads an image PNG locally"""
    try:
        resp = requests.get(link, stream=True, timeout=15)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            safe_name = name.replace("/", "_")[:120]
            img.save(f"images/{safe_name}.png", format="PNG")
            print(f"\rSaved image: {safe_name}", end="", flush=True)
        else:
            print("Failed image download:", resp.status_code)
    except Exception as e:
        print("Image error:", e)

def send_telegram(price, title, link, value, img_tag):
    """Sends a telegram message to a specific bot on telegram"""
    print("DEAL FOUND - DEAL FOUND - DEAL FOUND")
    print(f"HIGH VALUE: \n {title} \n Price: £{price} \n Value: £{round(value, 2)} \n {link}")
    if price < 15:
        value = value + SELL_MARKUP
        if value - price > 12:
            message = f"HIGH VALUE: \n {title} \n Price: £{price} \n Value: £{round(value, 2)} \n {link}"

        elif value - price > 8:
            message = f"MEDIUM VALUE: \n {title} \n Price: £{price} \n Value: £{round(value, 2)} \n {link}"

        elif value - price > 4:
            message = f"LOW VALUE: \n {title} \n Price: £{price} \n Value: £{round(value, 2)} \n {link}"

        else:
            message = f"IDK VALUE: \n {title} \n Price: £{price} \n Value: £{round(value, 2)} \n {link}"

        url = f"https://api.telegram.org/bot8552201082:AAEGd7zpkz2yGY8OQkEKpWq2n_yO3LIXqn0/sendMessage"
        payload = {"chat_id": '8506286983', "text": message}
        requests.post(url, data=payload)

class PricePredictorResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet34(weights="IMAGENET1K_V1")
        for name, param in self.backbone.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)

        self.backbone.fc = nn.Identity()  # remove classification head
        self.regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x).squeeze(1)

def dump_seen_ids(seen_ids):
    """Save the set of seen item IDs to a pickle file."""
    with open(SEEN_IDS_PATH, "wb") as f:
        pickle.dump(seen_ids, f)
def dump_price(item_id, price):
    """Append (item_id, price) to a CSV file."""
    file_exists = os.path.exists(PRICE_CSV_PATH)

    with open(PRICE_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        # if not file_exists:
            # writer.writerow(["item_id", "price"])

        writer.writerow([item_id, price])
def load_seen_ids():
    """Load the set of previously seen item IDs."""
    try:
        with open(SEEN_IDS_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return set()
def load_model(path=MODEL_PATH, device="cpu"):
    """Loads a trained model for evaluation, handling DataParallel checkpoints."""
    
    # Initialize model
    model = PricePredictorResNet()
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Fix 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")  # remove 'module.' if present
        new_state_dict[name] = v
    
    # Load state dict
    model.load_state_dict(new_state_dict)
    
    # Set model to eval mode and move to device
    model.to(device)
    model.eval()
    
    return model

def load_tensors(path="data/tensors.pkl"):
    """Loads the saved data for further training"""
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return None, None

    if os.path.getsize(path) == 0:
        print(f"Path {path} is empty")
        return None, None

    try:
        with open(path, "rb") as f:
            images, labels = pickle.load(f)
            return images, labels

    except Exception as e:
        print(f"Could not open file, exception: {e}")

