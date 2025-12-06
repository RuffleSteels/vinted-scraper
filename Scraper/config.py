import os

# Get the root of the project dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chrome driver path (update if using webdriver-manager or system chromedriver)
DRIVER_PATH = os.path.join(PROJECT_ROOT, "chrome_driver", "chromedriver")

# General settings
WIDTH, HEIGHT = 400, 600
PRICE_THRESHOLD = 5

BOT_TOKEN = "7883319571:AAFmWkHKXKt6UkYlUqT6DSrNsXy2gk0jhzg"
CHAT_ID = "8500213100"
ITEM = ["stussy teeshirt"]

Y_MEAN, Y_STD = 31.185, 14.549
SLEEP_INTERVAL = (1, 1)
SHIPPING = 2
BUYER_PROTECTION = 0.95
SELL_MARKUP = 3


# Paths inside project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = '//models/CNN3.pt'
DATA_PATH = '//data/tensors.pkl'
SEEN_IDS_PATH = './seen_ids.pkl'
PRICE_CSV_PATH = './prices.csv'
