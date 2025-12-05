import time
import datetime
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup

# from ML import CNNRegressor
from Scraper.utils import *
from Scraper.config import *

import sys
sys.stdout.reconfigure(line_buffering=True)
print("Starting Scraper...")

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using Cuda")
else:
    device = "cpu"
    print("Using CPU")

def scrape_page(page, url, isPred=True):
    """Scrape a single webpage and evaluate new products."""
    try:
        page.goto(url, timeout=30000)
        time.sleep(1)  # wait for JS to load content
        soup = BeautifulSoup(page.content(), "lxml")
        listings = soup.select("div.feed-grid__item")

        items = []
        for item in listings:
            img_tag = item.img
            title_a = item.select_one("a.new-item-box__overlay--clickable")
            price_tag = item.select_one("p[data-testid*=price-text]")
            link_tag = item.select_one("a[data-testid*=--overlay-link]")

            if not (img_tag and title_a and price_tag and link_tag):
                continue

            image_src = img_tag.get("src") or img_tag.get("data-src")
            if not image_src:
                continue

            href = link_tag["href"]
            item_id = href[href.rfind('/') + 1:].split('-')[0]

            # print(href)

            if item_id in seen_ids:
                continue
            # print(href)
            title = title_a.get("title", "").partition(",")[0]
            price_float = float(price_tag.text[1:])
            items.append((image_src, item_id, title, price_float, href))

        # print(items)

        for (src, item_id, title, price_float, href) in items:
            img_path = save_jpeg(src, item_id)

            # print("Scraping image:", img_path)

            if img_path is None:
                continue

            print(f"New item: {title}, Price: £{price_float}", href)
            seen_ids.add(item_id)
            dump_price(item_id, ((price_float - Y_MEAN) / Y_STD))
            dump_seen_ids(seen_ids)

            # if isPred:
            #     image_tensor = url_to_tensor(src)
            #     image_tensor = image_tensor.to(device)
            #     with torch.no_grad():
            #         pred_price = model(image_tensor.unsqueeze(0)).item()
            #
            #     print(f"Predicted price: {(pred_price * Y_STD + Y_MEAN)}")

                # value = (pred * Y_STD + Y_MEAN).item()
            #
            #     if (value - (price_float + BUYER_PROTECTION + SHIPPING)) > price_threshold and pred < 30: # removes weird high hallucinations
            #         send_telegram(price_float, title, link_tag.get("href"), value, image_src)


    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return

def scrape_category(browser, item_name, pages=10):
    """Scrape multiple pages for a single item."""
    for i in range(pages):
        url = f"https://www.vinted.co.uk/catalog?order=newest_first&search_id=29086618135&time=1764965529&catalog%5B%5D=77&search_by_image_uuid=&brand_ids%5B%5D=362&page={i+1}"
        print(f"Scraping page {i+1} -> {url}")

        page = browser.new_page()
        try:
            scrape_page(page, url)
        finally:
            page.close()
            print('Sleeping...')
            time.sleep(random.uniform(*SLEEP_INTERVAL))


if __name__ == "__main__":
    # model = CNNRegressor().to(device)

    # Load weights
    # model.load_state_dict(torch.load("./model1.pt", map_location=device))
    # model.eval()
    while True:
        print("Starting new browser session...")
        p, browser, context, page = get_driver()
        try:
            seen_ids = load_seen_ids()
            while True:
                hour = datetime.datetime.now().hour

                if 8 <= hour < 24:
                    for item in ITEM:
                        scrape_category(browser, item, pages=6)
                    sys.exit(0)
                    time.sleep(10)
                else:
                    print("Outside working hours, sleeping ...")
                    time.sleep(300)
        except Exception as e:
            print("CRASH — restarting browser:", e)
        finally:
            browser.close()
            p.stop()
            print("Browser closed. Restarting in 5 seconds...")
            time.sleep(5)
