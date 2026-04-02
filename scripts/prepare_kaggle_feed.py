"""
prepare_kaggle_feed.py
======================
Download the Flickr8k dataset from Kaggle, pick a random subset of images,
and write them as a feedData JSON file that the app can load.

Flickr8k is a safe, widely-used dataset of 8,000 general-interest photos
with captions — a good stand-in for a real social feed.

Setup
-----
1.  pip install kaggle pillow
2.  Get a Kaggle API token: https://www.kaggle.com/docs/api
    Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
3.  Run: python scripts/prepare_kaggle_feed.py

Outputs
-------
expo-pytorch/data/kaggle_feed.json   — feed items JSON (import into feedData.ts)
expo-pytorch/assets/feed/            — resized images bundled in the app

To use in the app, replace FEED_ITEMS in expo-pytorch/data/feedData.ts:
  import kaggleFeed from './kaggle_feed.json';
  export const FEED_ITEMS: FeedItem[] = kaggleFeed;
"""

import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────────
KAGGLE_DATASET   = "adityajn105/flickr8k"
NUM_IMAGES       = 50          # number of feed items to generate
IMAGE_WIDTH      = 400
IMAGE_HEIGHT     = 700
RANDOM_SEED      = 42

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
DOWNLOAD_DIR = SCRIPT_DIR / "_kaggle_flickr8k"
OUTPUT_JSON  = REPO_ROOT / "expo-pytorch" / "data" / "kaggle_feed.json"
OUTPUT_TS    = REPO_ROOT / "expo-pytorch" / "data" / "kaggleFeedItems.ts"
OUTPUT_IMGS  = REPO_ROOT / "expo-pytorch" / "assets" / "feed"

# ── Helpers ────────────────────────────────────────────────────────────────────

def download_dataset() -> None:
    print(f"Downloading {KAGGLE_DATASET} …")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "--unzip", "-p", str(DOWNLOAD_DIR)],
        check=True,
    )
    print("Download complete.")


def find_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def find_captions(root: Path) -> dict[str, str]:
    """
    Parse Flickr8k captions file (captions.txt or Flickr8k.token.txt).
    Returns a dict: image_filename → first caption.
    """
    captions: dict[str, str] = {}

    for candidate in root.rglob("*"):
        if candidate.suffix in (".txt",) and "caption" in candidate.name.lower():
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Flickr8k format: "filename.jpg#0\tcaption text"
                    if "\t" in line:
                        key, _, cap = line.partition("\t")
                        fname = key.split("#")[0].strip()
                        if fname not in captions:
                            captions[fname] = cap.strip()
                    # Alternative format: "filename.jpg,caption text"
                    elif "," in line:
                        fname, _, cap = line.partition(",")
                        fname = fname.strip()
                        if fname not in captions:
                            captions[fname] = cap.strip()
            if captions:
                print(f"Loaded {len(captions)} captions from {candidate.name}")
                return captions

    print("Warning: no captions file found — captions will be empty.")
    return captions


def resize_image(src: Path, dst: Path, w: int, h: int) -> bool:
    try:
        from PIL import Image  # type: ignore
        img = Image.open(src).convert("RGB")
        img.thumbnail((w, h), Image.LANCZOS)  # preserve aspect ratio, no crop
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"  Failed to resize {src.name}: {e}")
        return False


def infer_tags(caption: str) -> list[str]:
    """Very simple tag inference from caption keywords."""
    tag_map = {
        "dog": "animals", "cat": "animals", "horse": "animals", "bird": "animals",
        "water": "water", "lake": "nature", "ocean": "nature", "river": "nature",
        "mountain": "nature", "forest": "nature", "tree": "nature",
        "city": "urban", "street": "urban", "building": "urban",
        "people": "people", "man": "people", "woman": "people", "child": "people",
        "snow": "snow", "beach": "beach", "sunset": "sky", "sky": "sky",
        "food": "food", "grass": "nature", "field": "nature",
    }
    caption_lower = caption.lower()
    found = list({v for k, v in tag_map.items() if k in caption_lower})
    return found[:3] if found else ["photo"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DOWNLOAD_DIR.exists() or not any(DOWNLOAD_DIR.iterdir()):
        download_dataset()
    else:
        print(f"Using cached download at {DOWNLOAD_DIR}")

    images = find_images(DOWNLOAD_DIR)
    print(f"Found {len(images)} images.")
    if not images:
        raise RuntimeError("No images found — check the download directory.")

    captions = find_captions(DOWNLOAD_DIR)

    random.seed(RANDOM_SEED)
    selected = random.sample(images, min(NUM_IMAGES, len(images)))

    OUTPUT_IMGS.mkdir(parents=True, exist_ok=True)

    feed_items: list[dict[str, Any]] = []
    for i, src in enumerate(selected):
        item_id = str(i + 1)
        dst_name = f"feed_{i + 1:03d}.jpg"
        dst = OUTPUT_IMGS / dst_name

        ok = resize_image(src, dst, IMAGE_WIDTH, IMAGE_HEIGHT)
        if not ok:
            continue

        caption = captions.get(src.name, captions.get(src.stem + ".jpg", ""))
        tags = infer_tags(caption)

        feed_items.append({
            "id":       item_id,
            "uri":      f"../../assets/feed/{dst_name}",   # relative from data/
            "caption":  caption or src.stem.replace("_", " "),
            "tags":     tags,
            "source":   "kaggle",
            "kaggleId": src.name,
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(feed_items, f, indent=2, ensure_ascii=False)

    # Generate a TypeScript file with explicit require() calls so Metro can
    # bundle the images as static assets (dynamic string paths don't work).
    lines = [
        "// Auto-generated by scripts/prepare_kaggle_feed.py — do not edit manually",
        "// Re-run: python scripts/prepare_kaggle_feed.py",
        "",
        "import type { FeedItem } from './feedData';",
        "",
        "export const KAGGLE_FEED_ITEMS: FeedItem[] = [",
    ]
    for item in feed_items:
        dst_name = Path(item["uri"]).name
        caption_escaped = item["caption"].replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
        tags_json = json.dumps(item["tags"])
        lines.append(f"  {{")
        lines.append(f"    id: '{item['id']}',")
        lines.append(f"    uri: require('../assets/feed/{dst_name}'),")
        lines.append(f"    caption: `{caption_escaped}`,")
        lines.append(f"    tags: {tags_json},")
        lines.append(f"    source: 'kaggle',")
        lines.append(f"    kaggleId: '{item['kaggleId']}',")
        lines.append(f"  }},")
    lines.append("];")

    with open(OUTPUT_TS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {len(feed_items)} items → {OUTPUT_JSON}")
    print(f"Generated TypeScript → {OUTPUT_TS}")
    print(f"Images → {OUTPUT_IMGS}/")


if __name__ == "__main__":
    main()
