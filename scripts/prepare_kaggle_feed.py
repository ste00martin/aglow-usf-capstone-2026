"""
prepare_kaggle_feed.py
======================
Download the Flickr8k dataset from Kaggle and generate feed files that the app
can load.

Flickr8k is a safe, widely-used dataset of 8,000 general-interest photos
with captions — a good stand-in for a real social feed.

Setup
-----
1.  pip install kaggle pillow
2.  Get a Kaggle API token: https://www.kaggle.com/docs/api
    Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
3.  Run: python scripts/prepare_kaggle_feed.py
4.  Optional full local dataset:
    python scripts/prepare_kaggle_feed.py --profile full --num-images 500

Outputs
-------
starter profile:
  expo-pytorch/data/kaggle_feed.json
  expo-pytorch/data/kaggleFeedItems.ts
  expo-pytorch/assets/feed/

full profile (local only, gitignored):
  expo-pytorch/data/kaggle_feed.local.json
  expo-pytorch/data/kaggleFeedItems.local.ts
  expo-pytorch/assets/feed-local/
"""

import argparse
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────────
KAGGLE_DATASET      = "adityajn105/flickr8k"
STARTER_NUM_IMAGES  = 24       # checked-in subset for the app feed
FULL_NUM_IMAGES     = 500      # local default; use --num-images all for everything
IMAGE_WIDTH         = 400
IMAGE_HEIGHT        = 700
RANDOM_SEED         = 42

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
DOWNLOAD_DIR = SCRIPT_DIR / "_kaggle_flickr8k"
DATA_DIR     = REPO_ROOT / "expo-pytorch" / "data"
ASSETS_DIR   = REPO_ROOT / "expo-pytorch" / "assets"


@dataclass(frozen=True)
class OutputProfile:
    name: str
    default_num_images: int
    output_json: Path
    output_ts: Path
    output_imgs: Path
    require_prefix: str
    uri_prefix: str
    asset_path_prefix: str


PROFILES: dict[str, OutputProfile] = {
    "starter": OutputProfile(
        name="starter",
        default_num_images=STARTER_NUM_IMAGES,
        output_json=DATA_DIR / "kaggle_feed.json",
        output_ts=DATA_DIR / "kaggleFeedItems.ts",
        output_imgs=ASSETS_DIR / "feed",
        require_prefix="../assets/feed",
        uri_prefix="../../assets/feed",
        asset_path_prefix="expo-pytorch/assets/feed",
    ),
    "full": OutputProfile(
        name="full",
        default_num_images=FULL_NUM_IMAGES,
        output_json=DATA_DIR / "kaggle_feed.local.json",
        output_ts=DATA_DIR / "kaggleFeedItems.local.ts",
        output_imgs=ASSETS_DIR / "feed-local",
        require_prefix="../assets/feed-local",
        uri_prefix="../../assets/feed-local",
        asset_path_prefix="expo-pytorch/assets/feed-local",
    ),
}

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
        candidate_name = candidate.name.lower()
        if candidate.suffix == ".txt" and (
            "caption" in candidate_name or "token" in candidate_name
        ):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate starter or full Kaggle feed assets for the Expo app."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="starter",
        help="starter writes the checked-in subset; full writes local-only ignored outputs.",
    )
    parser.add_argument(
        "--num-images",
        default=None,
        help="Number of images to generate, or 'all' to use the whole dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def resolve_num_images(raw_value: str | None, profile: OutputProfile, available: int) -> int:
    if raw_value is None:
        return min(profile.default_num_images, available)
    if raw_value.lower() == "all":
        return available

    value = int(raw_value)
    if value <= 0:
        raise ValueError("--num-images must be a positive integer or 'all'.")
    return min(value, available)


def reset_output_dir(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_feed_outputs(profile: OutputProfile, feed_items: list[dict[str, Any]]) -> None:
    profile.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(profile.output_json, "w", encoding="utf-8") as f:
        json.dump(feed_items, f, indent=2, ensure_ascii=False)

    # Generate a TypeScript file with explicit require() calls so Metro can
    # bundle the images as static assets (dynamic string paths don't work).
    lines = [
        "// Auto-generated by scripts/prepare_kaggle_feed.py — do not edit manually",
        f"// Re-run: python scripts/prepare_kaggle_feed.py --profile {profile.name}",
        "",
        "import type { FeedItem } from './feedData';",
        "",
        "export const KAGGLE_FEED_ITEMS: FeedItem[] = [",
    ]
    for item in feed_items:
        dst_name = Path(item["uri"]).name
        caption_escaped = item["caption"].replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
        tags_json = json.dumps(item["tags"])
        lines.append("  {")
        lines.append(f"    id: '{item['id']}',")
        lines.append(f"    uri: require('{profile.require_prefix}/{dst_name}'),")
        lines.append(f"    assetPath: '{item['assetPath']}',")
        lines.append(f"    caption: `{caption_escaped}`,")
        lines.append(f"    tags: {tags_json},")
        lines.append("    source: 'kaggle',")
        lines.append(f"    kaggleId: '{item['kaggleId']}',")
        lines.append("  },")
    lines.append("];")

    with open(profile.output_ts, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]

    if not DOWNLOAD_DIR.exists() or not any(DOWNLOAD_DIR.iterdir()):
        download_dataset()
    else:
        print(f"Using cached download at {DOWNLOAD_DIR}")

    images = find_images(DOWNLOAD_DIR)
    print(f"Found {len(images)} images.")
    if not images:
        raise RuntimeError("No images found — check the download directory.")

    captions = find_captions(DOWNLOAD_DIR)
    num_images = resolve_num_images(args.num_images, profile, len(images))

    random.seed(args.seed)
    selected = random.sample(images, num_images)

    reset_output_dir(profile.output_imgs)

    feed_items: list[dict[str, Any]] = []
    for i, src in enumerate(selected):
        item_id = str(i + 1)
        dst_name = f"feed_{i + 1:03d}.jpg"
        dst = profile.output_imgs / dst_name

        ok = resize_image(src, dst, IMAGE_WIDTH, IMAGE_HEIGHT)
        if not ok:
            continue

        caption = captions.get(src.name, captions.get(src.stem + ".jpg", ""))
        tags = infer_tags(caption)

        feed_items.append({
            "id":       item_id,
            "uri":      f"{profile.uri_prefix}/{dst_name}",
            "assetPath": f"{profile.asset_path_prefix}/{dst_name}",
            "caption":  caption or src.stem.replace("_", " "),
            "tags":     tags,
            "source":   "kaggle",
            "kaggleId": src.name,
        })

    write_feed_outputs(profile, feed_items)

    print(f"\nProfile: {profile.name}")
    print(f"Wrote {len(feed_items)} items → {profile.output_json}")
    print(f"Generated TypeScript → {profile.output_ts}")
    print(f"Images → {profile.output_imgs}/")


if __name__ == "__main__":
    main()
