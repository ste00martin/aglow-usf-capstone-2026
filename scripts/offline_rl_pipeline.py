"""
offline_rl_pipeline.py
======================
Offline RL fine-tuning of the NSFW model using interaction logs from the app.

The problem framing
-------------------
We treat content moderation as a contextual bandit:
  - State   s: image (encoded via the NSFW model's penultimate features)
  - Action  a: model's output score / threshold decision (show vs filter)
  - Reward  r: user behavior signal collected in-app
                 report        → -5.0  (user found it harmful)
                 skip (<2s)    → -0.1
                 view (2-5s)   → +0.3
                 view (>5s)    → +1.0

The simplest effective algorithm here is Reward-Weighted Regression (RWR) /
reward-weighted fine-tuning:
  - Treat reports as hard negative training examples (label = "unsafe")
  - Treat long views of content that passes NSFW check as positives (label = "safe")
  - Weight each example by |reward|
  - Fine-tune only the classifier head (last linear layer) of the NSFW ViT

Why not a full Q-learning approach?
  The NSFW classifier is a 3-class softmax model. Its action space is effectively
  the predicted label. With limited data from a phone, reward-weighted regression
  is more stable. Once you have >10k interactions, you can graduate to IQL or CQL
  (a sketch is included below in `offline_rl_iql_sketch`).

Pipeline
--------
  1. Export interaction_log.jsonl from device (Share → Files, or via Xcode)
  2. Place it at: logs/interaction_log.jsonl
  3. Run: python scripts/offline_rl_pipeline.py
  4. Output: models/nsfw_finetuned.pt  (PyTorch checkpoint)
  5. Re-export to ExecuTorch: python scripts/export_nsfw.py --weights models/nsfw_finetuned.pt

Requirements
------------
  pip install torch torchvision transformers pillow tqdm
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────

LOG_PATH       = Path("logs/interaction_log.jsonl")
IMAGE_ROOT     = Path("expo-pytorch/assets/feed")   # where Kaggle images live
OUTPUT_DIR     = Path("models")
OUTPUT_WEIGHTS = OUTPUT_DIR / "nsfw_finetuned.pt"

NSFW_MODEL_ID  = "Falconsai/nsfw_image_detection"   # HuggingFace model
NSFW_LABELS    = ["gore_bloodshed_violent", "nudity_pornography", "safe_normal"]

# Training hyperparameters
LR             = 1e-4
EPOCHS         = 5
BATCH_SIZE     = 8
MIN_REWARD_ABS = 0.05   # ignore near-zero reward samples (uninformative)
REPORT_LABEL   = "nudity_pornography"  # map generic 'report' to this class
DEVICE         = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_interactions(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_path}\n"
            "Export interaction_log.jsonl from the device and place it here."
        )
    interactions = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                interactions.append(json.loads(line))
    print(f"Loaded {len(interactions)} interactions from {log_path}")
    return interactions


def interactions_to_examples(
    interactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert raw interactions to (image_path, label_idx, weight) triples.

    Label assignment:
      - action == 'report'        → label = class for REPORT_LABEL
      - action == 'view' (>5s)   → label = 'safe_normal' (content was OK)
      - action == 'skip'          → skip (uninformative for label assignment)
      - negative reward           → label = most-reported unsafe class
    """
    label_map = {name: i for i, name in enumerate(NSFW_LABELS)}
    safe_idx   = label_map["safe_normal"]
    report_idx = label_map.get(REPORT_LABEL, 1)

    examples = []
    for item in interactions:
        reward   = float(item["reward"])
        action   = item["action"]
        uri      = item.get("contentUri", "")
        weight   = abs(reward)

        if weight < MIN_REWARD_ABS:
            continue  # near-zero reward = uninformative

        # Determine training label
        if action == "report":
            label = report_idx
            # Use the user-provided category if available
            cat = item.get("reportCategory", "")
            if "gore" in cat.lower() or "violence" in cat.lower():
                label = label_map["gore_bloodshed_violent"]
            elif "nsfw" in cat.lower() or "nudity" in cat.lower():
                label = label_map["nudity_pornography"]
            else:
                label = report_idx

        elif action == "view" and reward > 0:
            label = safe_idx  # long view of content = implicitly safe
        else:
            continue  # skip or neutral — don't add supervised signal

        # Resolve image path
        img_path = _resolve_image_path(uri)
        if img_path is None:
            continue

        examples.append({
            "image_path": img_path,
            "label":      label,
            "weight":     weight,
        })

    print(f"  → {len(examples)} usable training examples "
          f"({sum(1 for e in examples if e['label'] != safe_idx)} negative, "
          f"{sum(1 for e in examples if e['label'] == safe_idx)} positive)")
    return examples


def _resolve_image_path(uri: str) -> Path | None:
    """Map a contentUri (remote URL or local file path) to a local Path."""
    if uri.startswith("file://"):
        p = Path(uri.removeprefix("file://"))
        return p if p.exists() else None
    if uri.startswith("https://picsum.photos"):
        return None  # can't use remote URLs as training data
    # Relative path (from kaggle_feed.json)
    for base in [IMAGE_ROOT, Path(".")]:
        p = base / uri.lstrip("./")
        if p.exists():
            return p
    return None


# ── Dataset ────────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        ex = self.examples[idx]
        img = Image.open(ex["image_path"]).convert("RGB")
        return self.transform(img), int(ex["label"]), float(ex["weight"])


def collate_fn(
    batch: list[tuple[torch.Tensor, int, float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images  = torch.stack([b[0] for b in batch])
    labels  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return images, labels, weights


# ── Model loading ──────────────────────────────────────────────────────────────

def load_nsfw_model() -> nn.Module:
    """
    Load the NSFW ViT from HuggingFace.
    We only fine-tune the classifier head — the ViT backbone is frozen.
    """
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification  # type: ignore

    print(f"Loading {NSFW_MODEL_ID} …")
    model = AutoModelForImageClassification.from_pretrained(NSFW_MODEL_ID)

    # Freeze ViT backbone, only train the classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  (head only)")
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    dataset: InteractionDataset,
) -> nn.Module:
    if len(dataset) == 0:
        print("No training examples — skipping fine-tuning.")
        return model

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    model.to(DEVICE)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_samples = 0

        for images, labels, weights in loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            weights = weights.to(DEVICE)

            outputs = model(pixel_values=images)
            logits  = outputs.logits  # [B, num_classes]

            # Reward-weighted cross-entropy loss
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            # Normalize weights within batch so large-|reward| samples dominate
            norm_weights = weights / (weights.sum() + 1e-8)
            loss = (per_sample_loss * norm_weights).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(images)
            total_samples += len(images)

        avg_loss = total_loss / max(total_samples, 1)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}")

    return model


# ── IQL Sketch (advanced — graduate to this at >10k interactions) ──────────────

def offline_rl_iql_sketch() -> None:
    """
    Sketch of Implicit Q-Learning (IQL) for content ranking.

    IQL (Kostrikov et al. 2021) learns a Q-function from offline data
    without ever querying the policy on out-of-distribution actions.
    It's well-suited to the logged interaction setting.

    State  s  = NSFW model penultimate features (768-dim ViT CLS token)
    Action a  = discretized content score bucket (0-4, low→high safety)
    Reward r  = interaction reward (-5 to +1)

    Key equations:
      V(s)   = E_τ[Q(s,a)]   (expectile regression, τ > 0.5 → optimistic value)
      Q(s,a) = r + γ * V(s') (TD target)
      π(s)   = argmax_a exp(β * A(s,a)) where A = Q - V  (advantage-weighted BC)

    NOT implemented here — requires significantly more data and infrastructure.
    See: https://arxiv.org/abs/2110.06169
    """
    raise NotImplementedError(
        "IQL is a sketch only. Graduate to it when you have >10k interactions. "
        "Start with reward-weighted fine-tuning (train() above)."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Log: {LOG_PATH}")

    interactions = load_interactions(LOG_PATH)

    # Print reward distribution
    rewards = [i["reward"] for i in interactions]
    print(f"\nReward stats: min={min(rewards):.2f}  max={max(rewards):.2f}  "
          f"mean={sum(rewards)/len(rewards):.3f}")
    action_counts = {}
    for i in interactions:
        action_counts[i["action"]] = action_counts.get(i["action"], 0) + 1
    print(f"Action counts: {action_counts}")

    examples = interactions_to_examples(interactions)
    if not examples:
        print("\nNo usable training examples found.")
        print("Tips:")
        print("  - Make sure interaction_log.jsonl contains 'report' or long 'view' entries")
        print("  - Images must be accessible at the resolved path (Kaggle data, not picsum URLs)")
        return

    dataset = InteractionDataset(examples)
    model   = load_nsfw_model()
    model   = train(model, dataset)

    # Save fine-tuned weights
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\nSaved fine-tuned weights → {OUTPUT_WEIGHTS}")
    print("Next: re-export to ExecuTorch with:")
    print(f"  python scripts/export_nsfw.py --weights {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
