"""
Export the NSFW classifier (Ateeqq/nsfw-image-detection) to ExecuTorch .pte format.

Labels: Typically SFW / NSFW (printed at runtime from model.config.id2label)

useClassification in react-native-executorch handles image preprocessing internally
(resize to 224x224, ImageNet normalization), so the app only needs to pass a URI.

Prerequisites:
    pip install torch torchvision
    pip install transformers
    pip install executorch
    # For iOS/CoreML export on macOS, also install ExecuTorch CoreML requirements

Usage:
    cd scripts/
    python export_nsfw.py
    # Output: nsfw_model.pte
    python export_nsfw.py --backend coreml
    # Output: nsfw_model_coreml.pte
"""

import argparse
import torch
from transformers import AutoModelForImageClassification
from torch.export import export

from executorch_export_utils import (
    lower_to_executorch,
    resolve_output_path,
    write_program,
)


MODEL_ID = "Ateeqq/nsfw-image-detection"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("xnnpack", "coreml"),
        default="xnnpack",
        help="ExecuTorch backend to target. Use coreml for iOS-specific export.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to nsfw_model.pte for xnnpack and nsfw_model_coreml.pte for coreml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {MODEL_ID} from HuggingFace...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Print labels if available
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        print(f"Labels ({len(labels)}): {labels}")
    else:
        print("Warning: No label mapping found in model config.")

    # Typical input shape for image classification models
    example_input = torch.randn(1, 3, 224, 224)

    # Sanity check forward pass
    with torch.no_grad():
        out = model(example_input)
        logits = out.logits if hasattr(out, "logits") else out
        print(f"Forward pass OK — logits shape: {logits.shape}")

    # Wrapper to return logits directly
    class NSFWModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = self.m(x)
            return outputs.logits if hasattr(outputs, "logits") else outputs

    wrapper = NSFWModelWrapper(model)
    wrapper.eval()

    print("Exporting to ExecuTorch...")
    exported_program = export(wrapper, (example_input,))

    print(f"Applying {args.backend.upper()} backend...")
    et_program = lower_to_executorch(exported_program, args.backend)

    output_path = resolve_output_path("nsfw_model.pte", args.backend, args.output)
    write_program(et_program, output_path)

    print(f"\nSuccess! Saved: {output_path}")
    print("Next steps:")
    print("  1. Copy nsfw_model.pte to expo-pytorch/assets/models/")


if __name__ == "__main__":
    main()
