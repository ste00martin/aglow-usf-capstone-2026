"""
Export the gender classifier (rizvandwiki/gender-classification) to ExecuTorch .pte format.

Labels: Male, Female (or similar — printed at runtime, check model.config.id2label).

useClassification in react-native-executorch handles image preprocessing internally
(resize to 224x224, ImageNet normalization), so the app only needs to pass a URI.

Prerequisites:
    pip install torch torchvision
    pip install transformers
    pip install executorch
    # For iOS/CoreML export on macOS, also install ExecuTorch CoreML requirements

Usage:
    cd scripts/
    python export_gender.py
    # Output: gender_model.pte
    python export_gender.py --backend coreml
    # Output: gender_model_coreml.pte
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


MODEL_ID = "rizvandwiki/gender-classification"


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
        help="Optional output path. Defaults to gender_model.pte for xnnpack and gender_model_coreml.pte for coreml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {MODEL_ID} from HuggingFace...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        print(f"Labels ({len(labels)}): {labels}")

    example_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(example_input)
        print(f"Forward pass OK — logits shape: {out.logits.shape}")

    class GenderModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x).logits  # shape: [1, num_classes]

    wrapper = GenderModelWrapper(model)
    wrapper.eval()

    print("Exporting to ExecuTorch...")
    exported_program = export(wrapper, (example_input,))

    print(f"Applying {args.backend.upper()} backend...")
    et_program = lower_to_executorch(exported_program, args.backend)

    output_path = resolve_output_path("gender_model.pte", args.backend, args.output)
    write_program(et_program, output_path)

    print(f"\nSuccess! Saved: {output_path}")
    print("Next steps:")
    print("  1. Copy gender_model.pte to expo-pytorch/assets/models/")


if __name__ == "__main__":
    main()
