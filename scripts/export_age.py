"""
Export the age classifier (nateraw/vit-age-classifier) to ExecuTorch .pte format.

The model is a Vision Transformer (ViT-base) fine-tuned on age ranges:
    1-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

useClassification in react-native-executorch handles image preprocessing internally
(resize to 224x224, ImageNet normalization), so the app only needs to pass a URI.

Prerequisites:
    pip install torch torchvision
    pip install transformers
    pip install executorch

Usage:
    cd scripts/
    python export_age.py
    # Output: age_model.pte
"""

import torch
from transformers import AutoModelForImageClassification
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge


MODEL_ID = "nateraw/vit-age-classifier"


def main():
    print(f"Loading {MODEL_ID} from HuggingFace...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Print labels so we know the output order
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        print(f"Labels ({len(labels)}): {labels}")

    # ViT standard input: [batch, 3, 224, 224]
    example_input = torch.randn(1, 3, 224, 224)

    # Verify forward pass
    with torch.no_grad():
        out = model(example_input)
        print(f"Forward pass OK — logits shape: {out.logits.shape}")

    # Wrap to return logits tensor directly (ExecuTorch needs a plain tensor or tuple)
    class AgeModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x).logits  # shape: [1, num_classes]

    wrapper = AgeModelWrapper(model)
    wrapper.eval()

    print("Exporting to ExecuTorch...")
    exported_program = export(wrapper, (example_input,))

    print("Applying XNNPACK backend...")
    edge_manager = to_edge(exported_program)
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    et_program = edge_manager.to_executorch()

    output_path = "age_model.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"\nSuccess! Saved: {output_path}")
    print("Next steps:")
    print("  1. Copy age_model.pte to expo-pytorch/assets/models/")
    print("  2. In aiScreen.tsx, set AGE_LABELS to:", labels if 'labels' in dir() else "[check model.config.id2label]")


if __name__ == "__main__":
    main()
