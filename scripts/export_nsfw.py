"""
Export the NSFW classifier (Ateeqq/nsfw-image-detection) to ExecuTorch .pte format.

Labels: Typically SFW / NSFW (printed at runtime from model.config.id2label)

useClassification in react-native-executorch handles image preprocessing internally
(resize to 224x224, ImageNet normalization), so the app only needs to pass a URI.

Prerequisites:
    pip install torch torchvision
    pip install transformers
    pip install executorch

Usage:
    cd scripts/
    python export_nsfw.py
    # Output: nsfw_model.pte
"""

import json
import torch
from transformers import AutoModelForImageClassification
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge


MODEL_ID = "Ateeqq/nsfw-image-detection"


def main():
    print(f"Loading {MODEL_ID} from HuggingFace...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    labels = []
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

    print("Applying XNNPACK backend...")
    edge_manager = to_edge(exported_program)
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    et_program = edge_manager.to_executorch()

    output_path = "nsfw_model.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    labels_path = "nsfw_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSuccess! Saved: {output_path}")
    print(f"Saved labels: {labels_path}")
    print("Next steps:")
    print("  1. Copy nsfw_model.pte and nsfw_labels.json to expo-pytorch/assets/models/")


if __name__ == "__main__":
    main()
