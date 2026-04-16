"""
Export the Text Moderation classifier (unitary/toxic-bert) to ExecuTorch .pte format.

This uses BERT-base (standard scaled dot-product attention), which is fully compatible
with the CoreML backend on iOS — unlike DeBERTa, whose disentangled attention drops
positional inputs in CoreML's gather/embedding layer and produces NaN.

Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate (multi-label sigmoid)

Prerequisites:
    pip install torch torchvision
    pip install transformers
    pip install executorch
    # For iOS/CoreML export on macOS, also install ExecuTorch CoreML requirements

Usage:
    cd scripts/
    python export_text.py
    # Output: text_model.pte
    python export_text.py --backend coreml
    # Output: text_model_coreml.pte
"""

import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.export import export

from executorch_export_utils import (
    lower_to_executorch,
    resolve_output_path,
    write_program,
)


# BERT-base multi-label toxic content classifier — CoreML compatible (standard attention).
# 6 labels from the Jigsaw Toxic Comment dataset: toxic, severe_toxic, obscene, threat, insult, identity_hate
MODEL_ID = "unitary/toxic-bert"


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
        help="Optional output path. Defaults to text_model.pte for xnnpack and text_model_coreml.pte for coreml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {MODEL_ID} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Print labels if available
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        print(f"Labels ({len(labels)}): {labels}")
    else:
        print("Warning: No label mapping found in model config.")

    # Typical input shape for text classification models
    example_text = "This is a test sentence."
    inputs = tokenizer(example_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    example_input_ids = inputs["input_ids"]
    example_attention_mask = inputs["attention_mask"]

    # Sanity check forward pass
    with torch.no_grad():
        out = model(input_ids=example_input_ids, attention_mask=example_attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        print(f"Forward pass OK — logits shape: {logits.shape}")

    # Wrapper to return logits directly and take input_ids and attention_mask
    class TextModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits if hasattr(outputs, "logits") else outputs

    wrapper = TextModelWrapper(model)
    wrapper.eval()

    print("Exporting to ExecuTorch...")
    # Provide the multiple inputs as a tuple for export
    exported_program = export(wrapper, (example_input_ids, example_attention_mask))

    print(f"Applying {args.backend.upper()} backend...")
    if args.backend == "coreml":
        # DeBERTa's first op is an integer embedding lookup which the MpsGraph engine
        # (GPU / Neural Engine) cannot handle — it causes NaN outputs on the iOS Simulator
        # and is unreliable on device with ComputeUnit.ALL.
        # CPU_ONLY + FLOAT32 sidesteps this entirely and works correctly everywhere.
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        from executorch.exir import to_edge_transform_and_lower

        et_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[
                CoreMLPartitioner(
                    compile_specs=[
                        CoreMLBackend.generate_minimum_deployment_target_compile_spec(
                            ct.target.iOS16
                        ),
                        CoreMLBackend.generate_compute_unit_compile_spec(
                            ct.ComputeUnit.CPU_ONLY
                        ),
                        CoreMLBackend.generate_compute_precision_compile_spec(
                            ct.precision.FLOAT32
                        ),
                    ]
                )
            ],
        ).to_executorch()
    else:
        et_program = lower_to_executorch(exported_program, args.backend)

    output_path = resolve_output_path("text_model.pte", args.backend, args.output)
    write_program(et_program, output_path)

    print(f"\nSuccess! Saved: {output_path}")
    print("Next steps:")
    print("  1. Copy text_model.pte to expo-pytorch/assets/models/")


if __name__ == "__main__":
    main()
