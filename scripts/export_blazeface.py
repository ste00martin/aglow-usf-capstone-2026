"""
Export BlazeFace face detector to ExecuTorch .pte format.

Only the neural network forward pass is exported. Preprocessing (resize + normalize)
and postprocessing (anchor decode + NMS) run in JavaScript on the app side.

Prerequisites:
    pip install torch torchvision
    pip install executorch  # follow https://pytorch.org/executorch/stable/getting-started.html

Files required in scripts/:
    blazeface.py     — BlazeFace class (download from hollance/BlazeFace-PyTorch)
    blazeface.pth    — pretrained weights (download from the same repo)
    anchors.npy      — precomputed anchors (download from the same repo)

Outputs:
    blazeface.pte    — ExecuTorch binary for on-device inference
    anchors.json     — anchors converted to JSON for JS postprocessing

Usage:
    cd scripts/
    python export_blazeface.py
"""

import sys
import json
import torch
import numpy as np

sys.path.insert(0, ".")

try:
    from blazeface import BlazeFace
except ImportError:
    print("ERROR: blazeface.py not found in scripts/.")
    print("Download from: https://github.com/hollance/BlazeFace-PyTorch")
    sys.exit(1)

from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge


class BlazeFaceExportWrapper(torch.nn.Module):
    """
    Thin wrapper so torch.export gets a clean tuple return signature.

    BlazeFace.forward() already does the right thing:
      - Expects input in [-1, 1] range (we normalize in JS before calling)
      - Returns [boxes (b,896,16), scores (b,896,1)] — raw logits, no NMS

    We wrap to return a tuple (boxes, scores) instead of a list,
    which torch.export handles more reliably.
    """

    def __init__(self, blazeface_model: BlazeFace):
        super().__init__()
        self.model = blazeface_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # BlazeFace.forward() returns [r (boxes), c (scores)]
        out = self.model(x)
        return out[0], out[1]  # boxes [b,896,16], scores [b,896,1]


def export_anchors(anchors_path: str, output_path: str) -> None:
    """Convert anchors.npy to anchors.json for use in JS postprocessing."""
    anchors = np.load(anchors_path, allow_pickle=True)
    anchors_list = anchors.tolist()
    with open(output_path, "w") as f:
        json.dump(anchors_list, f, separators=(",", ":"))
    print(f"Anchors saved: {output_path}  (shape: {anchors.shape})")


def main():
    # ── Load model ──────────────────────────────────────────────────────────────
    print("Loading BlazeFace...")
    detector = BlazeFace()
    detector.load_weights("blazeface.pth")
    anchors_array = np.load("anchors.npy", allow_pickle=True)
    detector.anchors = torch.tensor(anchors_array, dtype=torch.float32)
    detector.eval()

    # Export anchors for JS postprocessing
    export_anchors("anchors.npy", "anchors.json")

    # ── Wrap for export ──────────────────────────────────────────────────────────
    wrapper = BlazeFaceExportWrapper(detector)
    wrapper.eval()

    example_input = torch.randn(1, 3, 128, 128)

    # Verify forward pass runs before export
    with torch.no_grad():
        try:
            out = wrapper(example_input)
            if isinstance(out, (tuple, list)):
                print(f"Forward pass OK — outputs: {[o.shape for o in out]}")
            else:
                print(f"Forward pass OK — output: {out.shape}")
                print("WARNING: expected tuple output (scores, boxes). Check blazeface.py.")
        except Exception as e:
            print(f"ERROR: Forward pass failed: {e}")
            print("Review BlazeFaceExportWrapper.forward() and adjust to match your blazeface.py.")
            sys.exit(1)

    # ── Export ───────────────────────────────────────────────────────────────────
    print("Exporting to ExecuTorch...")
    exported_program = export(wrapper, (example_input,))

    print("Applying XNNPACK backend...")
    edge_manager = to_edge(exported_program)
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    et_program = edge_manager.to_executorch()

    output_path = "blazeface.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"\nSuccess! Saved: {output_path}")
    print("Next steps:")
    print("  1. Copy blazeface.pte and anchors.json to expo-pytorch/assets/models/")
    print("  2. Run export_age.py and export_gender.py")


if __name__ == "__main__":
    main()
