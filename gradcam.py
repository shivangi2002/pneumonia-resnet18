"""
Grad-CAM for pneumonia-resnet18
===============================

Generates heatmaps showing which regions of a chest X-ray drove the model's
prediction.

How it works (the 5 steps you explain in the interview):
  1. Grab the feature maps from the last conv layer  (layer4[-1] -> [1, 512, 7, 7])
  2. Compute the gradient of the target class score w.r.t. those feature maps
  3. Average those gradients spatially -> one importance weight per feature map
  4. Weighted sum of the feature maps, then ReLU (keep only positive evidence)
  5. Upscale 7x7 -> 224x224 and overlay on the original image

Usage (run from the project root):
    python gradcam.py
    python gradcam.py --n-per-class 4
    python gradcam.py --errors            # find FALSE POSITIVES (best story!)
    python gradcam.py --checkpoint checkpoints/<other>.pth
"""

import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")          # save to file, no GUI needed
import matplotlib.pyplot as plt
from matplotlib import cm

from src.model import get_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Must match XRayDataset.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_EXTS = (".jpeg", ".jpg", ".png")

# Same transform as your val/test pipeline in main.py / test_eval.py
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activation = None
        # Forward hook: stash the feature maps produced by the target layer
        self.handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activation = output          # [1, 512, 7, 7] for ResNet18 layer4

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)

        # ---- forward (NO torch.no_grad here - we need the graph) ----
        logits = self.model(input_tensor)                 # [1, 2]
        probs = F.softmax(logits, dim=1)[0]
        pred = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred                            # explain what it predicted
        score = logits[0, target_class]                    # scalar

        # ---- STEP 2: gradient of class score w.r.t. feature maps ----
        grads = torch.autograd.grad(score, self.activation)[0]   # [1, 512, 7, 7]
        acts = self.activation                                    # [1, 512, 7, 7]

        # ---- STEP 3: average gradients spatially -> importance weights ----
        weights = grads.mean(dim=(2, 3), keepdim=True)            # [1, 512, 1, 1]

        # ---- STEP 4: weighted sum of feature maps, then ReLU ----
        cam = (weights * acts).sum(dim=1, keepdim=True)           # [1, 1, 7, 7]
        cam = F.relu(cam)

        # ---- STEP 5: upscale to image size + normalise to [0, 1] ----
        cam = F.interpolate(cam, size=(224, 224),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)                              # no positive evidence

        return cam, pred, probs.detach().cpu().numpy(), target_class

    def close(self):
        self.handle.remove()


def load_model(checkpoint_path):
    model = get_model(fine_tune_layers=0)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # IMPORTANT: get_model() freezes the backbone (requires_grad=False), so no
    # autograd graph is built through the conv layers and autograd.grad would
    # fail with "does not require grad". We re-enable grads purely so the graph
    # exists - we never call optimizer.step(), so the weights never change.
    for p in model.parameters():
        p.requires_grad_(True)

    return model


def list_images(split_dir, class_name):
    d = os.path.join(split_dir, class_name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing folder: {d}")
    files = [os.path.join(d, f) for f in sorted(os.listdir(d))
             if f.lower().endswith(IMG_EXTS)]
    return files


def make_figure(img_path, cam, true_idx, pred_idx, probs, out_path):
    """Save a 3-panel figure: original | heatmap | overlay."""
    pil = Image.open(img_path).convert("RGB").resize((224, 224))
    display = np.asarray(pil).astype(np.float32) / 255.0

    heat = cm.jet(cam)[..., :3]                      # [224, 224, 3] in [0,1]
    overlay = np.clip(0.5 * display + 0.5 * heat, 0, 1)

    correct = (true_idx == pred_idx)
    mark = "CORRECT" if correct else "WRONG"
    title = (f"True: {CLASS_NAMES[true_idx]}   |   "
             f"Pred: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.2f})   |   {mark}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(display);  axes[0].set_title("X-ray")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay);  axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(title, fontsize=13,
                 color=("green" if correct else "red"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(PROJECT_ROOT, "checkpoints",
                                         "lr0.001_bs32_epochs30_w2.pth"),
                    help="path to your .pth checkpoint")
    ap.add_argument("--data-dir",
                    default=os.path.join(PROJECT_ROOT, "data", "test"),
                    help="folder containing NORMAL/ and PNEUMONIA/")
    ap.add_argument("--outdir",
                    default=os.path.join(PROJECT_ROOT, "gradcam_outputs"))
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--errors", action="store_true",
                    help="hunt for FALSE POSITIVES (normals predicted pneumonia)")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        available = os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else "no checkpoints/ dir"
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Available: {available}"
        )

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.checkpoint)
    target_layer = model.layer4[-1]        # last BasicBlock -> [1, 512, 7, 7]
    gradcam = GradCAM(model, target_layer)

    # ---- pick which images to explain ----
    jobs = []   # (img_path, true_idx)

    if args.errors:
        print("Searching test/NORMAL for false positives...")
        found = 0
        for p in list_images(args.data_dir, "NORMAL"):
            x = val_transform(Image.open(p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                pred = int(model(x).argmax(dim=1).item())
            if pred == 1:                       # normal misclassified as pneumonia
                jobs.append((p, 0))
                found += 1
                if found >= args.n_per_class:
                    break
        print(f"Found {found} false positive(s).")
        if not jobs:
            print("No false positives found - falling back to random sampling.")

    if not jobs:
        for idx, cname in enumerate(CLASS_NAMES):
            files = list_images(args.data_dir, cname)
            picks = random.sample(files, min(args.n_per_class, len(files)))
            jobs.extend((p, idx) for p in picks)

    # ---- run Grad-CAM on each ----
    print(f"\nGenerating Grad-CAM for {len(jobs)} image(s)...\n")
    for i, (img_path, true_idx) in enumerate(jobs):
        x = val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        cam, pred, probs, target = gradcam(x)

        tag = "WRONG" if pred != true_idx else "ok"
        out_name = f"{CLASS_NAMES[true_idx]}_{i:02d}_{tag}.png"
        out_path = os.path.join(args.outdir, out_name)
        make_figure(img_path, cam, true_idx, pred, probs, out_path)

        print(f"  {os.path.basename(img_path):35s} "
              f"true={CLASS_NAMES[true_idx]:9s} "
              f"pred={CLASS_NAMES[pred]:9s} "
              f"p={probs[pred]:.3f}  ->  {out_name}")

    gradcam.close()

    print(f"\nDone. Heatmaps saved to: {args.outdir}\n")
    print("WHAT TO LOOK FOR:")
    print("  * Heat over the lung fields / areas of opacity  -> model is looking")
    print("    at clinically relevant regions. Reassuring.")
    print("  * Heat on borders, corners, text markers, shoulders, or the")
    print("    diaphragm edge -> possible spurious correlation. This is the")
    print("    MORE interesting finding - it's exactly why explainability matters.")
    print("  * Run with --errors to see where it looks on FALSE POSITIVES.")
    print("    Those 82 misclassified normals are your best interview story.")


if __name__ == "__main__":
    main()