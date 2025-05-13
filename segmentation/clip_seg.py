#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from clipseg.models.clipseg import CLIPDensePredT


def load_model(weights_path, device="cpu"):
    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def read_image(path):
    img = Image.open(path)
    return img, (img.size[1], img.size[0]), Path(path).stem


def build_preprocess():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])


def infer(model, img, prompts, preprocess, device="cpu"):
    x = preprocess(img).unsqueeze(0)
    x = x.repeat(len(prompts), 1, 1, 1).to(device)
    with torch.no_grad():
        logits, *_ = model(x, prompts)
    return logits


def logits_to_labels(logits, threshold=0.5):
    probs = torch.sigmoid(logits[:, 0])
    max_probs, cls_idx = probs.max(dim=0)
    labels = torch.where(max_probs > threshold, cls_idx + 1, torch.zeros_like(cls_idx))
    return labels.cpu().numpy().astype(np.uint8)


def resize_labels(labels, size):
    h, w = size
    return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def build_color_mask(labels, palette):
    return palette[labels]


def alpha_blend(img, mask, alpha=0.5):
    return (img * (1 - alpha) + mask * alpha).astype(np.uint8)


def save_numpy(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def render_panels(input_rgb, color_mask, overlay):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, ttl in zip(axes, (input_rgb, color_mask, overlay), ("Input", "Mask", "Overlay")):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(ttl)
    fig.tight_layout()
    return fig


def main(image_path, weights_path="clipseg/weights/rd64-uni.pth",
         prompts=["window", "door", "wall", "ceiling", "floor"],
         out_dir="outputs", threshold=0.5, device="cpu"):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights_path, device)
    img, size, name = read_image(image_path)

    logits = infer(model, img, prompts, build_preprocess(), device)
    labels = logits_to_labels(logits, threshold)
    labels_up = resize_labels(labels, size)

    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ], dtype=np.uint8)

    rgb_352 = np.array(img.resize((352, 352)))
    color_mask = build_color_mask(labels, palette)
    overlay = alpha_blend(rgb_352, color_mask)

    save_numpy(labels_up, out_dir / f"{name}.npy")

    fig = render_panels(rgb_352, color_mask, overlay)
    fig.savefig(out_dir / f"{name}_viz.png", dpi=300)
    plt.close(fig)

    cv2.imwrite(str(out_dir / f"{name}_mask.png"), color_mask[:, :, ::-1])

import argparse
# ---- entry point with recursive directory traversal ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano_root", required=True, help="Root directory of pano folders (e.g., panos/)")
    parser.add_argument("--weights", default="clipseg/weights/rd64-uni.pth")
    parser.add_argument("--out_root", default="segs", help="Root directory to store segmentations")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pano_root = Path(args.pano_root)
    out_root = Path(args.out_root)

    for subfolder in sorted(pano_root.iterdir()):
        if subfolder.is_dir():
            out_dir = out_root / subfolder.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(subfolder.glob("*.jpg")):
                print(f"Processing: {img_path}")
                main(str(img_path), args.weights, out_dir=str(out_dir), threshold=args.threshold)
