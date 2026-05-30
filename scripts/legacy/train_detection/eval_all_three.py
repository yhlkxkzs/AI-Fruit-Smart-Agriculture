#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 val/test 上评估三种分类模型（mobilenet_v3 / shufflenet_v2 / efficientnet_lite0）。

用法：
  python eval_all_three.py --data /path/to/dataset_cls
默认读取：
  runs/mobilenet_v3/weights/best.pt
  runs/shufflenet_v2/weights/best.pt
  runs/efficientnet_lite0/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"))


class ImageFolderOnlyNonEmpty(ImageFolder):
    @staticmethod
    def find_classes(directory: str):
        entries = sorted(e.name for e in os.scandir(directory) if e.is_dir() and not e.name.startswith("."))
        class_to_idx = {}
        for cls in entries:
            cls_dir = os.path.join(directory, cls)
            for f in os.listdir(cls_dir):
                if os.path.isfile(os.path.join(cls_dir, f)) and _is_image_file(f):
                    class_to_idx[cls] = len(class_to_idx)
                    break
        classes = list(class_to_idx.keys())
        return classes, class_to_idx


def align_dataset_to_ref_classes(ds: ImageFolderOnlyNonEmpty, ref_classes: list) -> None:
    """将 val/test 的类别索引对齐到训练时的 ref_classes（与 train_mobilenet_v3 一致）。"""
    ref_idx = {c: i for i, c in enumerate(ref_classes)}
    new_samples = []
    for p, c in ds.samples:
        name = ds.classes[c]
        if name in ref_idx:
            new_samples.append((p, ref_idx[name]))
    ds.samples = new_samples
    if hasattr(ds, "targets"):
        ds.targets = [tpl[1] for tpl in new_samples]
    ds.classes = list(ref_classes)
    ds.class_to_idx = {c: i for i, c in enumerate(ref_classes)}


@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for xs, ys in loader:
        xs, ys = xs.to(device), ys.to(device)
        logits = model(xs)
        preds = logits.argmax(1)
        correct += (preds == ys).sum().item()
        total += ys.size(0)
    return (correct / total) if total else 0.0, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="分类数据集根目录（含 val/、可选 test/）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mobilenet", type=str, default=str(RUNS_DIR / "mobilenet_v3" / "weights" / "best.pt"))
    p.add_argument("--shufflenet", type=str, default=str(RUNS_DIR / "shufflenet_v2" / "weights" / "best.pt"))
    p.add_argument("--efficientnet", type=str, default=str(RUNS_DIR / "efficientnet_lite0" / "weights" / "best.pt"))
    args = p.parse_args()

    def _build_model_from_ckpt(ckpt: dict):
        """与 api_mobilenet.py 保持一致：根据 model_type 构建模型并加载权重。"""
        model_type = (ckpt.get("model_type") or "mobilenet_v3").lower()
        class_names = ckpt.get("classes")
        num_classes = ckpt.get("num_classes")
        if class_names is None:
            class_names = [str(i) for i in range(num_classes or 0)]
        if num_classes is None:
            num_classes = len(class_names)
        state = ckpt["model_state_dict"]

        if model_type == "efficientnet_lite0":
            try:
                import timm  # type: ignore
            except ImportError as e:
                raise ImportError("EfficientNet-Lite0 权重需要 timm: pip install timm") from e
            model = timm.create_model("tf_efficientnet_lite0", pretrained=False, num_classes=num_classes)
            model.load_state_dict(state, strict=True)
            model.eval()
            return model

        if model_type == "shufflenet_v2":
            variant = ckpt.get("variant", "x1_0")
            if variant == "x0_5":
                model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
            else:
                model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            model.load_state_dict(state, strict=True)
            model.eval()
            return model

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    weights = {
        "mobilenet_v3": Path(args.mobilenet),
        "shufflenet_v2": Path(args.shufflenet),
        "efficientnet_lite0": Path(args.efficientnet),
    }
    ckpts = {}
    models = {}
    classes_ref = None
    for name, wp in weights.items():
        if not wp.is_absolute():
            wp = PROJECT_ROOT / wp
        if not wp.exists():
            raise FileNotFoundError(f"权重不存在: {wp}")
        ckpt = torch.load(wp, map_location="cpu", weights_only=False)
        ckpts[name] = ckpt
        classes = ckpt.get("classes")
        if classes_ref is None:
            classes_ref = classes
        model = _build_model_from_ckpt(ckpt).to(device)
        models[name] = model
        print(f"Loaded {name}: num_classes={ckpt.get('num_classes')} classes={classes}")

    data_root = Path(args.data)
    ref_classes = classes_ref
    if ref_classes is None:
        raise RuntimeError("无法从权重中读取 classes，无法对齐 val 标签。")

    for split in ["val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        ds = ImageFolderOnlyNonEmpty(str(split_dir), transform=transform)
        if len(ds) == 0:
            continue
        align_dataset_to_ref_classes(ds, ref_classes)
        if len(ds) == 0:
            print(f"\n[{split}] 对齐后无样本（train/val 类名不一致？），跳过。")
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"\n[{split}] samples={len(ds)} (aligned to train class order, nc={len(ref_classes)})")
        for name, model in models.items():
            acc, n = eval_one(model, loader, device)
            print(f"  {name}: acc={acc:.4f} (n={n})")


if __name__ == "__main__":
    main()

