#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet-Lite0 分类训练：轻量级移动端模型，适合水果识别。

依赖：pip install timm

用法：
  首次训练：
    python train_efficientnet_lite0.py --data /path/to/dataset_cls --epochs 50

  继续训练：
    python train_efficientnet_lite0.py --data /path/to/dataset_cls --resume runs/efficientnet_lite0/weights/best.pt --epochs 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import timm
except ImportError:
    print("请先安装 timm: pip install timm")
    sys.exit(1)


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"))


class ImageFolderOnlyNonEmpty(ImageFolder):
    """只加载有图片的类别目录，跳过空目录。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
DEFAULT_SAVE_DIR = RUNS_DIR / "efficientnet_lite0"
MODEL_TYPE = "efficientnet_lite0"


def get_args():
    p = argparse.ArgumentParser(description="EfficientNet-Lite0 分类训练（支持从已有权重继续训练）")
    p.add_argument("--data", type=str, required=True, help="分类数据集根目录，含 train/ 与 val/")
    p.add_argument("--resume", type=str, default="", help="已有权重路径，继续训练时使用")
    p.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR), help="权重与日志保存目录")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="学习率（继续训练时建议 1e-4）")
    p.add_argument("--device", type=str, default="", help="留空则自动选择 cuda/cpu")
    return p.parse_args()


def build_model(num_classes: int, resume_path: str | None = None):
    """构建 EfficientNet-Lite0（timm）；若提供 resume_path 则加载已有权重。"""
    # 预训练权重需要联网下载；若下载失败则自动回退到随机初始化，保证训练可继续进行。
    try:
        model = timm.create_model("tf_efficientnet_lite0", pretrained=True, num_classes=num_classes)
    except Exception as e:
        print(f"  [Warn] 预训练权重下载/加载失败，将使用 pretrained=False 继续训练。原因: {e}")
        model = timm.create_model("tf_efficientnet_lite0", pretrained=False, num_classes=num_classes)

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
        if state is None:
            state = ckpt
        model_dict = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
        if matched:
            model.load_state_dict(matched, strict=False)
            print(f"  [Resume] 已加载 {len(matched)}/{len(state)} 个兼容参数")
        else:
            print("  [Resume] 无兼容参数，从预训练 backbone 开始训练")
    return model


def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += x.size(0)
    return correct / max(n, 1)


def main():
    args = get_args()
    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        print(f"错误：训练集目录不存在 {train_dir}")
        sys.exit(1)
    if not val_dir.exists():
        print(f"错误：验证集目录不存在 {val_dir}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  model: {MODEL_TYPE}")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolderOnlyNonEmpty(str(train_dir), transform=transform_train)
    val_ds = ImageFolderOnlyNonEmpty(str(val_dir), transform=transform_val)
    if val_ds.classes != train_ds.classes:
        train_idx = {c: i for i, c in enumerate(train_ds.classes)}
        new_samples = [(p, train_idx[val_ds.classes[c]]) for p, c in val_ds.samples if val_ds.classes[c] in train_idx]
        val_ds.samples = new_samples
        val_ds.classes = train_ds.classes
        val_ds.class_to_idx = train_ds.class_to_idx

    num_classes = len(train_ds.classes)
    print(f"Training with {num_classes} classes: {train_ds.classes}")

    model = build_model(num_classes, resume_path=args.resume if args.resume else None)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    best_acc = 0.0
    ckpt_extra = {"model_type": MODEL_TYPE, "num_classes": num_classes, "classes": train_ds.classes}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), **ckpt_extra},
                weights_dir / "best.pt",
            )
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  best={best_acc:.4f}")

    print(f"Best val_acc={best_acc:.4f}, saved to {weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
