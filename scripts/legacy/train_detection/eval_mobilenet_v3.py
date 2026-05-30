#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在验证集/测试集上评估训练好的 MobileNet，并可选保存一批「预测结果」样例图供查看。

用法：
  python eval_mobilenet_v3.py --data datasetlocal_cls --weights train_detection/runs/mobilenet_v3/weights/best.pt
  python eval_mobilenet_v3.py --data datasetlocal_cls --save-samples 20   # 额外保存 20 张样例图
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from torchvision.datasets import ImageFolder

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


def load_model(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("classes")
    num_classes = ckpt.get("num_classes")
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    if num_classes is None:
        num_classes = len(class_names)

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, class_names, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "datasetlocal_cls"), help="分类数据集根目录（含 val/、可选 test/）")
    parser.add_argument("--weights", type=str, default=str(RUNS_DIR / "mobilenet_v3" / "weights" / "best.pt"))
    parser.add_argument("--save-samples", type=int, default=0, help="保存多少张预测样例图到 runs/mobilenet_v3/eval_samples，0 不保存")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_root = Path(args.data)
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"权重不存在: {weights_path}")

    model, class_names, device = load_model(weights_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = {}
    for split in ["val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        ds = ImageFolderOnlyNonEmpty(str(split_dir), transform=transform)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        correct, total = 0, 0
        per_class_correct = {c: 0 for c in class_names}
        per_class_total = {c: 0 for c in class_names}
        all_paths, all_preds, all_labels, all_probs = [], [], [], []

        model.eval()
        with torch.no_grad():
            for xs, ys in loader:
                xs, ys = xs.to(device), ys.to(device)
                logits = model(xs)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(1)
                correct += (preds == ys).sum().item()
                total += ys.size(0)
                for i in range(ys.size(0)):
                    c = class_names[ys[i].item()]
                    per_class_total[c] = per_class_total.get(c, 0) + 1
                    if preds[i].item() == ys[i].item():
                        per_class_correct[c] = per_class_correct.get(c, 0) + 1

        acc = correct / total if total else 0
        results[split] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "per_class_acc": {c: per_class_correct.get(c, 0) / max(per_class_total.get(c, 1), 1) for c in class_names},
            "per_class_total": per_class_total,
        }
        print(f"\n===== {split.upper()} =====\n总样本: {total}  正确: {correct}  准确率: {acc:.4f} ({acc*100:.2f}%)")
        print("各类别准确率:")
        for c in class_names:
            n = per_class_total.get(c, 0)
            a = per_class_correct.get(c, 0) / max(n, 1)
            print(f"  {c}: {a:.4f}  ({per_class_correct.get(c, 0)}/{n})")

    # 保存样例图（仅 val）
    if args.save_samples > 0 and (data_root / "val").exists():
        val_dir = data_root / "val"
        ds_val = ImageFolderOnlyNonEmpty(str(val_dir), transform=transform)
        if len(ds_val) > 0:
            out_dir = RUNS_DIR / "mobilenet_v3" / "eval_samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            indices = random.sample(range(len(ds_val)), min(args.save_samples, len(ds_val)))
            transform_inv = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
            for idx in indices:
                path, true_idx = ds_val.samples[idx]
                true_label = ds_val.classes[true_idx]
                img_pil = Image.open(path).convert("RGB")
                img_t = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(img_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                pred_label = class_names[pred_idx]
                conf = float(probs[pred_idx])
                ok = "✓" if pred_label == true_label else "✗"
                name = Path(path).name
                out_path = out_dir / f"{ok}_{pred_label}_{conf:.2f}_true_{true_label}_{name}"
                img_pil.save(out_path)
            print(f"\n已保存 {len(indices)} 张样例到: {out_dir}")

    print("\n评估完成。")
    return results


if __name__ == "__main__":
    main()
