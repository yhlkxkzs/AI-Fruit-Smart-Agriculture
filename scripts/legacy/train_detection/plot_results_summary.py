#!/usr/bin/env python3
"""生成多模型结果对比表与柱状图，保存到 runs/结果对比图.png"""
from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# 数据：从 results.csv 与日志整理
MODELS = [
    ("YOLOv8\n(github)", 0.723, 0.463),
    ("YOLO11\n(github)", 0.724, 0.467),
    ("YOLOv8\n(local)", 0.923, 0.803),
    ("YOLO11\n(local)", 0.920, 0.799),
]
CLASS_NAMES = ["apple", "mango", "grape", "tomato", "strawberry", "broccoli"]
YOLOV8_LOCAL_MAP = [0.782, 0.990, 0.872, 0.913, 0.995, 0.995]
YOLO11_LOCAL_MAP = [0.784, 0.992, 0.881, 0.915, 0.995, 0.995]

OUT_DIR = Path(__file__).resolve().parents[1] / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not HAS_MPL:
        print("matplotlib not installed, skip figure.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # 1) 柱状图：各模型 mAP50 / mAP50-95
    ax1 = axes[0]
    x = np.arange(len(MODELS))
    w = 0.35
    map50 = [m[1] for m in MODELS]
    map5095 = [m[2] for m in MODELS]
    ax1.bar(x - w / 2, map50, w, label="mAP50", color="steelblue")
    ax1.bar(x + w / 2, map5095, w, label="mAP50-95", color="coral")
    ax1.set_ylabel("mAP")
    ax1.set_title("Detection models: mAP50 vs mAP50-95 (val)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in MODELS])
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # 2) 柱状图：Local 两模型各类别 mAP50
    ax2 = axes[1]
    x2 = np.arange(len(CLASS_NAMES))
    width = 0.35
    ax2.bar(x2 - width / 2, YOLOV8_LOCAL_MAP, width, label="YOLOv8 local", color="steelblue")
    ax2.bar(x2 + width / 2, YOLO11_LOCAL_MAP, width, label="YOLO11 local", color="seagreen")
    ax2.set_ylabel("mAP50")
    ax2.set_title("Local detection: mAP50 per class (val)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "结果对比图.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")

    # 3) 表格图：检测模型整体对比
    fig2, ax3 = plt.subplots(figsize=(8, 3))
    ax3.axis("off")
    table_data = [
        ["Model", "Data", "Precision", "Recall", "mAP50", "mAP50-95"],
        ["YOLOv8", "github", "0.709", "0.739", "0.723", "0.463"],
        ["YOLO11", "github", "0.693", "0.748", "0.724", "0.467"],
        ["YOLOv8", "local", "0.922", "0.900", "0.923", "0.803"],
        ["YOLO11", "local", "0.925", "0.892", "0.920", "0.799"],
    ]
    t = ax3.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.2, 2.2)
    ax3.set_title("Detection models (val)", fontsize=12)
    plt.tight_layout()
    out_table = OUT_DIR / "结果对比表.png"
    plt.savefig(out_table, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_table}")


if __name__ == "__main__":
    main()
