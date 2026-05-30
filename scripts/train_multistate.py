#!/usr/bin/env python3
"""Dual-head fruit classification: species + state from multistate manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    print("pip install timm", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TASK_DIR / "configs" / "multistate.yaml"


def load_yaml_simple(path: Path) -> dict:
    cfg: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            cfg[k] = v.lower() == "true"
        else:
            try:
                cfg[k] = int(v) if v.isdigit() else float(v)
            except ValueError:
                cfg[k] = v.strip('"').strip("'")
    return cfg


def resolve_path(raw: str) -> Path | None:
    p = Path(raw.strip().strip('"'))
    candidates = [p, Path(str(p).replace("/data1/yuhanlin", "/home/yuhanlin"))]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_labelmap(path: Path) -> tuple[list[str], dict[str, int]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    names = [x["object_name"] for x in items]
    idx = {n: i for i, n in enumerate(names)}
    return names, idx


class MultistateDataset(Dataset):
    def __init__(self, manifest_csv: Path, species_idx: dict[str, int], state_idx: dict[str, int], transform):
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []
        with open(manifest_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = resolve_path(row["path"])
                if path is None:
                    continue
                sp = row["species"]
                st = row["state"]
                if sp not in species_idx or st not in state_idx:
                    continue
                self.samples.append((path, species_idx[sp], state_idx[st]))
        if not self.samples:
            raise RuntimeError(f"No valid samples in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, sp, st = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sp, st


class DualHeadClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_species: int, num_states: int):
        super().__init__()
        self.backbone = backbone
        self.head_species = nn.Linear(feat_dim, num_species)
        self.head_state = nn.Linear(feat_dim, num_states)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return self.head_species(feat), self.head_state(feat)


def infer_feat_dim(backbone: nn.Module, img_size: int) -> int:
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, img_size, img_size))
    if out.dim() > 2:
        return int(out.flatten(1).shape[-1])
    return int(out.shape[-1])


def build_backbone(model_id: str, timm_name: str, pretrained: bool, img_size: int) -> nn.Module:
    if model_id == "mobilenet_v3":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_large(weights=weights)
        m.classifier = nn.Identity()
        return m
    if model_id == "shufflenet_v2":
        from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0

        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        m = shufflenet_v2_x1_0(weights=weights)
        m.fc = nn.Identity()
        return m
    try:
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    except Exception as e:
        print(f"[warn] pretrained=False fallback: {e}")
        return timm.create_model(timm_name, pretrained=False, num_classes=0)


def build_dual_head(model_id: str, timm_name: str, num_species: int, num_states: int, pretrained: bool, img_size: int):
    backbone = build_backbone(model_id, timm_name, pretrained, img_size)
    feat_dim = infer_feat_dim(backbone, img_size)
    return DualHeadClassifier(backbone, feat_dim, num_species, num_states)


def train_one_epoch(model, loader, crit_sp, crit_st, opt, device, state_weight: float):
    model.train()
    sp_ok = st_ok = n = 0
    loss_sum = 0.0
    for x, sp_y, st_y in loader:
        x = x.to(device)
        sp_y = sp_y.to(device)
        st_y = st_y.to(device)
        opt.zero_grad()
        sp_logits, st_logits = model(x)
        loss = crit_sp(sp_logits, sp_y) + state_weight * crit_st(st_logits, st_y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        sp_ok += (sp_logits.argmax(1) == sp_y).sum().item()
        st_ok += (st_logits.argmax(1) == st_y).sum().item()
        n += x.size(0)
    return loss_sum / n, sp_ok / n, st_ok / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sp_ok = st_ok = n = 0
    for x, sp_y, st_y in loader:
        x = x.to(device)
        sp_y = sp_y.to(device)
        st_y = st_y.to(device)
        sp_logits, st_logits = model(x)
        sp_ok += (sp_logits.argmax(1) == sp_y).sum().item()
        st_ok += (st_logits.argmax(1) == st_y).sum().item()
        n += x.size(0)
    return sp_ok / n, st_ok / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--model-id", default="", help="Export folder name under exports/")
    parser.add_argument("--timm-name", default="", help="Override timm backbone")
    args = parser.parse_args()

    cfg = load_yaml_simple(args.config)
    repo = REPO_ROOT
    manifest_dir = repo / cfg["manifest_dir"]
    species_names, species_idx = load_labelmap(repo / cfg["labelmap_species"])
    state_names, state_idx = load_labelmap(repo / cfg["labelmap_state"])

    epochs = args.epochs or int(cfg.get("epochs", 50))
    batch_size = args.batch_size or int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 1e-3))
    img_size = int(cfg.get("image_size", 224))
    model_id = args.model_id or cfg.get("model", "efficientnet_lite0")
    timm_name = args.timm_name or cfg.get("timm_name", "tf_efficientnet_lite0")
    pretrained = bool(cfg.get("pretrained", True))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = repo / cfg.get("runs_dir", "tasks/fruit_classification/runs/multistate") / run_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = MultistateDataset(manifest_dir / "train.csv", species_idx, state_idx, tf_train)
    val_ds = MultistateDataset(manifest_dir / "val.csv", species_idx, state_idx, tf_val)
    print(f"train={len(train_ds)} val={len(val_ds)} device={device} model={model_id} timm={timm_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_dual_head(
        model_id, timm_name, len(species_names), len(state_names), pretrained, img_size
    ).to(device)
    crit_sp = nn.CrossEntropyLoss()
    crit_st = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_meta = {
        "model_type": f"dual_head_{model_id}",
        "model_id": model_id,
        "timm_name": timm_name,
        "num_species": len(species_names),
        "num_states": len(state_names),
        "species_classes": species_names,
        "state_classes": state_names,
        "task_type": "classification_multistate",
    }

    best_score = 0.0
    log_path = save_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_sp, tr_st = train_one_epoch(
                model, train_loader, crit_sp, crit_st, opt, device, args.state_weight
            )
            va_sp, va_st = evaluate(model, val_loader, device)
            score = 0.7 * va_sp + 0.3 * va_st
            line = (
                f"Epoch {epoch}/{epochs} loss={tr_loss:.4f} "
                f"train_sp={tr_sp:.4f} train_st={tr_st:.4f} "
                f"val_sp={va_sp:.4f} val_st={va_st:.4f} score={score:.4f}"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()
            if score > best_score:
                best_score = score
                torch.save({"model_state_dict": model.state_dict(), **ckpt_meta}, weights_dir / "best.pt")

    export_dir = repo / cfg.get("exports_dir", "tasks/fruit_classification/exports") / model_id
    export_dir.mkdir(parents=True, exist_ok=True)
    best = weights_dir / "best.pt"
    if best.is_file():
        shutil_copy = __import__("shutil").copy2
        shutil_copy(best, export_dir / "best.pt")
        (export_dir / "classes.json").write_text(
            json.dumps(
                {
                    **ckpt_meta,
                    "best_score": best_score,
                    "run_dir": str(save_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Done. best_score={best_score:.4f} weights={weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
