#!/usr/bin/env python3
"""Run all category models listed in models_manifest.json on changed images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MANIFEST = SCRIPT_DIR / "models_manifest.json"

try:
    import timm
except ImportError:
    timm = None

from afsa_write_predictions import OUT, append_row, ensure_parent


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


def build_backbone(model_id: str, timm_name: str, img_size: int = 224) -> nn.Module:
    if model_id == "mobilenet_v3":
        from torchvision.models import mobilenet_v3_large

        m = mobilenet_v3_large(weights=None)
        m.classifier = nn.Identity()
        return m
    if model_id == "shufflenet_v2":
        from torchvision.models import shufflenet_v2_x1_0

        m = shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Identity()
        return m
    if timm is None:
        raise RuntimeError("timm required for this backbone")
    return timm.create_model(timm_name, pretrained=False, num_classes=0)


def infer_feat_dim(backbone: nn.Module, img_size: int = 224) -> int:
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, img_size, img_size))
    if out.dim() > 2:
        return int(out.flatten(1).shape[-1])
    return int(out.shape[-1])


def load_model(export_dir: Path, meta: dict, device: torch.device) -> tuple[nn.Module, list[str]]:
    ckpt_path = export_dir / "best.pt"
    classes_path = export_dir / "classes.json"
    cls_meta = json.loads(classes_path.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt

    if cls_meta.get("task_type") == "classification_multistate" or "species_classes" in cls_meta:
        model_id = cls_meta.get("model_id") or export_dir.name
        timm_name = cls_meta.get("timm_name") or "tf_efficientnet_lite0"
        species = cls_meta["species_classes"]
        states = cls_meta.get("state_classes") or []
        backbone = build_backbone(model_id, timm_name)
        feat_dim = infer_feat_dim(backbone)
        model = DualHeadClassifier(backbone, feat_dim, len(species), len(states) or 1)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model, species

    classes = cls_meta.get("classes") or cls_meta.get("species_classes") or []
    model_type = cls_meta.get("model_type", export_dir.name)
    timm_name = cls_meta.get("timm_name")
    if timm_name and timm is not None:
        model = timm.create_model(timm_name, pretrained=False, num_classes=len(classes))
    elif model_type in ("efficientnet_lite0", "dual_head_efficientnet_lite0"):
        if timm is None:
            raise RuntimeError("timm required for efficientnet_lite0")
        model = timm.create_model("tf_efficientnet_lite0", pretrained=False, num_classes=len(classes))
    elif model_type == "mobilenet_v3":
        from torchvision.models import mobilenet_v3_large

        model = mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    elif model_type == "shufflenet_v2":
        from torchvision.models import shufflenet_v2_x1_0

        model = shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    else:
        raise RuntimeError(f"unsupported single-head model: {export_dir}")
    if isinstance(state, dict) and any(k.startswith("backbone.") for k in state):
        raise RuntimeError(f"dual-head checkpoint in single-head loader: {export_dir}")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, classes


def predict_one(model: nn.Module, img: Image.Image, classes: list[str], device: torch.device) -> tuple[str, float]:
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = prob.max(0)
    return classes[int(idx)], float(conf.item())


def load_routes(images_file: Path) -> list[dict]:
    routes_file = Path("/tmp/afsa_routes.json")
    if routes_file.is_file():
        data = json.loads(routes_file.read_text(encoding="utf-8"))
        routes = data.get("routes") or []
        if routes:
            return routes
    routes = []
    for line in images_file.read_text(encoding="utf-8").splitlines():
        p = line.strip()
        if p:
            routes.append({"image_path": p, "task_type": "fruit_category"})
    return routes


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: afsa_run_category_models.py /tmp/images.txt", file=sys.stderr)
        sys.exit(2)
    images_file = Path(sys.argv[1])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exports_root = REPO_ROOT / manifest.get("exports_root", "exports")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    routes = [r for r in load_routes(images_file) if str(r.get("task_type", "")).startswith("fruit")]
    if not routes:
        print("[error] no routes to run (check sidecar + image_path)", file=sys.stderr)
        sys.exit(1)
    ensure_parent()
    OUT.write_text(json.dumps({"predictions": []}, indent=2) + "\n", encoding="utf-8")

    for spec in manifest.get("category_models", []):
        github_model_id = spec["github_model_id"]
        export_dir = exports_root / spec["export_dir"]
        pt = export_dir / "best.pt"
        if not pt.is_file():
            print(f"[skip] missing {pt}")
            continue
        try:
            model, classes = load_model(export_dir, {}, device)
        except Exception as exc:
            print(f"[skip] load failed {github_model_id}: {exc}")
            continue
        for route in routes:
            rel = route["image_path"]
            img_path = REPO_ROOT / rel
            if not img_path.is_file():
                print(f"[warn] missing image {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            pred_class, conf = predict_one(model, img, classes, device)
            append_row(
                {
                    "image": route["image_path"],
                    "github_path": route["image_path"],
                    "afsa_detection_id": route.get("afsa_detection_id"),
                    "github_model_id": github_model_id,
                    "predicted_class": pred_class,
                    "confidence": conf,
                }
            )
        print(f"[ok] {github_model_id} routes={len(routes)}")


if __name__ == "__main__":
    main()
