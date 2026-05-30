# 导出模型清单 — fruit_classification

统一目录：`exports/<model_id>/` 内含 `best.pt` + `classes.json`（可选 `README.md`）。

Multistate 标签：11 品种 + 7 状态，见各目录 `classes.json` 与 `_shared_classes.json`。

## 全部推荐模型

| # | 目录 | timm 骨干 | 用途 | `best.pt` |
|---|------|-----------|------|-----------|
| — | `deploy_efficientnet_lite0/` | `tf_efficientnet_lite0` | **GitHub / App 上线**（与推理仓一致） | ✅ GitHub |
| 1 | `efficientnet_lite0_multistate/` | `tf_efficientnet_lite0` | multistate 主结果（score 0.9886） | ✅ GitHub |
| 2 | `efficientnet_lite0/` | `tf_efficientnet_lite0` | 本地对比（与 multistate 同轮） | ✅ GitHub |
| 3 | `mobilenet_v3/` | `mobilenetv3_large_100` | 旧单头对比（2026-03，非 multistate） | ✅ GitHub |
| 4 | `shufflenet_v2/` | `shufflenet_v2_x1_0` | 旧单头对比（2026-03，非 multistate） | ✅ GitHub |
| 5 | `efficientnet_lite4/` | `tf_efficientnet_lite4` | multistate 对比 | ✅ GitHub |
| 6 | `mobilevit_s/` | `mobilevit_s` | multistate 对比 | ✅ GitHub |
| 7 | `efficientnetv2_s/` | `tf_efficientnetv2_s` | multistate 对比（score **0.9888** 最高） | ⚠️ 仅本地（78 MB，>50 MB 未 push） |
| 8 | `convnext_tiny/` | `convnext_tiny` | server 参考 | ⚠️ 仅本地（107 MB，>100 MB 未 push） |
| 9 | `regnety_400mf/` | `regnety_004` | multistate 对比 | ✅ GitHub |
| 10 | `resnet18/` | `resnet18` | multistate 对比 | ✅ GitHub |
| 11 | `vit_tiny/` | `deit_tiny_patch16_224` | multistate 对比 | ✅ GitHub |

> 原 `production/` 已改名为 **`deploy_efficientnet_lite0/`**，避免误以为「所有模型的统一上线目录」。

## 训练后写入权重

```bash
cp tasks/fruit_classification/runs/<model_run>/weights/best.pt \
   tasks/fruit_classification/exports/<model_id>/best.pt
```

若该模型成为新的 GitHub 上线版，再复制到 `deploy_<model_id>/` 并运行 `scripts/sync_exports_to_github.sh`。

## GitHub 大文件说明

根目录 `.gitignore` 排除超过 GitHub 限制/建议大小的权重：

- `exports/convnext_tiny/best.pt`（硬限制 100 MB）
- `exports/efficientnetv2_s/best.pt`（建议上限 50 MB）

本地文件仍在；GitHub 上仅有 `classes.json` 与 README。

## GitHub 推理仓

| 本地目录 | GitHub 仓 |
|----------|-----------|
| `deploy_efficientnet_lite0/` | EfficientNet-Lite0_fruitsclassify |
| `mobilenet_v3/` | mobileNetV3large_classify |
| `shufflenet_v2/` | ShuffleNetV2_fruitsclassify |
