# 导出模型清单 — fruit_classification

统一目录：`exports/<model_id>/` 内含 `best.pt` + `classes.json`（可选 `README.md`）。

10 类标准标签见 `_shared_classes.json` 与 `deploy_efficientnet_lite0/classes.json`。

## 全部推荐模型

| # | 目录 | timm 骨干 | 用途 | `best.pt` |
|---|------|-----------|------|-----------|
| — | `deploy_efficientnet_lite0/` | `tf_efficientnet_lite0` | **GitHub / App 上线**（与推理仓一致） | ✅ |
| 1 | `efficientnet_lite0/` | `tf_efficientnet_lite0` | 本地对比训练（另一轮） | ✅ |
| 2 | `mobilenet_v3/` | `mobilenetv3_large_100` | 移动端对比 | ✅ |
| 3 | `shufflenet_v2/` | `shufflenet_v2_x1_0` | 极轻量对比 | ✅ |
| 4 | `efficientnet_lite4/` | `tf_efficientnet_lite4` | 待训练 | ⏳ |
| 5 | `mobilevit_s/` | `mobilevit_s` | 待训练 | ⏳ |
| 6 | `efficientnetv2_s/` | `tf_efficientnetv2_s` | 待训练 | ⏳ |
| 7 | `convnext_tiny/` | `convnext_tiny` | 待训练 | ⏳ |
| 8 | `regnety_400mf/` | `regnety_004` | 待训练 | ⏳ |
| 9 | `resnet18/` | `resnet18` | 待训练 | ⏳ |
| 10 | `vit_tiny/` | `deit_tiny_patch16_224` | 待训练 | ⏳ |

> 原 `production/` 已改名为 **`deploy_efficientnet_lite0/`**，避免误以为「所有模型的统一上线目录」。

## 训练后写入权重

```bash
cp tasks/fruit_classification/runs/<model_run>/weights/best.pt \
   tasks/fruit_classification/exports/<model_id>/best.pt
```

若该模型成为新的 GitHub 上线版，再复制到 `deploy_<model_id>/` 并运行 `scripts/sync_exports_to_github.sh`。

## GitHub 推理仓

| 本地目录 | GitHub 仓 |
|----------|-----------|
| `deploy_efficientnet_lite0/` | EfficientNet-Lite0_fruitsclassify |
| `mobilenet_v3/` | mobileNetV3large_classify |
| `shufflenet_v2/` | ShuffleNetV2_fruitsclassify |
