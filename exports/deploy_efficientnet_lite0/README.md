# deploy_efficientnet_lite0

**GitHub / App 上线用** EfficientNet-Lite0 权重，与 `EfficientNet-Lite0_fruitsclassify` 推理仓一致。

| 项目 | 值 |
|------|-----|
| 模型类型 | **单头品种分类**（10 类，非 multistate） |
| timm 骨干 | `tf_efficientnet_lite0` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 16 MB；同步到 GitHub `models/efficientnet_lite0_fruit_cls_best.pt` |
| `classes.json` | 10 类品种标签 |

与 `../efficientnet_lite0_multistate/` 的区别：后者为 **11 品种 + 7 状态双头** multistate 权重，不要混用。

同步命令见仓库根目录 `scripts/sync_exports_to_github.sh`（若存在）。
