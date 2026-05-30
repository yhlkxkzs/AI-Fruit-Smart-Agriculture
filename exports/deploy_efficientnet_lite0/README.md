# deploy_efficientnet_lite0

**GitHub 上线用**的 EfficientNet-Lite0 权重，与 `EfficientNet-Lite0_fruitsclassify` 仓库一致。

| 文件 | 说明 |
|------|------|
| `best.pt` | 同步到 GitHub `models/efficientnet_lite0_fruit_cls_best.pt` |
| `classes.json` | 10 类品种标签 |

与 `../efficientnet_lite0/` 的区别：后者是**另一轮本地对比训练**的 Lite0，不要混用。

同步命令见仓库根目录 `scripts/sync_exports_to_github.sh`。
