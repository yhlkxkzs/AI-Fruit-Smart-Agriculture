# efficientnet_lite0_multistate

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-27 单独训练；与并行 batch 中 Lite0 结果一致。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `tf_efficientnet_lite0` |
| best_score | 0.9886 |
| val 品种 | 99.83% |
| val 状态 | 96.45% |
| 训练 run | `runs/multistate/efficientnet_lite0_20260527_085925` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 14 MB；**已在 GitHub** |
| `classes.json` | 品种/状态标签与指标 |

与 `../deploy_efficientnet_lite0/` 的区别：后者为 **旧单头 10 类品种** 上线权重；本目录为 **multistate 双头** 权重。

```bash
cp tasks/fruit_classification/runs/multistate/efficientnet_lite0_20260527_085925/weights/best.pt \
   tasks/fruit_classification/exports/efficientnet_lite0_multistate/best.pt
```
