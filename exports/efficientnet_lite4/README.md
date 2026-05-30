# efficientnet_lite4

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `tf_efficientnet_lite4` |
| best_score | 0.9883 |
| val 品种 | 99.79% |
| val 状态 | **96.57%** |
| 训练 run | `runs/multistate/efficientnet_lite4_20260530_003433` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 45 MB；**已在 GitHub** |
| `classes.json` | 品种/状态标签与指标 |

```bash
cp tasks/fruit_classification/runs/multistate/efficientnet_lite4_20260530_003433/weights/best.pt \
   tasks/fruit_classification/exports/efficientnet_lite4/best.pt
```
