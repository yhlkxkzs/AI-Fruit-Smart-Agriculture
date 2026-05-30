# vit_tiny

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `deit_tiny_patch16_224` |
| best_score | 0.9853 |
| val 品种 | 99.49% |
| val 状态 | 96.17% |
| 训练 run | `runs/multistate/vit_tiny_20260530_003433` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 21 MB；**已在 GitHub** |
| `classes.json` | 品种/状态标签与指标 |

```bash
cp tasks/fruit_classification/runs/multistate/vit_tiny_20260530_003433/weights/best.pt \
   tasks/fruit_classification/exports/vit_tiny/best.pt
```
