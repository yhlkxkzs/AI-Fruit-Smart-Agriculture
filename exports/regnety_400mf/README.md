# regnety_400mf

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `regnety_004` |
| best_score | 0.9887 |
| val 品种 | **99.85%** |
| val 状态 | 96.39% |
| 训练 run | `runs/multistate/regnety_400mf_20260530_003432` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 15 MB；**已在 GitHub** |
| `classes.json` | 品种/状态标签与指标 |

```bash
cp tasks/fruit_classification/runs/multistate/regnety_400mf_20260530_003432/weights/best.pt \
   tasks/fruit_classification/exports/regnety_400mf/best.pt
```
