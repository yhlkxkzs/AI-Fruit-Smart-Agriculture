# mobilevit_s

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `mobilevit_s` |
| best_score | 0.9877 |
| val 品种 | 99.78% |
| val 状态 | 96.33% |
| 训练 run | `runs/multistate/mobilevit_s_20260530_003433` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 约 19 MB；**已在 GitHub** |
| `classes.json` | 品种/状态标签与指标 |

```bash
cp tasks/fruit_classification/runs/multistate/mobilevit_s_20260530_003433/weights/best.pt \
   tasks/fruit_classification/exports/mobilevit_s/best.pt
```
