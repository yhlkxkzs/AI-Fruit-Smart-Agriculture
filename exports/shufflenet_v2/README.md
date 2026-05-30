# shufflenet_v2

**Multistate 尚未训成**；当前权重为 **2026-03 旧单头 checkpoint**（10 类品种），**不可**用于 multistate 推理。

| 项目 | 值 |
|------|-----|
| 当前类型 | 单头品种分类 |
| 类别数 | 10（见 `classes.json`） |
| multistate | 待重训（代码已改走 torchvision） |

| 文件 | 说明 |
|------|------|
| `best.pt` | 旧权重约 4 MB；**已在 GitHub**（推理仓 ShuffleNetV2_fruitsclassify 同步用） |
| `classes.json` | 旧 10 类标签 |

Multistate 训练完成后：

```bash
cp tasks/fruit_classification/runs/multistate/shufflenet_v2_<run>/weights/best.pt \
   tasks/fruit_classification/exports/shufflenet_v2/best.pt
```
