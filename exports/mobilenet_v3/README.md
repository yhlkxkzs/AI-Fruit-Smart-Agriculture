# mobilenet_v3

**Multistate 尚未训成**；当前权重为 **2026-03 旧单头 checkpoint**（仅 6 类品种），**不可**用于 multistate 推理。

| 项目 | 值 |
|------|-----|
| 当前类型 | 单头品种分类 |
| 类别数 | 6（见 `classes.json`） |
| multistate | 待重训（代码已修：torchvision 输出维 1280 vs head 960） |

| 文件 | 说明 |
|------|------|
| `best.pt` | 旧权重；**已在 GitHub**（推理仓 mobileNetV3large_classify 同步用） |
| `classes.json` | 旧 6 类标签 |

Multistate 训练完成后：

```bash
cp tasks/fruit_classification/runs/multistate/mobilenet_v3_<run>/weights/best.pt \
   tasks/fruit_classification/exports/mobilenet_v3/best.pt
```
