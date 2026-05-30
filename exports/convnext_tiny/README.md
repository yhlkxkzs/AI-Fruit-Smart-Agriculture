# 权重待训练

本目录结构已就绪，训练完成后将 checkpoint 复制为：

```bash
cp tasks/fruit_classification/runs/<run>/weights/best.pt \
   tasks/fruit_classification/exports/convnext_tiny/best.pt
```

类别定义见 `classes.json`（10 类，与 `production/` 一致）。
