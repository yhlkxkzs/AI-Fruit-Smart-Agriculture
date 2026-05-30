# convnext_tiny

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `convnext_tiny` |
| best_score | 0.9850 |
| val 品种 | 99.42% |
| val 状态 | 95.53% |
| 训练 run | `runs/multistate/convnext_tiny_20260530_003433` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 本地约 **107 MB**；**未上传 GitHub**（超过 100 MB 限制，见根目录 `.gitignore`） |
| `classes.json` | 品种/状态标签与指标；**已在 GitHub** |

从训练 run 复制权重：

```bash
cp tasks/fruit_classification/runs/multistate/convnext_tiny_20260530_003433/weights/best.pt \
   tasks/fruit_classification/exports/convnext_tiny/best.pt
```

若需分享大权重，可用 GitHub Release、Git LFS 或网盘/Hugging Face，并在文档中附下载链接。

## 为何 GitHub 上没有 `best.pt`

| 项目 | 说明 |
|------|------|
| 本地文件 | `exports/convnext_tiny/best.pt`（约 **107 MB**） |
| 未上传原因 | GitHub **硬性拒绝单文件 > 100 MB**；push 时会被 pre-receive hook 拦截 |
| 仓库处理 | 根目录 `.gitignore` 已排除该文件；**训练已完成，权重仅保存在本机/服务器** |
| GitHub 上有什么 | 本 README、`classes.json`（含 score 与 run 路径）；**不含** `.pt` |
| 本地备用路径 | `runs/multistate/convnext_tiny_20260530_003433/weights/best.pt`（同样未进 Git） |

其他模型权重若 < 100 MB，一般可正常 push；本模型因体积超限必须本地保存或通过 Release / LFS / 外链分发。
