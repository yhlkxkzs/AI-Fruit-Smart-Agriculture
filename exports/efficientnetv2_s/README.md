# efficientnetv2_s

**Multistate 双头模型**（11 品种 + 7 状态），2026-05-30 并行训练已完成；**10 模型中 score 最高**。

| 项目 | 值 |
|------|-----|
| timm 骨干 | `tf_efficientnetv2_s` |
| best_score | **0.9888** |
| val 品种 | 99.77% |
| val 状态 | 96.46% |
| 训练 run | `runs/multistate/efficientnetv2_s_20260530_003433` |

| 文件 | 说明 |
|------|------|
| `best.pt` | 本地约 **78 MB**；**未上传 GitHub**（超过 50 MB 建议上限，见根目录 `.gitignore`） |
| `classes.json` | 品种/状态标签与指标；**已在 GitHub** |

```bash
cp tasks/fruit_classification/runs/multistate/efficientnetv2_s_20260530_003433/weights/best.pt \
   tasks/fruit_classification/exports/efficientnetv2_s/best.pt
```

## 为何 GitHub 上没有 `best.pt`

| 项目 | 说明 |
|------|------|
| 本地文件 | `exports/efficientnetv2_s/best.pt`（约 **78 MB**） |
| 未上传原因 | 超过 GitHub **50 MB 建议上限**（虽未达 100 MB 硬限，仍易触发警告；与 ConvNeXt 一并从 Git 排除以保持 push 稳定） |
| 仓库处理 | 根目录 `.gitignore` 已排除该文件；**训练已完成，权重仅保存在本机/服务器** |
| GitHub 上有什么 | 本 README、`classes.json`（含 score 与 run 路径）；**不含** `.pt` |
| 本地备用路径 | `runs/multistate/efficientnetv2_s_20260530_003433/weights/best.pt`（同样未进 Git） |

若需对外发布，请用 GitHub Release 附件、Git LFS 或网盘/Hugging Face 链接；不要指望 `git clone` 能拉到此权重。
