# convnext_tiny

**Multistate 双头**（**275** 品种 + 7 状态），timm 骨干 `convnext_tiny`。

## 训练状态（2026-06-04）

| 项目 | 说明 |
|------|------|
| 全量 manifest | ~55 万样本，50 epoch 并行训练 **进行中** |
| 当前 `exports/best.pt` | 约 **107 MB**（可能为旧 run 权重，训练结束后由脚本覆盖） |
| GitHub | **`best.pt` 暂不上传**（单文件 >100 MB 会被拒绝） |
| GitHub 已含 | 本 README、`classes.json`（训练结束后更新指标） |

本地权重路径（训练完成后）：

```bash
# 自动写入
ls -lh exports/convnext_tiny/best.pt

# 或从 run 目录复制
cp runs/multistate/convnext_tiny_*/weights/best.pt exports/convnext_tiny/best.pt
```

## 为何 GitHub 上没有 `best.pt`

| 项目 | 说明 |
|------|------|
| 体积 | 约 **107 MB** |
| 限制 | GitHub **硬性拒绝** >100 MB 的单文件 |
| 仓库 | 根目录 `.gitignore` 已排除 |
| 分发建议 | GitHub Release 附件、Git LFS、网盘或 Hugging Face |

Actions 推理：缺少 `best.pt` 时该模型会被 **skip**，不影响其它 10 个模型。
