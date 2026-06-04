# efficientnetv2_s

**Multistate 双头**（**275** 品种 + 7 状态），timm 骨干 `tf_efficientnetv2_s`。

## 训练状态（2026-06-04）

| 项目 | 说明 |
|------|------|
| 全量 manifest | 50 epoch 训练 **进行中**（约 31/50 epoch） |
| 本地 `best.pt` | 约 **78 MB**（训练过程中为 checkpoint，结束后由脚本覆盖） |
| GitHub | **`best.pt` 暂不上传**（>50 MB 建议限，且避免推送未完成权重） |
| GitHub 已含 | 本 README、`classes.json` |

```bash
# 训练结束后自动写入 exports/；或手动：
cp runs/multistate/efficientnetv2_s_*/weights/best.pt exports/efficientnetv2_s/best.pt
```

## 为何 GitHub 上没有 `best.pt`

| 项目 | 说明 |
|------|------|
| 体积 | 约 **78 MB**（<100 MB 硬限，但 >50 MB 易触发警告） |
| 策略 | 与 `convnext_tiny` 一并列入 `.gitignore`，待训练完成且需对外分发时用 **Release / LFS** |
| Actions | 缺少权重时 **skip** 本模型 |
