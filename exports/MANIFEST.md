# 导出模型清单 — fruit_classification

统一目录：`exports/<model_id>/` 内含 `best.pt` + `classes.json`（部分大文件见下方「GitHub 大文件」）。

## 命名说明

| 目录 | 含义 |
|------|------|
| `production_lite0_singlehead_10class/` | App 主模型槽位（`github_model_id`: **`deploy_efficientnet_lite0`**） |
| `efficientnet_lite0/` … `vit_tiny/` | **10** 个对比 backbone（multistate **275** 品种 + 7 状态双头） |
| `_archive/multistate_v1_20260527_lite0_11species/` | **历史归档**（11 类双头，不参与 Actions） |

## App 推理 manifest（11 个 `github_model_id`）

| `github_model_id` | `exports/` 目录 | `best.pt` 本次 push |
|-------------------|-----------------|---------------------|
| `deploy_efficientnet_lite0` | `production_lite0_singlehead_10class/` | 是（10 类单头，待 multistate 替换） |
| `efficientnet_lite0` | `efficientnet_lite0/` | 是 |
| `mobilenet_v3` | `mobilenet_v3/` | 是 |
| `shufflenet_v2` | `shufflenet_v2/` | 是 |
| `efficientnet_lite4` | `efficientnet_lite4/` | 是 |
| `mobilevit_s` | `mobilevit_s/` | 是 |
| `efficientnetv2_s` | `efficientnetv2_s/` | **是**（约 80 MB，已 push） |
| `convnext_tiny` | `convnext_tiny/` | **否**（约 108 MB，>100 MB 硬限） |
| `regnety_400mf` | `regnety_400mf/` | 是 |
| `resnet18` | `resnet18/` | 是 |
| `vit_tiny` | `vit_tiny/` | 是 |

配置源：`.github/scripts/models_manifest.json`

## 2026-06-04 变更摘要

1. 完成 8 个 backbone 的 **275 类 multistate** 导出并推送权重。
2. 新增 **GitHub Actions** 与 `incoming/` 目录结构。
3. 重命名主槽位目录；旧 `deploy_*` / `*_multistate` 已移除或归档。
4. **`convnext_tiny`**：`classes.json` + README 推送；`best.pt` 仅本地（>100 MB）。**`efficientnetv2_s`** 权重已推送。

## 训练后写入

```bash
# 自动：train_multistate.py 结束 → exports/<model_id>/
# 手动指定 App 主模型：
cp exports/efficientnet_lite0/best.pt exports/production_lite0_singlehead_10class/best.pt
cp exports/efficientnet_lite0/classes.json exports/production_lite0_singlehead_10class/classes.json
```

## 同步到 GitHub

本目录即 GitHub 仓本地 clone，直接 `git push`，无需复制到其他路径。

```bash
bash scripts/sync_exports_to_github.sh
cd tasks/fruit_classification && git add -A && git commit && git push
```

## GitHub 大文件

根目录 `.gitignore` 排除：

- `exports/convnext_tiny/best.pt`（>100 MB）
- ~~`exports/efficientnetv2_s/best.pt`~~（约 80 MB，已纳入 Git）
- `runs/` 全部训练产物

Workflow 在缺少 `best.pt` 时会 **skip** 对应模型；`classes.json` 仍可用于元数据。
