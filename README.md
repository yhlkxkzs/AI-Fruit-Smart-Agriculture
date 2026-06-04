# 任务 1：水果分类（Fruit Classification）

**输入**：果实图片（健康 / 病害 / 未成熟等状态均可）  
**输出**：水果品种（如 apple, mango, grape…）  
**范式**：整图分类

本目录即 GitHub 仓 [`fruits-classifiaction-for-afsa`](https://github.com/ai-agriculture-circuits-and-systems/fruits-classifiaction-for-afsa) 的本地 clone（本地文件夹名可与远程仓名不同，以 `git remote` 为准）。

## 目录

```text
tasks/fruit_classification/
├── .github/                      # Actions workflow + 推理脚本
├── configs/
├── exports/                      # 模型权重与 classes.json
├── incoming/category/uploads/    # App 上传触发推理
├── docs/
├── runs/                         # 训练日志与 checkpoint（不进 Git）
└── scripts/
```

## 数据

| 目录 | 用途 |
|------|------|
| `data/fruit_classification/` | 仅 **品种** 10 类（ImageFolder） |
| `data/fruit_classification_multistate/` | **品种 + 状态** 双标签（manifest，275 品种，见 [docs/DATA_MULTISTATE.md](docs/DATA_MULTISTATE.md)） |

生成多状态 manifest：

```bash
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py
```

## 训练

```bash
cd /home/yuhanlin/APP/AFSA
bash tasks/fruit_classification/scripts/run_parallel_multistate.sh
```

`train_multistate.py` 在全部 epoch 结束后将 `best.pt` + `classes.json` 写入 `exports/<model_id>/`。

## exports 布局

见 [exports/MANIFEST.md](exports/MANIFEST.md)。App 主模型槽位：`production_lite0_singlehead_10class/`（`github_model_id`: `deploy_efficientnet_lite0`）。

## 上线到 GitHub

```bash
bash scripts/sync_exports_to_github.sh   # 校验 exports（可选）
cd tasks/fruit_classification
git add -A
git status   # 确认未包含 runs/ 与大文件 best.pt
git commit -m "..."
git push
```

---

## 2026-06-04 更新说明

### 1. Multistate 全量训练（275 品种 + 7 状态）

- 数据 manifest 约 **55 万** 样本（`data/fruit_classification_multistate/`）。
- **8/10** 对比 backbone 已完成 50 epoch 并写入 `exports/`（lite0、mobilenet_v3、shufflenet_v2、resnet18、lite4、mobilevit_s、regnety_400mf、vit_tiny）。
- **进行中**：`convnext_tiny`、`efficientnetv2_s`（本地训练，完成后自动更新 `exports/`）。

### 2. exports 目录整理

| 变更 | 说明 |
|------|------|
| `production_lite0_singlehead_10class/` | App 主槽位；当前仍为 **10 类单头** 旧权重，multistate 选定后再替换 |
| 删除 `deploy_efficientnet_lite0/`、`efficientnet_lite0_multistate/` | 合并/归档，避免与 manifest 重名混淆 |
| `exports/_archive/multistate_v1_20260527_lite0_11species/` | 历史 11 类双头备份 |
| 各 `exports/<model_id>/` | 新 multistate `classes.json`（275 类）+ `best.pt`（训练完成者） |

### 3. GitHub Actions 推理流水线

- 新增 `.github/workflows/afsa-incoming-dispatch.yml`：监听 `incoming/**` 图片 push，跑 11 模型推理并上传 `predictions` artifact。
- 脚本：`.github/scripts/`（`models_manifest.json`、`afsa_resolve_sidecar.py`、`afsa_run_category_models.py` 等）。
- 契约文档：[docs/GitHub流水线对接说明_给仓库维护者.md](docs/GitHub流水线对接说明_给仓库维护者.md)。

### 4. 大文件与本次 push 策略

| 路径 | GitHub |
|------|--------|
| `exports/convnext_tiny/best.pt`（约 **107 MB**） | **不上传**（>100 MB 硬限）；见该目录 README |
| `exports/efficientnetv2_s/best.pt`（约 **78 MB**） | **不上传**（>50 MB 建议限；训练未结束前为旧权重） |
| `runs/` | **不上传**（训练中间产物，已 `.gitignore`） |
| 其余 `exports/*/best.pt`（均 <50 MB） | **上传** |
| 各目录 `classes.json`、README | **上传** |

`convnext_tiny` / `efficientnetv2_s` 训练结束后在服务器保留 `best.pt`；若需进仓可用 Release、Git LFS 或外链。

### 5. 移除 `github_repos/` 符号链接

Plan A：不再维护 `github_repos/` 软链，统一在本仓 `exports/` + `git push`。

### 6. 脚本与配置

- `train_multistate.py`：截断图容错、`num_species: 275`。
- `run_parallel_multistate.sh`：10 GPU 并行，跳过已有 multistate export。
- `prepare_multistate_dataset.py`：temp + database 合并 manifest。
