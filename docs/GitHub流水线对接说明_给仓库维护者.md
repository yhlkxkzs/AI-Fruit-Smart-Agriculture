# GitHub 流水线对接说明（给推理仓维护者）

> **Audience**：负责在 **[fruits-classifiaction-for-afsa](https://github.com/ai-agriculture-circuits-and-systems/fruits-classifiaction-for-afsa)** 内 **编写并部署 GitHub Actions workflow** 的后台同事。  
> **App 端**已改为 **单仓 11 模型**（1 主模型 + 10 对比）；本文是 workflow 设计与契约的**唯一说明档**（App 仓内不再附带 `.yml` / 脚本模板，请按本文在推理仓自行创建）。  
> **用户图片收集仓**、**知识库 `afsa-knowledge`** 仍用原仓库，**不在本文范围**。

---

## 1. 必须满足的契约（Summary）

| 项 | 要求 |
|----|------|
| 推理仓 | `ai-agriculture-circuits-and-systems/fruits-classifiaction-for-afsa` |
| 模型权重 | `exports/<export_dir>/best.pt` + `classes.json`（见推理仓 `exports/MANIFEST.md`） |
| 触发路径 | `on.push.paths` 包含 `incoming/**`，排除 `!incoming/**/*.afsa.json` |
| 输入图片 | `incoming/category/uploads/xxx.jpg`（及 growth/disease 子路径，见 sidecar 文档） |
| 元数据 | 同目录 `<同名>.afsa.json` |
| Artifact | 名称固定 **`predictions`** |
| `predictions.json` | **每个模型一行**，必须含 **`github_model_id`** |

App 上传策略：**每张照片只 push 1 次**到本仓；sidecar 中 `github_model_target_id` 为 **`category_all`**；workflow 一次 run 内跑完全部可用模型。

---

## 2. `predictions.json` 格式（App 匹配用）

单仓多模型示例（一张图、11 行，每个 `github_model_id` 一行）：

```json
{
  "predictions": [
    {
      "image": "incoming/category/uploads/20260519_143052_a1b2c3d4.jpg",
      "github_path": "incoming/category/uploads/20260519_143052_a1b2c3d4.jpg",
      "afsa_detection_id": "rec_1716123456789",
      "github_model_id": "deploy_efficientnet_lite0",
      "predicted_class": "apple",
      "confidence": 0.92
    },
    {
      "image": "incoming/category/uploads/20260519_143052_a1b2c3d4.jpg",
      "github_path": "incoming/category/uploads/20260519_143052_a1b2c3d4.jpg",
      "afsa_detection_id": "rec_1716123456789",
      "github_model_id": "mobilenet_v3",
      "predicted_class": "apple",
      "confidence": 0.88
    }
  ]
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `image` | **是** | 与 sidecar `image_path` / App `github_path` **完全一致** |
| `github_path` | 强烈建议 | 与 `image` 相同 |
| `afsa_detection_id` | 强烈建议 | 与 sidecar 相同 |
| **`github_model_id`** | **是（单仓）** | 与下表及 App `lib/config/github_fruit_classification_targets.dart` 一致 |
| `predicted_class` | 是 | 模型输出类别（英文，与 `classes.json` 一致） |
| `confidence` | 是 | 0～1 浮点数（若写 92 表示 92%，App 会归一化） |

### App 匹配顺序

1. 轮询时带 `github_model_id` → 先筛 **`github_model_id` 相等** 的行  
2. `afsa_detection_id` 相等  
3. `github_path` / `image` 与上传路径完全相等  

---

## 3. 十一个模型 ID 与 `exports/` 目录

| `github_model_id` | `exports/` 目录 | 备注 |
|-------------------|-----------------|------|
| `deploy_efficientnet_lite0` | `production_lite0_singlehead_10class/` | App 主模型槽位（当前 10 类单头） |
| `efficientnet_lite0` | `efficientnet_lite0/` | multistate 对比 |
| `mobilenet_v3` | `mobilenet_v3/` | |
| `shufflenet_v2` | `shufflenet_v2/` | |
| `efficientnet_lite4` | `efficientnet_lite4/` | |
| `mobilevit_s` | `mobilevit_s/` | |
| `efficientnetv2_s` | `efficientnetv2_s/` | 权重可能未 push（>50MB） |
| `convnext_tiny` | `convnext_tiny/` | 权重可能未 push（>100MB） |
| `regnety_400mf` | `regnety_400mf/` | |
| `resnet18` | `resnet18/` | |
| `vit_tiny` | `vit_tiny/` | DeiT-Tiny 双头 multistate |

建议在推理仓维护清单（任选其一）：

- 硬编码在上表；或  
- 推理仓内 `.github/scripts/models_manifest.json`（结构见 **§6.3**）

缺失 `best.pt` 的目录：**跳过该模型并打日志**，不要导致整 job 失败。

---

## 4. sidecar 示例（App 实际上传）

种类识别（单仓跑全模型）：

```json
{
  "schema_version": 1,
  "app": "afsa_mobile",
  "detection_type": "category",
  "task_type": "fruit_category",
  "image_path": "incoming/category/uploads/20260519_143052_a1b2c3d4.jpg",
  "github_model_target_id": "category_all",
  "afsa_detection_id": "rec_1716123456789",
  "uploaded_at": "2026-05-19T06:30:52.123Z"
}
```

`task_type` 分支规则见 `github_incoming_path_and_sidecar.md`：

| task_type | workflow 行为 |
|-----------|----------------|
| `fruit_category` | 对 `exports/` 下全部种类模型推理 |
| `growth_*` | 生长专用脚本（未就绪可占位，勿输出水果名） |
| `disease_*` | 病害专用脚本（未就绪可占位） |

---

## 5. 推理仓目录结构（建议）

在 **`tasks/fruit_classification/`**（GitHub 仓 `fruits-classifiaction-for-afsa` 的本地 clone）内维护：

```text
tasks/fruit_classification/    # 本地路径；git push 到 fruits-classifiaction-for-afsa
├── .github/
│   ├── workflows/
│   │   └── afsa-incoming-dispatch.yml    ← 按 §6.1 编写
│   └── scripts/
│       ├── afsa_resolve_sidecar.py       ← 按 §6.2
│       ├── afsa_run_category_models.py   ← 按 §6.4（核心推理）
│       ├── afsa_write_predictions.py     ← 按 §6.5（可选辅助）
│       └── models_manifest.json          ← 可选，见 §6.3
├── exports/
│   ├── production_lite0_singlehead_10class/
│   │   ├── best.pt
│   │   └── classes.json
│   └── …（其余 10 个活跃模型目录；历史见 `_archive/`）
└── incoming/
    └── category/uploads/                 ← App PUT 后自动出现
```

---

## 6. Workflow 与脚本编写说明

### 6.1 `.github/workflows/afsa-incoming-dispatch.yml`

**目标**：图片 push 触发 **一次** job → 读 sidecar → 对变更图片跑全部种类模型 → 上传 Artifact `predictions`。

**推荐 YAML 骨架**（后台可直接在此基础上改推理命令）：

```yaml
name: AFSA incoming dispatch

on:
  push:
    branches: [main]
    paths:
      - 'incoming/**'
      - '!incoming/**/*.afsa.json'

permissions:
  contents: read
  actions: write

jobs:
  predict:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install inference deps
        run: |
          python -m pip install --upgrade pip
          pip install torch torchvision timm pillow numpy

      - name: List changed incoming images
        id: chg
        run: |
          git diff --name-only HEAD^ HEAD -- \
            'incoming/**/*.jpg' 'incoming/**/*.jpeg' \
            'incoming/**/*.png' 'incoming/**/*.webp' > /tmp/images.txt || true
          echo "count=$(wc -l < /tmp/images.txt | tr -d ' ')" >> "$GITHUB_OUTPUT"
          cat /tmp/images.txt

      - name: Resolve routes from sidecar
        if: steps.chg.outputs.count != '0'
        run: python3 .github/scripts/afsa_resolve_sidecar.py /tmp/images.txt

      - name: Run category models (exports/)
        if: steps.chg.outputs.count != '0'
        working-directory: .github/scripts
        run: python3 afsa_run_category_models.py /tmp/images.txt

      - name: Upload predictions artifact
        if: steps.chg.outputs.count != '0'
        uses: actions/upload-artifact@v4
        with:
          name: predictions
          path: out/predictions.json
          if-no-files-found: error
```

**注意**：

- 仅 **图片** commit 触发；sidecar 单独 push 不应再触发（故排除 `*.afsa.json`）。App 会先 push 图片再 push sidecar，以**图片那次 push** 为准。  
- Artifact 名必须是 **`predictions`**。  
- 输出路径必须是仓库根下 **`out/predictions.json`**（或与 upload-artifact 的 `path` 一致）。

---

### 6.2 `afsa_resolve_sidecar.py`

**输入**：`/tmp/images.txt`（每行一个仓库内相对路径）。

**逻辑**：

1. 对每张图，读取同目录 sidecar：将 `foo.jpg` → `foo.afsa.json`（替换扩展名，不是追加）。  
2. 解析 JSON，取出 `image_path`、`task_type`、`github_model_target_id`、`afsa_detection_id`。  
3. 写入 `/tmp/afsa_routes.json`：

```json
{
  "routes": [
    {
      "image_path": "incoming/category/uploads/….jpg",
      "task_type": "fruit_category",
      "github_model_target_id": "category_all",
      "afsa_detection_id": "rec_…"
    }
  ]
}
```

---

### 6.3 `models_manifest.json`（可选）

若希望模型列表可配置而不改 Python，在 `.github/scripts/` 放置：

```json
{
  "repo": "fruits-classifiaction-for-afsa",
  "exports_root": "exports",
  "category_models": [
    { "github_model_id": "deploy_efficientnet_lite0", "export_dir": "production_lite0_singlehead_10class" },
    { "github_model_id": "efficientnet_lite0", "export_dir": "efficientnet_lite0" },
    { "github_model_id": "mobilenet_v3", "export_dir": "mobilenet_v3" },
    { "github_model_id": "shufflenet_v2", "export_dir": "shufflenet_v2" },
    { "github_model_id": "efficientnet_lite4", "export_dir": "efficientnet_lite4" },
    { "github_model_id": "mobilevit_s", "export_dir": "mobilevit_s" },
    { "github_model_id": "efficientnetv2_s", "export_dir": "efficientnetv2_s" },
    { "github_model_id": "convnext_tiny", "export_dir": "convnext_tiny" },
    { "github_model_id": "regnety_400mf", "export_dir": "regnety_400mf" },
    { "github_model_id": "resnet18", "export_dir": "resnet18" },
    { "github_model_id": "vit_tiny", "export_dir": "vit_tiny" }
  ]
}
```

---

### 6.4 `afsa_run_category_models.py`（核心）

**职责**：

1. 读取 `/tmp/afsa_routes.json` 或 `/tmp/images.txt`，筛 `task_type` 以 `fruit` 开头的条目。  
2. 按 manifest（§3 表或 §6.3）遍历每个模型：  
   - 权重：`exports/<export_dir>/best.pt`  
   - 类别：`exports/<export_dir>/classes.json` 的 `classes` 数组  
3. 对每张变更图片跑推理；**每 (图片, 模型) 追加一行**到 `out/predictions.json`。  
4. 若 `best.pt` 不存在 → `SKIP` 并 continue。

**加载权重**：与你们训练脚本一致（可能是完整 `nn.Module`、或 `state_dict` + timm 骨干名）。推理仓 `scripts/train.py` / `evaluate.py` 为参考。

**预处理建议**（与常见 ImageNet 分类一致，可按训练配置调整）：

- Resize 224×224  
- ToTensor + Normalize mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`  
- softmax 取 argmax → `predicted_class` + `confidence`

**每行写入字段**（必须）：

```python
{
    "image": meta["image_path"],           # 与 sidecar 完全一致
    "github_path": meta["image_path"],
    "afsa_detection_id": meta.get("afsa_detection_id"),
    "github_model_id": "<当前模型 id>",
    "predicted_class": "<类别英文名>",
    "confidence": 0.92,
}
```

---

### 6.5 `afsa_write_predictions.py`（可选辅助）

小模块即可：维护 `out/predictions.json` 的 `predictions` 数组，`append_row(...)` 供 §6.4 调用。首次写入前 `mkdir -p out`。

---

## 7. 自测清单（合并 PR 前）

- [ ] 测试账号对 `fruits-classifiaction-for-afsa` 有 **write** + 读 Actions Artifacts 权限  
- [ ] 手动或 App 上传 1 图 + sidecar 到 `incoming/category/uploads/`  
- [ ] push 后 Actions **只触发 1 次**，`conclusion: success`  
- [ ] Artifact **`predictions`** 内 `predictions.json` 含 **11 行**（或可用模型数），且 `github_model_id` 各不同  
- [ ] 每行 `image` 与 sidecar `image_path` **字节级一致**  
- [ ] App 识别结果页：已 push 权重的模型显示「已匹配 predictions」

---

## 8. 常见问题

**Q：App 显示「已上传，predictions 未匹配」**  
A：几乎总是 `github_model_id` 或 `image`/`github_path` 与 App 不一致；用 sidecar 的 `image_path` 原样写入。

**Q：Actions 不触发**  
A：检查 `paths` 是否包含 `incoming/category/` 等新路径；不要只监听旧的 `incoming/uploads/**`。

**Q：生长识别却显示 apple/pear**  
A：未按 `task_type` 分支，误跑了水果分类脚本。

**Q：图片收集 / 知识库要不要改？**  
A：**不要**。用户数据集仍进 `{user}/afsa-incoming-*` 与 `yhlkxkzs/incomingpictureforafsa`；知识库仍用 `afsa-knowledge`。

---

## 9. 相关文档（App 仓）

| 文档 | 路径 |
|------|------|
| 路径 + sidecar 契约 | `docs/07_features_and_docs/github_incoming_path_and_sidecar.md` |
| App 模型 ID 注册 | `lib/config/github_fruit_classification_targets.dart`（需含 **`vit_tiny`**） |
| OAuth / 上传 API | `docs/07_features_and_docs/github_oauth_pkce_setup.md` |

---

*请后台在 **fruits-classifiaction-for-afsa** 按本文创建 workflow 与脚本；合并 PR 时附 Actions run 链接与 `predictions.json` 片段。*
