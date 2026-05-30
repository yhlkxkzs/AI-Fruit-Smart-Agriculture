# 任务 1：水果分类（Fruit Classification）

**输入**：果实图片（健康 / 病害 / 未成熟等状态均可）  
**输出**：水果品种（如 apple, mango, grape…）  
**范式**：整图分类

## 目录

```text
tasks/fruit_classification/
├── configs/
├── exports/
│   ├── deploy_efficientnet_lite0/  # GitHub / App 上线（与推理仓一致）
│   ├── efficientnet_lite0/         # 本地对比训练
│   └── .../                        # 其它 backbone，见 MANIFEST.md
├── github_repos/                 # symlink → ~/EfficientNet-* 等
├── runs/
└── scripts/
```

## 数据

| 目录 | 用途 |
|------|------|
| `data/fruit_classification/` | 仅 **品种** 10 类（ImageFolder） |
| `data/fruit_classification_multistate/` | **品种 + 状态** 双标签（manifest，见 [docs/DATA_MULTISTATE.md](docs/DATA_MULTISTATE.md)） |

生成多状态 manifest：

```bash
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py
```

旧数据位置：`../task1_fruit_classification/dataset_detection/`（迁移见 [docs/MIGRATION.md](../../docs/MIGRATION.md)）

## 训练

```bash
cd /home/yuhanlin/APP/AFSA
python tasks/fruit_classification/scripts/train.py --config tasks/fruit_classification/configs/default.yaml
```

## 对比实验 backbone（10 个推荐模型）

见 `exports/MANIFEST.md` 与 `configs/models/`（含 timm 骨干名）。

```text
exports/
├── deploy_efficientnet_lite0/  # GitHub 上线版 Lite0
├── efficientnet_lite0/         # 本地对比 Lite0（另一轮训练）
├── mobilenet_v3/            # 已训练
├── shufflenet_v2/           # 已训练
├── efficientnet_lite4/     # 目录已建，待训练
├── mobilevit_s/
├── efficientnetv2_s/
├── convnext_tiny/
├── regnety_400mf/
├── resnet18/
└── vit_tiny/
```

每个目录：`best.pt` + `classes.json`（10 类与 `deploy_efficientnet_lite0/` 一致）。

## 已训练模型

- **GitHub 上线**：`exports/deploy_efficientnet_lite0/best.pt`
- **对比实验**：`exports/efficientnet_lite0/`、`mobilenet_v3/`、`shufflenet_v2/`
- 旧 YOLO / local 实验已归档至 `_legacy/fruit_classification_old_experiments/`

## GitHub 推理仓

`EfficientNet-Lite0_fruitsclassify`（已有）
