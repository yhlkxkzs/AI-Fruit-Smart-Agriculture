# 水果分类多状态数据集（Fruit-State v1）

## 目标

品种 + 状态双标签，训练双头模型（见 `prepare_multistate_dataset.py`）。

- 输出：`data/fruit_classification_multistate/`
- 配置：`configs/multistate.yaml`

## 数据源（脚本已接入）

| source_id | 路径 |
|-----------|------|
| fruitvision | `Database/local/database/FruitVision.../` |
| rawripe | `Database/github/database/RawRipe/` |
| banana_ripening | `Database/local/database/Banana_Ripening_Process/banana/` |
| tomato_ripeness | `Database/local/database/tomato_ripeness_detection/` |
| riseholme_strawberry | `Database/local/database/riseholme_strawberry_classification_2021/` |
| tomato_plant | `Database/github/database/tomato_plant/` |
| lemon_qc | `Database/github/database/lemon-datase/` |
| apple_scab | `Database/github/database/AppleScabFDs/` |
| afsa_species_only | `data/fruit_classification/`（state=unknown） |

## 状态 6+1 类

`healthy` `immature` `mature` `aged` `diseased` `pest` `unknown`

## 品种

标准 10 类 + `other`（香蕉等记入 other）。

## 命令

```bash
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py --seed 42
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py --copy-images
```
