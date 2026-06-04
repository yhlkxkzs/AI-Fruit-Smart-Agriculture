# 水果分类多状态数据集（Fruit-State v1）

## 目标

品种 + 状态双标签，训练双头模型（见 `prepare_multistate_dataset.py`）。

- 输出：`data/fruit_classification_multistate/`
- 配置：`configs/multistate.yaml`

## 数据源（脚本已接入）

| source_id | 路径 |
|-----------|------|
| fruitvision | `Database/datasets/database/FruitVision.../` |
| rawripe | `Database/datasets/database/RawRipe/` |
| banana_ripening | `Database/datasets/database/Banana_Ripening_Process/banana/` |
| tomato_ripeness | `Database/datasets/database/tomato_ripeness_detection/` |
| riseholme_strawberry | `Database/datasets/database/riseholme_strawberry_classification_2021/` |
| tomato_plant | `Database/datasets/database/tomato_plant/` |
| lemon_qc | `Database/datasets/database/lemon-datase/` |
| apple_scab | `Database/datasets/database/AppleScabFDs/` |
| afsa_species_only | `data/fruit_classification/`（state=unknown） |

## 状态 6+1 类

`healthy` `immature` `mature` `aged` `diseased` `pest` `unknown`

## 品种

26 类：原 17 类 + `avocado` `bean` `broccoli` `chestnut` `cucumber` `eggplant` `hogplum` `jackfruit` `soybean` + `other`（未识别 fallback）。

各数据源按**原始种类文件夹/标注**映射为对应品种。

## 新增数据源（2026 审查接入）

| source_id | 路径 | 品种 | 说明 |
|-----------|------|------|------|
| rangeland_avocado | `rangeland_weeds_australia/avocados/` | avocado | 仅取 avocados 子集 |
| bean_disease_uganda | `bean_disease_uganda/beans/` | bean | 子目录为病害/健康状态 |
| ghai_broccoli_detection | `ghai_broccoli_detection/broccolis/` | broccoli | |
| chestnut | `chestnut/chestnuts/` | chestnut | 产区子目录 → state mostly unknown |
| cucumber_disease | `cucumber_disease_classification/cucumbers/` | cucumber | 病害子目录 |
| eggplant_preprocessed | `eggplant_preprocessed_2026/eggplants/` | eggplant | healthy / rejected |
| fruits_od_detection | `Fruits Images Dataset_Object Detection/fruits/` | hogplum, jackfruit | 仅这两类 |
| vegann_soybean | `vegann_multicrop_presence_segmentation/soybeans/` | soybean | images/ 仅 |

## 命令

```bash
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py --seed 42
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py --copy-images
```
