# Dataset 形成逻辑说明

## 一、Dataset 是如何形成的

### 1. 核心脚本

**主脚本**: `src/prepare_dataset.py`

- **数据源**（只读）:
  - `GITHUB_DB` = `/home/yuhanlin/Database/github/database`
  - `LOCAL_DB` = `/home/yuhanlin/Database/local/database`
- **输出目录**: `OUTPUT_DIR` = `/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset`

### 2. 形成流程概览

1. **遍历两个数据库**  
   对 `GITHUB_DB` 和 `LOCAL_DB` 下的每个子目录（每个子目录视为一个“数据集”）进行处理。

2. **判定数据集类型**  
   - 若数据集名称能在 `FRUIT_MAPPING` 中匹配到某个标准水果名（如 `apple_detection_drone_brazil` → `apple`）→ 按**单水果数据集**处理。  
   - 若数据集在“多水果列表”中（如 `deepfruits`, `fruit-salad`, `plant_village` 等）→ 按**多水果数据集**处理（有单独逻辑，如按子目录名、labelmap 等推断水果类别）。

3. **单水果数据集处理**（`process_single_fruit_dataset`）  
   - 递归收集该数据集目录下**所有图片**（扩展名：jpg/jpeg/png/bmp/tiff 等），并跳过目录：`__pycache__`, `.git`, `scripts`, `labels`, `annotations`, `data`, `docs`, `sets`, `origin`。  
   - 用 PIL 校验图片是否可读，无效的跳过。  
   - 将有效图片列表**随机打乱**，按 **70% / 15% / 15%** 划分为 train / val / test。  
   - 复制到：
     - `dataset/train/{水果名}/`
     - `dataset/val/{水果名}/`
     - `dataset/test/{水果名}/`
   - 重命名规则：`{水果名}_{index:06d}.{原后缀}`，例如 `apple_000042.jpg`。  
   - **重要**：此脚本**只复制图片文件**，不复制、不生成任何 JSON 或 CSV。

4. **多水果数据集处理**（`process_multifruit_dataset` 及子函数）  
   - 根据数据集类型（如 deepfruits、fruit-salad、plant_village）用不同方式从子目录或 labelmap 中解析出“图片 → 水果类别”。  
   - 对每种水果同样做 70/15/15 划分，复制到 `dataset/{split}/{水果名}/`，命名方式同上。  
   - 同样**只复制图片**，不处理 JSON。

5. **Plant Village 类数据集**  
   - 名称中含 `Plant_Village` 的走 `process_plant_village_dataset`：在数据集内找“水果主目录”（如 apples、cherries），再递归收集图片、划分、复制，依旧**仅图片**。

6. **统计与副产品**  
   - 脚本会统计每个水果在 train/val/test 下的数量，并写入 `dataset/dataset_stats.json`。  
   - 多水果数据集中无法归类到已知水果的图片会记录到 `dataset/unknown_sources.json`（若有）。

### 3. 当前 dataset 目录结构（脚本直接产生的部分）

```
dataset/
├── train/
│   ├── apple/          # 直接放图片：apple_000000.bmp, apple_000001.bmp, ...
│   ├── cherry/
│   ├── fig/
│   ├── orange/
│   └── ... (其他水果)
├── val/
│   ├── apple/
│   └── ...
├── test/
│   ├── apple/
│   └── ...
├── sets/               # 非 prepare_dataset.py 所建，为后续脚本生成
├── dataset_stats.json
└── unknown_sources.json (可选)
```

也就是说：**由 prepare_dataset.py 形成的“标准产物”只有“按 split + 水果名”的图片目录，没有任何 JSON/CSV/sets。**

### 4. 为何会出现“图片没有对应 JSON”的问题

- **prepare_dataset.py 从不拷贝或生成 JSON**。  
- 当前你在部分水果下看到的 `json/`、`csv/`、`sets/`、`single/`、`multi/` 等，是**后续其他脚本**（如 `process_one_fruit.py`、`scripts/dedupe_one_fruit.py`、各类标注/修复脚本）在**部分**水果或**部分**图片上做的增量处理。  
- 因此会出现：
  - 有些水果完全没有 json；
  - 有些水果只有部分图片有对应 json；
  - 图片在“水果目录根下”，而 json 在“水果目录下的 json 子目录”，且命名可能不一致（例如根目录是 `apple_000042.jpg`，json 里可能是别的 filename），导致**结构和一一对应关系不统一**。

### 5. 目标结构参考：apple_detection_drone_brazil

你希望新 dataset 的**每个子文件夹**在结构上向 `/home/yuhanlin/Database/local/database/apple_detection_drone_brazil` 看齐。该目录下与“单类水果”对应的部分是 `apples/`：

```
apple_detection_drone_brazil/
├── apples/
│   ├── images/     # 图片：gebler-000-00.jpg, ...
│   ├── json/       # 与图片同名的 JSON：gebler-000-00.json, ...
│   ├── csv/
│   ├── sets/
│   └── labelmap.json
├── data/
├── annotations/
└── scripts/
```

要点：

- 每个类别（水果）下有 **images/** 与 **json/**，且**一张图对应一个同 stem 的 .json**（如 `gebler-000-00.jpg` ↔ `gebler-000-00.json`）。  
- JSON 为“单图单标注”格式（含 `info`, `images` 单条, `annotations`, `categories` 等），与当前 `api_mobilenet` 等使用的格式一致。

---

## 二、总结

| 项目         | 说明 |
|--------------|------|
| 谁生成 dataset | `src/prepare_dataset.py` |
| 数据从哪来   | `Database/github/database` + `Database/local/database` |
| 生成了什么   | 仅图片：按 train/val/test + 水果名分目录，重命名为 `{水果}_{六位序号}.ext` |
| 未做哪些事   | 未拷贝/未生成 JSON、CSV、sets；未按“每图一个 JSON”组织 |
| 当前问题     | 图片与 JSON 的对应关系不完整、不统一，部分水果无 JSON |
| 目标         | 重新生成 dataset，在保留现有层级（train/val/test + 水果）的基础上，每个子文件夹采用与 `apple_detection_drone_brazil` 相同的结构（images/ + json/ 且 1:1 对应） |

如果你需要，下一步可以基于这份逻辑，写一个“重新生成 dataset”的脚本：从两个数据源重新扫图、按同一划分逻辑生成 split，并**从源里拷贝或生成与每张图对应的 JSON**，使每个水果子目录都符合 apple_detection_drone_brazil 的 images + json 结构。
