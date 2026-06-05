# 病害识别数据集索引（本地盘点）

扫描日期：2026-06-05

扫描路径：
- `/home/yuhanlin/Database/datasets/database`
- `/home/yuhanlin/Database/datasets/temp`（**无可用叶片病害集**，均为果实 COCO）

## 质量分级

| 等级 | 含义 | 建议 |
|------|------|------|
| **A** | 规模大、病类文件夹清晰、含健康/病害对照 | 优先纳入 `prepare_disease_dataset.py` |
| **B** | 可用，规模中等或标签略粗 | 第二期或按作物合并 |
| **C** | 体量大但目录结构复杂（如 PlantVillage 分仓、增强重复） | 需去重后使用 |
| **D** | 样本过少或标签不完整 | 暂作补充 |

**合计**：46 个数据集，约 **1,578,919** 张图片（跨集有重复，PlantVillage 含增强副本）。

## 与水果分类相关的宿主（建议第一期）

tomato · citrus/orange · grape · apple · strawberry · peach · cherry · plum · papaya · cucumber · dragonfruit · coconut · bean

## A 级数据集（10 个）

| 数据集 | 路径根 | 图片数 | 宿主 | 输入 | 标签文件夹数 | 主要病类（样本数） |
|--------|--------|--------|------|------|--------------|-------------------|
| `plant_village` | `database/` | 171,238 | plant_village | leaf | 38 | Orange___Haunglongbing_(Citrus_greening)(11014)；Tomato___Tomato_Yellow_Leaf_Curl_Virus(10714)；Soybean___healthy(10180)；Peach___Bacterial_spot(4594) |
| `paddy_disease_classification` | `database/` | 24,283 | rice | leaf | 6 | normal(1764)；brown_spot(965)；downy_mildew(620)；bacterial_leaf_blight(479) |
| `tomato_leaf_disease` | `database/` | 21,168 | tomato | leaf | 10 | Tomato___Leaf_Mold(1100)；Tomato___Bacterial_spot(1100)；Tomato___Septoria_leaf_spot(1100)；Tomato___Early_blight(1100) |
| `cucumber_disease_classification` | `database/` | 14,258 | cucumber | leaf | 8 | Pythium Fruit Rot(969)；Downy Mildew(960)；Bacterial Wilt(960)；Belly Rot(960) |
| `tea_leaf_disease_classification` | `database/` | 11,734 | tea | leaf | 5 | algal_spot(1000)；healthy(1000)；red_spot(1000)；gray_blight(1000) |
| `orange_leaf_disease_classification` | `database/` | 11,462 | orange/citrus | leaf | 3 | Powdery mildew(598)；Citrus canker(588)；Healthy leaf(547) |
| `corn_maize_leaf_disease` | `database/` | 8,376 | corn | leaf | 4 | Common_Rust(1306)；Healthy(1162)；Blight(1146)；Gray_Leaf_Spot(574) |
| `rice_leaf_disease_classification` | `database/` | 7,658 | rice | leaf | 5 | Healthy Rice Leaf(653)；Brown Spot(646)；Bacterial Leaf Blight(636)；Sheath Blight(632) |
| `java_plum_leaf_disease_classification` | `database/` | 7,200 | plum | leaf | 5 | Healthy(800)；Bacterial_Spot(800)；Brown_Blight(800)；Sooty_Mold(800) |
| `Soybean` | `database/` | 5,684 | bean | leaf | 4 | rust(1000)；Soyabean Semilooper_Pest_Attack(790)；Soyabean_Mosaic(772)；Healthy_Soyabean(280) |

## B 级数据集（11 个）

| 数据集 | 路径根 | 图片数 | 宿主 | 输入 | 标签文件夹数 | 主要病类（样本数） |
|--------|--------|--------|------|------|--------------|-------------------|
| `FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection` | `database/` | 177,438 | — | fruit (rot/state) | 2 | Fresh(34200)；Rotten(23939) |
| `arabica_coffee_leaf_disease_classification` | `database/` | 71,924 | coffee | leaf | 1 | Healthy(18984) |
| `cassava` | `database/` | 28,121 | cassava | leaf | 1 | healthy(374) |
| `papaya_leaf_disease_classification` | `database/` | 4,318 | papaya | leaf | 5 | Curl(585)；RingSpot(533)；BacterialSpot(458)；Anthracnose(355) |
| `sunflower_disease_classification` | `database/` | 4,250 | sunflower | leaf | 3 | Fresh Leaf(649)；Downy mildew(590)；Gray mold(470) |
| `bean_disease_uganda` | `database/` | 2,590 | bean | leaf | 3 | bean_rust(436)；angular_leaf_spot(432)；healthy(427) |
| `plant_leaves` | `database/` | 2,590 | — | leaf | 3 | bean_rust(436)；angular_leaf_spot(432)；healthy(427) |
| `blackgram_plant_leaf_disease_classification` | `database/` | 2,014 | bean | leaf | 4 | Anthracnose 230(230)；Yellow Mosaic 220(224)；Healthy 220(221)；Powdery Mildew 180(180) |
| `citrus_leaves` | `database/` | 1,518 | citrus | leaf | 4 | canker(241)；black_spot(190)；healthy(80)；scab(15) |
| `DragonFruitLeafDiseaseDataset_Clean` | `database/` | 796 | dragonfruit | leaf | 4 | StemCanker(122)；Healthy(96)；BrownSpot_Early(74)；Anthracnose(68) |
| `grapevine_leaf_diseases` | `database/` | 494 | grape | leaf | 8 | Downy_Mildew(129)；Healthy(37)；Anthracnose(31)；Powdery_Mildew(18) |

## C 级数据集（18 个）

| 数据集 | 路径根 | 图片数 | 宿主 | 输入 | 标签文件夹数 | 主要病类（样本数） |
|--------|--------|--------|------|------|--------------|-------------------|
| `Plant_Village_Orange` | `database/` | 121,570 | orange/citrus | leaf | — | 见子目录（需解析） |
| `Plant_Village_Soybean` | `database/` | 110,944 | bean | leaf | — | 见子目录（需解析） |
| `Plant_Village_Grape` | `database/` | 92,696 | grape | leaf | — | 见子目录（需解析） |
| `Plant_Village_Tomato` | `database/` | 91,475 | tomato | leaf | — | 见子目录（需解析） |
| `Plant_Village_Corn` | `database/` | 88,192 | corn | leaf | — | 见子目录（需解析） |
| `Plant_Village_Apple` | `database/` | 78,460 | apple | leaf | — | 见子目录（需解析） |
| `Plant_Village_Peach` | `database/` | 64,844 | peach | leaf | — | 见子目录（需解析） |
| `Plant_Village_Pepper` | `database/` | 58,652 | pepper | leaf | — | 见子目录（需解析） |
| `Plant_Village_Potato` | `database/` | 55,576 | potato | leaf | — | 见子目录（需解析） |
| `Plant_Village_Cherry` | `database/` | 47,848 | cherry | leaf | — | 见子目录（需解析） |
| `Plant_Village_Squash` | `database/` | 45,844 | squash | leaf | — | 见子目录（需解析） |
| `Plant_Village_Blueberry` | `database/` | 44,474 | blueberry | leaf | — | 见子目录（需解析） |
| `Plant_Village_Strawberry` | `database/` | 42,620 | strawberry | leaf | — | 见子目录（需解析） |
| `Plant_Village_Raspberry` | `database/` | 19,080 | raspberry | leaf | — | 见子目录（需解析） |
| `LeavesDisease_FFinal` | `database/` | 12,995 | — | leaf | — | 见子目录（需解析） |
| `coconut_tree_disease_classification` | `database/` | 5,798 | coconut | tree/leaf | — | 见子目录（需解析） |
| `strawberry_disease` | `database/` | 5,752 | strawberry | leaf | — | 见子目录（需解析） |
| `plant_doc_detection` | `database/` | 5,156 | — | leaf | — | 见子目录（需解析） |

## D 级数据集（7 个）

| 数据集 | 路径根 | 图片数 | 宿主 | 输入 | 标签文件夹数 | 主要病类（样本数） |
|--------|--------|--------|------|------|--------------|-------------------|
| `PlantDoc` | `database/` | 4,607 | — | leaf | — | 见子目录（需解析） |
| `chilli_leaf_classification` | `database/` | 1,066 | pepper | leaf | — | 见子目录（需解析） |
| `tomato_plant` | `database/` | 1,040 | tomato | leaf | — | 见子目录（需解析） |
| `Strawberry` | `database/` | 599 | strawberry | leaf | — | 见子目录（需解析） |
| `AppleScabFDs` | `database/` | 297 | apple | leaf | — | 见子目录（需解析） |
| `carrot_weeds_germany` | `database/` | 180 | — | leaf | — | 见子目录（需解析） |
| `lemon_leaves` | `database/` | 40 | — | leaf | — | 见子目录（需解析） |

## 说明

1. **叶片病害**与 **果实腐烂**（FruitVision Fresh/Rotten）是不同任务；后者可补 multistate 的 `diseased` 状态，不替代叶病细分类。
2. **`plant_village`** 与 **`Plant_Village_*`** 内容重叠；训练时二选一，并去掉 color/grayscale/segmented 等增强重复。
3. **`temp/`** 目录当前无叶片病害数据，不参与病害训练。
4. 机器可读清单：`data/disease_classification/dataset_inventory.json`（由脚本生成）。

## 推荐训练顺序

1. **番茄**：`tomato_leaf_disease` + `Plant_Village_Tomato`（去重）
2. **柑橘**：`orange_leaf_disease_classification` + `citrus_leaves` + `Plant_Village_Orange`
3. **全局**：consolidated `plant_village` + `PlantDoc` + `LeavesDisease_FFinal`

