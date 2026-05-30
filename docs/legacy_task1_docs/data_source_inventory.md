# 数据源用途清单（识别水果种类 · 果实数据）

目标任务：**识别水果种类**，需要的是**有果实（水果本体）**的数据集，而不是叶片病害、花朵、杂草、作物等。

以下根据各数据集 README 的 Task/描述 判断：**USE = 用于果实种类识别，SKIP = 不采用**。

---

## Local database (`/home/yuhanlin/Database/local/database`)

| 数据集 | 用途（README 摘要） | 判定 |
|--------|---------------------|------|
| **apple_detection_drone_brazil** | Apple **detection** in orchards (UAV), 果实 | **USE** |
| **apple_flower_segmentation** | Apple/peach/pear **flower** segmentation | SKIP（花） |
| **apple_segmentation_minnesota** | MinneApple: apple **fruit** detection | **USE** |
| arabica_coffee_leaf_disease_classification | Coffee **leaf** disease | SKIP |
| bean_disease_uganda | Bean **leaf** disease | SKIP |
| blackgram_plant_leaf_disease_classification | **Leaf** disease | SKIP |
| carrot_weeds_germany | **Weeds** in carrot fields | SKIP |
| chilli_leaf_classification | **Leaf** classification | SKIP |
| **citrus_leaves** | Citrus **fruits** (150) + leaves (609) | **USE**（仅用果实子集） |
| coconut_tree_disease_classification | **Coconut tree** disease (树/叶) | SKIP |
| corn_maize_leaf_disease | **Leaf** disease | SKIP |
| cucumber_disease_classification | Cucumber disease（叶/果混合，偏病害） | SKIP |
| deep_weeds | **Weeds** | SKIP |
| **embrapa_wgisd_grape_detection** | **Grape** detection in vineyard | **USE** |
| ghai_broccoli_detection | **Broccoli** detection（蔬菜，非水果） | SKIP |
| **ghai_strawberry_fruit_detection** | Strawberry **fruit** detection | **USE** |
| java_plum_leaf_disease_classification | **Leaf** disease | SKIP |
| **mango_detection_australia** | Mango **fruit** detection | **USE** |
| orange_leaf_disease_classification | **Orange leaf** disease | SKIP |
| paddy_disease_classification | **Rice** disease | SKIP |
| papaya_leaf_disease_classification | **Papaya leaf** disease | SKIP |
| peachpear_flower_segmentation | **Flower** segmentation | SKIP（花） |
| plantae_k | 植物种类（叶等） | SKIP |
| plant_doc_detection | Document/leaf 检测 | SKIP |
| plant_leaves | **Leaves** | SKIP |
| **Plant_Village_Apple** | Apple **leaf** disease (healthy/scab/black_rot…) | **SKIP**（叶） |
| **Plant_Village_Blueberry** | Blueberry **leaf** / 病害 | **SKIP**（叶） |
| **Plant_Village_Cherry** | Cherry **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Corn** | Corn **leaf** 等 | **SKIP**（叶） |
| **Plant_Village_Grape** | Grape **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Orange** | Orange **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Peach** | Peach **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Pepper** | Pepper **leaf** 等 | **SKIP**（叶） |
| **Plant_Village_Potato** | Potato **leaf** 等 | **SKIP**（叶） |
| **Plant_Village_Raspberry** | Raspberry **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Soybean** | Soybean **leaf** 等 | **SKIP**（叶） |
| **Plant_Village_Squash** | Squash **leaf** 等 | **SKIP**（叶） |
| **Plant_Village_Strawberry** | Strawberry **leaf** 病害 | **SKIP**（叶） |
| **Plant_Village_Tomato** | Tomato **leaf** 病害 | **SKIP**（叶） |
| rangeland_weeds_australia | **Weeds** | SKIP |
| rice_leaf_disease_classification | **Leaf** disease | SKIP |
| rice_seedling_segmentation | **Seedling** | SKIP |
| **riseholme_strawberry_classification_2021** | Strawberry **fruit** (ripe/unripe/occluded) | **USE** |
| soybean_weed_uav_brazil | **Soybean/weed** | SKIP |
| sugarbeet_weed_segmentation | **Weed** | SKIP |
| sunflower_disease_classification | **Leaf**/disease | SKIP |
| tea_leaf_disease_classification | **Leaf** disease | SKIP |
| tomato_leaf_disease | **Leaf** disease | SKIP |
| **tomato_ripeness_detection** | Tomato **fruit** ripeness | **USE** |
| vegann_multicrop_presence_segmentation | **Crop** presence | SKIP |
| wheat_head_counting | **Wheat head** counting | SKIP |

---

## GitHub database (`/home/yuhanlin/Database/github/database`)

| 数据集 | 用途（README 摘要） | 判定 |
|--------|---------------------|------|
| **acfr-multifruit-2016** | **Fruit** detection (apples, mangoes, almonds) | **USE** |
| **AppleBBCH81** | Apple **fruit** detection (BBCH 81–85) | **USE** |
| **apple_minnesota** | Apple **fruit** (同 MinneApple 等) | **USE** |
| AppleScabFDs | Apple **leaves/fruits** for scab（叶/果混合） | 可选（暂不纳入，偏病害） |
| **cassava** | **Cassava** root 病害（块根，非果实） | **SKIP** |
| **CherryBBCH72** | Cherry **fruit** detection (BBCH 72–73) | **USE** |
| **CherryBBCH81** | Cherry **fruit** (成熟期) | **USE** |
| citrus_leaves | 同 local：果实+叶，可取果实部分 | **USE**（仅果实） |
| **deepfruits** | **Fruit** images, 20 kinds | **USE** |
| **embrapa_add_256** | Apple **fruit** detection (UAV 256×256) | **USE** |
| **fruit-salad** | **Fruit** depictions (synthetic) | **USE** |
| **lemon-datase** | **Lemon** fruit quality control | **USE** |
| **merged_fruit_detection** | **Fruit** detection（合并果实检测） | **USE** |
| Multi-species-fruit-flower | **Flower** detection (apple/peach/pear) | SKIP（花） |
| **Pear640** | **Pear fruit** detection | **USE** |
| **Pistachio** | **Pistachio** image classification | **USE** |
| plant_leaves | **Leaves** | SKIP |
| plant_village | 同 Plant_Village_*，**叶**病害 | **SKIP** |
| **RawRipe** | **Fruit** raw/ripe (10 fruit types) | **USE** |
| **tomato_plant** | Tomato **fruit** (green/red) in plant factory | **USE** |

---

## 汇总：仅用于「识别水果种类」的数据源（USE）

- **Local**:  
  `apple_detection_drone_brazil`, `apple_segmentation_minnesota`, `citrus_leaves`（仅果实）, `embrapa_wgisd_grape_detection`, `ghai_strawberry_fruit_detection`, `mango_detection_australia`, `riseholme_strawberry_classification_2021`, `tomato_ripeness_detection`

- **GitHub**:  
  `acfr-multifruit-2016`, `AppleBBCH81`, `apple_minnesota`, `CherryBBCH72`, `CherryBBCH81`, `deepfruits`, `embrapa_add_256`, `fruit-salad`, `lemon-datase`, `merged_fruit_detection`, `Pear640`, `Pistachio`, `RawRipe`, `tomato_plant`

**不采用**：所有 Plant_Village_*（叶病害）、各类 leaf/weed/flower/seedling/crop 数据集。
