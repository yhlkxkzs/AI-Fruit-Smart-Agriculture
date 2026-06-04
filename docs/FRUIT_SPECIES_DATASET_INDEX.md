# 水果种类 × 数据集索引

路径：`/home/yuhanlin/Database/datasets/database`（原 `github/database` + `local/database` 已合并）

统计方式：从各数据集的顶层目录、`Training/Test` 类文件夹、`labelmap.json` 解析水果名，归一化为英文基础名（如 `Apple 10` → `apple`）。

## 1. 共有多少种水果

- **GitHub 库去重**：33 种
- **Local 库去重**：63 种
- **两库合并去重**：**63 种**

全部种类：

`almond, apple, apricot, avocado, banana, bean, beetroot, blackberry, blueberry, broccoli, carrot, cauliflower, cherry, chestnut, citrus, clementine, coconut, corn, cranberry, cucumber, date, eggplant, fig, gooseberry, grape, grapefruit, guava, hogplum, jackfruit, kiwi, kumquat, lemon, lime, litchi, lychee, mango, melon, mulberry, nectarine, orange, papaya, passion, passionfruit, peach, pear, pepper, pineapple, pistachio, plum, pomegranate, pomelo, potato, pumpkin, quince, rambutan, raspberry, soybean, strawberry, tangerine, tomato, walnut, watermelon, zucchini`

## 2. 每个数据集有多少种水果

| 库 | 数据集 | 水果种类数 | 水果列表 |
|----|--------|-----------|----------|
| github | AppleBBCH81 | 1 | apple |
| github | AppleScabFDs | 1 | apple |
| github | CherryBBCH72 | 1 | cherry |
| github | CherryBBCH81 | 1 | cherry |
| github | Multi-species-fruit-flower | 3 | apple, peach, pear |
| github | Pear640 | 1 | pear |
| github | Pistachio | 1 | pistachio |
| github | RawRipe | 10 | apple, banana, coconut, guava, litchi, mango, orange, papaya, pomegranate, strawberry |
| github | acfr-multifruit-2016 | 3 | almond, apple, mango |
| github | apple_minnesota | 1 | apple |
| github | cassava | 0 | — |
| github | citrus_leaves | 1 | citrus |
| github | deepfruits | 19 | apple, apricot, avocado, banana, fig, grape, guava, kiwi, lemon, lime, mango, orange, peach, pear, pineapple, plum, pomegranate, raspberry, strawberry |
| github | embrapa_add_256 | 2 | apple, grape |
| github | fruit-salad | 10 | apple, avocado, banana, blueberry, fig, kiwi, orange, pear, pineapple, strawberry |
| github | lemon-datase | 1 | lemon |
| github | merged_fruit_detection | 0 | — |
| github | plant_leaves | 1 | bean |
| github | plant_village | 13 | apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, strawberry, tomato |
| github | scripts | 0 | — |
| github | tomato_plant | 1 | tomato |
| local | Banana_Ripening_Process | 1 | banana |
| local | DragonFruitLeafDiseaseDataset_Clean | 0 | — |
| local | FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection | 5 | apple, banana, grape, mango, orange |
| local | Fruits Images Dataset_Object Detection | 10 | apple, banana, grape, guava, hogplum, jackfruit, litchi, mango, orange, papaya |
| local | LeavesDisease_FFinal | 11 | apple, cherry, citrus, corn, grape, orange, peach, pepper, potato, strawberry, tomato |
| local | North_American_Mushrooms | 0 | — |
| local | PlantDoc | 10 | apple, blueberry, cherry, corn, grape, peach, potato, raspberry, strawberry, tomato |
| local | Plant_Village_Apple | 1 | apple |
| local | Plant_Village_Blueberry | 1 | blueberry |
| local | Plant_Village_Cherry | 1 | cherry |
| local | Plant_Village_Corn | 1 | corn |
| local | Plant_Village_Grape | 1 | grape |
| local | Plant_Village_Orange | 1 | orange |
| local | Plant_Village_Peach | 1 | peach |
| local | Plant_Village_Pepper | 1 | pepper |
| local | Plant_Village_Potato | 1 | potato |
| local | Plant_Village_Raspberry | 1 | raspberry |
| local | Plant_Village_Soybean | 1 | soybean |
| local | Plant_Village_Squash | 0 | — |
| local | Plant_Village_Strawberry | 1 | strawberry |
| local | Plant_Village_Tomato | 1 | tomato |
| local | SPVD | 0 | — |
| local | Soybean | 1 | soybean |
| local | Strawberry | 1 | strawberry |
| local | apple_detection_drone_brazil | 2 | apple, strawberry |
| local | apple_flower_segmentation | 1 | apple |
| local | apple_segmentation_minnesota | 1 | apple |
| local | arabica_coffee_leaf_disease_classification | 0 | — |
| local | banana | 1 | banana |
| local | bean_disease_uganda | 1 | bean |
| local | betelnut | 0 | — |
| local | blackgram_plant_leaf_disease_classification | 0 | — |
| local | carrot_weeds_germany | 1 | carrot |
| local | chestnut | 1 | chestnut |
| local | chilli_leaf_classification | 0 | — |
| local | citrus_leaves | 1 | citrus |
| local | coconut_tree_disease_classification | 1 | coconut |
| local | corn_maize_leaf_disease | 1 | corn |
| local | cucumber_disease_classification | 1 | cucumber |
| local | deep_weeds | 0 | — |
| local | eggplant_preprocessed_2026 | 1 | eggplant |
| local | embrapa_wgisd_grape_detection | 1 | grape |
| local | fruits | 5 | apple, banana, grape, orange, pear |
| local | fruits-360_100x100 | 53 | almond, apple, apricot, avocado, banana, bean, beetroot, blackberry, blueberry, carrot, cauliflower, cherry, chestnut, clementine, corn, cucumber, date, eggplant, fig, gooseberry, grape, grapefruit, guava, kiwi, kumquat, lemon, lime, lychee, mango, melon, mulberry, nectarine, orange, papaya, passion, passionfruit, peach, pear, pepper, pineapple, pistachio, plum, pomegranate, pomelo, potato, quince, rambutan, raspberry, strawberry, tomato, walnut, watermelon, zucchini |
| local | fruits-360_3-body-problem | 2 | apple, cherry |
| local | fruits-360_meta | 0 | — |
| local | fruits-360_multi | 33 | apple, apricot, banana, blackberry, carrot, cherry, chestnut, cucumber, date, grape, kiwi, lemon, lime, lychee, mango, melon, mulberry, nectarine, orange, peach, pear, pepper, plum, pomegranate, potato, pumpkin, rambutan, raspberry, strawberry, tangerine, tomato, watermelon, zucchini |
| local | fruits-360_original-size | 25 | almond, apple, avocado, banana, bean, blackberry, carrot, cherry, cucumber, eggplant, gooseberry, grape, nectarine, orange, papaya, peach, pear, pepper, pistachio, plum, quince, raspberry, strawberry, tomato, zucchini |
| local | ghai_broccoli_detection | 1 | broccoli |
| local | ghai_strawberry_fruit_detection | 1 | strawberry |
| local | grapevine_leaf_diseases | 0 | — |
| local | java_plum_leaf_disease_classification | 1 | plum |
| local | lemon_leaves | 1 | lemon |
| local | mango_detection_australia | 1 | mango |
| local | orange_leaf_disease_classification | 2 | citrus, orange |
| local | paddy_disease_classification | 0 | — |
| local | papaya_leaf_disease_classification | 1 | papaya |
| local | peachpear_flower_segmentation | 3 | apple, peach, pear |
| local | plant_doc_detection | 10 | apple, blueberry, cherry, corn, grape, peach, potato, raspberry, strawberry, tomato |
| local | plant_leaves | 1 | bean |
| local | plantae_k | 8 | apple, apricot, cherry, cranberry, grape, peach, pear, walnut |
| local | rangeland_weeds_australia | 5 | apple, avocado, mango, orange, strawberry |
| local | rice_leaf_disease_classification | 0 | — |
| local | rice_seedling_segmentation | 0 | — |
| local | riseholme_strawberry_classification_2021 | 1 | strawberry |
| local | scripts | 0 | — |
| local | soybean_weed_uav_brazil | 1 | soybean |
| local | strawberry_disease | 1 | strawberry |
| local | strawberry_multiclass | 1 | strawberry |
| local | sugarbeet_weed_segmentation | 0 | — |
| local | sunflower_disease_classification | 0 | — |
| local | tea_leaf_disease_classification | 0 | — |
| local | tomato_leaf_disease | 1 | tomato |
| local | tomato_preprocessed | 1 | tomato |
| local | tomato_ripeness_detection | 1 | tomato |
| local | vegann_multicrop_presence_segmentation | 5 | bean, pepper, potato, raspberry, soybean |
| local | wheat_head_counting | 0 | — |

## 3. 每种水果在哪些数据集

| 水果 | 数据集数量 | 所在数据集（完整列表） |
|------|-----------|------------------------|
| almond | 3 | github/acfr-multifruit-2016; local/fruits-360_100x100; local/fruits-360_original-size |
| apple | 27 | github/AppleBBCH81; github/AppleScabFDs; github/Multi-species-fruit-flower; github/RawRipe; github/acfr-multifruit-2016; github/apple_minnesota; github/deepfruits; github/embrapa_add_256; github/fruit-salad; github/plant_village; local/FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection; local/Fruits Images Dataset_Object Detection; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Apple; local/apple_detection_drone_brazil; local/apple_flower_segmentation; local/apple_segmentation_minnesota; local/fruits; local/fruits-360_100x100; local/fruits-360_3-body-problem; local/fruits-360_multi; local/fruits-360_original-size; local/peachpear_flower_segmentation; local/plant_doc_detection; local/plantae_k; local/rangeland_weeds_australia |
| apricot | 4 | github/deepfruits; local/fruits-360_100x100; local/fruits-360_multi; local/plantae_k |
| avocado | 5 | github/deepfruits; github/fruit-salad; local/fruits-360_100x100; local/fruits-360_original-size; local/rangeland_weeds_australia |
| banana | 11 | github/RawRipe; github/deepfruits; github/fruit-salad; local/Banana_Ripening_Process; local/FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection; local/Fruits Images Dataset_Object Detection; local/banana; local/fruits; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
| bean | 6 | github/plant_leaves; local/bean_disease_uganda; local/fruits-360_100x100; local/fruits-360_original-size; local/plant_leaves; local/vegann_multicrop_presence_segmentation |
| beetroot | 1 | local/fruits-360_100x100 |
| blackberry | 3 | local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
| blueberry | 6 | github/fruit-salad; github/plant_village; local/PlantDoc; local/Plant_Village_Blueberry; local/fruits-360_100x100; local/plant_doc_detection |
| broccoli | 1 | local/ghai_broccoli_detection |
| carrot | 4 | local/carrot_weeds_germany; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
| cauliflower | 1 | local/fruits-360_100x100 |
| cherry | 12 | github/CherryBBCH72; github/CherryBBCH81; github/plant_village; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Cherry; local/fruits-360_100x100; local/fruits-360_3-body-problem; local/fruits-360_multi; local/fruits-360_original-size; local/plant_doc_detection; local/plantae_k |
| chestnut | 3 | local/chestnut; local/fruits-360_100x100; local/fruits-360_multi |
| citrus | 4 | github/citrus_leaves; local/LeavesDisease_FFinal; local/citrus_leaves; local/orange_leaf_disease_classification |
| clementine | 1 | local/fruits-360_100x100 |
| coconut | 2 | github/RawRipe; local/coconut_tree_disease_classification |
| corn | 7 | github/plant_village; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Corn; local/corn_maize_leaf_disease; local/fruits-360_100x100; local/plant_doc_detection |
| cranberry | 1 | local/plantae_k |
| cucumber | 4 | local/cucumber_disease_classification; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
| date | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| eggplant | 3 | local/eggplant_preprocessed_2026; local/fruits-360_100x100; local/fruits-360_original-size |
| fig | 3 | github/deepfruits; github/fruit-salad; local/fruits-360_100x100 |
| gooseberry | 2 | local/fruits-360_100x100; local/fruits-360_original-size |
| grape | 15 | github/deepfruits; github/embrapa_add_256; github/plant_village; local/FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection; local/Fruits Images Dataset_Object Detection; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Grape; local/embrapa_wgisd_grape_detection; local/fruits; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/plant_doc_detection; local/plantae_k |
| grapefruit | 1 | local/fruits-360_100x100 |
| guava | 4 | github/RawRipe; github/deepfruits; local/Fruits Images Dataset_Object Detection; local/fruits-360_100x100 |
| hogplum | 1 | local/Fruits Images Dataset_Object Detection |
| jackfruit | 1 | local/Fruits Images Dataset_Object Detection |
| kiwi | 4 | github/deepfruits; github/fruit-salad; local/fruits-360_100x100; local/fruits-360_multi |
| kumquat | 1 | local/fruits-360_100x100 |
| lemon | 5 | github/deepfruits; github/lemon-datase; local/fruits-360_100x100; local/fruits-360_multi; local/lemon_leaves |
| lime | 3 | github/deepfruits; local/fruits-360_100x100; local/fruits-360_multi |
| litchi | 2 | github/RawRipe; local/Fruits Images Dataset_Object Detection |
| lychee | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| mango | 9 | github/RawRipe; github/acfr-multifruit-2016; github/deepfruits; local/FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection; local/Fruits Images Dataset_Object Detection; local/fruits-360_100x100; local/fruits-360_multi; local/mango_detection_australia; local/rangeland_weeds_australia |
| melon | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| mulberry | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| nectarine | 3 | local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
| orange | 14 | github/RawRipe; github/deepfruits; github/fruit-salad; github/plant_village; local/FruitVision A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection; local/Fruits Images Dataset_Object Detection; local/LeavesDisease_FFinal; local/Plant_Village_Orange; local/fruits; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/orange_leaf_disease_classification; local/rangeland_weeds_australia |
| papaya | 5 | github/RawRipe; local/Fruits Images Dataset_Object Detection; local/fruits-360_100x100; local/fruits-360_original-size; local/papaya_leaf_disease_classification |
| passion | 1 | local/fruits-360_100x100 |
| passionfruit | 1 | local/fruits-360_100x100 |
| peach | 12 | github/Multi-species-fruit-flower; github/deepfruits; github/plant_village; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Peach; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/peachpear_flower_segmentation; local/plant_doc_detection; local/plantae_k |
| pear | 10 | github/Multi-species-fruit-flower; github/Pear640; github/deepfruits; github/fruit-salad; local/fruits; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/peachpear_flower_segmentation; local/plantae_k |
| pepper | 7 | github/plant_village; local/LeavesDisease_FFinal; local/Plant_Village_Pepper; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/vegann_multicrop_presence_segmentation |
| pineapple | 3 | github/deepfruits; github/fruit-salad; local/fruits-360_100x100 |
| pistachio | 3 | github/Pistachio; local/fruits-360_100x100; local/fruits-360_original-size |
| plum | 5 | github/deepfruits; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/java_plum_leaf_disease_classification |
| pomegranate | 4 | github/RawRipe; github/deepfruits; local/fruits-360_100x100; local/fruits-360_multi |
| pomelo | 1 | local/fruits-360_100x100 |
| potato | 8 | github/plant_village; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Potato; local/fruits-360_100x100; local/fruits-360_multi; local/plant_doc_detection; local/vegann_multicrop_presence_segmentation |
| pumpkin | 1 | local/fruits-360_multi |
| quince | 2 | local/fruits-360_100x100; local/fruits-360_original-size |
| rambutan | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| raspberry | 9 | github/deepfruits; github/plant_village; local/PlantDoc; local/Plant_Village_Raspberry; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/plant_doc_detection; local/vegann_multicrop_presence_segmentation |
| soybean | 5 | github/plant_village; local/Plant_Village_Soybean; local/Soybean; local/soybean_weed_uav_brazil; local/vegann_multicrop_presence_segmentation |
| strawberry | 18 | github/RawRipe; github/deepfruits; github/fruit-salad; github/plant_village; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Strawberry; local/Strawberry; local/apple_detection_drone_brazil; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/ghai_strawberry_fruit_detection; local/plant_doc_detection; local/rangeland_weeds_australia; local/riseholme_strawberry_classification_2021; local/strawberry_disease; local/strawberry_multiclass |
| tangerine | 1 | local/fruits-360_multi |
| tomato | 12 | github/plant_village; github/tomato_plant; local/LeavesDisease_FFinal; local/PlantDoc; local/Plant_Village_Tomato; local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size; local/plant_doc_detection; local/tomato_leaf_disease; local/tomato_preprocessed; local/tomato_ripeness_detection |
| walnut | 2 | local/fruits-360_100x100; local/plantae_k |
| watermelon | 2 | local/fruits-360_100x100; local/fruits-360_multi |
| zucchini | 3 | local/fruits-360_100x100; local/fruits-360_multi; local/fruits-360_original-size |
