# 水果分类多状态训练报告（Fruit-State v1）

**项目**：AFSA — `fruit_classification`  
**日期**：2026-05-27 ~ 2026-05-30  
**任务类型**：双头分类（品种 + 生理/健康状态）  
**数据版本**：`data/fruit_classification_multistate/`（manifest 索引，130,554 样本）

---

## 1. 摘要

本次实验在 **9 个公开/本地数据集** 合并的 multistate manifest 上，对 **10 个推荐 backbone** 进行 **共享骨干 + 双分类头** 的并行训练（50 epoch，224×224，Adam lr=1e-3）。

| 指标 | 结果 |
|------|------|
| 完成 multistate 训练 | **8 / 10** |
| 未完成 | MobileNetV3、ShuffleNetV2（backbone 兼容问题，仍为 3 月单头旧权重） |
| 验证集最佳综合得分 | **0.9888**（EfficientNetV2-S） |
| 验证集品种准确率（最佳） | **99.85%**（RegNetY-400MF） |
| 验证集状态准确率（最佳） | **96.57%**（EfficientNet-Lite4） |

**推荐部署候选（移动端）**：EfficientNet-Lite0 / EfficientNet-Lite4（体积与得分平衡）  
**推荐部署候选（精度优先）**：EfficientNetV2-S / RegNetY-400MF

---

## 2. 任务定义

### 2.1 网络结构

```text
Input (3×224×224)
       ↓
  Shared Backbone (timm / torchvision)
       ↓
   Feature vector
     ↙         ↘
Head-Species   Head-State
  11 classes     7 classes
```

### 2.2 损失函数

```text
L = CE(species) + 0.5 × CE(state)
```

### 2.3 综合得分（模型选择）

```text
score = 0.7 × val_species_acc + 0.3 × val_state_acc
```

### 2.4 标签空间

**品种（11 类）**：apple, cherry, grape, lemon, mango, orange, pear, pistachio, strawberry, tomato, **other**（含 banana 等）

**状态（7 类）**：healthy, immature, mature, aged, diseased, pest, **unknown**

---

## 3. 数据集

### 3.1 规模与划分

| 划分 | 样本数 |
|------|--------|
| 训练 | 91,373 |
| 验证 | 19,568 |
| 测试 | 19,613 |
| **合计** | **130,554** |

划分方式：按 `(species, state)` 分层，约 70% / 15% / 15%，seed=42。  
**说明**：manifest 仅存源文件路径，**不复制图片**。

### 3.2 数据来源

| source_id | 样本数 | 贡献 |
|-----------|--------|------|
| fruitvision | 78,565 | 5 品种 × fresh/rotten/formalin |
| afsa_species_only | 24,454 | 原 10 类数据，state=unknown |
| banana_ripening | 18,074 | 香蕉成熟度 → species=other |
| rawripe | 1,630 | 10 品种 raw/ripe |
| riseholme_strawberry | 3,520 | 草莓 ripe/unripe/anomalous |
| lemon_qc | 2,690 | 柠檬（多数 state=unknown） |
| tomato_ripeness | 804 | 番茄成熟度 |
| tomato_plant | 520 | 番茄 green/red |
| apple_scab | 297 | 苹果 healthy/scab |

### 3.3 类别分布（需注意）

| 维度 | 观察 |
|------|------|
| **状态** | `aged` 55,000、`unknown` 28,447 占主导；`diseased` 仅 360；**`pest` 为 0** |
| **品种** | `other` 35,337（主要为 banana）；pear 仅 712 |
| **含义** | val 指标偏高部分来自「品种/状态较易区分」的大类；病害、虫害泛化尚未充分验证 |

---

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 50 |
| Batch size | 32~64（按模型/GPU 调整） |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Image size | 224 |
| Pretrained | ImageNet（timm / torchvision） |
| 硬件 | 10×GPU（4×RTX 2080 Ti + 6×V100 32GB） |
| 并行策略 | 每模型 1 GPU，`CUDA_VISIBLE_DEVICES` 隔离 |

**脚本**：

- 数据：`tasks/fruit_classification/scripts/prepare_multistate_dataset.py`
- 训练：`tasks/fruit_classification/scripts/train_multistate.py`
- 并行：`tasks/fruit_classification/scripts/run_parallel_multistate.sh`

---

## 5. 实验结果（验证集）

### 5.1 已完成模型（按 score 排序）

| 排名 | 模型 | timm/backbone | score | val 品种 | val 状态 | 权重路径 |
|------|------|---------------|-------|----------|----------|----------|
| 1 | **EfficientNetV2-S** | tf_efficientnetv2_s | **0.9888** | 99.77% | 96.46% | `exports/efficientnetv2_s/best.pt` |
| 2 | RegNetY-400MF | regnety_004 | 0.9887 | **99.85%** | 96.39% | `exports/regnety_400mf/best.pt` |
| 3 | EfficientNet-Lite0 | tf_efficientnet_lite0 | 0.9886 | 99.83% | 96.45% | `exports/efficientnet_lite0_multistate/best.pt` |
| 4 | EfficientNet-Lite4 | tf_efficientnet_lite4 | 0.9883 | 99.79% | **96.57%** | `exports/efficientnet_lite4/best.pt` |
| 5 | ResNet-18 | resnet18 | 0.9881 | 99.80% | 96.32% | `exports/resnet18/best.pt` |
| 6 | MobileViT-S | mobilevit_s | 0.9877 | 99.78% | 96.33% | `exports/mobilevit_s/best.pt` |
| 7 | ViT-Tiny | deit_tiny_patch16_224 | 0.9853 | 99.49% | 96.17% | `exports/vit_tiny/best.pt` |
| 8 | ConvNeXt-Tiny | convnext_tiny | 0.9850 | 99.42% | 95.53% | `exports/convnext_tiny/best.pt` |

### 5.2 未完成 / 旧权重

| 模型 | 状态 | 原因 |
|------|------|------|
| MobileNetV3 | ❌ multistate 未训成 | torchvision 输出维 1280 vs head 960 不匹配（代码已修，待重跑） |
| ShuffleNetV2 | ❌ multistate 未训成 | timm 无 shufflenet 名，需 torchvision（代码已修，待重跑） |

当前 `exports/mobilenet_v3/`、`exports/shufflenet_v2/` 内为 **2026-03 单头、仅品种** 旧 checkpoint，**不可**用于 multistate 推理。

### 5.3 EfficientNet-Lite0 单独首轮（参考）

| 项目 | 值 |
|------|-----|
| Run | `runs/multistate/efficientnet_lite0_20260527_085925` |
| Epoch 50 | train_sp=99.68%, train_st=95.61%, val_sp=99.83%, val_st=96.45% |
| best_score | 0.9886 |

与并行 batch 中其他模型可比；作为 **移动端默认候选** 仍合理。

---

## 6. 结论与建议

### 6.1 结论

1. **双头 multistate 方案可行**：8 个 backbone 在 val 上品种准确率均 >99.4%，状态 >95.5%。
2. **品种任务相对容易**：大量 `unknown`/`aged` 与明确成熟度标签使 state 头也有较高 acc，但不等于田间真实场景已覆盖。
3. **精度最高**：EfficientNetV2-S、RegNetY-400MF；**轻量与精度平衡**：EfficientNet-Lite0/Lite4。
4. **数据短板**：diseased/pest 样本极少，不宜过度解读「病害/虫害识别能力」。

### 6.2 下一步

| 优先级 | 行动 |
|--------|------|
| P0 | 补训 **MobileNetV3**、**ShuffleNetV2** multistate |
| P0 | 在 **test.csv（19,613）** 上统一评估 8 个已完成模型 |
| P1 | 选定 1 个移动端模型同步至 `exports/deploy_efficientnet_lite0/` 或新建 deploy 目录 |
| P1 | 增加果实虫害/病害自建数据，平衡 `diseased`/`pest` |
| P2 | 导出 ONNX/TFLite，对接 App 端侧推理 |
| P2 | GitHub Actions 推理仓需适配双头输出 JSON |

---

## 7. 附录

### 7.1 复现命令

```bash
cd /home/yuhanlin/APP/AFSA

# 生成 manifest
python tasks/fruit_classification/scripts/prepare_multistate_dataset.py --seed 42

# 单模型
python tasks/fruit_classification/scripts/train_multistate.py \
  --config tasks/fruit_classification/configs/multistate.yaml \
  --model-id efficientnet_lite0 --timm-name tf_efficientnet_lite0 \
  --epochs 50 --batch-size 64 --device cuda

# 10 模型并行（10 GPU）
bash tasks/fruit_classification/scripts/run_parallel_multistate.sh
```

### 7.2 相关文档

- `tasks/fruit_classification/docs/DATA_MULTISTATE.md`
- `tasks/fruit_classification/exports/MANIFEST.md`
- `data/fruit_classification_multistate/stats.json`

### 7.3 Run 目录

| 模型 | Run 路径 |
|------|----------|
| efficientnet_lite0 | `runs/multistate/efficientnet_lite0_20260527_085925/` |
| 并行 batch | `runs/multistate/parallel_20260530_003428/`（各模型 `.log`） |
| 其余 7 模型 | `runs/multistate/<model_id>_20260530_00343*/` |

---

*Report generated: 2026-05-30*
