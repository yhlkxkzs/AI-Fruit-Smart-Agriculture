# 任务1: 水果种类识别 - 脚本使用说明

## 📁 脚本列表

### 1. `train.py` - 训练脚本
**功能**: 使用YOLOv8分类模型进行水果种类识别训练

**使用方法**:
```bash
conda activate fruit_classification
cd /home/yuhanlin/APP/AFSA/task1_fruit_classification
python scripts/train.py
```

**功能特点**:
- 自动检测数据集中的类别
- 创建YOLOv8数据集配置文件
- 支持GPU/CPU训练
- 自动数据增强
- 保存最佳模型和训练配置

**输出**:
- `results/fruit_classification_YYYYMMDD_HHMMSS/` - 训练结果目录
- `models/best_model_YYYYMMDD_HHMMSS.pt` - 最佳模型权重
- `config/training_config_YYYYMMDD_HHMMSS.yaml` - 训练配置

---

### 2. `collect_metrics.py` - 指标收集脚本
**功能**: 从训练结果中收集并整理各种评估指标

**使用方法**:
```bash
conda activate fruit_classification
python scripts/collect_metrics.py --results_dir <训练结果目录>
```

**参数**:
- `--results_dir`: 训练结果目录路径（必需）
- `--output`: 输出JSON文件路径（可选，默认：results_dir/metrics.json）

**示例**:
```bash
python scripts/collect_metrics.py --results_dir results/fruit_classification_20240211_120000
```

**输出**:
- `metrics.json` - 包含所有训练指标的JSON文件

**收集的指标**:
- 训练损失（最终值、最小值、平均值）
- 验证损失（最终值、最小值、平均值）
- 学习率变化
- 最佳epoch信息
- 模型文件大小

---

### 3. `visualize_results.py` - 结果可视化脚本
**功能**: 生成训练曲线、混淆矩阵、准确率等图表

**使用方法**:
```bash
conda activate fruit_classification
python scripts/visualize_results.py --results_dir <训练结果目录>
```

**参数**:
- `--results_dir`: 训练结果目录路径（必需）
- `--output_dir`: 输出目录（可选，默认：results_dir/visualizations）

**示例**:
```bash
python scripts/visualize_results.py --results_dir results/fruit_classification_20240211_120000
```

**输出图表**:
- `training_curves.png` - 训练曲线（损失、学习率、准确率）
- `confusion_matrix.png` - 混淆矩阵
- `class_accuracy.png` - 各类别准确率
- `metrics_summary.png` - 指标摘要

---

## 🔄 完整工作流程

### 步骤1: 训练模型
```bash
conda activate fruit_classification
cd /home/yuhanlin/APP/AFSA/task1_fruit_classification
python scripts/train.py
```

### 步骤2: 收集指标
```bash
# 找到训练结果目录（通常在results/下）
python scripts/collect_metrics.py --results_dir results/fruit_classification_YYYYMMDD_HHMMSS
```

### 步骤3: 生成可视化图表
```bash
python scripts/visualize_results.py --results_dir results/fruit_classification_YYYYMMDD_HHMMSS
```

---

## 📊 训练配置说明

### 默认配置
- **模型**: YOLOv8n-cls (轻量级分类模型)
- **训练轮数**: 100 epochs
- **图像大小**: 224x224
- **批次大小**: 16
- **设备**: GPU (如果可用)

### 修改配置
编辑 `scripts/train.py` 中的以下变量：
```python
model_name = 'yolov8n-cls.pt'  # 可选: yolov8s-cls.pt, yolov8m-cls.pt等
epochs = 100
imgsz = 224
batch_size = 16
device = 0  # 0表示GPU，'cpu'表示CPU
```

---

## 📁 目录结构

```
task1_fruit_classification/
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── collect_metrics.py    # 指标收集脚本
│   ├── visualize_results.py  # 可视化脚本
│   └── README.md             # 本文件
├── dataset/                  # 数据集目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── config/                   # 配置文件目录
│   └── dataset.yaml         # 数据集配置（自动生成）
├── models/                   # 模型权重目录
├── results/                  # 训练结果目录
└── logs/                     # 日志目录
```

---

## ⚠️ 注意事项

1. **环境要求**: 确保已激活 `fruit_classification` conda环境
2. **数据集**: 确保数据集已准备好，结构为 `dataset/{train,val,test}/{class_name}/`
3. **GPU**: 如果有GPU，确保CUDA已正确配置
4. **磁盘空间**: 训练过程会生成大量文件，确保有足够的磁盘空间

---

## 🐛 常见问题

### Q1: 训练时出现CUDA错误
**A**: 检查CUDA是否可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
如果返回False，在`train.py`中将`device = 0`改为`device = 'cpu'`

### Q2: 内存不足
**A**: 减小批次大小：
```python
batch_size = 8  # 或更小
```

### Q3: 找不到数据集
**A**: 确保数据集目录结构正确：
```
dataset/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ...
├── val/
└── test/
```

---

**最后更新**: 2024-02-11
