#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务1: 水果种类识别 - 训练脚本
使用YOLOv8分类模型进行训练
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

# 添加项目根目录到路径（脚本在scripts目录下，需要向上两级到项目根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


def create_dataset_yaml(dataset_dir, output_path, class_names):
    """创建YOLOv8数据集配置文件"""
    dataset_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✅ 数据集配置文件已创建: {output_path}")
    return output_path


def get_class_names(dataset_dir):
    """从数据集目录获取类别名称"""
    train_dir = dataset_dir / 'train'
    if not train_dir.exists():
        raise ValueError(f"训练目录不存在: {train_dir}")
    
    # 获取所有类别目录，排除隐藏文件和文件
    class_names = sorted([
        d.name for d in train_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    # 移除unknown类别（如果存在）
    if 'unknown' in class_names:
        class_names.remove('unknown')
        print("⚠️  警告: 检测到'unknown'类别，已从训练类别中移除")
    
    print(f"✅ 检测到 {len(class_names)} 个类别: {', '.join(class_names)}")
    return class_names


def main():
    """主训练函数"""
    print("=" * 60)
    print("任务1: 水果种类识别 - 模型训练")
    print("=" * 60)
    print()
    
    # 配置路径
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / 'dataset'
    config_dir = project_root / 'config'
    models_dir = project_root / 'models'
    logs_dir = project_root / 'logs'
    results_dir = project_root / 'results'
    
    # 创建必要的目录
    config_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 检查数据集是否存在
    if not dataset_dir.exists():
        raise ValueError(f"数据集目录不存在: {dataset_dir}")
    
    # 获取类别名称
    class_names = get_class_names(dataset_dir)
    num_classes = len(class_names)
    
    print(f"\n数据集信息:")
    print(f"  数据集路径: {dataset_dir}")
    print(f"  类别数量: {num_classes}")
    print(f"  类别列表: {', '.join(class_names)}")
    print()
    
    # 创建数据集配置文件
    dataset_yaml = config_dir / 'dataset.yaml'
    create_dataset_yaml(dataset_dir, dataset_yaml, class_names)
    
    # 训练配置
    model_name = 'yolov8n-cls.pt'  # 使用YOLOv8n分类模型（轻量级）
    epochs = 100
    imgsz = 224  # 图像大小
    batch_size = 16
    device = 0  # 使用GPU，如果只有CPU则改为 'cpu'
    
    print(f"\n训练配置:")
    print(f"  模型: {model_name}")
    print(f"  训练轮数: {epochs}")
    print(f"  图像大小: {imgsz}")
    print(f"  批次大小: {batch_size}")
    print(f"  设备: {'GPU' if device != 'cpu' else 'CPU'}")
    print()
    
    # 加载模型
    print("加载预训练模型...")
    model = YOLO(model_name)
    print(f"✅ 模型加载成功: {model_name}")
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"fruit_classification_{timestamp}"
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(results_dir),
            name=project_name,
            exist_ok=True,
            save=True,
            save_period=10,  # 每10个epoch保存一次
            val=True,  # 启用验证
            plots=True,  # 生成训练图表
            verbose=True,
            # 数据增强
            hsv_h=0.015,  # 色调增强
            hsv_s=0.7,    # 饱和度增强
            hsv_v=0.4,    # 明度增强
            degrees=10,   # 旋转角度
            translate=0.1,  # 平移
            scale=0.5,    # 缩放
            flipud=0.0,   # 上下翻转概率
            fliplr=0.5,  # 左右翻转概率
            mosaic=1.0,   # Mosaic增强概率
            mixup=0.1,   # Mixup增强概率
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print()
        
        # 保存最佳模型
        best_model_path = results_dir / project_name / 'weights' / 'best.pt'
        if best_model_path.exists():
            # 复制到models目录
            import shutil
            final_model_path = models_dir / f'best_model_{timestamp}.pt'
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ 最佳模型已保存: {final_model_path}")
        
        # 保存训练配置信息
        config_info = {
            'timestamp': timestamp,
            'model_name': model_name,
            'num_classes': num_classes,
            'class_names': class_names,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch_size': batch_size,
            'device': device,
            'project_name': project_name,
            'best_model_path': str(final_model_path) if best_model_path.exists() else None,
            'results_dir': str(results_dir / project_name)
        }
        
        config_file = config_dir / f'training_config_{timestamp}.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_info, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✅ 训练配置已保存: {config_file}")
        print(f"✅ 训练结果目录: {results_dir / project_name}")
        print()
        
        return results
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
