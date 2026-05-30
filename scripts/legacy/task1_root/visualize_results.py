#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务1: 水果种类识别 - 结果可视化脚本
生成训练曲线、混淆矩阵、准确率等图表
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from PIL import Image
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径（脚本在scripts目录下，需要向上两级到项目根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


def load_training_results(results_dir):
    """加载训练结果CSV"""
    results_dir = Path(results_dir)
    results_csv = results_dir / 'results.csv'
    
    if not results_csv.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_csv}")
    
    df = pd.read_csv(results_csv)
    return df


def plot_training_curves(df, output_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(df) + 1)
    
    # 1. 训练和验证损失
    ax1 = axes[0, 0]
    if 'train/loss' in df.columns:
        ax1.plot(epochs, df['train/loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    if 'val/loss' in df.columns:
        ax1.plot(epochs, df['val/loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 学习率
    ax2 = axes[0, 1]
    if 'lr/pg0' in df.columns:
        ax2.plot(epochs, df['lr/pg0'], label='Learning Rate', linewidth=2, color='#2ca02c')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Top-1准确率（如果有）
    ax3 = axes[1, 0]
    if 'metrics/accuracy_top1' in df.columns:
        ax3.plot(epochs, df['metrics/accuracy_top1'], label='Top-1 Accuracy', linewidth=2, color='#d62728')
    if 'metrics/accuracy_top5' in df.columns:
        ax3.plot(epochs, df['metrics/accuracy_top5'], label='Top-5 Accuracy', linewidth=2, color='#9467bd')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. 损失对比（训练vs验证）
    ax4 = axes[1, 1]
    if 'train/loss' in df.columns and 'val/loss' in df.columns:
        # 计算移动平均
        window = min(5, len(df) // 10)
        if window > 1:
            train_smooth = df['train/loss'].rolling(window=window, center=True).mean()
            val_smooth = df['val/loss'].rolling(window=window, center=True).mean()
            ax4.plot(epochs, train_smooth, label='Train Loss (Smoothed)', linewidth=2, alpha=0.7, color='#1f77b4')
            ax4.plot(epochs, val_smooth, label='Val Loss (Smoothed)', linewidth=2, alpha=0.7, color='#ff7f0e')
        else:
            ax4.plot(epochs, df['train/loss'], label='Train Loss', linewidth=2, color='#1f77b4')
            ax4.plot(epochs, df['val/loss'], label='Val Loss', linewidth=2, color='#ff7f0e')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Loss Comparison (Smoothed)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ 训练曲线已保存: {output_path}")
    plt.close()


def plot_confusion_matrix(results_dir, output_path):
    """绘制混淆矩阵（如果存在）"""
    results_dir = Path(results_dir)
    confusion_matrix_path = results_dir / 'val' / 'confusion_matrix.png'
    
    if confusion_matrix_path.exists():
        # 复制现有的混淆矩阵
        import shutil
        shutil.copy2(confusion_matrix_path, output_path)
        print(f"✅ 混淆矩阵已复制: {output_path}")
    else:
        print(f"⚠️  混淆矩阵文件不存在: {confusion_matrix_path}")


def plot_class_accuracy(results_dir, output_path, class_names=None):
    """绘制各类别准确率（如果测试结果存在）"""
    results_dir = Path(results_dir)
    test_results_file = results_dir / 'test' / 'results.json'
    
    if not test_results_file.exists():
        print(f"⚠️  测试结果文件不存在: {test_results_file}")
        return
    
    try:
        with open(test_results_file, 'r') as f:
            test_results = json.load(f)
        
        # 提取类别准确率（如果存在）
        if 'per_class_accuracy' in test_results:
            per_class_acc = test_results['per_class_accuracy']
            
            if class_names is None:
                class_names = list(per_class_acc.keys())
            
            accuracies = [per_class_acc.get(name, 0) for name in class_names]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            bars = ax.barh(class_names, accuracies, color='#2ca02c', alpha=0.7)
            ax.set_xlabel('Accuracy', fontsize=12)
            ax.set_ylabel('Class', fontsize=12)
            ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', facecolor='white')
            print(f"✅ 类别准确率图表已保存: {output_path}")
            plt.close()
        else:
            print("⚠️  测试结果中未找到类别准确率数据")
    
    except Exception as e:
        print(f"⚠️  读取测试结果失败: {str(e)}")


def plot_metrics_summary(results_dir, output_path):
    """绘制指标摘要图表"""
    results_dir = Path(results_dir)
    metrics_file = results_dir / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"⚠️  指标文件不存在: {metrics_file}")
        return
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Training Metrics Summary', fontsize=16, fontweight='bold')
        
        # 1. 损失对比
        ax1 = axes[0]
        if 'training_metrics' in metrics:
            tm = metrics['training_metrics']
            if 'train_loss' in tm and 'val_loss' in tm:
                categories = ['Final', 'Minimum', 'Mean']
                train_values = [
                    tm['train_loss'].get('final', 0),
                    tm['train_loss'].get('min', 0),
                    tm['train_loss'].get('mean', 0)
                ]
                val_values = [
                    tm['val_loss'].get('final', 0),
                    tm['val_loss'].get('min', 0),
                    tm['val_loss'].get('mean', 0)
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax1.bar(x - width/2, train_values, width, label='Train Loss', color='#1f77b4', alpha=0.7)
                ax1.bar(x + width/2, val_values, width, label='Val Loss', color='#ff7f0e', alpha=0.7)
                ax1.set_xlabel('Metric Type', fontsize=11)
                ax1.set_ylabel('Loss', fontsize=11)
                ax1.set_title('Loss Statistics', fontsize=12, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories)
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 最佳模型信息
        ax2 = axes[1]
        if 'best_epoch' in metrics and metrics['best_epoch']:
            best_epoch = metrics['best_epoch']
            if 'best_metrics' in metrics:
                bm = metrics['best_metrics']
                train_loss = bm.get('train_loss', 0)
                val_loss = bm.get('val_loss', 0)
                
                ax2.bar(['Train Loss', 'Val Loss'], [train_loss, val_loss], 
                       color=['#1f77b4', '#ff7f0e'], alpha=0.7)
                ax2.set_ylabel('Loss', fontsize=11)
                ax2.set_title(f'Best Model (Epoch {best_epoch})', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # 添加数值标签
                ax2.text(0, train_loss + 0.01, f'{train_loss:.4f}', ha='center', fontsize=10)
                ax2.text(1, val_loss + 0.01, f'{val_loss:.4f}', ha='center', fontsize=10)
        
        # 3. 模型大小
        ax3 = axes[2]
        if 'best_model_size_mb' in metrics:
            model_size = metrics['best_model_size_mb']
            ax3.bar(['Model Size'], [model_size], color='#2ca02c', alpha=0.7)
            ax3.set_ylabel('Size (MB)', fontsize=11)
            ax3.set_title('Model Size', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.text(0, model_size + 0.5, f'{model_size:.2f} MB', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"✅ 指标摘要图表已保存: {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"⚠️  读取指标文件失败: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='训练结果目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：results_dir/visualizations）')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ 错误: 结果目录不存在: {results_dir}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 训练曲线
    try:
        df = load_training_results(results_dir)
        plot_training_curves(df, output_dir / 'training_curves.png')
    except Exception as e:
        print(f"⚠️  生成训练曲线失败: {str(e)}")
    
    # 2. 混淆矩阵
    plot_confusion_matrix(results_dir, output_dir / 'confusion_matrix.png')
    
    # 3. 类别准确率
    plot_class_accuracy(results_dir, output_dir / 'class_accuracy.png')
    
    # 4. 指标摘要
    plot_metrics_summary(results_dir, output_dir / 'metrics_summary.png')
    
    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print(f"所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
