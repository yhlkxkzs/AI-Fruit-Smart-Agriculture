#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务1: 水果种类识别 - 训练指标收集脚本
从训练结果中收集并整理各种评估指标
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径（脚本在scripts目录下，需要向上两级到项目根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_results_csv(csv_path):
    """解析YOLOv8训练结果CSV文件"""
    if not csv_path.exists():
        print(f"⚠️  警告: 结果CSV文件不存在: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取结果文件: {csv_path}")
        print(f"   包含 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {str(e)}")
        return None


def collect_training_metrics(results_dir):
    """收集训练指标"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise ValueError(f"结果目录不存在: {results_dir}")
    
    print("=" * 60)
    print("收集训练指标")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print()
    
    metrics = {
        'results_dir': str(results_dir),
        'timestamp': datetime.now().isoformat(),
        'training_metrics': {},
        'validation_metrics': {},
        'test_metrics': {},
        'best_epoch': None,
        'final_metrics': {}
    }
    
    # 查找results.csv文件
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        df = parse_results_csv(results_csv)
        if df is not None and len(df) > 0:
            # 提取关键指标
            metrics['training_metrics'] = {
                'total_epochs': len(df),
                'train_loss': {
                    'final': float(df['train/loss'].iloc[-1]) if 'train/loss' in df.columns else None,
                    'min': float(df['train/loss'].min()) if 'train/loss' in df.columns else None,
                    'mean': float(df['train/loss'].mean()) if 'train/loss' in df.columns else None,
                },
                'val_loss': {
                    'final': float(df['val/loss'].iloc[-1]) if 'val/loss' in df.columns else None,
                    'min': float(df['val/loss'].min()) if 'val/loss' in df.columns else None,
                    'mean': float(df['val/loss'].mean()) if 'val/loss' in df.columns else None,
                },
                'learning_rate': {
                    'initial': float(df['lr/pg0'].iloc[0]) if 'lr/pg0' in df.columns else None,
                    'final': float(df['lr/pg0'].iloc[-1]) if 'lr/pg0' in df.columns else None,
                }
            }
            
            # 找到最佳epoch（验证损失最低）
            if 'val/loss' in df.columns:
                best_idx = df['val/loss'].idxmin()
                metrics['best_epoch'] = int(best_idx) + 1
                metrics['best_metrics'] = {
                    'epoch': int(best_idx) + 1,
                    'train_loss': float(df['train/loss'].iloc[best_idx]),
                    'val_loss': float(df['val/loss'].iloc[best_idx]),
                }
            
            # 最终指标
            metrics['final_metrics'] = {
                'epoch': len(df),
                'train_loss': float(df['train/loss'].iloc[-1]) if 'train/loss' in df.columns else None,
                'val_loss': float(df['val/loss'].iloc[-1]) if 'val/loss' in df.columns else None,
            }
    
    # 查找验证结果（如果有）
    val_results_dir = results_dir / 'val'
    if val_results_dir.exists():
        # 查找混淆矩阵、PR曲线等
        confusion_matrix = val_results_dir / 'confusion_matrix.png'
        if confusion_matrix.exists():
            metrics['validation_metrics']['confusion_matrix'] = str(confusion_matrix)
        
        pr_curve = val_results_dir / 'PR_curve.png'
        if pr_curve.exists():
            metrics['validation_metrics']['pr_curve'] = str(pr_curve)
    
    # 查找测试结果（如果有）
    test_results_dir = results_dir / 'test'
    if test_results_dir.exists():
        test_results_file = test_results_dir / 'results.json'
        if test_results_file.exists():
            try:
                with open(test_results_file, 'r') as f:
                    test_results = json.load(f)
                metrics['test_metrics'] = test_results
            except Exception as e:
                print(f"⚠️  读取测试结果失败: {str(e)}")
    
    # 查找最佳模型权重
    weights_dir = results_dir / 'weights'
    if weights_dir.exists():
        best_model = weights_dir / 'best.pt'
        last_model = weights_dir / 'last.pt'
        
        if best_model.exists():
            metrics['best_model_path'] = str(best_model)
            metrics['best_model_size_mb'] = best_model.stat().st_size / (1024 * 1024)
        
        if last_model.exists():
            metrics['last_model_path'] = str(last_model)
            metrics['last_model_size_mb'] = last_model.stat().st_size / (1024 * 1024)
    
    return metrics


def save_metrics(metrics, output_path):
    """保存指标到JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 指标已保存: {output_path}")


def print_metrics_summary(metrics):
    """打印指标摘要"""
    print("\n" + "=" * 60)
    print("训练指标摘要")
    print("=" * 60)
    
    if 'training_metrics' in metrics and metrics['training_metrics']:
        tm = metrics['training_metrics']
        print(f"\n训练过程:")
        print(f"  总轮数: {tm.get('total_epochs', 'N/A')}")
        
        if 'train_loss' in tm and tm['train_loss']['final']:
            print(f"  训练损失: {tm['train_loss']['final']:.4f} (最低: {tm['train_loss']['min']:.4f})")
        
        if 'val_loss' in tm and tm['val_loss']['final']:
            print(f"  验证损失: {tm['val_loss']['final']:.4f} (最低: {tm['val_loss']['min']:.4f})")
    
    if 'best_epoch' in metrics and metrics['best_epoch']:
        print(f"\n最佳模型:")
        print(f"  轮次: {metrics['best_epoch']}")
        if 'best_metrics' in metrics:
            bm = metrics['best_metrics']
            print(f"  训练损失: {bm.get('train_loss', 'N/A'):.4f}")
            print(f"  验证损失: {bm.get('val_loss', 'N/A'):.4f}")
    
    if 'best_model_path' in metrics:
        print(f"\n模型文件:")
        print(f"  最佳模型: {metrics['best_model_path']}")
        if 'best_model_size_mb' in metrics:
            print(f"  模型大小: {metrics['best_model_size_mb']:.2f} MB")
    
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='收集训练指标')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='训练结果目录路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径（默认：results_dir/metrics.json）')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ 错误: 结果目录不存在: {results_dir}")
        return
    
    # 收集指标
    metrics = collect_training_metrics(results_dir)
    
    # 保存指标
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / 'metrics.json'
    
    save_metrics(metrics, output_path)
    
    # 打印摘要
    print_metrics_summary(metrics)


if __name__ == "__main__":
    main()
