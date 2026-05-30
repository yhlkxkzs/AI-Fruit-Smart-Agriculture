#!/usr/bin/env python3
"""
根据JSON标注文件中的原始来源信息，将unknown图片重新分类到对应的水果类别
"""
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# 路径配置
UNKNOWN_DIRS = {
    'train': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    'val': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    'test': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
}

OUTPUT_BASE = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset'

# 从原始数据集路径推断水果类别的映射规则
FRUIT_INFERENCE_RULES = {
    # 数据集名称 -> 水果类别
    'almonds': 'almond',
    'almond': 'almond',
    'acfr-multifruit-2016': 'almond',  # 这个数据集主要是almonds
    
    # 子目录名称 -> 水果类别
    'strawberries': 'strawberry',
    'strawberry': 'strawberry',
    'litchis': 'lychee',
    'litchi': 'lychee',
    'lychee': 'lychee',
    'papayas': 'papaya',
    'papaya': 'papaya',
    'coconuts': 'coconut',
    'coconut': 'coconut',
    'apples': 'apple',
    'apple': 'apple',
    'mangoes': 'mango',
    'mango': 'mango',
    'bananas': 'banana',
    'banana': 'banana',
    'pomegranates': 'pomegranate',
    'pomegranate': 'pomegranate',
    'oranges': 'orange',
    'orange': 'orange',
    'guavas': 'guava',
    'guava': 'guava',
    
    # deepfruits数据集中的组合目录名
    'mango_grape_plum_kiwi_pear': ['mango', 'grape', 'plum', 'kiwi', 'pear'],
    'pineapple_fig_peach_apricot_avocado': ['pineapple', 'fig', 'peach', 'apricot', 'avocado'],
    'summer_squash_lemon_lime_guava_raspberry': ['summer squash', 'lemon', 'lime', 'guava', 'raspberry'],
}

def infer_fruit_from_path(original_path, dataset, subdirectory):
    """从原始路径推断水果类别"""
    if not original_path:
        return None
    
    # 方法1: 从子目录名推断
    if subdirectory:
        subdir_lower = subdirectory.lower()
        # 检查是否直接匹配
        if subdir_lower in FRUIT_INFERENCE_RULES:
            result = FRUIT_INFERENCE_RULES[subdir_lower]
            if isinstance(result, list):
                # 如果是组合类别，返回第一个（或者可以返回列表让用户选择）
                return result[0]
            return result
        
        # 检查子目录名中是否包含水果关键词
        for key, value in FRUIT_INFERENCE_RULES.items():
            if key in subdir_lower:
                if isinstance(value, list):
                    return value[0]
                return value
    
    # 方法2: 从数据集名称推断
    if dataset:
        dataset_lower = dataset.lower()
        if dataset_lower in FRUIT_INFERENCE_RULES:
            result = FRUIT_INFERENCE_RULES[dataset_lower]
            if isinstance(result, list):
                return result[0]
            return result
    
    # 方法3: 从完整路径推断
    path_lower = original_path.lower()
    
    # 检查路径中的关键词
    fruit_keywords = {
        'almond': 'almond',
        'almonds': 'almond',
        'strawberry': 'strawberry',
        'strawberries': 'strawberry',
        'litchi': 'lychee',
        'lychee': 'lychee',
        'papaya': 'papaya',
        'papayas': 'papaya',
        'coconut': 'coconut',
        'coconuts': 'coconut',
        'apple': 'apple',
        'apples': 'apple',
        'mango': 'mango',
        'mangoes': 'mango',
        'banana': 'banana',
        'bananas': 'banana',
        'pomegranate': 'pomegranate',
        'pomegranates': 'pomegranate',
        'orange': 'orange',
        'oranges': 'orange',
        'guava': 'guava',
        'guavas': 'guava',
        'grape': 'grape',
        'grapes': 'grape',
        'plum': 'plum',
        'plums': 'plum',
        'kiwi': 'kiwi',
        'pear': 'pear',
        'pears': 'pear',
        'pineapple': 'pineapple',
        'fig': 'fig',
        'peach': 'peach',
        'peaches': 'peach',
        'apricot': 'apricot',
        'avocado': 'avocado',
        'lemon': 'lemon',
        'lime': 'lime',
        'raspberry': 'raspberry',
        'raspberries': 'raspberry',
    }
    
    # 按长度排序，优先匹配较长的关键词
    for keyword, fruit in sorted(fruit_keywords.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in path_lower:
            return fruit
    
    return None

def reclassify_unknown_images():
    """重新分类unknown图片"""
    print("=" * 70)
    print("根据JSON标注重新分类Unknown图片")
    print("=" * 70)
    
    stats = defaultdict(lambda: {'moved': 0, 'skipped': 0, 'no_match': 0})
    reclassification_log = []
    
    for split, unknown_dir in UNKNOWN_DIRS.items():
        if not os.path.exists(unknown_dir):
            continue
        
        print(f"\n处理 {split} 集...")
        split_dir = os.path.join(OUTPUT_BASE, split)
        
        for filename in os.listdir(unknown_dir):
            # 只处理图片文件，跳过JSON文件
            if filename.endswith('.json') or not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                continue
            
            # 读取对应的JSON文件
            json_filename = filename + '.json'
            json_filepath = os.path.join(unknown_dir, json_filename)
            
            if not os.path.exists(json_filepath):
                print(f"  警告: {filename} 没有对应的JSON文件")
                stats[split]['skipped'] += 1
                continue
            
            # 读取JSON内容
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"  错误: 无法读取 {json_filename}: {e}")
                stats[split]['skipped'] += 1
                continue
            
            # 获取原始来源信息
            original_source = json_data.get('original_source')
            if not original_source:
                # 未匹配的图片，跳过
                stats[split]['no_match'] += 1
                continue
            
            dataset = original_source.get('dataset')
            subdirectory = original_source.get('subdirectory')
            original_path = original_source.get('original_path')
            
            # 推断水果类别
            fruit_category = infer_fruit_from_path(original_path, dataset, subdirectory)
            
            if not fruit_category:
                print(f"  无法推断 {filename} 的水果类别 (dataset: {dataset}, subdir: {subdirectory})")
                stats[split]['no_match'] += 1
                continue
            
            # 目标目录
            target_dir = os.path.join(split_dir, fruit_category)
            os.makedirs(target_dir, exist_ok=True)
            
            # 目标文件路径（保持原文件名）
            target_filepath = os.path.join(target_dir, filename)
            
            # 如果目标文件已存在，跳过
            if os.path.exists(target_filepath):
                print(f"  跳过 {filename} (目标文件已存在)")
                stats[split]['skipped'] += 1
                continue
            
            # 移动图片文件
            source_filepath = os.path.join(unknown_dir, filename)
            try:
                shutil.move(source_filepath, target_filepath)
                
                # 同时移动JSON文件
                target_json_path = os.path.join(target_dir, json_filename)
                shutil.move(json_filepath, target_json_path)
                
                stats[split]['moved'] += 1
                reclassification_log.append({
                    'filename': filename,
                    'split': split,
                    'from': 'unknown',
                    'to': fruit_category,
                    'dataset': dataset,
                    'subdirectory': subdirectory
                })
                
                if stats[split]['moved'] % 50 == 0:
                    print(f"  已移动 {stats[split]['moved']} 张图片...")
            except Exception as e:
                print(f"  错误: 无法移动 {filename}: {e}")
                stats[split]['skipped'] += 1
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("重新分类完成！")
    print("=" * 70)
    
    total_moved = 0
    total_skipped = 0
    total_no_match = 0
    
    for split in ['train', 'val', 'test']:
        moved = stats[split]['moved']
        skipped = stats[split]['skipped']
        no_match = stats[split]['no_match']
        total_moved += moved
        total_skipped += skipped
        total_no_match += no_match
        print(f"\n{split} 集:")
        print(f"  已移动: {moved} 张")
        print(f"  跳过: {skipped} 张")
        print(f"  无法匹配: {no_match} 张")
    
    print(f"\n总计:")
    print(f"  已移动: {total_moved} 张")
    print(f"  跳过: {total_skipped} 张")
    print(f"  无法匹配: {total_no_match} 张")
    
    # 按水果类别统计
    fruit_stats = defaultdict(int)
    for log_entry in reclassification_log:
        fruit_stats[log_entry['to']] += 1
    
    if fruit_stats:
        print(f"\n重新分类的水果类别分布:")
        print("-" * 70)
        for fruit, count in sorted(fruit_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fruit:20s}: {count:5d} 张")
    
    # 保存重新分类日志
    log_file = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/reclassification_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_moved': total_moved,
                'total_skipped': total_skipped,
                'total_no_match': total_no_match
            },
            'fruit_distribution': dict(fruit_stats),
            'details': reclassification_log
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n重新分类日志已保存到: {log_file}")

if __name__ == '__main__':
    reclassify_unknown_images()
