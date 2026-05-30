#!/usr/bin/env python3
"""
为每张unknown图片创建同名的JSON文件，包含原始来源信息
"""
import json
import os
from pathlib import Path

# 路径配置
UNKNOWN_DIRS = {
    'train': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    'val': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    'test': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
}

TRACE_FILE = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_images_trace.json'

def create_individual_json_files():
    """为每张unknown图片创建对应的JSON文件"""
    print("=" * 70)
    print("为每张Unknown图片创建对应的JSON文件")
    print("=" * 70)
    
    # 1. 读取追踪结果
    print("\n1. 读取追踪结果...")
    with open(TRACE_FILE, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)
    
    images_info = trace_data.get('images', {})
    print(f"   读取了 {len(images_info)} 张图片的信息")
    
    # 2. 为每张图片创建JSON文件
    print("\n2. 创建JSON文件...")
    created_count = 0
    skipped_count = 0
    
    for split, unknown_dir in UNKNOWN_DIRS.items():
        if not os.path.exists(unknown_dir):
            continue
        
        print(f"\n处理 {split} 集...")
        split_count = 0
        
        for filename in os.listdir(unknown_dir):
            # 只处理图片文件
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                continue
            
            # 获取图片信息
            if filename not in images_info:
                print(f"  警告: {filename} 未在追踪结果中找到")
                skipped_count += 1
                continue
            
            img_info = images_info[filename]
            
            # 创建JSON文件名（与图片同名，扩展名为.json）
            json_filename = filename + '.json'
            json_filepath = os.path.join(unknown_dir, json_filename)
            
            # 准备JSON内容
            json_content = {
                'image_filename': filename,
                'image_path': os.path.join(unknown_dir, filename),
                'split': split,
                'matched': img_info.get('matched', False),
                'original_source': {
                    'original_path': img_info.get('original_path'),
                    'original_relative_path': img_info.get('original_relative_path'),
                    'dataset': img_info.get('dataset'),
                    'dataset_source': img_info.get('dataset_source'),
                    'subdirectory': img_info.get('subdirectory')
                } if img_info.get('matched') else None
            }
            
            # 写入JSON文件
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)
                created_count += 1
                split_count += 1
                
                if created_count % 100 == 0:
                    print(f"  已创建 {created_count} 个JSON文件...")
            except Exception as e:
                print(f"  错误: 无法创建 {json_filename}: {e}")
                skipped_count += 1
        
        print(f"  {split} 集: 创建了 {split_count} 个JSON文件")
    
    # 3. 统计信息
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"成功创建: {created_count} 个JSON文件")
    print(f"跳过/错误: {skipped_count} 个")
    
    # 显示示例
    print("\n示例JSON文件内容:")
    print("-" * 70)
    for split, unknown_dir in UNKNOWN_DIRS.items():
        if not os.path.exists(unknown_dir):
            continue
        for filename in os.listdir(unknown_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                json_filename = filename + '.json'
                json_filepath = os.path.join(unknown_dir, json_filename)
                if os.path.exists(json_filepath):
                    with open(json_filepath, 'r', encoding='utf-8') as f:
                        example = json.load(f)
                    print(f"\n文件: {split}/unknown/{json_filename}")
                    print(json.dumps(example, indent=2, ensure_ascii=False))
                    break
        break

if __name__ == '__main__':
    create_individual_json_files()
