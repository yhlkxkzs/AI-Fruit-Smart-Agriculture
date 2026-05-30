#!/usr/bin/env python3
"""
将unknown文件夹中的图片移动到指定类别文件夹，并重命名，同时删除JSON文件
"""
import os
import shutil
import re
from pathlib import Path

def move_and_rename_unknown_images(unknown_dir, target_dir, target_prefix):
    """
    将unknown文件夹中的图片移动到目标文件夹并重命名
    
    Args:
        unknown_dir: unknown文件夹路径
        target_dir: 目标文件夹路径
        target_prefix: 新的文件名前缀（如'almond'）
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 统计
    image_count = 0
    json_count = 0
    moved_count = 0
    skipped_count = 0
    deleted_json_count = 0
    
    # 获取已存在的文件编号，避免冲突
    existing_numbers = set()
    if os.path.exists(target_dir):
        for filename in os.listdir(target_dir):
            match = re.search(rf'{target_prefix}_(\d+)\.', filename)
            if match:
                existing_numbers.add(int(match.group(1)))
    
    # 找到下一个可用的编号
    next_number = max(existing_numbers) + 1 if existing_numbers else 0
    
    print(f"开始处理: {unknown_dir}")
    print(f"目标目录: {target_dir}")
    print(f"目标前缀: {target_prefix}")
    print(f"下一个可用编号: {next_number}\n")
    
    # 先收集所有图片文件
    image_files = []
    for filename in sorted(os.listdir(unknown_dir)):
        filepath = os.path.join(unknown_dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_files.append(filename)
    
    print(f"找到 {len(image_files)} 张图片\n")
    
    for filename in image_files:
        filepath = os.path.join(unknown_dir, filename)
        
        # 获取文件扩展名
        ext = Path(filename).suffix
        
        # 生成新文件名
        new_filename = f"{target_prefix}_{next_number:05d}{ext}"
        target_path = os.path.join(target_dir, new_filename)
        
        # 如果目标文件已存在，跳过
        if os.path.exists(target_path):
            skipped_count += 1
            print(f"跳过（已存在）: {new_filename}")
            next_number += 1
            continue
        
        # 移动并重命名图片文件
        try:
            shutil.move(filepath, target_path)
            moved_count += 1
            image_count += 1
            
            if moved_count % 50 == 0:
                print(f"已移动 {moved_count} 张图片...")
        except Exception as e:
            print(f"错误: 无法移动 {filename}: {e}")
            continue
        
        # 删除对应的JSON文件
        json_filename = filename + '.json'
        json_filepath = os.path.join(unknown_dir, json_filename)
        if os.path.exists(json_filepath):
            try:
                os.remove(json_filepath)
                deleted_json_count += 1
                json_count += 1
            except Exception as e:
                print(f"警告: 无法删除JSON文件 {json_filename}: {e}")
        
        next_number += 1
    
    print(f"\n完成！")
    print(f"图片文件: {image_count} 个")
    print(f"JSON文件: {json_count} 个（已删除）")
    print(f"已移动并重命名: {moved_count} 张图片")
    print(f"跳过: {skipped_count} 张图片（已存在）")
    
    return moved_count, deleted_json_count

if __name__ == '__main__':
    # 配置
    unknown_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
    target_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/almond'
    target_prefix = 'almond'
    
    move_and_rename_unknown_images(unknown_dir, target_dir, target_prefix)
