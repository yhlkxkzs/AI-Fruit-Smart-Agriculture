#!/usr/bin/env python3
"""
将unknown文件夹中的所有图片移动到对应的almond文件夹，并重命名为almond格式
同时删除JSON文件
"""
import os
import re
import shutil
from pathlib import Path

def move_unknown_to_almond(unknown_dir, almond_dir, split_name):
    """
    将unknown文件夹中的图片移动到almond文件夹并重命名
    
    Args:
        unknown_dir: unknown文件夹路径
        almond_dir: almond文件夹路径
        split_name: split名称（train/val/test）
    """
    if not os.path.exists(unknown_dir):
        print(f"跳过: {unknown_dir} 不存在")
        return 0, 0, 0
    
    if not os.path.exists(almond_dir):
        print(f"创建目录: {almond_dir}")
        os.makedirs(almond_dir, exist_ok=True)
    
    # 统计
    moved_count = 0
    json_deleted_count = 0
    error_count = 0
    
    print(f"\n处理 {split_name}/unknown -> {split_name}/almond")
    print(f"源目录: {unknown_dir}")
    print(f"目标目录: {almond_dir}\n")
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP')
    all_files = sorted(os.listdir(unknown_dir))
    image_files = [f for f in all_files if f.lower().endswith(image_extensions) and 
                   f.lower().startswith('unknown_')]
    
    if len(image_files) == 0:
        print("没有找到以unknown开头的图片文件")
        return 0, 0, 0
    
    print(f"找到 {len(image_files)} 个图片文件\n")
    
    # 正则表达式匹配 unknown_数字.扩展名
    pattern = re.compile(r'^unknown_(\d+)(\.[^.]+)$', re.IGNORECASE)
    
    for filename in image_files:
        match = pattern.match(filename)
        if not match:
            print(f"警告: 无法解析文件名格式: {filename}")
            error_count += 1
            continue
        
        number = match.group(1)
        ext = match.group(2)
        new_filename = f"almond_{number}{ext}"
        
        old_path = os.path.join(unknown_dir, filename)
        new_path = os.path.join(almond_dir, new_filename)
        
        # 如果目标文件已存在，先删除它
        if os.path.exists(new_path):
            try:
                os.remove(new_path)
                print(f"覆盖: 删除已存在的 {new_filename}")
            except Exception as e:
                print(f"警告: 无法删除已存在的文件 {new_filename}: {e}")
        
        # 移动并重命名图片文件
        try:
            shutil.move(old_path, new_path)
            moved_count += 1
            
            if moved_count % 50 == 0:
                print(f"已移动 {moved_count} 个文件...")
        except Exception as e:
            print(f"错误: 无法移动 {filename} -> {new_filename}: {e}")
            error_count += 1
            continue
        
        # 删除对应的JSON文件（在unknown文件夹中）
        json_filename = filename + '.json'
        json_path = os.path.join(unknown_dir, json_filename)
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
                json_deleted_count += 1
            except Exception as e:
                print(f"警告: 无法删除JSON文件 {json_filename}: {e}")
    
    print(f"\n完成 {split_name}:")
    print(f"  已移动: {moved_count} 个图片文件")
    print(f"  已删除: {json_deleted_count} 个JSON文件")
    print(f"  错误: {error_count} 个文件")
    
    return moved_count, json_deleted_count, error_count

def cleanup_empty_unknown_dirs(base_dir):
    """清理空的unknown文件夹"""
    for split in ['train', 'val', 'test']:
        unknown_dir = os.path.join(base_dir, split, 'unknown')
        if os.path.exists(unknown_dir):
            # 检查是否还有文件
            remaining_files = [f for f in os.listdir(unknown_dir) 
                             if os.path.isfile(os.path.join(unknown_dir, f))]
            if len(remaining_files) == 0:
                try:
                    os.rmdir(unknown_dir)
                    print(f"已删除空文件夹: {unknown_dir}")
                except Exception as e:
                    print(f"警告: 无法删除文件夹 {unknown_dir}: {e}")
            else:
                print(f"保留文件夹 {unknown_dir} (还有 {len(remaining_files)} 个文件)")

if __name__ == '__main__':
    base_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset'
    
    total_moved = 0
    total_json_deleted = 0
    total_errors = 0
    
    # 处理所有split（train, val, test）
    for split in ['train', 'val', 'test']:
        unknown_dir = os.path.join(base_dir, split, 'unknown')
        almond_dir = os.path.join(base_dir, split, 'almond')
        
        moved, json_deleted, errors = move_unknown_to_almond(unknown_dir, almond_dir, split)
        total_moved += moved
        total_json_deleted += json_deleted
        total_errors += errors
    
    print(f"\n{'='*60}")
    print(f"总计:")
    print(f"  已移动: {total_moved} 个图片文件")
    print(f"  已删除: {total_json_deleted} 个JSON文件")
    print(f"  错误: {total_errors} 个文件")
    print(f"{'='*60}\n")
    
    # 清理空的unknown文件夹
    print("清理空的unknown文件夹...")
    cleanup_empty_unknown_dirs(base_dir)
