#!/usr/bin/env python3
"""
重命名almond文件夹中所有unknown开头的图片文件，并删除对应的JSON文件
"""
import os
import re
from pathlib import Path

def rename_unknown_files_in_almond(almond_dir):
    """
    重命名almond文件夹中所有unknown开头的图片文件为almond格式，并删除JSON文件
    
    Args:
        almond_dir: almond文件夹路径
    """
    if not os.path.exists(almond_dir):
        print(f"错误: 目录不存在: {almond_dir}")
        return
    
    # 统计
    renamed_count = 0
    json_deleted_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"开始处理: {almond_dir}\n")
    
    # 获取所有以unknown开头的图片文件
    all_files = sorted(os.listdir(almond_dir))
    unknown_images = [f for f in all_files if f.lower().startswith('unknown_') and 
                      f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    print(f"找到 {len(unknown_images)} 个以unknown开头的图片文件\n")
    
    # 正则表达式匹配 unknown_数字.扩展名
    pattern = re.compile(r'^unknown_(\d+)(\.[^.]+)$', re.IGNORECASE)
    
    for filename in unknown_images:
        match = pattern.match(filename)
        if not match:
            print(f"警告: 无法解析文件名格式: {filename}")
            error_count += 1
            continue
        
        number = match.group(1)
        ext = match.group(2)
        new_filename = f"almond_{number}{ext}"
        
        old_path = os.path.join(almond_dir, filename)
        new_path = os.path.join(almond_dir, new_filename)
        
        # 检查新文件名是否已存在
        if os.path.exists(new_path):
            print(f"跳过: {filename} -> {new_filename} (目标文件已存在)")
            skipped_count += 1
            
            # 即使目标文件存在，也删除JSON文件
            json_filename = filename + '.json'
            json_path = os.path.join(almond_dir, json_filename)
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                    json_deleted_count += 1
                except Exception as e:
                    print(f"  警告: 无法删除JSON {json_filename}: {e}")
            
            # 删除旧图片文件（因为目标已存在）
            try:
                os.remove(old_path)
            except Exception as e:
                print(f"  警告: 无法删除旧文件 {filename}: {e}")
            continue
        
        # 重命名图片文件
        try:
            os.rename(old_path, new_path)
            renamed_count += 1
            
            if renamed_count % 50 == 0:
                print(f"已重命名 {renamed_count} 个文件...")
        except Exception as e:
            print(f"错误: 无法重命名 {filename} -> {new_filename}: {e}")
            error_count += 1
            continue
        
        # 删除对应的JSON文件
        json_filename = filename + '.json'
        json_path = os.path.join(almond_dir, json_filename)
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
                json_deleted_count += 1
            except Exception as e:
                print(f"警告: 无法删除JSON文件 {json_filename}: {e}")
    
    print(f"\n完成！")
    print(f"已重命名: {renamed_count} 个图片文件")
    print(f"已删除: {json_deleted_count} 个JSON文件")
    print(f"跳过: {skipped_count} 个文件（目标已存在）")
    print(f"错误: {error_count} 个文件")
    
    return renamed_count, json_deleted_count

if __name__ == '__main__':
    almond_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/almond'
    rename_unknown_files_in_almond(almond_dir)
