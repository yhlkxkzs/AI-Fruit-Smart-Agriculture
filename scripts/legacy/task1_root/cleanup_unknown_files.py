#!/usr/bin/env python3
"""
清理almond文件夹中所有以unknown开头的文件（包括图片和JSON）
对于图片文件，重命名为almond格式；对于JSON文件，直接删除
"""
import os
import re
from pathlib import Path

def cleanup_unknown_files(almond_dir):
    """
    清理almond文件夹中所有以unknown开头的文件
    
    Args:
        almond_dir: almond文件夹路径
    """
    if not os.path.exists(almond_dir):
        print(f"错误: 目录不存在: {almond_dir}")
        return
    
    # 统计
    renamed_images = 0
    deleted_json = 0
    deleted_other = 0
    error_count = 0
    
    print(f"开始处理: {almond_dir}\n")
    
    # 获取所有以unknown开头的文件
    all_files = sorted(os.listdir(almond_dir))
    unknown_files = [f for f in all_files if f.lower().startswith('unknown_')]
    
    if len(unknown_files) == 0:
        print("没有找到以unknown开头的文件")
        return
    
    print(f"找到 {len(unknown_files)} 个以unknown开头的文件\n")
    
    # 正则表达式匹配 unknown_数字.扩展名
    pattern = re.compile(r'^unknown_(\d+)(\.[^.]+)$', re.IGNORECASE)
    
    for filename in unknown_files:
        # 判断是图片文件还是JSON文件
        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        is_json = filename.lower().endswith('.json')
        
        if is_image:
            # 处理图片文件：重命名为almond格式
            match = pattern.match(filename)
            if not match:
                print(f"警告: 无法解析图片文件名格式: {filename}")
                error_count += 1
                continue
            
            number = match.group(1)
            ext = match.group(2)
            new_filename = f"almond_{number}{ext}"
            
            old_path = os.path.join(almond_dir, filename)
            new_path = os.path.join(almond_dir, new_filename)
            
            # 如果目标文件已存在，先删除它
            if os.path.exists(new_path):
                try:
                    os.remove(new_path)
                    print(f"覆盖: 删除已存在的 {new_filename}")
                except Exception as e:
                    print(f"警告: 无法删除已存在的文件 {new_filename}: {e}")
            
            # 重命名图片文件
            try:
                os.rename(old_path, new_path)
                renamed_images += 1
                
                if renamed_images % 50 == 0:
                    print(f"已重命名 {renamed_images} 个图片文件...")
            except Exception as e:
                print(f"错误: 无法重命名 {filename} -> {new_filename}: {e}")
                error_count += 1
                continue
        
        elif is_json:
            # 处理JSON文件：直接删除
            json_path = os.path.join(almond_dir, filename)
            try:
                os.remove(json_path)
                deleted_json += 1
            except Exception as e:
                print(f"错误: 无法删除JSON文件 {filename}: {e}")
                error_count += 1
        
        else:
            # 其他类型的unknown_文件：直接删除
            other_path = os.path.join(almond_dir, filename)
            try:
                os.remove(other_path)
                deleted_other += 1
                print(f"删除其他文件: {filename}")
            except Exception as e:
                print(f"错误: 无法删除文件 {filename}: {e}")
                error_count += 1
    
    print(f"\n完成！")
    print(f"已重命名图片: {renamed_images} 个")
    print(f"已删除JSON: {deleted_json} 个")
    print(f"已删除其他: {deleted_other} 个")
    print(f"错误: {error_count} 个")
    
    return renamed_images, deleted_json, deleted_other

if __name__ == '__main__':
    base_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset'
    
    # 处理所有split（train, val, test）
    for split in ['train', 'val', 'test']:
        almond_dir = os.path.join(base_dir, split, 'almond')
        if os.path.exists(almond_dir):
            print(f"\n{'='*60}")
            print(f"处理 {split} split")
            print(f"{'='*60}\n")
            cleanup_unknown_files(almond_dir)
        else:
            print(f"跳过: {almond_dir} 不存在")
