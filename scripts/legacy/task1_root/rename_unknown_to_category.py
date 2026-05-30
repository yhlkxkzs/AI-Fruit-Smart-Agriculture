#!/usr/bin/env python3
"""
将指定目录中所有unknown_开头的图片文件重命名为对应的类别格式
例如：unknown_000093.png -> almond_000093.png
同时删除对应的JSON文件
"""
import os
import re
from pathlib import Path

def rename_unknown_files(target_dir, category_name=None):
    """
    将目录中所有unknown_开头的图片文件重命名为类别格式
    
    Args:
        target_dir: 目标目录路径
        category_name: 类别名称（如 'almond'），如果不提供则从目录名推断
    """
    if not os.path.exists(target_dir):
        print(f"错误: 目录不存在: {target_dir}")
        return
    
    # 如果没有提供类别名，从目录名推断
    if category_name is None:
        category_name = os.path.basename(os.path.abspath(target_dir))
        print(f"从目录名推断类别: {category_name}")
    
    print(f"\n处理目录: {target_dir}")
    print(f"目标类别: {category_name}\n")
    
    # 统计
    renamed_images = 0
    deleted_json = 0
    skipped = 0
    errors = 0
    
    # 获取所有以unknown开头的文件
    all_files = sorted(os.listdir(target_dir))
    unknown_files = [f for f in all_files if f.lower().startswith('unknown_')]
    
    if len(unknown_files) == 0:
        print("没有找到以unknown开头的文件")
        return
    
    print(f"找到 {len(unknown_files)} 个以unknown开头的文件\n")
    
    # 正则表达式匹配 unknown_数字.扩展名
    pattern = re.compile(r'^unknown_(\d+)(\.[^.]+)$', re.IGNORECASE)
    
    for filename in unknown_files:
        file_path = os.path.join(target_dir, filename)
        
        # 判断是图片文件还是JSON文件
        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG'))
        is_json = filename.lower().endswith('.json')
        
        if is_image:
            # 处理图片文件：重命名为类别格式
            match = pattern.match(filename)
            if not match:
                print(f"警告: 无法解析文件名格式: {filename}")
                errors += 1
                continue
            
            number = match.group(1)
            ext = match.group(2)
            new_filename = f"{category_name}_{number}{ext}"
            
            new_path = os.path.join(target_dir, new_filename)
            
            # 如果目标文件已存在，先检查是否相同
            if os.path.exists(new_path):
                # 检查文件大小是否相同
                old_size = os.path.getsize(file_path)
                new_size = os.path.getsize(new_path)
                if old_size == new_size:
                    print(f"跳过: {filename} -> {new_filename} (目标文件已存在且大小相同)")
                    # 删除旧的unknown文件
                    try:
                        os.remove(file_path)
                        skipped += 1
                    except Exception as e:
                        print(f"警告: 无法删除 {filename}: {e}")
                    continue
                else:
                    # 大小不同，覆盖
                    try:
                        os.remove(new_path)
                        print(f"覆盖: 删除已存在的 {new_filename} (大小不同)")
                    except Exception as e:
                        print(f"警告: 无法删除已存在的文件 {new_filename}: {e}")
            
            # 重命名图片文件
            try:
                os.rename(file_path, new_path)
                renamed_images += 1
                if renamed_images % 10 == 0:
                    print(f"已重命名 {renamed_images} 个图片文件...")
            except Exception as e:
                print(f"错误: 无法重命名 {filename} -> {new_filename}: {e}")
                errors += 1
                continue
        
        elif is_json:
            # 处理JSON文件：直接删除
            try:
                os.remove(file_path)
                deleted_json += 1
            except Exception as e:
                print(f"错误: 无法删除JSON文件 {filename}: {e}")
                errors += 1
    
    print(f"\n完成！")
    print(f"已重命名图片: {renamed_images} 个")
    print(f"已删除JSON: {deleted_json} 个")
    print(f"已跳过: {skipped} 个")
    print(f"错误: {errors} 个")
    
    return renamed_images, deleted_json, skipped, errors

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python rename_unknown_to_category.py <目录路径> [类别名称]")
        print("示例: python rename_unknown_to_category.py /path/to/almond almond")
        print("      python rename_unknown_to_category.py /path/to/almond")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    category_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    rename_unknown_files(target_dir, category_name)
