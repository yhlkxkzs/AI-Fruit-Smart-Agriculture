#!/usr/bin/env python3
"""
健壮的重命名脚本：将unknown_开头的图片文件重命名为类别格式
支持多种文件格式，提供详细的日志输出
"""
import os
import re
from pathlib import Path

def find_unknown_files(target_dir):
    """查找所有unknown_开头的文件"""
    if not os.path.exists(target_dir):
        return []
    
    all_files = os.listdir(target_dir)
    # 支持大小写不敏感的匹配
    unknown_files = [f for f in all_files if f.lower().startswith('unknown_')]
    return sorted(unknown_files)

def check_file_status(target_dir, filename):
    """检查文件状态"""
    file_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(file_path):
        return "不存在"
    
    stat = os.stat(file_path)
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime
    }

def rename_unknown_files(target_dir, category_name=None, dry_run=False):
    """
    将目录中所有unknown_开头的图片文件重命名为类别格式
    
    Args:
        target_dir: 目标目录路径
        category_name: 类别名称（如 'almond'），如果不提供则从目录名推断
        dry_run: 如果为True，只显示将要执行的操作，不实际执行
    """
    if not os.path.exists(target_dir):
        print(f"错误: 目录不存在: {target_dir}")
        return
    
    # 如果没有提供类别名，从目录名推断
    if category_name is None:
        category_name = os.path.basename(os.path.abspath(target_dir))
    
    print(f"\n{'='*60}")
    print(f"处理目录: {target_dir}")
    print(f"目标类别: {category_name}")
    if dry_run:
        print(f"模式: 预览模式（不会实际修改文件）")
    print(f"{'='*60}\n")
    
    # 查找所有unknown_文件
    unknown_files = find_unknown_files(target_dir)
    
    if len(unknown_files) == 0:
        print("✓ 没有找到以unknown开头的文件")
        print("\n检查是否有已重命名的文件...")
        # 检查是否有对应编号的almond文件
        all_files = os.listdir(target_dir)
        pattern = re.compile(rf'^{category_name}_(\d+)(\.[^.]+)$', re.IGNORECASE)
        numbered_files = []
        for f in all_files:
            match = pattern.match(f)
            if match:
                numbered_files.append(int(match.group(1)))
        
        if numbered_files:
            print(f"找到 {len(numbered_files)} 个已重命名的{category_name}文件")
            print(f"编号范围: {min(numbered_files)} - {max(numbered_files)}")
        return
    
    print(f"找到 {len(unknown_files)} 个以unknown开头的文件:\n")
    
    # 统计
    renamed_images = 0
    deleted_json = 0
    skipped = 0
    errors = 0
    
    # 正则表达式匹配 unknown_数字.扩展名
    pattern = re.compile(r'^unknown_(\d+)(\.[^.]+)$', re.IGNORECASE)
    
    for filename in unknown_files:
        file_path = os.path.join(target_dir, filename)
        
        # 检查文件状态
        status = check_file_status(target_dir, filename)
        if status == "不存在":
            print(f"⚠ 跳过: {filename} (文件不存在)")
            errors += 1
            continue
        
        # 判断是图片文件还是JSON文件
        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        is_json = filename.lower().endswith('.json')
        
        if is_image:
            # 处理图片文件：重命名为类别格式
            match = pattern.match(filename)
            if not match:
                print(f"⚠ 警告: 无法解析文件名格式: {filename}")
                errors += 1
                continue
            
            number = match.group(1)
            ext = match.group(2)
            new_filename = f"{category_name}_{number}{ext}"
            new_path = os.path.join(target_dir, new_filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(new_path):
                old_size = status["size"]
                new_size = os.path.getsize(new_path)
                
                if old_size == new_size:
                    print(f"⊘ 跳过: {filename} -> {new_filename} (目标文件已存在且大小相同)")
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            skipped += 1
                        except Exception as e:
                            print(f"  错误: 无法删除旧文件: {e}")
                            errors += 1
                    else:
                        skipped += 1
                    continue
                else:
                    print(f"⚠ 覆盖: {filename} -> {new_filename} (目标文件存在但大小不同: {old_size} vs {new_size})")
                    if not dry_run:
                        try:
                            os.remove(new_path)
                        except Exception as e:
                            print(f"  错误: 无法删除已存在的文件: {e}")
                            errors += 1
                            continue
            
            # 执行重命名
            if dry_run:
                print(f"→ 将重命名: {filename} -> {new_filename}")
                renamed_images += 1
            else:
                try:
                    os.rename(file_path, new_path)
                    renamed_images += 1
                    if renamed_images % 10 == 0:
                        print(f"  进度: 已重命名 {renamed_images} 个文件...")
                except Exception as e:
                    print(f"✗ 错误: 无法重命名 {filename} -> {new_filename}: {e}")
                    errors += 1
                    continue
        
        elif is_json:
            # 处理JSON文件：直接删除
            if dry_run:
                print(f"→ 将删除: {filename}")
                deleted_json += 1
            else:
                try:
                    os.remove(file_path)
                    deleted_json += 1
                except Exception as e:
                    print(f"✗ 错误: 无法删除JSON文件 {filename}: {e}")
                    errors += 1
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"已重命名图片: {renamed_images} 个")
    print(f"已删除JSON: {deleted_json} 个")
    print(f"已跳过: {skipped} 个")
    print(f"错误: {errors} 个")
    print(f"{'='*60}\n")
    
    return renamed_images, deleted_json, skipped, errors

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python rename_unknown_files_robust.py <目录路径> [类别名称] [--dry-run]")
        print("示例: python rename_unknown_files_robust.py /path/to/almond almond")
        print("      python rename_unknown_files_robust.py /path/to/almond almond --dry-run")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    category_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    dry_run = '--dry-run' in sys.argv
    
    rename_unknown_files(target_dir, category_name, dry_run)
