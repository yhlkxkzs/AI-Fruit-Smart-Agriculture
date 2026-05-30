#!/usr/bin/env python3
"""
修复所有同名但内容不同的图片，并重新生成所有标注文件

1. 检查所有同名但不同格式的图片（包括大小写不同）
2. 使用图像哈希比较内容是否相同
3. 如果内容不同，重命名为不同的编号
4. 删除旧的JSON/CSV文件
5. 重新生成所有标注文件
"""
import os
import json
import shutil
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash
from tqdm import tqdm
import sys

def calculate_image_hash(img_path):
    """计算图片的感知哈希值"""
    try:
        with Image.open(img_path) as img:
            # 使用感知哈希，对相同内容但不同格式的图片也能识别
            return str(imagehash.phash(img)), None
    except Exception as e:
        return None, str(e)

def extract_number_from_name(name):
    """从文件名中提取数字部分"""
    match = re.search(r'(\d+)$', name)
    if match:
        return int(match.group(1))
    return None

def find_next_available_number(category_dir, prefix, used_numbers, start_num=0):
    """找到下一个可用的编号"""
    num = start_num
    max_attempts = 100000
    attempts = 0
    while num in used_numbers and attempts < max_attempts:
        num += 1
        attempts += 1
    if attempts >= max_attempts:
        raise ValueError(f"无法找到可用编号，已尝试 {max_attempts} 次")
    used_numbers.add(num)
    return num

def process_category(category_dir, category_name, split_name):
    """处理单个类别，修复同名但内容不同的图片"""
    print(f"\n处理 {split_name}/{category_name}...")
    
    # 收集所有图片文件（包括.tif等）
    image_extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']
    all_images = []
    for ext in image_extensions:
        all_images.extend(category_dir.glob(f'*{ext}'))
    
    if not all_images:
        return 0, 0, []
    
    # 按文件名stem分组（不区分大小写）
    files_by_stem = defaultdict(list)
    for img_file in all_images:
        stem = img_file.stem  # 保持原始大小写
        files_by_stem[stem.lower()].append(img_file)  # 但用lower作为key
    
    # 找出需要处理的组（多个格式的）
    groups_to_process = {}
    for stem_lower, files in files_by_stem.items():
        if len(files) > 1:
            # 检查是否有不同的扩展名或大小写不同的文件名
            exts = set(f.suffix.lower() for f in files)
            names = set(f.name for f in files)
            if len(exts) > 1 or len(names) > 1:
                groups_to_process[stem_lower] = files
    
    if not groups_to_process:
        return 0, 0, []
    
    print(f"  发现 {len(groups_to_process)} 个同名不同格式的图片组")
    
    # 获取已使用的编号
    used_numbers = set()
    for img_file in all_images:
        num = extract_number_from_name(img_file.stem)
        if num is not None:
            used_numbers.add(num)
    
    renamed_count = 0
    deleted_count = 0
    rename_log = []
    
    # 处理每个组
    for stem_lower, files in tqdm(groups_to_process.items(), desc=f"  处理 {category_name}", leave=False):
        # 计算每个文件的哈希值
        file_hashes = {}
        for img_file in files:
            img_hash, error = calculate_image_hash(img_file)
            if img_hash:
                file_hashes[img_file] = img_hash
            elif error:
                rename_log.append(f"跳过 {img_file.name}: 无法读取图片 ({error})")
        
        if not file_hashes:
            continue
        
        # 按哈希值分组
        hash_groups = defaultdict(list)
        for img_file, img_hash in file_hashes.items():
            hash_groups[img_hash].append(img_file)
        
        # 如果所有文件哈希相同，说明内容相同，保留一个即可
        if len(hash_groups) == 1:
            # 保留第一个，删除其他的
            keep_file = files[0]
            for img_file in files[1:]:
                # 删除图片和对应的JSON
                json_file = category_dir / 'json' / f"{img_file.stem}.json"
                if json_file.exists():
                    json_file.unlink()
                img_file.unlink()
                deleted_count += 1
                rename_log.append(f"删除重复: {img_file.name} (与 {keep_file.name} 内容相同)")
        else:
            # 内容不同，需要重命名
            # 第一个文件保持原名
            keep_file = files[0]
            keep_hash = file_hashes.get(keep_file)
            
            # 提取前缀和编号
            match = re.search(r'^(.+?)(\d+)$', keep_file.stem)
            if match:
                prefix = match.group(1)
                base_num = int(match.group(2))
            else:
                prefix = keep_file.stem
                base_num = 0
            
            # 其他不同内容的文件需要重命名
            for img_file in files[1:]:
                img_hash = file_hashes.get(img_file)
                if img_hash and img_hash != keep_hash:
                    # 生成新名称
                    try:
                        new_num = find_next_available_number(category_dir, prefix, used_numbers, base_num + 1)
                        new_stem = f"{prefix}{new_num:05d}"
                        new_name = f"{new_stem}{img_file.suffix}"
                        new_path = category_dir / new_name
                        
                        # 如果目标文件已存在，跳过
                        if new_path.exists():
                            rename_log.append(f"跳过 {img_file.name}: 目标文件已存在 {new_name}")
                            continue
                        
                        # 重命名图片文件
                        old_json = category_dir / 'json' / f"{img_file.stem}.json"
                        new_json = category_dir / 'json' / f"{new_stem}.json"
                        
                        try:
                            img_file.rename(new_path)
                            renamed_count += 1
                            
                            # 重命名JSON文件（如果存在）
                            if old_json.exists():
                                if new_json.exists():
                                    old_json.unlink()
                                else:
                                    old_json.rename(new_json)
                                    # 更新JSON内容中的文件名
                                    try:
                                        with open(new_json, 'r+', encoding='utf-8') as f:
                                            data = json.load(f)
                                            if 'images' in data and len(data['images']) > 0:
                                                data['images'][0]['file_name'] = new_name
                                                data['images'][0]['original_filename'] = new_name
                                            f.seek(0)
                                            f.truncate()
                                            json.dump(data, f, indent=2, ensure_ascii=False)
                                    except Exception as e:
                                        rename_log.append(f"警告: 无法更新JSON {new_json}: {e}")
                            
                            rename_log.append(f"重命名: {img_file.name} -> {new_name} (内容不同)")
                        except Exception as e:
                            rename_log.append(f"错误: 无法重命名 {img_file.name}: {e}")
                    except ValueError as e:
                        rename_log.append(f"错误: {img_file.name}: {e}")
    
    return renamed_count, deleted_count, rename_log

def regenerate_all_annotations():
    """重新生成所有标注文件"""
    print("\n" + "="*60)
    print("重新生成所有标注文件...")
    print("="*60)
    
    # 导入生成标注的脚本
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    try:
        from generate_annotations import process_dataset
        process_dataset('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset', output_structure=False)
        return True
    except ImportError as e:
        print(f"错误: 无法导入 generate_annotations 模块: {e}")
        return False
    except Exception as e:
        print(f"错误: 生成标注时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    dataset_dir = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    splits = ['train', 'val', 'test']
    
    print("="*60)
    print("修复所有同名但内容不同的图片")
    print("="*60)
    
    total_renamed = 0
    total_deleted = 0
    all_logs = []
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\n处理 {split.upper()} split:")
        print("-"*60)
        
        categories = sorted([d for d in split_dir.iterdir() 
                          if d.is_dir() and d.name not in ['json', 'csv', 'sets']])
        
        for category_dir in categories:
            category_name = category_dir.name
            renamed, deleted, logs = process_category(category_dir, category_name, split)
            total_renamed += renamed
            total_deleted += deleted
            all_logs.extend(logs)
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"总计重命名: {total_renamed} 个文件")
    print(f"总计删除重复: {total_deleted} 个文件")
    
    if all_logs:
        # 保存日志到文件
        log_file = Path('/tmp/fix_duplicates_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            for log in all_logs:
                f.write(log + '\n')
        print(f"\n详细日志已保存到: {log_file}")
        
        print(f"\n操作日志（前50个）:")
        for log in all_logs[:50]:
            print(f"  {log}")
        if len(all_logs) > 50:
            print(f"  ... 还有 {len(all_logs) - 50} 个操作，请查看日志文件")
    
    # 重新生成所有标注文件
    if total_renamed > 0 or total_deleted > 0:
        print("\n" + "="*60)
        print("开始重新生成标注文件...")
        print("="*60)
        success = regenerate_all_annotations()
        if success:
            print("\n所有标注文件已重新生成！")
        else:
            print("\n警告: 标注文件生成可能未完成，请检查错误信息")
    else:
        print("\n无需重新生成标注文件（没有文件被修改）")

if __name__ == '__main__':
    # 检查依赖
    try:
        import imagehash
    except ImportError:
        print("错误: 需要安装 imagehash 库")
        print("请运行: pip install imagehash")
        sys.exit(1)
    
    main()
