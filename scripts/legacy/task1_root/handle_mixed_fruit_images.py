#!/usr/bin/env python3
"""
处理来自DeepFruits数据集的多水果组合图片
提供多种处理方案：
1. 删除这些组合图片（如果不需要）
2. 将这些图片移到mixed_fruits类别
3. 在JSON标注中添加标记，标记为multi_fruit
4. 创建单独的apricot_in_combination类别
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

def identify_deepfruits_images(base_dir):
    """识别来自DeepFruits数据集的图片"""
    deepfruits_images = defaultdict(list)
    
    # DeepFruits图片的特征：
    # 1. 尺寸通常是64x64像素
    # 2. 来自deepfruits数据集
    
    for split in ['train', 'val', 'test']:
        apricot_dir = base_dir / split / 'apricot'
        if not apricot_dir.exists():
            continue
        
        # 检查所有图片
        for img_path in apricot_dir.glob('*.jpg'):
            if img_path.stem.startswith('apricot_'):
                # 检查图片尺寸（DeepFruits通常是64x64）
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    if img.size == (64, 64):
                        deepfruits_images[split].append(img_path)
                except:
                    pass
    
    return deepfruits_images

def option1_delete_images(deepfruits_images, base_dir):
    """方案1：删除这些组合图片"""
    print("\n方案1：删除DeepFruits组合图片")
    print("="*60)
    
    total_deleted = 0
    for split, images in deepfruits_images.items():
        print(f"\n处理 {split.upper()} split...")
        deleted = 0
        
        for img_path in images:
            try:
                # 删除图片
                img_path.unlink()
                deleted += 1
                
                # 删除对应的JSON和CSV文件
                json_path = base_dir / split / 'apricot' / 'json' / f"{img_path.stem}.json"
                csv_path = base_dir / split / 'apricot' / 'csv' / f"{img_path.stem}.csv"
                
                if json_path.exists():
                    json_path.unlink()
                if csv_path.exists():
                    csv_path.unlink()
                    
            except Exception as e:
                print(f"  错误删除 {img_path.name}: {e}")
        
        total_deleted += deleted
        print(f"  已删除 {deleted} 张图片及其标注文件")
    
    print(f"\n总计删除 {total_deleted} 张图片")
    return total_deleted

def option2_move_to_mixed_fruits(deepfruits_images, base_dir):
    """方案2：将这些图片移到mixed_fruits类别"""
    print("\n方案2：将DeepFruits组合图片移到mixed_fruits类别")
    print("="*60)
    
    total_moved = 0
    for split, images in deepfruits_images.items():
        print(f"\n处理 {split.upper()} split...")
        
        # 创建mixed_fruits目录
        mixed_dir = base_dir / split / 'mixed_fruits'
        mixed_dir.mkdir(exist_ok=True)
        (mixed_dir / 'json').mkdir(exist_ok=True)
        (mixed_dir / 'csv').mkdir(exist_ok=True)
        
        moved = 0
        index = 0
        
        for img_path in images:
            try:
                # 移动图片
                new_name = f"mixed_fruits_{index:06d}{img_path.suffix}"
                new_img_path = mixed_dir / new_name
                shutil.move(str(img_path), str(new_img_path))
                
                # 移动JSON文件
                json_path = base_dir / split / 'apricot' / 'json' / f"{img_path.stem}.json"
                if json_path.exists():
                    new_json_path = mixed_dir / 'json' / f"{Path(new_name).stem}.json"
                    # 更新JSON内容
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 更新文件名和类别
                    json_data['images'][0]['file_name'] = new_name
                    json_data['categories'][0]['name'] = 'mixed_fruits'
                    json_data['annotations'][0]['category_id'] = 2000000000 + 999  # 使用特殊ID
                    
                    with open(new_json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    json_path.unlink()
                
                # 移动CSV文件
                csv_path = base_dir / split / 'apricot' / 'csv' / f"{img_path.stem}.csv"
                if csv_path.exists():
                    new_csv_path = mixed_dir / 'csv' / f"{Path(new_name).stem}.csv"
                    shutil.move(str(csv_path), str(new_csv_path))
                
                moved += 1
                index += 1
                
            except Exception as e:
                print(f"  错误移动 {img_path.name}: {e}")
        
        total_moved += moved
        print(f"  已移动 {moved} 张图片到mixed_fruits类别")
    
    print(f"\n总计移动 {total_moved} 张图片")
    return total_moved

def option3_add_multi_fruit_flag(deepfruits_images, base_dir):
    """方案3：在JSON标注中添加multi_fruit标记"""
    print("\n方案3：在JSON标注中添加multi_fruit标记")
    print("="*60)
    
    total_updated = 0
    for split, images in deepfruits_images.items():
        print(f"\n处理 {split.upper()} split...")
        
        updated = 0
        for img_path in images:
            json_path = base_dir / split / 'apricot' / 'json' / f"{img_path.stem}.json"
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 添加标记
                    json_data['images'][0]['is_multi_fruit'] = True
                    json_data['images'][0]['source_dataset'] = 'deepfruits'
                    json_data['images'][0]['image_type'] = 'combination'
                    
                    json_data['annotations'][0]['is_multi_fruit'] = True
                    json_data['categories'][0]['is_multi_fruit'] = True
                    json_data['categories'][0]['note'] = 'This image contains multiple fruits, not suitable for single fruit classification'
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    updated += 1
                except Exception as e:
                    print(f"  错误更新 {json_path.name}: {e}")
        
        total_updated += updated
        print(f"  已更新 {updated} 个JSON文件")
    
    print(f"\n总计更新 {total_updated} 个JSON文件")
    print("\n注意：训练时可以过滤掉is_multi_fruit=True的图片")
    return total_updated

def option4_create_separate_category(deepfruits_images, base_dir):
    """方案4：创建单独的apricot_in_combination类别"""
    print("\n方案4：创建单独的apricot_in_combination类别")
    print("="*60)
    
    total_moved = 0
    for split, images in deepfruits_images.items():
        print(f"\n处理 {split.upper()} split...")
        
        # 创建apricot_in_combination目录
        new_category_dir = base_dir / split / 'apricot_in_combination'
        new_category_dir.mkdir(exist_ok=True)
        (new_category_dir / 'json').mkdir(exist_ok=True)
        (new_category_dir / 'csv').mkdir(exist_ok=True)
        
        moved = 0
        index = 0
        
        for img_path in images:
            try:
                # 移动图片
                new_name = f"apricot_in_combination_{index:06d}{img_path.suffix}"
                new_img_path = new_category_dir / new_name
                shutil.move(str(img_path), str(new_img_path))
                
                # 移动并更新JSON文件
                json_path = base_dir / split / 'apricot' / 'json' / f"{img_path.stem}.json"
                if json_path.exists():
                    new_json_path = new_category_dir / 'json' / f"{Path(new_name).stem}.json"
                    
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 更新文件名和类别
                    json_data['images'][0]['file_name'] = new_name
                    json_data['categories'][0]['name'] = 'apricot_in_combination'
                    json_data['categories'][0]['note'] = 'Apricot in multi-fruit combination scene'
                    json_data['images'][0]['is_multi_fruit'] = True
                    json_data['images'][0]['source_dataset'] = 'deepfruits'
                    
                    with open(new_json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    json_path.unlink()
                
                # 移动CSV文件
                csv_path = base_dir / split / 'apricot' / 'csv' / f"{img_path.stem}.csv"
                if csv_path.exists():
                    new_csv_path = new_category_dir / 'csv' / f"{Path(new_name).stem}.csv"
                    shutil.move(str(csv_path), str(new_csv_path))
                
                moved += 1
                index += 1
                
            except Exception as e:
                print(f"  错误移动 {img_path.name}: {e}")
        
        total_moved += moved
        print(f"  已移动 {moved} 张图片到apricot_in_combination类别")
    
    print(f"\n总计移动 {total_moved} 张图片")
    return total_moved

def main():
    """主函数"""
    import sys
    
    base_dir = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    
    print("="*60)
    print("处理DeepFruits多水果组合图片")
    print("="*60)
    
    # 识别DeepFruits图片
    print("\n正在识别DeepFruits组合图片...")
    deepfruits_images = identify_deepfruits_images(base_dir)
    
    total_images = sum(len(images) for images in deepfruits_images.values())
    print(f"\n识别到 {total_images} 张DeepFruits组合图片:")
    for split, images in deepfruits_images.items():
        print(f"  {split.upper()}: {len(images)} 张")
    
    if total_images == 0:
        print("\n未找到DeepFruits组合图片")
        return
    
    # 显示处理方案
    print("\n" + "="*60)
    print("可用的处理方案：")
    print("="*60)
    print("1. 删除这些组合图片（不推荐，会丢失所有apricot数据）")
    print("2. 将这些图片移到mixed_fruits类别（推荐，保留数据但标记为混合）")
    print("3. 在JSON标注中添加multi_fruit标记（推荐，保留在原位置但添加标记）")
    print("4. 创建单独的apricot_in_combination类别（推荐，明确区分单一和组合）")
    print("\n建议：使用方案3或方案4，这样可以保留数据但明确标记不适合单一水果识别")
    
    if len(sys.argv) > 1:
        option = sys.argv[1]
    else:
        option = input("\n请选择处理方案 (1/2/3/4，默认3): ").strip() or "3"
    
    if option == "1":
        confirm = input(f"\n⚠️  警告：这将删除 {total_images} 张图片！确认吗？(yes/no): ")
        if confirm.lower() == 'yes':
            option1_delete_images(deepfruits_images, base_dir)
        else:
            print("已取消")
    elif option == "2":
        option2_move_to_mixed_fruits(deepfruits_images, base_dir)
    elif option == "3":
        option3_add_multi_fruit_flag(deepfruits_images, base_dir)
    elif option == "4":
        option4_create_separate_category(deepfruits_images, base_dir)
    else:
        print(f"无效的选项: {option}")
        return
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()
