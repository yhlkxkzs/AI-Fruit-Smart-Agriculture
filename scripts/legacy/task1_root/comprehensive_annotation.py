#!/usr/bin/env python3
"""
全面标注脚本：为所有图片添加详细标注
1. 内容类型（花、叶、果实）- 特别是apple类别
2. 图片风格（真实、油画、卡通、素描等）
3. 单/多水果检测
4. 更新所有JSON文件
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

def detect_image_style(img_path):
    """
    检测图片风格：真实照片、油画、卡通、素描等
    使用图像特征分析：
    - 边缘检测（油画通常有更多笔触边缘）
    - 颜色分布（卡通通常颜色更饱和、更均匀）
    - 纹理分析（真实照片有更多细节纹理）
    """
    try:
        img = Image.open(img_path)
        
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # 转换为OpenCV格式（BGR）
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. 边缘检测 - 计算边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 2. 颜色分析
        # 计算颜色饱和度
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # 计算颜色数量（唯一颜色数量）
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (img_array.shape[0] * img_array.shape[1])
        
        # 3. 纹理分析 - 使用拉普拉斯算子计算纹理复杂度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # 4. 亮度分布
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 5. 检查是否为灰度图（可能是素描）
        if img.mode == 'L' or (saturation < 0.1 and brightness_std < 30):
            return 'sketch'  # 素描
        
        # 启发式规则判断风格
        # 油画特征：边缘密度高，纹理方差中等，颜色饱和度较高
        if edge_density > 0.15 and 0.3 < saturation < 0.7 and 100 < texture_variance < 1000:
            return 'painting'  # 油画
        
        # 卡通特征：颜色饱和度很高，颜色多样性低，边缘清晰
        if saturation > 0.7 and color_diversity < 0.1 and edge_density > 0.1:
            return 'cartoon'  # 卡通
        
        # 真实照片特征：纹理方差高，颜色多样性高，边缘密度中等
        if texture_variance > 500 and color_diversity > 0.15:
            return 'real'  # 真实照片
        
        # 默认判断
        if texture_variance > 300:
            return 'real'
        elif saturation > 0.6:
            return 'cartoon'
        else:
            return 'real'  # 默认真实照片
            
    except Exception as e:
        print(f"  错误检测风格 {img_path.name}: {e}")
        return 'real'  # 默认真实照片

def detect_content_type(img_path, category_name):
    """
    检测图片内容类型：花、叶、果实
    特别针对apple类别，需要区分花、叶、果实
    """
    if category_name.lower() != 'apple':
        # 对于非apple类别，默认为果实
        return 'fruit'
    
    try:
        # 检查文件名
        name_lower = img_path.stem.lower()
        if any(keyword in name_lower for keyword in ['flower', 'blossom', 'bloom', '花']):
            return 'flower'
        if any(keyword in name_lower for keyword in ['leaf', 'leaves', '叶', '叶子']):
            return 'leaf'
        
        img = Image.open(img_path)
        
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # 计算平均RGB值
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # 计算平均亮度
        avg_brightness = (avg_r + avg_g + avg_b) / 3
        
        # 计算颜色饱和度
        max_channel = max(avg_r, avg_g, avg_b)
        min_channel = min(avg_r, avg_g, avg_b)
        saturation = (max_channel - min_channel) / max_channel if max_channel > 0 else 0
        
        # 判断规则：
        # 1. 花：通常较亮（>180），白色/粉色（红色和绿色通道都较高）
        # 2. 叶：绿色通道明显高于其他通道
        # 3. 果实：红色或绿色通道较高，但整体较暗
        
        # 检查是否为叶子（绿色为主）
        if avg_g > avg_r + 40 and avg_g > avg_b + 40 and avg_brightness < 150:
            return 'leaf'  # 叶子
        
        # 检查是否为花（较亮，白色/粉色）
        if avg_brightness > 180 and saturation < 0.4:
            return 'flower'  # 花
        
        # 检查是否为花（粉色/白色，红色和绿色通道都较高）
        if avg_r > 150 and avg_g > 150 and avg_b > 120 and avg_brightness > 170:
            return 'flower'  # 花
        
        # 默认是果实
        return 'fruit'
        
    except Exception as e:
        print(f"  错误检测内容类型 {img_path.name}: {e}")
        return 'fruit'

def detect_multi_fruit(img_path):
    """
    检测图片是否包含多个水果
    使用多种启发式方法：
    1. 图片尺寸（64x64通常是DeepFruits数据集）
    2. 颜色区域分析（多个水果通常有多个明显的颜色区域）
    3. 边缘检测（多个物体通常有多个分离的边缘区域）
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        # DeepFruits数据集特征：64x64像素
        if width == 64 and height == 64:
            return True, 'deepfruits'
        
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 使用轮廓检测来识别多个物体
        # 应用阈值
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓，计算有效物体数量
        min_area = (width * height) * 0.01  # 至少占图片1%的面积
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # 如果有多个明显分离的轮廓，可能是多个水果
        if len(valid_contours) >= 3:  # 至少3个明显区域
            return True, 'multiple_objects'
        
        # 使用颜色直方图分析
        # 计算RGB三个通道的直方图，如果有多个明显的峰值，可能是多个水果
        hist_r = cv2.calcHist([img_cv], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_cv], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_cv], [2], None, [256], [0, 256])
        
        # 查找直方图的峰值（局部最大值）
        def find_peaks(hist, threshold=0.1):
            peaks = []
            max_val = np.max(hist)
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > max_val * threshold:
                    peaks.append(i)
            return peaks
        
        peaks_r = find_peaks(hist_r)
        peaks_g = find_peaks(hist_g)
        peaks_b = find_peaks(hist_b)
        
        # 如果有多个明显的颜色峰值，可能是多个水果
        if len(peaks_r) >= 3 or len(peaks_g) >= 3 or len(peaks_b) >= 3:
            return True, 'color_peaks'
        
        return False, None
        
    except Exception as e:
        # 如果分析失败，使用简单的尺寸判断
        if width == 64 and height == 64:
            return True, 'deepfruits'
        return False, None

def update_json_annotation(json_path, content_type=None, image_style=None, is_multi_fruit=None, source_dataset=None):
    """更新JSON标注文件，添加所有标注信息"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新images字段
        if 'images' in data and len(data['images']) > 0:
            image_info = data['images'][0]
            
            if content_type:
                image_info['content_type'] = content_type
                # 对于apple类别，更新subcategory
                if 'file_name' in image_info:
                    category = Path(image_info['file_name']).stem.split('_')[0]
                    if category.lower() == 'apple':
                        image_info['subcategory'] = content_type
            
            if image_style:
                image_info['image_style'] = image_style
                image_info['image_type'] = image_style  # 保持兼容性
            
            if is_multi_fruit is not None:
                image_info['is_multi_fruit'] = is_multi_fruit
                if is_multi_fruit:
                    if source_dataset:
                        image_info['source_dataset'] = source_dataset
                    image_info['image_type'] = 'combination'
                else:
                    image_info['image_type'] = 'single_fruit'
        
        # 更新annotations字段
        if 'annotations' in data and len(data['annotations']) > 0:
            if is_multi_fruit is not None:
                data['annotations'][0]['is_multi_fruit'] = is_multi_fruit
            if content_type:
                data['annotations'][0]['content_type'] = content_type
        
        # 更新categories字段
        if 'categories' in data and len(data['categories']) > 0:
            category = data['categories'][0]
            category_name = category.get('name', '').lower()
            
            if content_type and 'apple' in category_name:
                category['subcategory'] = content_type
                category['name'] = f"apple_{content_type}"
            
            if is_multi_fruit is not None:
                category['is_multi_fruit'] = is_multi_fruit
                if is_multi_fruit:
                    category['note'] = 'This image contains multiple fruits, not suitable for single fruit classification'
                elif 'note' in category and 'multiple fruits' in category.get('note', ''):
                    del category['note']
            
            if image_style:
                category['image_style'] = image_style
        
        # 保存更新后的JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"  错误更新 {json_path.name}: {e}")
        return False

def process_category(category_dir, split_name):
    """处理单个类别的所有图片"""
    category = category_dir.name
    json_dir = category_dir / 'json'
    
    # 初始化统计字典
    stats = {
        'content_types': {'flower': 0, 'leaf': 0, 'fruit': 0},
        'styles': {'real': 0, 'painting': 0, 'cartoon': 0, 'sketch': 0},
        'multi_fruit': 0,
        'single_fruit': 0
    }
    
    if not json_dir.exists():
        return stats
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_files = [f for f in category_dir.iterdir() 
                   if f.suffix in image_extensions and f.is_file()]
    
    if not image_files:
        return stats
    
    for img_path in tqdm(image_files, desc=f"  {category}", leave=False):
        # 找到对应的JSON文件
        json_path = json_dir / f"{img_path.stem}.json"
        
        if not json_path.exists():
            continue
        
        # 检测内容类型（特别是apple类别）
        content_type = None
        if category.lower() == 'apple':
            content_type = detect_content_type(img_path, category)
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
        
        # 检测图片风格
        image_style = detect_image_style(img_path)
        stats['styles'][image_style] = stats['styles'].get(image_style, 0) + 1
        
        # 检测单/多水果
        is_multi, source = detect_multi_fruit(img_path)
        if is_multi:
            stats['multi_fruit'] += 1
        else:
            stats['single_fruit'] += 1
        
        # 更新JSON文件
        update_json_annotation(
            json_path,
            content_type=content_type,
            image_style=image_style,
            is_multi_fruit=is_multi,
            source_dataset=source
        )
    
    return stats

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    
    if not dataset_root.exists():
        print(f"错误: 数据集目录不存在: {dataset_root}")
        return
    
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    print("=" * 60)
    print("全面标注所有图片")
    print("=" * 60)
    print()
    
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        
        print(f"处理 {split.upper()} split:")
        print("-" * 60)
        
        categories = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for category_dir in tqdm(categories, desc=f"  {split}"):
            category = category_dir.name
            stats = process_category(category_dir, split)
            
            if category not in all_stats:
                all_stats[category] = {
                    'content_types': {'flower': 0, 'leaf': 0, 'fruit': 0},
                    'styles': {'real': 0, 'painting': 0, 'cartoon': 0, 'sketch': 0},
                    'multi_fruit': 0,
                    'single_fruit': 0
                }
            
            # 合并统计
            for key in ['content_types', 'styles']:
                for subkey in stats[key]:
                    all_stats[category][key][subkey] += stats[key][subkey]
            all_stats[category]['multi_fruit'] += stats['multi_fruit']
            all_stats[category]['single_fruit'] += stats['single_fruit']
        
        print()
    
    # 打印统计信息
    print("=" * 60)
    print("标注统计")
    print("=" * 60)
    for category, stats in sorted(all_stats.items()):
        print(f"\n{category}:")
        if category.lower() == 'apple':
            print(f"  内容类型: {stats['content_types']}")
        print(f"  图片风格: {stats['styles']}")
        print(f"  多水果: {stats['multi_fruit']}, 单水果: {stats['single_fruit']}")

if __name__ == '__main__':
    main()
