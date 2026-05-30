#!/usr/bin/env python3
"""
手动审查苹果图片，生成详细报告供人工判断
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_image_detailed(img_path):
    """详细分析图片"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        # 红色区域（成熟果实）
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 粉色区域（花）
        pink_lower = np.array([140, 30, 150])
        pink_upper = np.array([170, 100, 255])
        pink_mask = cv2.inRange(img_hsv, pink_lower, pink_upper)
        pink_ratio = np.sum(pink_mask > 0) / (width * height)
        
        # 白色区域（花）
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(img_hsv, white_lower, white_upper)
        white_ratio = np.sum(white_mask > 0) / (width * height)
        
        # 绿色区域（叶子/未成熟果实）
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        # 平均颜色
        avg_r = np.mean(img_rgb[:, :, 0])
        avg_g = np.mean(img_rgb[:, :, 1])
        avg_b = np.mean(img_rgb[:, :, 2])
        
        # HSV平均值
        avg_h = np.mean(img_hsv[:, :, 0])
        avg_s = np.mean(img_hsv[:, :, 1])
        avg_v = np.mean(img_hsv[:, :, 2])
        
        # 边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        return {
            'size': (width, height),
            'red_ratio': red_ratio,
            'pink_ratio': pink_ratio,
            'white_ratio': white_ratio,
            'green_ratio': green_ratio,
            'avg_rgb': (avg_r, avg_g, avg_b),
            'avg_hsv': (avg_h, avg_s, avg_v),
            'edge_density': edge_density,
            'brightness': avg_v,
            'saturation': avg_s
        }
    except Exception as e:
        return None

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    json_files = sorted(list(json_dir.glob('*.json')))
    
    # 按标注类型分组
    by_type = defaultdict(list)
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content_type = data['images'][0].get('content_type', 'unknown')
            by_type[content_type].append(json_path)
        except:
            continue
    
    print("="*80)
    print("苹果图片分类统计和样本分析")
    print("="*80)
    print()
    
    for content_type in ['flower', 'fruit', 'leaf']:
        files = by_type.get(content_type, [])
        print(f"{content_type.upper()}: {len(files)} 个文件")
        
        # 分析前10个样本
        print(f"\n前10个 {content_type} 样本的详细特征：")
        print("-" * 80)
        
        for json_path in files[:10]:
            img_stem = json_path.stem
            img_files = list(apple_dir.glob(f"{img_stem}.*"))
            img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if not img_files:
                continue
            
            img_path = img_files[0]
            features = analyze_image_detailed(img_path)
            
            if features:
                print(f"\n{img_path.name}:")
                print(f"  尺寸: {features['size'][0]}x{features['size'][1]}")
                print(f"  红色: {features['red_ratio']:.2%}, 粉色: {features['pink_ratio']:.2%}, "
                      f"白色: {features['white_ratio']:.2%}, 绿色: {features['green_ratio']:.2%}")
                print(f"  平均RGB: ({features['avg_rgb'][0]:.1f}, {features['avg_rgb'][1]:.1f}, {features['avg_rgb'][2]:.1f})")
                print(f"  亮度: {features['brightness']:.1f}, 饱和度: {features['saturation']:.1f}")
                print(f"  边缘密度: {features['edge_density']:.2%}")
        
        print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
