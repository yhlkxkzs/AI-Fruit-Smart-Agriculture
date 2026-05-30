#!/usr/bin/env python3
"""
找出标注可能与图片内容不匹配的苹果图片
生成详细报告供人工审查
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_image(img_path):
    """分析图片特征"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]
        
        # 红色区域
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(img_hsv, red_lower1, red_upper1),
            cv2.inRange(img_hsv, red_lower2, red_upper2)
        )
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 粉色区域
        pink_mask = cv2.inRange(img_hsv, np.array([140, 30, 150]), np.array([170, 100, 255]))
        pink_ratio = np.sum(pink_mask > 0) / (width * height)
        
        # 白色区域
        white_mask = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        white_ratio = np.sum(white_mask > 0) / (width * height)
        
        # 绿色区域
        green_mask = cv2.inRange(img_hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        brightness = np.mean(img_hsv[:, :, 2])
        saturation = np.mean(img_hsv[:, :, 1])
        
        return {
            'red_ratio': red_ratio,
            'pink_ratio': pink_ratio,
            'white_ratio': white_ratio,
            'green_ratio': green_ratio,
            'brightness': brightness,
            'saturation': saturation
        }
    except:
        return None

def predict_type(features):
    """基于特征预测类型"""
    if features is None:
        return None
    
    red = features['red_ratio']
    pink = features['pink_ratio']
    white = features['white_ratio']
    green = features['green_ratio']
    brightness = features['brightness']
    
    # 改进的判断逻辑
    # 花：粉色/白色多，亮度高
    if (pink + white) > 0.2 or (brightness > 160 and (pink + white) > 0.1):
        return 'flower'
    
    # 果实：红色多，或绿色多但红色少（未成熟）
    if red > 0.15:
        return 'fruit'
    if green > 0.3 and red < 0.05:
        return 'fruit'
    
    # 叶子：绿色多，红色和粉色都少
    if green > 0.3 and red < 0.05 and (pink + white) < 0.1:
        return 'leaf'
    
    # 默认判断
    if brightness > 150:
        return 'flower'
    elif red > 0.05 or green > 0.2:
        return 'fruit'
    else:
        return 'fruit'  # 默认

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    json_files = sorted(list(json_dir.glob('*.json')))
    
    mismatches = []
    
    print("分析所有文件，找出可能不匹配的标注...")
    print()
    
    for json_path in tqdm(json_files, desc="处理进度"):
        img_stem = json_path.stem
        img_files = list(apple_dir.glob(f"{img_stem}.*"))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            continue
        
        img_path = img_files[0]
        
        # 读取JSON标注
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_type = data['images'][0].get('content_type', 'unknown')
        except:
            continue
        
        # 分析图片
        features = analyze_image(img_path)
        if features is None:
            continue
        
        # 预测类型
        predicted_type = predict_type(features)
        
        if predicted_type and predicted_type != json_type:
            mismatches.append({
                'file': img_path.name,
                'json_type': json_type,
                'predicted_type': predicted_type,
                'features': features
            })
    
    print()
    print("="*80)
    print(f"发现 {len(mismatches)} 个可能不匹配的标注")
    print("="*80)
    print()
    
    # 按类型分组显示
    by_json_type = {}
    for m in mismatches:
        json_type = m['json_type']
        if json_type not in by_json_type:
            by_json_type[json_type] = []
        by_json_type[json_type].append(m)
    
    for json_type in ['flower', 'fruit', 'leaf']:
        if json_type not in by_json_type:
            continue
        
        items = by_json_type[json_type]
        print(f"\n标注为 {json_type.upper()} 但可能是其他类型的文件 ({len(items)} 个):")
        print("-" * 80)
        
        for m in items[:30]:  # 显示前30个
            f = m['features']
            print(f"{m['file']}:")
            print(f"  JSON标注: {m['json_type']} -> 预测: {m['predicted_type']}")
            print(f"  特征: 红={f['red_ratio']:.2%}, 粉={f['pink_ratio']:.2%}, "
                  f"白={f['white_ratio']:.2%}, 绿={f['green_ratio']:.2%}, "
                  f"亮度={f['brightness']:.1f}")
            print()
        
        if len(items) > 30:
            print(f"... 还有 {len(items) - 30} 个文件")
    
    print()
    print("="*80)
    print(f"总计: {len(mismatches)} 个文件可能需要修正")
    print("="*80)

if __name__ == '__main__':
    main()
