#!/usr/bin/env python3
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def analyze_image_style(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None, "无法读取"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        features['unique_colors'] = len(np.unique(img_rgb.reshape(-1, 3), axis=0))
        blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        features['color_smoothness'] = float(np.mean(np.abs(img_rgb.astype(float) - blurred.astype(float))))
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (h * w))
        features['contrast'] = float(gray.std())
        quantized = (img_rgb // 64) * 64
        features['unique_quantized'] = len(np.unique(quantized.reshape(-1, 3), axis=0))
        features['white_bg_ratio'] = float(np.sum(gray > 240) / (h * w))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_complexity'] = float(laplacian.var())
        features['mean_brightness'] = float(np.mean(gray))
        
        scores = {'real': 0, 'cartoon': 0, 'sketch': 0, 'painting': 0}
        
        # Cartoon
        if features['color_smoothness'] < 8:
            scores['cartoon'] += 2
        if features['unique_quantized'] < 40:
            scores['cartoon'] += 2
        if features['edge_density'] < 0.08:
            scores['cartoon'] += 1
        if features['texture_complexity'] < 500:
            scores['cartoon'] += 1
        if features['unique_colors'] < 10000:
            scores['cartoon'] += 1
            
        # Sketch
        if features['white_bg_ratio'] > 0.2:
            scores['sketch'] += 2
        if features['edge_density'] > 0.18:
            scores['sketch'] += 2
        if features['contrast'] > 55:
            scores['sketch'] += 1
        if features['unique_colors'] < 10000 and features['edge_density'] > 0.12:
            scores['sketch'] += 1
        if features['mean_brightness'] > 200:
            scores['sketch'] += 1
            
        # Real
        if features['unique_colors'] > 10000:
            scores['real'] += 1
        if features['texture_complexity'] > 1000:
            scores['real'] += 1
        if 0.05 < features['edge_density'] < 0.18:
            scores['real'] += 1
        if features['white_bg_ratio'] < 0.15:
            scores['real'] += 1
            
        max_score = max(scores.values())
        predicted = max(scores, key=scores.get)
        if max_score < 3:
            predicted = 'uncertain'
            
        return {
            'predicted': predicted,
            'confidence': max_score,
            'scores': scores,
            'features': features
        }
    except Exception as e:
        return None, str(e)

def main():
    banana_dir = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/banana')
    json_dir = banana_dir / 'json'
    
    json_files = sorted(json_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件\n")
    
    to_fix = []
    uncertain = []
    already_correct = []
    
    for json_path in tqdm(json_files[:3000], desc="分析中"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if not data.get('images'):
                continue
            
            img_info = data['images'][0]
            filename = img_info.get('file_name', '')
            current_style = img_info.get('image_style', 'N/A')
            
            img_path = banana_dir / filename
            if not img_path.exists():
                continue
            
            result = analyze_image_style(img_path)
            if result is None or isinstance(result, tuple):
                continue
            
            predicted = result['predicted']
            confidence = result['confidence']
            
            if predicted == 'uncertain':
                uncertain.append({'file': filename, 'current': current_style, 'scores': result['scores']})
            elif current_style == 'N/A':
                to_fix.append({'file': filename, 'current': 'N/A', 'predicted': predicted, 'confidence': confidence})
            elif current_style != predicted:
                to_fix.append({'file': filename, 'current': current_style, 'predicted': predicted, 'confidence': confidence, 'scores': result['scores']})
            else:
                already_correct.append(filename)
        except:
            pass
    
    print(f"\n✓ 已正确: {len(already_correct)} 个")
    print(f"⚠ 需修正: {len(to_fix)} 个")
    print(f"? 不确定: {len(uncertain)} 个\n")
    
    if to_fix:
        print("需要修正的文件（按置信度排序前50）:")
        print("-" * 80)
        to_fix_sorted = sorted(to_fix, key=lambda x: -x['confidence'])
        for item in to_fix_sorted[:50]:
            print(f"{item['file']:<35} {item['current']:<10} -> {item['predicted']:<10} (置信度:{item['confidence']})")
    
    # 保存报告
    report_path = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/banana_fix_report_v2.txt')
    with open(report_path, 'w') as f:
        f.write("香蕉图片标注修正报告 v2\n")
        f.write("="*80 + "\n\n")
        f.write(f"已正确: {len(already_correct)} 个\n")
        f.write(f"需修正: {len(to_fix)} 个\n")
        f.write(f"不确定: {len(uncertain)} 个\n\n")
        f.write("需修正文件列表:\n")
        for item in sorted(to_fix, key=lambda x: -x['confidence']):
            f.write(f"{item['file']}: {item['current']} -> {item['predicted']}\n")
    print(f"\n报告已保存: {report_path}")

if __name__ == '__main__':
    main()
