#!/usr/bin/env python3
"""
分析banana文件夹中所有图片的标注，并生成详细的markdown报告
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

def analyze_image_style(img_path):
    """分析图片风格特征"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 基本特征
        unique_colors = len(np.unique(img_rgb.reshape(-1, 3), axis=0))
        
        # 颜色平滑度
        blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        color_smoothness = np.mean(np.abs(img_rgb.astype(float) - blurred.astype(float)))
        
        # 纹理分析
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_variance = np.var(gradient_magnitude)
        
        # 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 对比度
        contrast = gray.std()
        
        # 笔触效果
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        brush_stroke_variance = np.mean(local_variance)
        
        # 颜色量化
        quantized = (img_rgb // 32) * 32
        unique_quantized = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        # 风格判断 - 调整后的逻辑，优先判断real
        scores = {'cartoon': 0, 'painting': 0, 'sketch': 0, 'real': 0}
        
        # Real特征：颜色多、纹理丰富（这是最关键的判断标准）
        # 如果颜色数量足够多且纹理丰富，优先判断为real
        if unique_colors > 10000 and texture_variance > 3000:
            scores['real'] += 5  # 高权重
        elif unique_colors > 5000 and texture_variance > 2000:
            scores['real'] += 3
        elif unique_colors > 2000 and texture_variance > 1000:
            scores['real'] += 2
        
        # 考虑图片尺寸：小图片可能颜色数较少，但不代表是cartoon
        # 如果图片很小（<100x100），放宽颜色数量要求
        if h * w < 10000:  # 小于100x100
            if unique_colors > 1000 and texture_variance > 2000:
                scores['real'] += 2
        
        # Cartoon特征：颜色非常少、纹理简单、且平滑度高
        # 只有在颜色数量很少且纹理简单时才判断为cartoon
        if unique_colors < 300 and texture_variance < 2000:
            scores['cartoon'] += 3
        elif unique_colors < 1000 and texture_variance < 3000 and color_smoothness < 5:
            scores['cartoon'] += 2
        elif unique_colors < 2000 and unique_quantized < 50:
            scores['cartoon'] += 1
        
        # Painting特征：笔触非常明显
        if brush_stroke_variance > 800:
            scores['painting'] += 4
        elif brush_stroke_variance > 500:
            scores['painting'] += 2
        elif brush_stroke_variance > 300:
            scores['painting'] += 1
        
        # Sketch特征：边缘非常明显、对比度很高，且颜色较少
        # 只有在边缘密度很高且颜色数量较少时才判断为sketch
        if edge_density > 0.25 and unique_colors < 5000:
            scores['sketch'] += 4
        elif edge_density > 0.20 and unique_colors < 3000:
            scores['sketch'] += 3
        elif edge_density > 0.15 and unique_colors < 2000 and contrast > 70:
            scores['sketch'] += 2
        elif edge_density > 0.12 and unique_colors < 1000:
            scores['sketch'] += 1
        
        predicted_style = max(scores, key=scores.get)
        max_score = scores[predicted_style]
        
        return {
            'predicted_style': predicted_style,
            'confidence': max_score,
            'scores': scores,
            'features': {
                'unique_colors': unique_colors,
                'color_smoothness': round(color_smoothness, 2),
                'texture_variance': round(texture_variance, 2),
                'edge_density': round(edge_density, 4),
                'contrast': round(contrast, 2),
                'brush_stroke_variance': round(brush_stroke_variance, 2),
                'unique_quantized': unique_quantized
            }
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    banana_dir = dataset_root / 'train' / 'banana'
    json_dir = banana_dir / 'json'
    
    json_files = sorted(json_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")
    print("开始分析...")
    
    results = []
    style_stats = Counter()
    annotation_issues = []
    
    # 分析所有文件（采样前1000个进行详细分析，其余只统计）
    sample_size = min(1000, len(json_files))
    detailed_files = json_files[:sample_size]
    
    for json_path in tqdm(detailed_files, desc="分析中"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data.get('images'):
                continue
            
            img_info = data['images'][0]
            filename = img_info.get('file_name', '')
            img_path = banana_dir / filename
            
            current_style = img_info.get('image_style', 'N/A')
            current_content = img_info.get('content_type', 'N/A')
            
            # 分析图片
            analysis = analyze_image_style(img_path) if img_path.exists() else None
            
            if analysis and 'error' not in analysis:
                predicted_style = analysis['predicted_style']
                confidence = analysis['confidence']
                
                # 检查标注是否一致
                style_match = (current_style == predicted_style) if current_style != 'N/A' else None
                
                if current_style != 'N/A' and not style_match:
                    annotation_issues.append({
                        'file': filename,
                        'current': current_style,
                        'predicted': predicted_style,
                        'confidence': confidence
                    })
                
                results.append({
                    'filename': filename,
                    'current_style': current_style,
                    'current_content': current_content,
                    'predicted_style': predicted_style,
                    'confidence': confidence,
                    'features': analysis['features'],
                    'scores': analysis['scores'],
                    'style_match': style_match
                })
                
                style_stats[predicted_style] += 1
            else:
                results.append({
                    'filename': filename,
                    'current_style': current_style,
                    'current_content': current_content,
                    'error': analysis.get('error', '无法分析') if analysis else '文件不存在'
                })
        except Exception as e:
            results.append({
                'filename': json_path.name,
                'error': str(e)
            })
    
    # 统计所有文件的标注
    all_annotations = Counter()
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if data.get('images'):
                all_annotations[data['images'][0].get('image_style', 'N/A')] += 1
        except:
            pass
    
    # 生成Markdown报告
    report_path = dataset_root / 'banana_annotation_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Banana数据集标注审查报告\n\n")
        f.write(f"生成时间: {Path(__file__).stat().st_mtime}\n\n")
        f.write("---\n\n")
        
        # 总体统计
        f.write("## 1. 总体统计\n\n")
        f.write(f"- **总文件数**: {len(json_files)}\n")
        f.write(f"- **详细分析文件数**: {sample_size}\n")
        f.write(f"- **发现标注问题数**: {len(annotation_issues)}\n\n")
        
        f.write("### 当前标注分布（全部文件）\n\n")
        f.write("| 风格 | 数量 | 占比 |\n")
        f.write("|------|------|------|\n")
        total = sum(all_annotations.values())
        for style, count in sorted(all_annotations.items(), key=lambda x: -x[1]):
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"| {style} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        f.write("### 预测风格分布（详细分析样本）\n\n")
        f.write("| 风格 | 数量 | 占比 |\n")
        f.write("|------|------|------|\n")
        total_predicted = sum(style_stats.values())
        for style, count in sorted(style_stats.items(), key=lambda x: -x[1]):
            percentage = (count / total_predicted * 100) if total_predicted > 0 else 0
            f.write(f"| {style} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        # 标注问题
        if annotation_issues:
            f.write("## 2. 标注问题列表\n\n")
            f.write("以下文件的标注风格与图片特征分析结果不一致：\n\n")
            f.write("| 文件名 | 当前标注 | 预测风格 | 置信度 | 建议 |\n")
            f.write("|--------|----------|----------|--------|------|\n")
            for issue in sorted(annotation_issues, key=lambda x: -x['confidence'])[:100]:  # 只显示前100个
                f.write(f"| {issue['file']} | {issue['current']} | {issue['predicted']} | {issue['confidence']} | 建议改为 {issue['predicted']} |\n")
            f.write("\n")
        
        # 详细分析结果（前50个）
        f.write("## 3. 详细分析结果（前50个文件）\n\n")
        f.write("| 文件名 | 当前风格 | 预测风格 | 置信度 | 特征 |\n")
        f.write("|--------|----------|----------|--------|------|\n")
        
        for result in results[:50]:
            if 'error' in result:
                f.write(f"| {result['filename']} | - | - | - | 错误: {result['error']} |\n")
            else:
                features = result['features']
                feature_str = f"颜色:{features['unique_colors']}, 平滑度:{features['color_smoothness']}, 边缘:{features['edge_density']:.3f}"
                match_str = "✓" if result.get('style_match') else ("✗" if result.get('style_match') is False else "?")
                f.write(f"| {result['filename']} | {result['current_style']} | {result['predicted_style']} | {result['confidence']} | {match_str} {feature_str} |\n")
        f.write("\n")
        
        # 标注规则说明
        f.write("## 4. 标注规则说明\n\n")
        f.write("### 风格判断标准\n\n")
        f.write("1. **real（真实照片）**:\n")
        f.write("   - 唯一颜色数 > 2000\n")
        f.write("   - 纹理方差 > 5000\n")
        f.write("   - 颜色平滑度 > 20\n\n")
        
        f.write("2. **cartoon（卡通）**:\n")
        f.write("   - 唯一颜色数 < 500\n")
        f.write("   - 颜色平滑度 < 10\n")
        f.write("   - 纹理方差 < 5000\n")
        f.write("   - 量化后颜色数 < 100\n\n")
        
        f.write("3. **painting（油画/绘画）**:\n")
        f.write("   - 笔触方差 > 500\n")
        f.write("   - 有明显的笔触效果\n\n")
        
        f.write("4. **sketch（素描/简笔画）**:\n")
        f.write("   - 边缘密度 > 0.15\n")
        f.write("   - 对比度 > 70\n\n")
        
        f.write("### 标注建议\n\n")
        f.write("- 如果预测风格置信度 >= 3，建议采用预测风格\n")
        f.write("- 如果置信度较低（< 2），建议人工审查\n")
        f.write("- 对于不确定的情况，可以标记为需要人工确认\n\n")
        
        # 样本文件列表
        f.write("## 5. 所有文件标注汇总\n\n")
        f.write("由于文件数量较多（7346个），详细分析结果请查看脚本输出或数据库。\n\n")
        f.write("### 按风格分类的文件列表（前100个）\n\n")
        
        by_style = defaultdict(list)
        for result in results:
            if 'predicted_style' in result:
                by_style[result['predicted_style']].append(result['filename'])
        
        for style in ['real', 'cartoon', 'painting', 'sketch']:
            if style in by_style:
                f.write(f"#### {style.upper()} 风格 ({len(by_style[style])} 个文件)\n\n")
                for filename in sorted(by_style[style])[:50]:
                    f.write(f"- {filename}\n")
                if len(by_style[style]) > 50:
                    f.write(f"- ... 还有 {len(by_style[style]) - 50} 个文件\n")
                f.write("\n")
    
    print(f"\n报告已生成: {report_path}")
    print(f"发现 {len(annotation_issues)} 个可能的标注问题")

if __name__ == '__main__':
    main()
